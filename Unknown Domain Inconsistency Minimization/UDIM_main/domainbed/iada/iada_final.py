import os
import torch
import torchvision
from torchvision.utils import save_image
from .util import enable_running_stats, disable_running_stats
import contextlib
from torch.distributed import ReduceOp
import torch.nn.functional as F
from backpack import extend, backpack
from backpack.extensions import Variance, BatchL2Grad, BatchGrad

from torch import nn
from torch.nn import CrossEntropyLoss, Linear
from .functorch_module import compute_distance_grads_var, per_sample_gradient, get_grads_var
from functorch.experimental import replace_all_batch_norm_modules_

from collections import OrderedDict
# import matplotlib.pyplot as plt

class IADA_FINAL(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, rho_scheduler, hparams,args,classifier, adaptive=False, perturb_eps=1e-12, grad_reduce='mean',**kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(IADA_FINAL, self).__init__(params, defaults)

        self.hparams = hparams
        self.model = model
        self.classifier = classifier

        self.loss_fn = extend(nn.CrossEntropyLoss(reduction='none'))


        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.args = args
        replace_all_batch_norm_modules_(self.model)


        # initialize self.rho_t
        self.update_rho_t()
        
        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        self.rho_s = self.rho_t

        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        '''Norm calculation over whole parameters'''
        grad_norm = self._grad_norm( weight_adaptive = self.adaptive)

        for group in self.param_groups:
            scale = (rho / (grad_norm + self.perturb_eps))

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
    
    @torch.no_grad()
    def perturb_instances(self, rho=0.0, cur_step=-1):
        '''Norm calculation over whole parameters'''
        if self.args.advstyle:
            self.adv_optim.step()
            self.adv_optim.zero_grad()
            self.base_optimizer.zero_grad(set_to_none=True)
        else:
            # plot figure
            self.syn_norm_grad = F.normalize(self.syn_inputs.grad, p=2.0, dim=0, eps=1e-12)
            if cur_step in [0,500,1000,1500,2000,2500,3000,3500,4000,4500,4999]:
                print(cur_step)
                # directory
                env_name = 0
                curdir = '/home/aailab/data2/SAM_DG_rebuttal/figure/env_'+str(env_name)+'/step_' + str(cur_step) +'/'
                while os.path.exists(curdir):
                    env_name+=1
                    curdir = '/home/aailab/data2/SAM_DG_rebuttal/figure/env_'+str(env_name)+'/step_'+str(cur_step)+'/'
                os.mkdir(curdir)

                inputs_ = self.inputs.clone().cpu()
                syn_inputs_ = self.syn_inputs.clone().detach().cpu()
                syn_grad = self.syn_norm_grad.clone().cpu()

                for i in range(inputs_.shape[0]):
                    input = inputs_[i]
                    save_image(input, curdir+str(i)+'_real.jpg')
                    for slr in [0.1, 0.5, 1., 5., 10., 50., 100.]:
                        syn_input =syn_inputs_[i]+syn_grad[i]*slr
                        save_image(syn_input, curdir+str(i)+'_perturb_'+str(slr)+'_syn.jpg')

            self.syn_inputs += self.syn_norm_grad * self.args.syn_lr
            self.syn_inputs.grad.zero_()
            self.base_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                sam_grad = self.state[p]['old_g'] * 0.5 - p.grad * 0.5
                p.grad.data.add_(sam_grad)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),p=2)
        return norm

    @torch.no_grad()
    def _grad_norm_tuple(self,grad_tuple, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                    torch.stack([
                        grad.norm(p=2)
                        for grad in grad_tuple
                        if grad is not None]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets,syn_inputs=None, **kwargs):
        def get_grad():
            self.base_optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                if syn_inputs != None:
                    if self.args.advstyle:
                        # Get style feature and normalized image
                        self.inputs = inputs
                        self.targets = targets
                        self.mu = self.inputs.mean(dim=[2, 3], keepdim=True)
                        self.var = self.inputs.var(dim=[2, 3], keepdim=True)
                        self.sig = (self.var + 1e-5).sqrt()
                        self.mu, self.sig = self.mu.detach(), self.sig.detach()
                        self.input_normed = (self.inputs - self.mu) / self.sig
                        self.input_normed = self.input_normed.detach().clone()

                        # Set learnable style feature and adv optimizer
                        self.adv_mu, self.adv_sig = self.mu, self.sig
                        self.adv_mu.requires_grad_(True)
                        self.adv_sig.requires_grad_(True)
                        self.adv_optim = torch.optim.SGD(params=[self.adv_mu, self.adv_sig], lr=self.args.syn_lr,
                                                         momentum=0, weight_decay=0)

                        # Optimize adversarial style feature
                        self.adv_optim.zero_grad()
                        self.syn_inputs = self.input_normed * self.adv_sig + self.adv_mu
                        outputs = self.model(self.syn_inputs)
                        loss = 0
                        loss += loss_fn(outputs, targets, **kwargs)
                        g_syn = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                        scale = self.rho_t / self._grad_norm_tuple(g_syn)
                        self.origin_grad_eps = [g.data.clone().detach() * scale for g in g_syn]
                        g_norm = torch.norm(torch.stack([g.norm(p=2) for g in g_syn]), p=2)
                        loss += self.rho_s * g_norm
                        loss_value = loss.data.clone().detach()
                        (-loss).backward()
                        return outputs, loss_value

                    else:
                        self.inputs = inputs
                        self.targets = targets
                        self.syn_inputs = syn_inputs

                        outputs = self.model(self.syn_inputs)
                        loss = 0
                        loss += loss_fn(outputs, targets, **kwargs)
                        g_syn = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                        scale = self.rho_t / self._grad_norm_tuple(g_syn)
                        '''Epsilon reflected gradient'''
                        self.origin_grad_eps = [g.data.clone().detach() * scale for g in g_syn]
                        g_norm = torch.norm(torch.stack([g.norm(p=2) for g in g_syn]), p=2)
                        loss += self.rho_s * g_norm
                        loss_value = loss.data.clone().detach()
                        loss.backward()

                        return outputs, loss_value

                else:
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets, **kwargs)
                    loss_value = loss.data.clone().detach()
                    loss.backward()
                    return outputs, loss_value

        #얘가 받는 건 output 및 detach된 loss인데 어떻게 gradient를 받지?
        self.forward_backward_func = get_grad

    @torch.no_grad()
    def sam_compute_grad(self,syn=False):
        with torch.no_grad():
            for p, v in zip(self.model.parameters(), self.origin_grad_eps):
                p.add_(v)
        with torch.enable_grad():
            loss = F.cross_entropy(self.model(self.inputs), self.targets)
            self.base_optimizer.zero_grad()
            loss.backward()
        # restore original network params (다시 update 전으로 돌려놓은 다음)
        with torch.no_grad():
            for p, v in zip(self.model.parameters(), self.origin_grad_eps):
                p.sub_(v)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if syn:
                    self.state[p]["syn_grad_sam"] = p.grad.data.clone().detach()
                else:
                    self.state[p]["origin_grad_sam"] = p.grad.data.clone().detach()

    def l2_between_dicts(self,dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
                torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
                torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()

    def inconsistency_compute_grad_previous(self):
        with torch.enable_grad():
            inconsistency_loss = 0
            self.base_optimizer.zero_grad(set_to_none=True)
            original_variance = []

            concat_inputs = torch.cat((self.inputs, self.syn_inputs), 0)

            per_sample_grad_origin = per_sample_gradient(self.model, self.inputs, self.targets)
            for i in range(len(per_sample_grad_origin)):
                if len(per_sample_grad_origin[i].shape) == 3:
                    original_variance.append(get_grads_var(per_sample_grad_origin[i].view(per_sample_grad_origin[i].size(0), -1)).data.clone().detach())

            # self.base_optimizer.zero_grad(set_to_none=True)
            for param in self.model.parameters():
                param.grad = None

            per_sample_grad_syn = per_sample_gradient(self.model, self.syn_inputs, self.targets)

            for i in range(len(per_sample_grad_syn)):
                if len(per_sample_grad_syn[i].shape) == 3:
                    inconsistency_loss+=torch.norm(per_sample_grad_syn[i])
                    grad_var_syn = get_grads_var(per_sample_grad_syn[i].view(per_sample_grad_syn[i].size(0), -1))
                    inconsistency_loss+=torch.norm(grad_var_syn-origin_variance[0])

            self.base_optimizer.zero_grad(set_to_none=True)
            print('========= inconsistency loss ========')
            print(inconsistency_loss)
            inconsistency_loss.backward()

            self.inconsistency_loss = inconsistency_loss.data.clone().detach()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["syn_grad_ilc"] = p.grad.data.clone().detach()
                    # print(torch.norm(self.state[p]["syn_grad_ilc"]))

    def _get_grads(self, logits, y):
        self.base_optimizer.zero_grad()
        loss = self.loss_fn(logits, y).sum()
        with backpack(BatchGrad(),BatchL2Grad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict([
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ])

        dict_grad_norms = OrderedDict([
                (name, weights.batch_l2.clone().view(weights.batch_l2.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ])

        return dict_grads, dict_grad_norms

    def _get_grads_var(self, dict_grads):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(2)]
        self.param_dim = []

        for name, _grads in dict_grads.items():
            last_idx = len(_grads)//2
            origin_grads = _grads[:last_idx]
            syn_grads = _grads[last_idx:]
            self.param_dim.append(origin_grads.shape[1])
            origin_mean = origin_grads.mean(dim=0, keepdim=True)
            origin_grads_centered = origin_grads - origin_mean
            origin_grads_var = (origin_grads_centered).pow(2).mean(dim=0)

            grads_var_per_domain[0][name] = origin_grads_var

            syn_mean = syn_grads.mean(dim=0, keepdim=True)
            syn_grads_centered = syn_grads - origin_mean
            syn_grads_var = (syn_grads_centered).pow(2).mean(dim=0)

            grads_var_per_domain[1][name] = syn_grads_var
        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):
        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(2)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )
        penalty = 0
        for domain_id in range(2):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / 2

    def _grads_norm_syn(self, dict_grad_norm):
        norm_loss = 0
        idx = 0
        for name, _grads_norm in dict_grad_norm.items():
            # print(_grads_norm[(len(_grads_norm)//2):])
            norm_loss += _grads_norm[(len(_grads_norm)//2):].mean()/self.param_dim[idx]
            idx +=1
        return norm_loss

    def inconsistency_compute_grad(self):
        with torch.enable_grad():
            inconsistency_loss = 0

            self.syn_inputs = self.syn_inputs.detach().clone()
            if self.args.domain_wise_inconsistency:
                domain_batch_num = len(self.inputs)//self.args.num_domains
                for i in range(self.args.num_domains):
                    concat_inputs = torch.cat((self.inputs[i*domain_batch_num:(i+1)*domain_batch_num], self.syn_inputs[i*domain_batch_num:(i+1)*domain_batch_num]), 0)
                    concat_targets = torch.cat((self.targets[i*domain_batch_num:(i+1)*domain_batch_num],self.targets[i*domain_batch_num:(i+1)*domain_batch_num]),0)


                    logits = self.model(concat_inputs)
                    dict_sample_grad, dict_grad_norm = self._get_grads(logits, concat_targets)

                    grads_var_per_domain_list = self._get_grads_var(dict_sample_grad)
                    inconsistency_loss += self._compute_distance_grads_var(grads_var_per_domain_list)
                    inconsistency_loss+= self._grads_norm_syn(dict_grad_norm)

            else:
                concat_inputs = torch.cat((self.inputs,self.syn_inputs), 0)
                concat_targets = torch.cat((self.targets,self.targets), 0)
                logits = self.model(concat_inputs)
                dict_sample_grad, dict_grad_norm = self._get_grads(logits, concat_targets)

                grads_var_per_domain_list = self._get_grads_var(dict_sample_grad)
                inconsistency_loss += self._compute_distance_grads_var(grads_var_per_domain_list)
                inconsistency_loss += self._grads_norm_syn(dict_grad_norm)

            self.base_optimizer.zero_grad(set_to_none=True)
            inconsistency_loss.backward()
            self.inconsistency_loss = inconsistency_loss.data.clone().detach()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["syn_grad_ilc"] = p.grad.data.clone().detach()

    @torch.no_grad()
    def merge_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'origin_grad_sam' in self.state[p].keys():
                    p.grad.data = self.state[p]['origin_grad_sam'] + self.args.worst_weight*self.state[p]['syn_grad_ilc']


    @torch.no_grad()
    def step(self, cur_step=-1, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()
            # perturb data instances
            self.perturb_instances(rho=self.rho_s, cur_step=cur_step)

            # store sam gradient from original input
            self.sam_compute_grad()

            # store ilc gradient
            self.inconsistency_compute_grad()
            # disable running stats for second pass
            disable_running_stats(self.model)

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.merge_grad()
        self.base_optimizer.step()
        self.base_optimizer.zero_grad(set_to_none=True)

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value, self.inconsistency_loss
