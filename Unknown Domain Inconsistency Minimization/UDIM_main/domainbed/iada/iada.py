import torch
from .util import enable_running_stats, disable_running_stats
import contextlib
from torch.distributed import ReduceOp
import torch.nn.functional as F

class IADA(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, rho_scheduler, hparams,args, adaptive=False, perturb_eps=1e-12, grad_reduce='mean',**kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(IADA, self).__init__(params, defaults)

        self.hparams = hparams
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.args = args

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
    def perturb_instances(self, rho=0.0):
        '''Norm calculation over whole parameters'''

        self.syn_inputs += self.syn_inputs.grad * self.args.syn_lr
        self.syn_inputs.grad.zero_()

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
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]),
                    p=2
               )

            # for group in self.param_groups:
            #     for p in group['params']:
            #         print(p.shape)

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
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                if syn_inputs != None:
                    self.inputs = inputs
                    self.targets = targets
                    self.syn_inputs = syn_inputs
                    outputs = self.model(self.syn_inputs)
                    loss=0
                    loss+= loss_fn(outputs, targets, **kwargs)
                    print('=================================')
                    print("classification loss : %.3f" % loss)
                    print('=================================')
                    g_syn = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                    g_norm = torch.norm(torch.stack([g.norm(p=2) for g in g_syn]),p=2)
                    print('=================================')
                    print("gradient norm : %.3f" % g_norm)
                    print("corresponding rho : %.5f" % self.rho_t)
                    print('=================================')
                    loss+= self.rho_t*g_norm

                else:
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets, **kwargs)

            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def sam_compute_grad(self,inputs,syn=False):
        with torch.enable_grad():
            loss = F.cross_entropy(self.model(inputs), self.targets)
            grad_w = torch.autograd.grad(loss, self.model.parameters())
            scale = self.rho_t / self._grad_norm_tuple(grad_w)
            '''Epsilon reflected gradient'''
            eps = [g * scale for g in grad_w]
            with torch.no_grad():
                for p, v in zip(self.model.parameters(), eps):
                    p.add_(v)
        with torch.enable_grad():
            loss = F.cross_entropy(self.model(inputs), self.targets)
            self.base_optimizer.zero_grad()
            loss.backward()
        with torch.no_grad():
            for p, v in zip(self.model.parameters(), eps):
                p.sub_(v)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if syn:
                    self.state[p]["syn_g"] = p.grad.data.clone()
                else:
                    self.state[p]["origin_g"] = p.grad.data.clone()

    @torch.no_grad()
    def sam_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'origin_g' in self.state[p].keys():
                    p.grad.data = self.state[p]['origin_g'] + self.args.worst_weight*self.state[p]['syn_g']

    @torch.no_grad()
    def step(self, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()
            # perturb data instances
            self.perturb_instances(rho=self.rho_s)
            self.sam_compute_grad(self.inputs)
            self.sam_compute_grad(self.syn_inputs,syn=True)

            # disable running stats for second pass
            disable_running_stats(self.model)
        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.sam_step()
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value
