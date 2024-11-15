# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import defaultdict, OrderedDict

from torch.nn import Module
import gc
import functorch
from functorch import make_functional_with_buffers, vmap, grad
import torch.nn.functional as F
from functorch import hessian



def per_sample_gradient(model,data, targets):
    def loss_fn(predictions, targets):
        return F.cross_entropy(predictions, targets)

    # should be fixed if network get different
    relevant_param_indices = [159]
    # assert len(model.parameters()[159].shape) == 2

    fmodel, params, buffers = make_functional_with_buffers(model)

    def split(params, relevant_param_indices):
        relevant_params = []
        other_params = []
        for i, param in enumerate(params):
            if i in relevant_param_indices:
                relevant_params.append(param)
            else:
                other_params.append(param)
        return tuple(relevant_params), tuple(other_params)

    def combine(relevant_params, other_params, relevant_param_indices):
        relevant_params_iter = iter(relevant_params)
        other_params_iter = iter(other_params)
        num_total_params = len(relevant_params) + len(other_params)
        params = []
        for i in range(num_total_params):
            if i in relevant_param_indices:
                params.append(next(relevant_params_iter))
            else:
                params.append(next(other_params_iter))
        return tuple(params)
    #
    # def compute_loss_stateless_model(params, buffers, sample, target):
    #     batch = sample.unsqueeze(0)
    #     targets = target.unsqueeze(0)
    #     predictions = fmodel(params, buffers, batch)
    #     loss = loss_fn(predictions, targets)
    #     return loss

    def compute_loss_stateless_model(relevant_params, other_params, buffers, sample, target):
        params = combine(relevant_params, other_params, relevant_param_indices)
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        # print(target.shape)
        predictions = fmodel(params, buffers, batch)
        loss = loss_fn(predictions, targets)
        return loss

    relevant_params, other_params = split(params, relevant_param_indices)
    # print(relevant_params)
    # print(other_params)

    ft_compute_grad = grad(compute_loss_stateless_model)
    # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0),randomness='different')
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None,0,0),randomness='different')

    ft_per_sample_grads = ft_compute_sample_grad(relevant_params, other_params, buffers, data, targets)

    return ft_per_sample_grads

# def per_sample_hessian(model,data, targets):
#     def loss_fn(predictions, targets):
#         return F.cross_entropy(predictions, targets)
#
#     def compute_loss_stateless_model(params, buffers, sample, target):
#         batch = sample.unsqueeze(0)
#         targets = target.unsqueeze(0)
#         predictions = fmodel(params, buffers, batch)
#         loss = loss_fn(predictions, targets)
#         return loss
#
#     fmodel, params, buffers = make_functional_with_buffers(model)
#     compute_batch_hessian = hessian(compute_loss_stateless_model)
#     ft_compute_sample_hessian = vmap(compute_batch_hessian, in_dims=(None, None, 0, 0))
#     ft_per_sample_hessian = ft_compute_sample_hessian(params, buffers, data, targets)
#
#     return ft_per_sample_hessian

def per_sample_hessian_classifier(model,data, targets):
    feature = model.forward(data,only_return_features=True)

    def loss_fn(predictions, targets):
        return F.cross_entropy(predictions, targets)

    def compute_loss_stateless_model(params, buffers, sample, target):
        predictions = fmodel(params, buffers, sample.unsqueeze(0))
        loss = loss_fn(predictions, target.unsqueeze(0))
        return loss

    fmodel, params, buffers = make_functional_with_buffers(model)
    sample_hessian = hessian(compute_loss_stateless_model,argnums=0)(params,buffers,data,targets)

    return sample_hessian


def memory_cleanup(module: Module) -> None:
    """Remove I/O stored by backpack during the forward pass.

    Deletes the attributes created by `hook_store_io`.

    Args:
        module: current module
    """
    if hasattr(module, "output"):
        delattr(module, "output")
    i = 0
    while hasattr(module, "input{}".format(i)):
        delattr(module, "input{}".format(i))
        i += 1



def get_grads_var(grads_temp):
    # grads var per domain
    env_mean = grads_temp.mean(dim=0, keepdim=True)
    env_grads_centered = grads_temp - env_mean
    '''
    # moving average
    for domain_id in range(num_domains):
        grads_var_per_domain[domain_id] = ema_per_domain[domain_id].update(
            grads_var_per_domain[domain_id]
        )
    '''
    return env_grads_centered.pow(2).mean(dim=0)


def compute_distance_grads_var(tensor_1,tensor_2):
    penalty = 0
    penalty += l2_between_tensors(tensor_1, tensor_2)
    return penalty

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).sum()

def l2_between_lists(list_1, list_2):
    assert len(list_1) == len(list_2)
    # dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    # dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in list_1])) -
        torch.cat(tuple([t.view(-1) for t in list_2]))
    ).pow(2).sum()

def l2_between_tensors(tensor_1, tensor_2):
    assert len(tensor_1) == len(tensor_2)
    # dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    # dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (tensor_1-tensor_2).pow(2).mean()

#
# def predict( x):
#     return network(x)
