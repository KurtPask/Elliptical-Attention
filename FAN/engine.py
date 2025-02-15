# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
from fvcore.nn import FlopCountAnalysis
# import wandb
import numpy as np
from autoattack import AutoAttack
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import torch.nn.functional as F

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import pdb

@torch.no_grad()
def evaluate(data_loader, model, device, attn_only=False, batch_limit=0, attack='none', eps=1/255, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # i = 0
    if not isinstance(batch_limit, int) or batch_limit < 0:
        batch_limit = 0
    attn = []
    pi = []
    if attack=='auto':
        adversary = AutoAttack(model, norm='Linf', eps=6/255, version='standard')
    for i, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if i >= batch_limit > 0:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bs = images.shape[0]
        if attack=='auto':
            images = adversary.run_standard_evaluation(images, target, bs=bs)
        elif attack == 'fgm':
            images = fast_gradient_method(model, images, eps, np.inf)
        elif attack == 'pgd':
            images = projected_gradient_descent(model, images, eps, 0.15 * eps, 20, np.inf)

        with torch.cuda.amp.autocast():
            if attn_only:
                output, _aux = model(images)
                attn.append(_aux[0].detach().cpu().numpy())
                pi.append(_aux[1].detach().cpu().numpy())
                del _aux
            else:
                output = model(images)
            loss = criterion(output, target)

        # print(output.shape,target.shape)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        r = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        # wandb.log({f'test_{k}': meter.global_avg for k, meter in metric_logger.meters.items()})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # for k, meter in metric_logger.meters.items():
    #     wandb.log({f'test_{k}': meter.global_avg, 'epoch':epoch})

    if attn_only:
        return r, (attn, pi)
    return r

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to('cpu').numpy()

@torch.no_grad()
def get_predictions(loader, net=None, mask=None):
    confidence = []
    correct = []
    num_correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)[:,mask]

            # accuracy
            pred = output.data.max(1)[1]
            num_correct += pred.eq(target.data).sum().item()

            confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())

    return np.array(confidence), np.array(correct), num_correct

@torch.no_grad()
def get_imagenet_a_results(loader, net, mask, epoch=0):
    net.eval()
    confidence, correct, num_correct = get_predictions(loader, net, mask)
    acc = num_correct / len(loader.dataset)
    print('Accuracy (%):', round(100*acc, 4))
    wandb.log({'ImnetA: Acc(%)':round(100*acc, 4), 'epoch':epoch})
    return round(100*acc, 4)
