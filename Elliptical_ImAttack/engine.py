"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import pdb
#import cleverhans

import torch
import os
import statistics

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import numpy as np
import wandb

from losses import DistillationLoss
import utils
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
#from cleverhans.torch.attacks.spsa import spsa
from spsa import spsa
from autoattack.autoattack import AutoAttack
#from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
#from cleverhans.torch.attacks.noise import noise
#from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
#from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
#from fgsm import fgsm
#from torchattacks import FGSM

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # pdb.set_trace()
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, ".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate(data_loader, model, device, attack='none', eps=0.03, use_wandb = False, output_dir = None):
    criterion = torch.nn.CrossEntropyLoss()

    if attack.startswith('auto'):
        log_path = os.makedirs(os.path.join(output_dir, 'verbose_log'), exist_ok=True)
        adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard', verbose = True, log_path=log_path)
        print(f'Auto attack using perturbation budget {eps}')
    
    if attack == 'fgsm':
        eps = eps/255
        adversary = FGSM(model, eps = eps)
        adversary.targetted = True
        print(f'Using FGSM with perturbation budget {eps}')
    
    if attack == 'fgm':
        #eps = eps/255
        print(f'Using fgm attack with budget {eps}')

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    i=-1
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bs = images.shape[0]
        i += 1
        if i%10==0: print('evalloop', i)
        if attack != 'none':
            # bad_indices = np.random.choice(bs, bs, replace=False)
            if attack == 'fgm':
                images = fast_gradient_method(model, images, eps, np.inf)
            elif attack == 'fgsm':
                #breakpoint()
                images = adversary(images, target)
            elif attack == 'pgd':
                images = projected_gradient_descent(model, images, eps, 0.15 * eps, 20, np.inf)
            elif attack == 'sld':
                images = sparse_l1_descent(model, images)
            elif attack == 'noise':
                images = noise(images)
            elif attack == 'cw':
                images = carlini_wagner_l2(model, images, 1000, confidence=eps, targeted=True, lr=0.1, max_iterations=10)
            elif attack == 'spsa':
                images = spsa(model, images, eps, 40)
                print("running spsa attack")
            elif attack == 'auto':
                #breakpoint()
                images = adversary.run_standard_evaluation(images, target, bs = bs)
            elif attack == 'auto-individual':
                #breakpoint()
                dict_images = adversary.run_standard_evaluation_individual(images, target, bs = bs)
                #breakpoint()
            elif attack == 'hsja':
                # can do targeted attack
                images = hop_skip_jump_attack(model, images, np.inf, num_iterations=1)
        # compute output
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if attack == 'auto-individual':
                    individual_outputs = {}
                    for name, images in dict_images.items():
                        output = model(images)
                        individual_outputs[name] = output
                else:
                    output = model(images)

                    ### l2 distances
                    # mod = model.module
                    # l2_distances = []
                    # for idx, block in enumerate(mod.blocks.children()):
                    #     l2_distances = l2_distances + block.attn.l2_distances
                    
                    # print(l2_distances)
                    # print(statistics.mean(l2_distances))

                    ### rep collapse
                    # mod = model.module
                    # cossim = []
                    # for idx, block in enumerate(mod.blocks.children()):
                    #     if idx in range(12):
                    #         cossim.append(block.attn.cossim.item())
            if attack == 'auto-individual':
                individual_losses = {}
                for name, images in dict_images.items():
                    loss = criterion(output, target)
                    individual_losses[name] = loss
                loss_list = [loss for loss in individual_losses.values()]
                loss = torch.mean(torch.stack(loss_list))
            else:
                loss = criterion(output, target)

        if not attack == 'auto-individual':
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print({'test_acc1':acc1,'test_acc5':acc5})
            if use_wandb:
                wandb.log({'test_acc1':acc1,'test_acc5':acc5})
        else:
            #breakpoint()
            top1_accuracies, top5_accuracies = [], []
            for name, output in individual_outputs.items():
                acc1, acc5 = accuracy(output, target, topk = (1,5))
                print(f'{name} | test_acc1: {acc1} , test_acc5: {acc5}')
                if use_wandb:
                    wandb.log({f'{name}_test_acc1':acc1, f'{name}_test_acc5':acc5})
                top1_accuracies.append(acc1)
                top5_accuracies.append(acc5)
            acc1, acc5 = torch.mean(torch.stack(top1_accuracies)), torch.mean(torch.stack(top5_accuracies))
            if use_wandb:
                wandb.log({'test_acc1':acc1,'test_acc5':acc5})


        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=bs)
        metric_logger.meters['acc5'].update(acc5.item(), n=bs)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if attack != 'none':
        print(f'Evaluating attack method {attack} with perturbation budget {eps}:')
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
