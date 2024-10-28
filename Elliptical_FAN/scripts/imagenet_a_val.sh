#v#!/bin/bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

### FAN ###
# CUDA_VISIBLE_DEVICES=6 ../validate_ood.py --data ../../ellattack-ly/data/imagenet-a/imagenet-a/ \
# 	--model fan_tiny_12_p16_224 \
#         --img-size 224 \
#         -b 128 \
#         -j 32 \
#         --imagenet_a \
# 	--results-file /aor/checkpoints/ \
# 	--checkpoint output/fan_tiny_12_p16_224/train/FAN_baseline-resume2/checkpoint.pth \
#         #--robust --no-test-pool

### FAN-ELLIPTICAL ###
# CUDA_VISIBLE_DEVICES=6 ../validate_ood.py ../../ellattack-ly/data/imagenet-a/imagenet-a/ \
# --model elliptical_fan_tiny_12_p16_224 \
# --img-size 224 \
# -b 128 \
# -j 32 \
# --imagenet_a \
# --results-file /aor/checkpoints/ \
# --checkpoint output/elliptical_fan_tiny_12_p16_224/train/elliptical_FAN-restart2/checkpoint.pth \
#--robust --no-test-pool \

### DEIT ###
# CUDA_VISIBLE_DEVICES=6 ../validate_ood.py --data ../../ellattack-ly/data/imagenet-a/imagenet-a/ \
# --model deit_tiny_patch16_224 \
# --img-size 224 \
# -b 128 \
# -j 32 \
# --imagenet_a \
# --results-file /aor/checkpoints/ \
# --checkpoint ../../imagenet/path/output/checkpoint.pth \

### DEIT - ELLIPTICAL ###
CUDA_VISIBLE_DEVICES=6 ../validate_ood.py --data ../../ellattack-ly/data/imagenet-a/imagenet-a/ \
--model mahala_deit_overlayers_tiny_patch16_224 \
--img-size 224 \
-b 128 \
-j 32 \
--imagenet_a \
--results-file /aor/checkpoints/ \
--checkpoint ../../imagenet/path/output-M1234567891011-dist-overlayers2-att5e-1/checkpoint.pth \

