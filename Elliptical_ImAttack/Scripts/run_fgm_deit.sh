#!/bin/bash

count=0
offset=1916

# for i in {1..6}
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#      eps=$(perl -e "print $i / 255")
#      CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_kde_tiny_patch16_224 --batch-size 48 --data-path /home/tongzheng/imagenet --output_dir ./files_kde --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_kde_eval --attack 'fgm' --eps $eps --finetune ./files_kde/checkpoint.pth --eval 1
# done

for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
     eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=1,2,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num \
     --use_env ~/trids/EllAttack/main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /vinai/tannhm/imagenet \
     --output_dir /vinai/tannhm/data/ --use_wandb 1 --project_name 'robust_attack' --job_name sym_6itper1stlayer_fgm_deit \
     --attack 'fgm' --eps $eps --finetune /vinai/tannhm/checkpoints/trids/deit-baseline-model.pth --eval 1 \
     --robust
done
# for i in {1..6}
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#      eps=$(perl -e "print $i / 255")
#      CUDA_VISIBLE_DEVICES=2,4,5,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num \
#      --use_env ~/trids/KPCA-S/Attack/main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /vinai/tannhm/imagenet \
#      --output_dir /vinai/tannhm/data/ --use_wandb 1 --project_name 'robust_attack' --job_name sym_6itper1stlayer_fgm \
#      --attack 'fgm' --eps $eps --finetune /vinai/tannhm/checkpoints/trids/deit-baseline-model.pth --eval 1 \
#      --robust
# done

# for i in {1..6}
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#      eps=$(perl -e "print $i / 255")
#      CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /home/tongzheng/imagenet --output_dir ./files_baseline --use_wandb 1 --project_name 'robust' --job_name imagenet_deit_baseline_eval --attack 'fgm' --eps $eps --finetune ./files_baseline/checkpoint.pth --eval 1
# done