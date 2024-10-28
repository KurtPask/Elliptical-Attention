#!/bin/bash

count=0
offset=2000
batchsize=48

# for i in $(seq 0 .005 0.1)
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=$port_num --use_env main.py --model deit_kde_tiny_patch16_224 --batch-size $batchsize --data-path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet/imagenet --output_dir /home/xing/robust_transformer/imagenet/files_kde --use_wandb 1 --project_name 'robust' --job_name cw_kde_eval --attack 'cw' --eps $i --finetune /home/xing/robust_transformer/imagenet/files_kde/checkpoint.pth --eval 1
# done

for i in {0..0}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
    #  eps=$(perl -e "print $i")
     CUDA_VISIBLE_DEVICES=0,6,1,2 python -m torch.distributed.launch --nproc_per_node=4 --master_port=9901 \
     --use_env /home/ubuntu/Kpca/Attack/main.py --model deit_tiny_patch16_224 --batch-size $batchsize --data-path /vinai/tannhm/imagenet/ \
     --output_dir /home/ubuntu/Kpca/checkpoints/ --use_wandb 1 --project_name 'robust_attack' --job_name sym_baseline_hsja --attack 'hsja' --eps 10 \
     --finetune /home/ubuntu/Kpca/checkpoints/rpc/sym_baseline.pth --eval 1 
done

# for i in $(seq 0 .005 0.1)
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=$port_num --use_env main.py --model deit_tiny_patch16_224 --batch-size $batchsize --data-path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet/imagenet --output_dir /home/xing/robust_transformer/imagenet/files_baseline --use_wandb 1 --project_name 'robust' --job_name cw_baseline_eval --attack 'cw' --eps $i --finetune /home/xing/robust_transformer/imagenet/files_baseline/checkpoint.pth --eval 1
# done
# done
