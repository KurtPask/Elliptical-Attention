#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1517 --use_env main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet/imagenet --output_dir /home/xing/robust_transformer/imagenet/files_baseline --use_wandb 0 --project_name 'robust' --job_name imagenet_deit_baseline --finetune /home/xing/robust_transformer/imagenet/files_baseline/checkpoint.pth --eval 1 --inc_path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet-c

#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1518 --use_env main.py --model deit_kde_tiny_patch16_224 --batch-size 48 --data-path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet/imagenet --output_dir /home/xing/robust_transformer/imagenet/files_kde --use_wandb 0 --project_name 'robust' --job_name imagenet_deit_kde --finetune /home/xing/robust_transformer/imagenet/files_kde/checkpoint.pth --eval 1 --inc_path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet-c

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1519 --use_env main.py --model deit_robust_tiny_patch16_224 \
--batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ --output_dir output/imnetc/Mfull-tiny-overlayers2-gradmed \
--use_wandb --project_name 'robust-imnetc' --job_name imnetc_Mfull-tiny-overlayers2-gradmed \
--finetune ../../imagenet/path/output-M1234567891011-dist-overlayers2-att5e-1/checkpoint.pth --eval 1 --inc_path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet-c



for i in {1..6}
  do
     (( count++ ))
     port_num=`expr $count + $offset`
    eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num \
     --use_env ../main.py --model deit_mahala_overlayers_tiny_patch16_224_wrapper --batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ \
     --output_dir output/spsa2/Mfull-tiny-overlayers2-gradmed --use_wandb --project_name 's-spsa' --job_name sym_2itperlayer_spsa2_Mfull-tiny-overlayers2-gradmed --M-positions 1 2 3 4 5 6 7 8 9 10 11 \
    --attack 'spsa' --eps $eps --finetune ../../imagenet/path/output-M1234567891011-dist-overlayers2-att5e-1/checkpoint.pth --eval 1 --robust --dist-eval --use_wandb --grad-median
done