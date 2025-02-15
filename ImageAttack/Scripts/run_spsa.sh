#!/bin/bash

count=0
offset=7500

# for i in $(seq 0 .005 0.1)
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#      CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port_num --use_env main.py --model deit_kde_tiny_patch16_224 --batch-size 48 --data-path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet/imagenet --output_dir /home/xing/robust_transformer/imagenet/files_kde --use_wandb 0 --project_name 'robust' --job_name imagenet_deit_kde_eval --attack 'spsa' --eps $i --finetune /home/xing/robust_transformer/imagenet/files_kde/checkpoint.pth --eval 1
# done

#### ORIGINAL CODE RUN ####
# for i in $(seq 0.1 .005 0.1)
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#     #  eps=$(perl -e "print $i / 255")
#      CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num \
#      --use_env ../main.py --model deit_mahala_overlayers_tiny_patch16_224_wrapper --batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ \
#      --output_dir output/spsa/Mfull-tiny-overlayers2-delta5-eps.1 --project_name 's-spsa' --job_name sym_2itperlayer_spsa_Mfull-tiny-overlayers2-delta5 --M-positions 1 2 3 4 5 6 7 8 9 10 11 \
#     --attack 'spsa' --eps $i --finetune ../../imagenet/path/output-Mfull-tiny-dist-overlayers2-delta5/checkpoint.pth --eval 1 --robust --dist-eval \
#     --use_wandb --grad-median
# done

# ablation run
for i in $(seq 0.1 .005 0.1)
  do
     (( count++ ))
     port_num=`expr $count + $offset`
    #  eps=$(perl -e "print $i / 255")
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num \
     --use_env ../main.py --model deit_mahala_overlayers_tiny_patch16_224 --batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ \
     --output_dir output/spsa/ablation-eps.1 --project_name 's-spsa' --job_name sym_2itperlayer_spsa_ablation --M-positions 1 2 3 4 5 6 7 8 9 10 11 \
    --attack 'spsa' --eps $i --finetune ../../imagenet/path/output-M-tiny-ablation2-bottom0/checkpoint.pth --eval 1 --dist-eval --ablation \
    --use_wandb
done




# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 \
# --use_env ../main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ \
# --output_dir output/spsa2/deit-baseline --project_name 's-spsa' --job_name deit-baseline-sanitytest \
# --attack 'none' --eps 1 --finetune ../../imagenet/path/output/checkpoint.pth --eval 1 --dist-eval

### changing epsilon to match paper ###
# for i in {1..6}
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#     eps=$(perl -e "print $i / 255")
#      CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num \
#      --use_env ../main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ \
#      --output_dir output/spsa2/deit-baseline --use_wandb --project_name 's-spsa' --job_name sym_2itperlayer_spsa2_deit-baseline --M-positions 1 2 3 4 5 6 7 8 9 10 11 \
#     --attack 'spsa' --eps $eps --finetune ../../imagenet/path/output/checkpoint.pth --eval 1 --robust --dist-eval --use_wandb --grad-median 
# done

# for i in $(seq 0 .005 0.1)
#   do
#      (( count++ ))
#      port_num=`expr $count + $offset`
#      CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port_num --use_env main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imagenet/imagenet --output_dir /home/xing/robust_transformer/imagenet/files_baseline --use_wandb 0 --project_name 'robust' --job_name imagenet_deit_baseline_eval --attack 'spsa' --eps $i --finetune /home/xing/robust_transformer/imagenet/files_baseline/checkpoint.pth --eval 1
# done
