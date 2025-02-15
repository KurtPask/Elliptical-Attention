### Mfull run ###
# CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5721 \
# --use_env ../main.py --model deit_mahala_overlayers_tiny_patch16_224_wrapper --batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ \
# --output_dir output/auto/Mfull-overlayers-delta5-gradmed-maxscale-eps1 --project_name 'autoattack' --job_name auto/Mfull-overlayers-delta5-gradmed-maxscale \
# --attack 'auto' --eps 1 --finetune ../../imagenet/path/output-Mfull-tiny-dist-overlayers2-delta5/checkpoint.pth --eval 1 --dist-eval --robust \
# --M-positions 1 2 3 4 5 6 7 8 9 10 11 --use_wandb --grad-median

### baseline run ###
CUDA_VISIBLE_DEVICES=0,1,2,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1720 \
--use_env ../main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ \
--output_dir output/auto/deit-baseline2-eps1 --project_name 'autoattack' --job_name deit-baseline-2 \
--attack 'auto' --eps 1 --finetune ../../imagenet/path/output/deit-baseline2/checkpoint.pth --eval 1 --dist-eval --robust --use_wandb \