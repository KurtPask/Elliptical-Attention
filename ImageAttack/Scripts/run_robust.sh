###### FOR INFERENCE
# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --master_port 10010 --nproc_per_node=3 --use_env ../eval_OOD.py \
# --model deit_mahala_overlayers_tiny_patch16_224_wrapper --model-path ../../imagenet/path/output-Mfull-tiny-dist-overlayers2-delta5/checkpoint.pth \
# --data-path /../../../../cm/shared/cuongdc10/datasets/image/ --output_dir output/aorc/ --which o --M-positions 1 2 3 4 5 6 7 8 9 10 11 \

#### Baseline Inference
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --master_port 10010 --nproc_per_node=3 --use_env ../eval_OOD.py \
--model deit_tiny_patch16_224 --model-path ../../imagenet/path/output/deit-baseline2/checkpoint.pth \
--data-path /../../../../cm/shared/cuongdc10/datasets/image/ --output_dir output/aorc/ --which o \


# CUDA_VISIBLE_DEVICES='0' python eval_imagenetp.py --model-name deit_tiny_patch16_224 --ngpu 1

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=5721 \
# --use_env ../main.py --model deit_mahala_overlayers_tiny_patch16_224_wrapper --batch-size 48 --data-path /../../../../cm/shared/cuongdc10/datasets/image/ \
# --output_dir output/auto-indi2/Mfull-overlayers-gradmed-eps1 --project_name 'autoattack' --job_name individual2/Mfull-overlayers-gradmed \
# --attack 'auto-individual' --eps 1 --finetune ../../imagenet/path/output-M1234567891011-dist-overlayers2-att5e-1/checkpoint.pth --eval 1 --dist-eval --robust \
# --M-positions 1 2 3 4 5 6 7 8 9 10 11 --grad-median --use_wandb
