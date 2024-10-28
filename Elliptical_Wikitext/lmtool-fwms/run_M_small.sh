CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 \
--adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 \
--dropatt 0.0 --optim adam --lr 0.00025 --max_step 500000 --attn_type 2 --warmup_step 2000 \
--tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --M-positions 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
--project_name 'fourierformer' --seed 1111 --job_name M_Mfull-overlayers-seed-1111-delta1 --work_dir ./saved_models/Mfull-delta1-maxscale \
--use_wandb --show-M --over-layers --attenuation 5e-2 --use_wandb