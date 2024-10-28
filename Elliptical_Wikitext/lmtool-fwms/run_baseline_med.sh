CUDA_VISIBLE_DEVICES=2 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 \
--adaptive --n_layer 16 --d_model 256 --n_head 8 --d_head 32 --d_inner 2048 --dropout 0.1 \
--dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 400000 --attn_type 2 \
--tgt_len 384 --mem_len 0 --eval_tgt_len 384 --batch_size 56 --multi_gpu --use_wandb \
--project_name 'fourierformer' --seed 1111 --job_name softmax-seed-1111 --work_dir ./softmax-baseline