#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7 python eval_sliding_window.py --cuda --data ../../wikitext103/lmtool-fwms/data/wikitext-103 \
--dataset wt103 --split 'test' --batch_size 1 --mem_len 0 --work_dir ../../wikitext103/lmtool-fwms/saved_models/Mfull/ \
--M-positions 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --output-dir output/elliptical --model-size small --over-layers
