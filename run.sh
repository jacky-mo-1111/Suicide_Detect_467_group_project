#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# python run_roberta.py \
#     --seed 1 \
#     --model_name roberta-base \
#     --dropout 0.1 \
#     --chckpt_dir roberta_finetuned/ \
#     --do_eval \

python run_gpt.py \
    --shots_num 2 \


    
    
