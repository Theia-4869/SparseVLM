#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="sparsevlm"
SCALE=$1
SHIFT=0
PARAM="scale_${SCALE}_shift_${SHIFT}"

python -m llava.eval.model_vqa_science \
    --model-path /mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/huggingface/${CKPT} \
    --question-file ./playground/data/eval/sqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/sqa/images/test \
    --answers-file ./playground/data/eval/sqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --single-pred-prompt \
    --sparse \
    --scale ${SCALE} \
    --shift ${SHIFT} \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/sqa \
    --result-file ./playground/data/eval/sqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --output-file ./playground/data/eval/sqa/answers/${CKPT}/${METHOD}/${PARAM}_output.jsonl \
    --output-result ./playground/data/eval/sqa/answers/${CKPT}/${METHOD}/${PARAM}_result.json \
