#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="sparsevlm"
SCALE=$1
SHIFT=0
PARAM="scale_${SCALE}_shift_${SHIFT}"

python -m llava.eval.model_vqa \
    --model-path /mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/huggingface/${CKPT} \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --sparse \
    --scale ${SCALE} \
    --shift ${SHIFT} \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${CKPT}/${METHOD}/${PARAM}.json

