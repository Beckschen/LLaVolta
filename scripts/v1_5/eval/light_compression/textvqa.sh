#!/bin/bash
#
ROOT_DATA=''
ROOT_WEIGHT=''
ROOT_LOG=""

run_textvqa() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local CKPT=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder $ROOT_DATA/eval/textvqa/train_images \
            --answers-file $ROOT_DATA/eval/textvqa/answers/$NAME.jsonl \
            --temperature 0 \
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

        python -m llava.eval.eval_textvqa \
            --annotation-file $ROOT_DATA/eval/textvqa/TextVQA_0.5.1_val.json \
            --result-file $ROOT_DATA/eval/textvqa/answers/$NAME.jsonl
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

NAME=light-compression
grouping=avgpool1d
layer=16
stride=8
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME
GPU_ID=0

run_textvqa $GPU_ID $NAME $layer $stride $grouping $CKPT

