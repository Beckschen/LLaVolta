#!/bin/bash
#
ROOT_DATA=''
ROOT_WEIGHT=''
ROOT_LOG=""

run_pope() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local CKPT=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/pope/llava_pope_test.jsonl \
            --image-folder $ROOT_DATA/eval/pope/val2014 \
            --answers-file $ROOT_DATA/eval/pope/answers/$NAME.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --grouping $grouping \
            --stride $stride \
            --layer $layer

        python llava/eval/eval_pope.py \
            --annotation-dir $ROOT_DATA/eval/pope/coco \
            --question-file $ROOT_DATA/eval/pope/llava_pope_test.jsonl \
            --result-file $ROOT_DATA/eval/pope/answers/$NAME.jsonl
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

NAME=4stage
grouping=none
layer=0
stride=1
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME
GPU_ID=0

run_pope $GPU_ID $NAME $layer $stride $grouping $CKPT