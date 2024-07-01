#!/bin/bash
#
ROOT_DATA=''
ROOT_WEIGHT=''
ROOT_LOG=""

run_vizwiz() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local CKPT=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/vizwiz/llava_test.jsonl \
            --image-folder $ROOT_DATA/eval/vizwiz/test \
            --answers-file $ROOT_DATA/eval/vizwiz/answers/$NAME.jsonl \
            --temperature 0 \
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

        python scripts/convert_vizwiz_for_submission.py \
            --annotation-file $ROOT_DATA/eval/vizwiz/llava_test.jsonl \
            --result-file $ROOT_DATA/eval/vizwiz/answers/$NAME.jsonl \
            --result-upload-file $ROOT_DATA/eval/vizwiz/answers_upload/$NAME.json
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

NAME=heavy-compression
grouping=avgpool1d
layer=2
stride=8
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME
GPU_ID=0

run_vizwiz $GPU_ID $NAME $layer $stride $grouping $CKPT