#!/bin/bash
#
ROOT_DATA=''
ROOT_WEIGHT=''
ROOT_LOG=""

run_mmvet() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local CKPT=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/mm-vet/llava-mm-vet.jsonl \
            --image-folder $ROOT_DATA/eval/mm-vet/images \
            --answers-file $ROOT_DATA/eval/mm-vet/answers/$NAME.jsonl \
            --temperature 0 \
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

        mkdir -p $ROOT_DATA/eval/mm-vet/results
        python scripts/convert_mmvet_for_eval.py \
            --src $ROOT_DATA/eval/mm-vet/answers/$NAME.jsonl \
            --dst $ROOT_DATA/eval/mm-vet/results/$NAME.json
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

NAME=reproduce
grouping=none
layer=0
stride=1
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME
GPU_ID=0

run_mmvet $GPU_ID $NAME $layer $stride $grouping $CKPT