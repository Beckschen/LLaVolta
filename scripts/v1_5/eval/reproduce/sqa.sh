#!/bin/bash
#
ROOT_DATA=''
ROOT_WEIGHT=''
ROOT_LOG=""

run_sqa() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local CKPT=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_science \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/scienceqa/llava_test_CQM-A.json \
            --image-folder $ROOT_DATA/eval/scienceqa/ScienceQA/test \
            --answers-file $ROOT_DATA/eval/scienceqa/answers/$NAME.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

        python llava/eval/eval_science_qa.py \
            --base-dir $ROOT_DATA/eval/scienceqa \
            --result-file $ROOT_DATA/eval/scienceqa/answers/$NAME.jsonl \
            --output-file $ROOT_DATA/eval/scienceqa/answers/$NAME-output.jsonl \
            --output-result $ROOT_DATA/eval/scienceqa/answers/$NAME-result.json
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

NAME=reproduce
grouping=none
layer=0
stride=1
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME
GPU_ID=0

run_sqa $GPU_ID $NAME $layer $stride $grouping $CKPT