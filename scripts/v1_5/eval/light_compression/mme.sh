#!/bin/bash
#


ROOT_DATA=''
ROOT_WEIGHT=''
ROOT_LOG=""

run_mme() {
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local CKPT=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/MME/llava_mme.jsonl \
            --image-folder $ROOT_DATA/eval/MME/MME_Benchmark_release_version \
            --answers-file $ROOT_DATA/eval/MME/answers/$NAME.jsonl \
            --temperature 0 \
            --grouping $grouping \
            --stride $stride \
            --layer $layer \
            --conv-mode vicuna_v1

        cd $ROOT_DATA/eval/MME
        python convert_answer_to_mme.py --experiment $NAME
        cd eval_tool
        python calculation.py --results_dir answers/$NAME > ./eval_result/$NAME.txt
        cd ~/llava_git/llava
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

NAME=light-compression
grouping=avgpool1d
layer=16
stride=8
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME
GPU_ID=0

run_mme $GPU_ID $NAME $layer $stride $grouping $CKPT