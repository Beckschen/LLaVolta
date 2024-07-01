#!/bin/bash
#
LLAVA_HOME=$(pwd)
ROOT_DATA=''
ROOT_WEIGHT=''
SPLIT="llava_gqa_testdev_balanced"
GQADIR="$ROOT_DATA/eval/gqa/data"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

run_gqa(){
    local CKPT=$1
    local layer=$2
    local stride=$3
    local grouping=$4
    local NAME=$5
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/gqa/$SPLIT.jsonl \
            --image-folder $ROOT_DATA/eval/gqa/images \
            --answers-file $ROOT_DATA/eval/gqa/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --grouping $grouping \
            --stride $stride \
            --layer $layer &
    done

    wait

    output_file=$ROOT_DATA/eval/gqa/answers/$SPLIT/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval/gqa/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

    cd $GQADIR
    python eval/eval.py --tier testdev_balanced
    cd $LLAVA_HOME
}

NAME=reproduce
grouping=none
layer=0
stride=1
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME

run_gqa $CKPT $layer $stride $grouping $NAME