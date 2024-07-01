#!/bin/bash
#
ROOT_DATA=''
ROOT_WEIGHT=''
ROOT_LOG=""
SPLIT="llava_vqav2_mscoco_test-dev2015"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

run_vqav2(){
    local CKPT=$1
    local layer=$2
    local stride=$3
    local grouping=$4
    local NAME=$5
    for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval/vqav2/$SPLIT.jsonl \
        --image-folder $ROOT_DATA/eval/vqav2/test2015 \
        --answers-file $ROOT_DATA/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --layer $layer \
        --stride $stride \
        --grouping $grouping &
    done

    wait

    output_file=$ROOT_DATA/eval/vqav2/answers/$SPLIT/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval/vqav2/answers/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $NAME --dir $ROOT_DATA/eval/vqav2
}

NAME=heavy-compression
PREFIX=$NAME-vqav2
grouping=avgpool1d
layer=2
stride=8
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME

run_vqav2 $CKPT $layer $stride $grouping $NAME