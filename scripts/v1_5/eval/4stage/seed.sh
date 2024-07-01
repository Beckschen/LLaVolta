#!/bin/bash
#
ROOT_DATA=''
ROOT_WEIGHT=''
ROOT_LOG=""

run_seed(){
    local CKPT=$1
    local NAME=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $ROOT_DATA/eval/seed_bench/llava-seed-bench-img.jsonl \
            --image-folder $ROOT_DATA/eval/seed_bench \
            --answers-file $ROOT_DATA/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --stride $stride \
            --grouping $grouping &
    done

    wait

    output_file=$ROOT_DATA/eval/seed_bench/answers/$NAME/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $ROOT_DATA/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    # Evaluate
    python scripts/convert_seed_for_submission.py \
        --annotation-file $ROOT_DATA/eval/seed_bench/SEED-Bench.json \
        --result-file $output_file \
        --result-upload-file $ROOT_DATA/eval/seed_bench/answers_upload/$NAME.jsonl
}

NAME=4stage
grouping=none
layer=0
stride=1
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME

run_seed $CKPT $NAME $layer $stride $grouping

