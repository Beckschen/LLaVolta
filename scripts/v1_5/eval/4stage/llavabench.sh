#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ROOT_DATA=''
ROOT_WEIGHT=''
export OPENAI_API_KEY=''

run_llavabench(){
    local GPU_ID=$1
    local LOG_PREFIX=$2
    local layer=$3
    local stride=$4
    local grouping=$5
    local CKPT=$6
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
    python -m llava.eval.model_vqa \
        --model-path $CKPT \
        --question-file $ROOT_DATA/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder $ROOT_DATA/eval/llava-bench-in-the-wild/images \
        --answers-file $ROOT_DATA/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --stride $stride \
        --layer $layer \
        --grouping $grouping

    mkdir -p $ROOT_DATA/eval/llava-bench-in-the-wild/reviews

    python llava/eval/eval_gpt_review_bench.py \
        --question $ROOT_DATA/eval/llava-bench-in-the-wild/questions.jsonl \
        --context $ROOT_DATA/eval/llava-bench-in-the-wild/context.jsonl \
        --rule llava/eval/table/rule.json \
        --answer-list \
            $ROOT_DATA/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
            $ROOT_DATA/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
        --output \
            $ROOT_DATA/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl

    python llava/eval/summarize_gpt_review.py -f $ROOT_DATA/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl > $ROOT_DATA/eval/llava-bench-in-the-wild/review_result/$NAME.txt
    "  
}

NAME=4stage
grouping=none
layer=0
stride=1
CKPT=$ROOT_WEIGHT/llava-v1.5-7b-$NAME
GPU_ID=0

run_llavabench $GPU_ID $NAME $layer $stride $grouping $CKPT