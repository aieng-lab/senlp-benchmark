#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# Variables to deal with NCCL timeout
export NCCL_TIMEOUT=3600
export TORCH_NCCL_TIMEOUT=3600
export DEEPSPEED_TIMEOUT=3600


DATASET=story_points
TASK_ARGS=""
# TASK_ARGS="--task_type multilabel --cv_strategy iterative"
# TASK_ARGS="--task_type regression"
TRAINING_ARGS="--omit_cv --batch_size_per_gpu 8 --epochs 3 --gradient_accumulation_steps 1 --learning_rate 1e-5"
TRAINING_ARGS_OOS="--omit_cv --batch_size_per_gpu 4 --epochs 3 --gradient_accumulation_steps 2 --learning_rate 1e-5"
TRAINING_ARGS_OOS_2="--omit_cv --batch_size_per_gpu 2 --epochs 3 --gradient_accumulation_steps 4 --learning_rate 1e-5"
TRAINING_ARGS_OOS_3="--omit_cv --batch_size_per_gpu 1 --epochs 3 --gradient_accumulation_steps 8 --learning_rate 1e-5"



# BERT

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path bert-base-cased \
    --tokenizer_name_or_path bert-base-cased \
    $TRAINING_ARGS \
    $TASK_ARGS

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path bert-large-cased \
    --tokenizer_name_or_path bert-large-cased \
    $TRAINING_ARGS \
    $TASK_ARGS


# RoBERTa

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path roberta-base \
    --tokenizer_name_or_path roberta-base \
    $TRAINING_ARGS \
    $TASK_ARGS

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path roberta-large \
    --tokenizer_name_or_path roberta-large \
    $TRAINING_ARGS \
    $TASK_ARGS


# ModernBERT

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path answerdotai/ModernBERT-base \
    --tokenizer_name_or_path answerdotai/ModernBERT-base \
    $TRAINING_ARGS \
    $TASK_ARGS \
    --flash_attention

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path answerdotai/ModernBERT-large \
    --tokenizer_name_or_path answerdotai/ModernBERT-large \
    $TRAINING_ARGS \
    $TASK_ARGS \
    --flash_attention


# GPT2

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path gpt2 \
    --tokenizer_name_or_path gpt2 \
    $TRAINING_ARGS \
    $TASK_ARGS

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path gpt2-medium \
    --tokenizer_name_or_path gpt2-medium \
    $TRAINING_ARGS \
    $TASK_ARGS

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path gpt2-large \
    --tokenizer_name_or_path gpt2-large \
    $TRAINING_ARGS \
    $TASK_ARGS

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path gpt2-xl \
    --tokenizer_name_or_path gpt2-xl \
    $TRAINING_ARGS \
    $TASK_ARGS


# Llama

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --tokenizer_name_or_path meta-llama/Llama-3.2-1B \
    $TRAINING_ARGS \
    $TASK_ARGS \
    --max_length 1024 \
    --flash_attention

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --tokenizer_name_or_path meta-llama/Llama-3.2-3B \
    $TRAINING_ARGS \
    $TASK_ARGS \
    --max_length 1024 \
    --flash_attention


# T5

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path t5-small \
    --tokenizer_name_or_path t5-small \
    $TRAINING_ARGS \
    $TASK_ARGS

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path t5-base \
    --tokenizer_name_or_path t5-base \
    $TRAINING_ARGS \
    $TASK_ARGS \
    --max_length 512

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path t5-large \
    --tokenizer_name_or_path t5-large \
    $TRAINING_ARGS \
    $TASK_ARGS \
    --max_length 512

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path t5-3b \
    --tokenizer_name_or_path t5-3b \
    $TRAINING_ARGS \
    $TASK_ARGS \
    --max_length 512


# CodeBERT

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path microsoft/codebert-base \
    --tokenizer_name_or_path microsoft/codebert-base \
    $TRAINING_ARGS \
    $TASK_ARGS


# CodeLlama

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path meta-llama/CodeLlama-7b-hf \
    --tokenizer_name_or_path meta-llama/CodeLlama-7b-hf \
    $TRAINING_ARGS_OOS \
    $TASK_ARGS \
    --max_length 1024 \
    --flash_attention


# StarCoder2

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path bigcode/starcoder2-3b \
    --tokenizer_name_or_path bigcode/starcoder2-3b \
    $TRAINING_ARGS \
    $TASK_ARGS \
    --max_length 1024 \
    --padding_side left \
    --flash_attention 

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path bigcode/starcoder2-7b \
    --tokenizer_name_or_path bigcode/starcoder2-7b \
    $TRAINING_ARGS_OOS \
    $TASK_ARGS \
    --max_length 1024 \
    --padding_side left \
    --flash_attention


# CodeT5+

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path Salesforce/codet5p-220m \
    --tokenizer_name_or_path Salesforce/codet5p-220m \
    $TRAINING_ARGS \
    $TASK_ARGS

deepspeed finetune_llm.py \
    --dataset_name $DATASET \
    --model_name_or_path Salesforce/codet5p-770m \
    --tokenizer_name_or_path Salesforce/codet5p-770m \
    $TRAINING_ARGS \
    $TASK_ARGS
