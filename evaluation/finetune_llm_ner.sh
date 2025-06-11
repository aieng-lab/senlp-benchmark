#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


DATASET=se_entities
TRAINING_ARGS="--batch_size_per_gpu 8 --epochs 50 --gradient_accumulation_steps 1 --learning_rate 1e-5"


# BERT

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path bert-base-cased \
    --tokenizer_name_or_path bert-base-cased \
    $TRAINING_ARGS

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path bert-large-cased \
    --tokenizer_name_or_path bert-large-cased \
    $TRAINING_ARGS


# RoBERTa

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path roberta-base \
    --tokenizer_name_or_path roberta-base \
    $TRAINING_ARGS

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path roberta-large \
    --tokenizer_name_or_path roberta-large \
    $TRAINING_ARGS


# ModernBERT

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path answerdotai/ModernBERT-base \
    --tokenizer_name_or_path answerdotai/ModernBERT-base \
    $TRAINING_ARGS \
    --flash_attention

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path answerdotai/ModernBERT-large \
    --tokenizer_name_or_path answerdotai/ModernBERT-large \
    $TRAINING_ARGS \
    --flash_attention


# T5

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path t5-small \
    --tokenizer_name_or_path t5-small \
    $TRAINING_ARGS

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path t5-base \
    --tokenizer_name_or_path t5-base \
    $TRAINING_ARGS \
    --max_length 512

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path t5-large \
    --tokenizer_name_or_path t5-large \
    $TRAINING_ARGS \
    --max_length 512

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path t5-3b \
    --tokenizer_name_or_path t5-3b \
    $TRAINING_ARGS \
    --max_length 512


# CodeBERT

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path microsoft/codebert-base \
    --tokenizer_name_or_path microsoft/codebert-base \
    $TRAINING_ARGS


# CodeT5+

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path Salesforce/codet5p-220m \
    --tokenizer_name_or_path Salesforce/codet5p-220m \
    $TRAINING_ARGS

deepspeed finetune_llm_ner.py \
    --dataset_name $DATASET \
    --model_name_or_path Salesforce/codet5p-770m \
    --tokenizer_name_or_path Salesforce/codet5p-770m \
    $TRAINING_ARGS
