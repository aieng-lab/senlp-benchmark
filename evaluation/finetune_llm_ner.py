"""Script to fine-tune a language model for a NER task.

This script implements the required steps to fine-tune a language model for a NER task including
data splitting, tokenization, training and evaluation.
"""

import argparse
import datetime
import logging
import pickle
import random
import sys
import time

from datasets import Dataset, DatasetDict  # type: ignore[attr-defined]
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

sys.path.append("../")
from config.label_matchers import matchers
from utils.evaluation import (
    compute_metrics_ner_hf as compute_metrics,
)


DATASETS_BASE_PATH = "../datasets"
RESULTS_BASE_PATH = "./results/finetuning"


logger = logging.getLogger(__name__)


def handle_args():
    """Handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a language model for a NER task."
    )

    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Name or path of the model to fine-tune"
    )
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true",
        help="Specify if the training must be resumed from a checkpoint located at the default path"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, help="Name or path of the tokenizer to use"
    )

    # Model hyper-parameters
    parser.add_argument(
        "--batch_size_per_gpu", type=int, required=True,
        help="Batch size for training and testing (per device)"
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--flash_attention", action="store_true",
        help="Specify if using flash attention 2"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, required=True,
        help="Number of update steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--learning_rate", type=float, required=True, help="Initial learning rate"
    )
    parser.add_argument(
        "--max_length", type=int,
        help="Max. length for sentence truncation"
    )

    # Aux: Required for compatibility with DeepSpeed
    parser.add_argument(
        "--local_rank", type=int,
        help="..."
    )

    return parser.parse_args()


def train_test_split_list(data, test_size=0.2, random_state=None):
    """Splits a list into train and test sets."""
    if random_state is not None:
        random.seed(random_state)  # Ensure reproducibility

    data_copy = data[:]  # Avoid modifying the original list
    random.shuffle(data_copy)  # Shuffle the data

    split_idx = int(len(data_copy) * (1 - test_size))  # Calculate split index

    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]

    return train_data, test_data


def align_labels_with_tokens(labels, word_ids):
    """Make sure that after tokenization labels have the same length than tokens."""
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def main(params):
    """Main function to run the script"""

    logger.info("RUNNING THE SCRIPT TO FINE-TUNE A LANGUAGE MODEL FOR A NER TASK")
    logger.info("Arguments passed by command-line: %s", params)
    logger.info("Execution date: %s", datetime.datetime.now())

    results_base_path = f"{RESULTS_BASE_PATH}/{params.dataset_name}/{params.model_name_or_path.split("/")[-1]}"

    label_names = matchers[params.dataset_name]
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    try:
        logger.info("Loading the %s dataset...", params.dataset_name)
        with open(f"{DATASETS_BASE_PATH}/{params.dataset_name}/{params.dataset_name}.pkl", "rb") as file:
            data = pickle.load(file)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    logger.info("Splitting the data for testing...")
    train, test = train_test_split_list(data, test_size=0.2, random_state=42)

    logger.info("Transforming the data to a HF dataset...")
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_dict({
        "tokens": [[j[0] for j in i] for i in train],
        "ner_tags": [[j[1] for j in i] for i in train]
    })
    dataset["test"] = Dataset.from_dict({
        "tokens": [[j[0] for j in i] for i in test],
        "ner_tags": [[j[1] for j in i] for i in test]
    })

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_name_or_path, add_prefix_space=True)

    def tokenize_and_align_labels(x):
        tokenized_inputs = tokenizer(
            x["tokens"],
            padding="max_length",
            truncation=True,
            max_length=params.max_length,
            is_split_into_words=True
        )
        all_labels = x["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    logger.info("Tokenizing train and test datasets...")
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Model to fine-tune
    model = AutoModelForTokenClassification.from_pretrained(
        params.model_name_or_path,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if params.flash_attention else None
    )

    training_args = TrainingArguments(
        output_dir=f"{results_base_path}/test",
        eval_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=params.batch_size_per_gpu,
        per_device_eval_batch_size=params.batch_size_per_gpu,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        learning_rate=params.learning_rate,
        weight_decay=0.01,
        num_train_epochs=params.epochs,
        save_strategy=IntervalStrategy.EPOCH,
        save_total_limit=1,
        seed=42,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        deepspeed="./deepspeed_config.json",
        optim="adamw_apex_fused",  # Unable when using deepspeed
        # torch_compile=True,  # Useful, but incompatible with DeepSpeed
    )

    # Instantiate the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, label_names=label_names),  # type: ignore
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Fine-tuning the %s model...", params.model_name_or_path)
    trainer.train(params.resume_from_checkpoint)

    logger.info("Reporting error metrics for the test dataset...")
    trainer.save_metrics(
        split="train",
        metrics=trainer.evaluate(tokenized_dataset["train"]),  # type: ignore
        combined=False
    )
    trainer.save_metrics(
        split="test",
        metrics=trainer.evaluate(tokenized_dataset["test"]),  # type: ignore
        combined=False
    )

    logger.info("Saving best model...")
    model.save_pretrained(f"{results_base_path}/test/best")


if __name__ == "__main__":
    start_time = time.time()
    args = handle_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
    logger.info("Execution time: %.2f seconds", time.time() - start_time)
