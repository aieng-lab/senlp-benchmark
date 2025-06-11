"""Script to fine-tune a language model.

This script implements the required steps to fine-tune a language model including tokenization,
training and evaluation, depending on the cross-validation strategy selected.

NOTE:
- Argument --cv_strategy leave-one-group-out will be removed in future versions.
"""

import argparse
import datetime
from glob import glob
import logging
import sys
import time

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value
)
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

sys.path.append("../")
from utils.evaluation import (
    compute_metrics_hf,
    compute_metrics_multilabel_hf,
    compute_metrics_regression_hf,
)


DATASETS_BASE_PATH = "../preprocessing/datasets"
SPLITS_BASE_PATH = "./splits"
RESULTS_BASE_PATH = "./results/finetuning"


logger = logging.getLogger(__name__)


def handle_args():
    """Handle command line arguments"""
    parser = argparse.ArgumentParser(description="Script to fine-tune a language model.")

    parser.add_argument(
        "--cv_strategy", type=str, default="k-fold",
        choices=["k-fold", "repeated-k-fold", "leave-one-group-out", "iterative"],
        help="Splitting strategy"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--dataset_version", type=str, default="cased", choices=["cased", "uncased"],
        help="Version of the dataset to use"
    )
    parser.add_argument(
        "--eval_accumulation_steps", type=int,
        help="Accumulate evaluation results to deal with memory error"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Name or path of the model to fine-tune"
    )
    parser.add_argument(
        "--omit_cv", action="store_true",
        help="Specify if omitting cv, fine-tuning on all the training dataset"
    )
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true",
        help="Specify if the training must be resumed from a checkpoint located at the default path"
    )
    parser.add_argument(
        "--task_type", type=str, default="classification",
        choices=["classification", "multilabel", "regression", "ner", "mlm"],
        help="Task type to solve"
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
    parser.add_argument(
        "--padding_side", type=str, choices=["right", "left"],
        help="The side on which the model should have padding applied"
    )

    # Aux: Required for compatibility with DeepSpeed
    parser.add_argument(
        "--local_rank", type=int,
        help="..."
    )

    return parser.parse_args()


def main(params):
    """Main function to run the script"""

    logger.info("RUNNING THE SCRIPT TO FINE-TUNE A LANGUAGE MODEL")
    logger.info("Arguments passed by command-line: %s", params)
    logger.info("Execution date: %s", datetime.datetime.now())

    results_base_path = f"{RESULTS_BASE_PATH}/{params.dataset_name}/{params.dataset_version}/{params.model_name_or_path.split("/")[-1]}"

    # Definitions based on task type
    if params.task_type == "classification":
        problem_type = None
        metric_for_best_model = "f1_macro"
        greater_is_better = True
        compute_metrics = compute_metrics_hf
    elif params.task_type == "multilabel":
        problem_type = "multi_label_classification"
        metric_for_best_model = "f1_macro"
        greater_is_better = True
        compute_metrics = compute_metrics_multilabel_hf
    elif params.task_type == "regression":
        problem_type = "regression"
        metric_for_best_model = "smape"
        greater_is_better = False
        compute_metrics = compute_metrics_regression_hf
    else:
        logger.error("Compute metrics function not implemented for %s", params.task_type)
        sys.exit(1)

    try:
        logger.info("Loading the %s %s dataset...", params.dataset_name, params.dataset_version)
        data_df = pd.read_parquet(
            f"{DATASETS_BASE_PATH}/{params.dataset_version}/{params.dataset_name}.parquet"
        )

        # Workaround to avoid T5 tokenization errors
        data_df["text_clean"] = data_df["text_clean"].str.replace("</s>", "")

        if params.task_type == "multilabel":
            labels = [c for c in data_df.columns if "label_" in c]
            num_labels = len(labels)
        elif params.task_type == "regression":
            num_labels = 1
        else:
            num_labels = data_df["label"].nunique()

            # Workaround for regression problems
            if params.task_type == "regression":
                data_df["label"] = data_df["label"].astype(float)

        splits_filename = f"{params.dataset_name}.{params.cv_strategy}.*.csv"
        logger.info("Loading the splits file named %s...", splits_filename)
        splits_df = pd.read_csv(glob(f"{SPLITS_BASE_PATH}/{splits_filename}")[0])

        if not params.omit_cv:
            logger.info("Number of folds to experiment with: %d", splits_df.shape[1])
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if params.padding_side is not None:
        tokenizer.padding_side = params.padding_side

    if params.task_type == "multilabel":
        def tokenize(x):
            example = tokenizer(
                x["text_clean"],
                padding="max_length",
                truncation=True,
                max_length=params.max_length
            )
            example["labels"] = [float(v) for k, v in x.items() if k.startswith("label_")]
            return example
    else:
        def tokenize(x):
            return tokenizer(
                x["text_clean"],
                padding="max_length",
                truncation=True,
                max_length=params.max_length
            )

    if not params.omit_cv:
        # Iterate on folds
        for i in range(splits_df.shape[1]):
            i += 1

            logger.info("- - - - -")
            logger.info("Splitting the data based on fold %d...", i)
            train_df = data_df.loc[splits_df[f"fold_{i}"] == 1]
            validation_df = data_df.loc[splits_df[f"fold_{i}"] == 0]

            logger.info("Transforming the data to a HF dataset...")
            dataset = DatasetDict()
            dataset["train"] = Dataset.from_pandas(train_df)
            dataset["validation"] = Dataset.from_pandas(validation_df)

            logger.info("Tokenizing train and validation datasets...")
            tokenized_datasets = dataset.map(tokenize)

            # Model to fine-tune
            model = AutoModelForSequenceClassification.from_pretrained(
                params.model_name_or_path,
                num_labels=num_labels,
                problem_type=problem_type,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if params.flash_attention else None
            )
            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.eos_token_id

            training_args = TrainingArguments(
                output_dir=f"{results_base_path}/{params.cv_strategy}.fold_{i}",
                eval_strategy=IntervalStrategy.STEPS,
                per_device_train_batch_size=params.batch_size_per_gpu,
                per_device_eval_batch_size=params.batch_size_per_gpu,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
                eval_accumulation_steps=params.eval_accumulation_steps,
                learning_rate=params.learning_rate,
                weight_decay=0.01,
                num_train_epochs=params.epochs,
                save_steps=0.1,
                save_total_limit=1,
                seed=42,
                bf16=True,
                eval_steps=0.1,
                load_best_model_at_end=True,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,
                deepspeed="./deepspeed_config.json",
                optim="adamw_apex_fused",  # Unable when using deepspeed
                torch_compile=True
            )

            # Instantiate the trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            logger.info("Fine-tuning the %s model...", params.model_name_or_path)
            trainer.train()

            logger.info("Reporting error metrics for the current fold...")
            trainer.save_metrics(
                split="train",
                metrics=trainer.evaluate(tokenized_datasets["train"]),  # type: ignore
                combined=False
            )
            trainer.save_metrics(
                split="validation",
                metrics=trainer.evaluate(tokenized_datasets["validation"]),  # type: ignore
                combined=False
            )

    # Evaluate results on test dataset

    logger.info("- - - - -")
    logger.info("Splitting the data for final testing...")
    train_df = data_df.loc[splits_df["fold_1"] != 2]
    test_df = data_df.loc[splits_df["fold_1"] == 2]

    logger.info("Transforming the data to a HF dataset...")
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_pandas(train_df)
    dataset["test"] = Dataset.from_pandas(test_df)

    logger.info("Tokenizing train and test datasets...")
    tokenized_datasets = dataset.map(tokenize)

    # Workaround for regression problems
    if params.task_type == "regression":
        tokenized_datasets["train"] = tokenized_datasets["train"].cast(
            Features({**tokenized_datasets["train"].features, "label": Value("float32")})
        )
        tokenized_datasets["test"] = tokenized_datasets["test"].cast(
            Features({**tokenized_datasets["test"].features, "label": Value("float32")})
        )

    # Model to fine-tune
    model = AutoModelForSequenceClassification.from_pretrained(
        params.model_name_or_path,
        num_labels=num_labels,
        problem_type=problem_type,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if params.flash_attention else None
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=f"{results_base_path}/test",
        eval_strategy=IntervalStrategy.STEPS,
        per_device_train_batch_size=params.batch_size_per_gpu,
        per_device_eval_batch_size=params.batch_size_per_gpu,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        eval_accumulation_steps=params.eval_accumulation_steps,  # Useful when evaluation dataset is too large
        learning_rate=params.learning_rate,
        weight_decay=0.01,
        num_train_epochs=params.epochs,
        save_steps=0.1,
        save_total_limit=1,
        seed=42,
        bf16=True,
        eval_steps=0.1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        deepspeed="./deepspeed_config.json",
        optim="adamw_apex_fused",  # Unable when using deepspeed
        # torch_compile=True,  # Useful, but incompatible with DeepSpeed
        # dataloader_drop_last=True,  # May improve efficiency dropping incomplete batches
        # gradient_checkpointing=True  # Useful to avoid NCCL timeout
    )

    # Instantiate the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # eval_dataset=tokenized_datasets["test"].train_test_split(test_size=0.01, seed=42)["test"],  # Simple strategy to speed up experimentation on large datasets
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Fine-tuning the %s model...", params.model_name_or_path)
    trainer.train(params.resume_from_checkpoint)

    logger.info("Reporting error metrics for the test dataset...")
    trainer.save_metrics(
        split="train",
        metrics=trainer.evaluate(tokenized_datasets["train"]),  # type: ignore
        combined=False
    )
    trainer.save_metrics(
        split="test",
        metrics=trainer.evaluate(tokenized_datasets["test"]),  # type: ignore
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
