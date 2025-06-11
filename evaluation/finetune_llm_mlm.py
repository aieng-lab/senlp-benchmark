"""Script to fine-tune a language model for a MLM task.

This script implements the required steps to fine-tune a language model for a MLM task including
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
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

sys.path.append("../")
from utils.evaluation import (
    compute_metrics_mlm_hf as compute_metrics,
)


DATASETS_BASE_PATH = "../datasets"
RESULTS_BASE_PATH = "./results/finetuning"


logger = logging.getLogger(__name__)


def handle_args():
    """Handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a language model for a MLM task."
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


def align_pos_tags_with_tokens(tags, word_ids):
    """Make sure that after tokenization POS tags have the same length than tokens."""
    new_tags = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            tag = -100 if word_id is None else tags[word_id]
            new_tags.append(tag)
        elif word_id is None:
            # Special token
            new_tags.append(-100)
        else:
            # Same word as previous token
            tag = tags[word_id]
            new_tags.append(tag)

    return new_tags


class VerbOnlyMLMCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, mask_replace_prob=0.8, random_replace_prob=0.2):
        # Initialize parent collator with MLM enabled
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

        # Define the probabilities for replacing masked tokens
        self.mask_replace_prob = mask_replace_prob  # 80% of the time replace masked tokens with [MASK]
        self.random_replace_prob = random_replace_prob  # 20% of the time replace masked tokens with a random word

    def __call__(self, examples):
        # Separate POS tag sequences from the examples
        pos_tag_sequences = [ex["pos_tags"] for ex in examples]
        # Remove POS tags from examples to let the tokenizer pad other fields
        features = [{k: v for k, v in ex.items() if k != "pos_tags"} for ex in examples]

        # Pad the batch using the tokenizer (this pads input_ids, attention_mask, etc.)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        # Pad POS tag sequences to the same length as input_ids
        max_len = batch["input_ids"].shape[1]
        padded_pos_tags = [
            seq + ["<pad>"] * (max_len - len(seq))
            for seq in pos_tag_sequences
        ]

        # Create a mask of shape (batch_size, max_len) where True = verb token, False = otherwise
        verb_mask = torch.zeros(batch["input_ids"].shape, dtype=torch.bool)
        for i, tags in enumerate(padded_pos_tags):
            for j, tag in enumerate(tags):
                # 16 corresponds to the tag VERB
                if tag == 16:  # mark this position as a verb
                    verb_mask[i, j] = True

        # Clone input_ids to create labels. We will mask some of these labels.
        labels = batch["input_ids"].clone()

        # Initialize a probability matrix for masking with the given mlm_probability
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Never mask special tokens (like [CLS], [SEP], [PAD]): set their probability to 0
        special_tokens_mask = batch.get("special_tokens_mask")
        if special_tokens_mask is None:
            # If no special token mask provided, compute from tokenizer
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val_ids, already_has_special_tokens=True) 
                for val_ids in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Also set probability to 0 for non-verb tokens – only verbs remain maskable
        non_verb_mask = ~verb_mask  # inverse mask: True for tokens that are NOT verbs
        probability_matrix.masked_fill_(non_verb_mask, value=0.0)

        # Sample which tokens to mask (True = masked, with given probability on verbs only)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time, replace masked tokens with [MASK] token
        mask_token_id = self.tokenizer.mask_token_id
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob)).bool() & masked_indices
        batch["input_ids"][indices_replaced] = mask_token_id

        # 10% of the time, replace masked tokens with a random token (keeping them masked in labels)
        if self.random_replace_prob > 0:
            indices_random = (
                torch.bernoulli(torch.full(labels.shape, self.random_replace_prob)).bool() 
                & masked_indices & ~indices_replaced
            )
            random_tokens = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
            batch["input_ids"][indices_random] = random_tokens[indices_random]

        # (The remaining 10% of masked_indices are left with their original tokens, per BERT training scheme)

        # Set the labels in the batch and return. (input_ids in batch are already modified with masking)
        batch["labels"] = labels
        return batch


def main(params):
    """Main function to run the script"""

    logger.info("RUNNING THE SCRIPT TO FINE-TUNE A LANGUAGE MODEL FOR A MLM TASK")
    logger.info("Arguments passed by command-line: %s", params)
    logger.info("Execution date: %s", datetime.datetime.now())

    results_base_path = f"{RESULTS_BASE_PATH}/{params.dataset_name}/{params.model_name_or_path.split("/")[-1]}"

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
        "pos_tags": [[j[1] for j in i] for i in train]
    })
    dataset["test"] = Dataset.from_dict({
        "tokens": [[j[0] for j in i] for i in test],
        "pos_tags": [[j[1] for j in i] for i in test]
    })

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_name_or_path, add_prefix_space=True)

    def tokenize_and_align_pos_tags(x):
        tokenized_inputs = tokenizer(x["tokens"], is_split_into_words=True)
        all_pos_tags = x["pos_tags"]
        new_pos_tags = []
        for i, tags in enumerate(all_pos_tags):
            word_ids = tokenized_inputs.word_ids(i)
            new_pos_tags.append(align_pos_tags_with_tokens(tags, word_ids))

        tokenized_inputs["pos_tags"] = new_pos_tags

        return tokenized_inputs

    logger.info("Tokenizing train and test datasets...")
    tokenized_dataset = dataset.map(
        tokenize_and_align_pos_tags,
        batched=True,
        remove_columns=["tokens", "pos_tags"]
    )

    if params.max_length is not None:
        max_length = params.max_length
    else:
        max_length = tokenizer.model_max_length

    def group_texts(x):
        """Group texts to fit the context window."""

        # Concatenate all texts
        concatenated_x = {k: sum(x[k], []) for k in x.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_x[list(x.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // max_length) * max_length
        # Split by chunks of max_length
        result = {
            k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_x.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Grouping train and test datasets...")
    grouped_dataset = tokenized_dataset.map(
        group_texts,
        batched=True
    )

    # Model to fine-tune
    model = AutoModelForMaskedLM.from_pretrained(
        params.model_name_or_path,
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
        metric_for_best_model="accuracy",
        deepspeed="./deepspeed_config.json",
        optim="adamw_apex_fused",  # Unable when using deepspeed
        # torch_compile=True,  # Useful, but incompatible with DeepSpeed
        remove_unused_columns=False
    )

    # Instantiate the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped_dataset["train"],
        eval_dataset=grouped_dataset["test"],
        data_collator=VerbOnlyMLMCollator(tokenizer=tokenizer, mlm_probability=0.5),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Fine-tuning the %s model...", params.model_name_or_path)
    trainer.train(params.resume_from_checkpoint)

    logger.info("Reporting error metrics for the test dataset...")
    trainer.save_metrics(
        split="train",
        metrics=trainer.evaluate(grouped_dataset["train"]),  # type: ignore
        combined=False
    )
    trainer.save_metrics(
        split="test",
        metrics=trainer.evaluate(grouped_dataset["test"]),  # type: ignore
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
