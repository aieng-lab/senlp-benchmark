# :wrench: Pre-processing

In this folder, we include the data preparation process previous to splitting and tokenization. We read each dataset in parquet format, apply the corresponding pre-processing pipeline and store it again in the same format.

A pipeline consists of the following steps:
1. Normalizing white spaces,
2. Removing any kind of markdown and HTML styling, and
3. Masking by some common regex patterns, such as URLs, hashes, user mentions, and code blocks.

For verification purposes, we also implement a subprocess to generate diffs from random samples of texts.

Be aware of these paths where the data is read and stored:

```python
DATASETS_BASE_PATH = "../datasets"
DIFFS_BASE_PATH = "./diffs"
NEW_DATASETS_BASE_PATH = "./datasets"
```

Note that the NER and MLM tasks are not considered for this stage. However, the datasets have to be available in `DATASETS_BASE_PATH` to be used in subsequent stages (i.e., data splitting, fine-tuning).
