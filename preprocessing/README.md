# :wrench: Pre-processing

In this folder, we include the main implementation to prepare the data previous to tokenization. We read each dataset in parquet format, apply the corresponding pre-processing pipeline and store them again in the same format. For verification purposes, we also implement a subprocess to generate diffs from random samples of texts.

Be aware of these paths where the data is read and stored:

```python
DATASETS_BASE_PATH = "../datasets"
DIFFS_BASE_PATH = "./diffs"
NEW_DATASETS_BASE_PATH = "./datasets"
```

Note that the NER and MLM tasks are not considered for this stage. However, the datasets have to be available in `DATASETS_BASE_PATH` to be used in subsequent stages (i.e., data splitting, fine-tuning).
