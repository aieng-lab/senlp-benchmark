# :microscope: Evaluation

In this folder, we include all the scripts to fine-tune/prompt and evaluate the LLMs on the different binary, multi-class, multi-label classification, regression, Named Entity Recognition (NER), and Masked Language Modeling (MLM) tasks.

## :carpentry_saw: Data splitting

The `split_data.sh` script is intended to split each dataset while controlling experimentation elements such as percentage of instances reserved for testing and number of folds for cross-validation. The resulting file is a CSV with the same number rows of the source dataset but with columns corresponding to the subset the instance is assigned to in each fold: training (1), validation (0), and testing (2).

**NOTE:** Although the implementation supports cross-validation, we only report results at hold-out level due to compute budget.

## :robot: Models

This is a list of the open-source and proprietary LLMs selected for evaluation. The scripts `finetune_llm.sh`, `finetune_llm_ner.sh`, `finetune_llm_mlm.sh`, and `prompt_llm.py` run the fine-tuning and prompting processes according to the parameters configured.

|     **Model**     | **Size** | **Architecture** | **Domain adaptation** | **License** |
|:-----------------:|:--------:|:----------------:|:---------------------:|:-----------:|
|     BERT base     |   110m   |   encoder-only   |       Generalist      | Open-source |
|     BERT large    |   340m   |   encoder-only   |                       |             |
|    RoBERTa base   |   125m   |   encoder-only   |                       |             |
|   RoBERTa large   |   355m   |   encoder-only   |                       |             |
|  ModernBERT base  |   150m   |   encoder-only   |                       |             |
|  ModernBERT large |   396m   |   encoder-only   |                       |             |
|    GPT-2 small    |   117m   |   decoder-only   |                       |             |
|    GPT-2 medium   |   345m   |   decoder-only   |                       |             |
|    GPT-2 large    |   774m   |   decoder-only   |                       |             |
|      GPT-2 xl     |   1.5b   |   decoder-only   |                       |             |
|    Llama 3.2 1b   |    1b    |   decoder-only   |                       |             |
|    Llama 3.2 3b   |    3b    |   decoder-only   |                       |             |
|      T5 small     |   60.5m  |  encoder-decoder |                       |             |
|      T5 base      |   223m   |  encoder-decoder |                       |             |
|      T5 large     |   738m   |  encoder-decoder |                       |             |
|       T5 3b       |    3b    |  encoder-decoder |                       |             |
|   CodeBERT base   |   125m   |   encoder-only   |     Domain-adapted    | Open-source |
|    CodeLlama 7b   |    7b    |   decoder-only   |                       |             |
|   StarCoder2 3b   |    3b    |   decoder-only   |                       |             |
|   StarCoder2 7b   |    7b    |   decoder-only   |                       |             |
|    CodeT5+ 220m   |   220m   |  encoder-decoder |                       |             |
|   CodeT5+ 7700m   |   770m   |  encoder-decoder |                       |             |
|       GPT-4o      |     -    |   decoder-only   |       Generalist      | Proprietary |
| Claude 3.5 Sonnet |     -    |   decoder-only   |                       |             |

Finally, the script `train_baseline.py` runs the training and model selection processes according to the parameters configured.

## :bar_chart: Results

In the subfolder `results`, we include the performace scores calculated for the models on each task, in this way:

- finetuning: Scores for the fine-tuned open-source LLMs.
- text_generation: Scores for the proprietary LLMs.
- sklearn: Scores for the TFIDF+XGBoost baselines.
- fasttext: Scores for the FastText baselines.

For both baselines, the subfolder for each task also includes the best hyper-parameters choosen during model selection.

Be aware of these paths where the data is read and stored:

```python
DATASETS_BASE_PATH = "../preprocessing/datasets"
SPLITS_BASE_PATH = "./splits"
```
