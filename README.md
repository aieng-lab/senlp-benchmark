## Evaluating Large Language Models on Non-Code Software Engineering Tasks

Source code to replicate the results reported in *F. Peña, S. Herbold, "Evaluating Large Language Models on Non-Code Software Engineering Tasks," 2025*.

### :memo: Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in code understanding and generation; however, their effectiveness on non‐code Software Engineering (SE) tasks remains underexplored. We present the first comprehensive benchmark, which we name 'Software Engineering Language Understanding' (SELU), for evaluating LLMs on 17 non‐code tasks, spanning from identifying whether a requirement is functional or non-functional to estimating the effort and complexity of backlog items. SELU covers classification, regression, Named Entity Recognition (NER), and Masked Language Modeling (MLM) targets, with data drawn from diverse sources such as code repositories, issue tracking systems, and developer forums. We fine-tune 22 open-source LLMs, prompt two proprietary alternatives, and train two baselines. Performance is measured using metrics such as F1-macro, SMAPE, F1-micro, and accuracy, and compared via the Bayesian signed-rank test. Our results show that moderate-scale decoder-only models consistently form a top-tier, exhibiting high mean performance and low across-task variance, while domain adaptation via code-focused pre-training might yield only modest improvements. These insights guide model selection for non-code SE workflows and highlight directions for expanding SELU to generative and design-oriented scenarios.

### :robot: Models

We publish all fine-tuned open-source LLMs on Hugging Face on the [Chair of AI Engineering, University of Passau
](https://huggingface.co/aieng-lab) organization under MIT license. Each model follows a common naming syntax: *`{model}_{task}`*. For instance, *`CodeLlama-7b-hf_bug-issue`* is the fine-tuned version of [meta-llama/CodeLlama-7b-hf](https://huggingface.co/meta-llama/CodeLlama-7b-hf) for the *bug_issue* task.

### :card_index_dividers: Repository organization

- `datasets`: Inventory of the 17 non-code SE tasks included in SELU and their respective datasets.
- `evaluation`: Scripts to fine-tune/prompt and evaluate the models on the different tasks.
- `preprocessing`: Scripts to prepare the data previous to splitting and tokenization.
- `utils`: Common functions used during pre-processing and evaluation.

### :gear: Setup

All of our experiments run on a server with 8 NVIDIA A100 GPUs.

### :bookmark: Cite as

```

@misc{peña2025evaluatinglargelanguagemodels,
  title={Evaluating Large Language Models on Non-Code Software Engineering Tasks}, 
  author={Fabian C. Peña and Steffen Herbold},
  year={2025},
  eprint={2506.10833},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2506.10833}, 
}
```
