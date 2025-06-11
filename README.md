## Evaluating Large Language Models on Non-Code Software Engineering Tasks

Source code to replicate the results reported in *F. Peña, S. Herbold, "Evaluating Large Language Models on Non-Code Software Engineering Tasks," 2025*.

### :memo: Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in code understanding and generation; however, their effectiveness on non‐code Software Engineering (SE) tasks remains underexplored. We present the first comprehensive benchmark for evaluating LLMs on 17 non‐code tasks, spanning from identifying whether a requirement is functional or non-functional to estimating the effort and complexity of backlog items. Our benchmark covers classification, regression, Named Entity Recognition (NER), and Masked Language Modeling (MLM) targets, with data drawn from diverse sources such as code repositories, issue tracking systems, and developer forums. We fine-tune 22 open-source LLMs, prompt 2 proprietary alternatives, and train 2 baselines. Performance is measured using metrics such as F1-macro, SMAPE, F1-micro, and accuracy, and compared via the Bayesian signed-rank test. Our results show that moderate-scale decoder-only models consistently form a top-tier, exhibiting high mean performance and low across-task variance, while domain adaptation via code-focused pre-training might yield only modest improvements. These insights guide model selection for non-code SE workflows and highlight directions for expanding the benchmark to generative and design-oriented scenarios.

### :card_index_dividers: Repository organization

...

### :gear: Setup

All of our experiments run on a server with 8 NVIDIA A100 GPUs.

### :bookmark: Cite as

```
@misc{pena2025benchmark,
  author    = {Fabian Peña and Steffen Herbold},
  title     = {Evaluating Large Language Models on Non-Code Software Engineering Tasks},
  year      = {2025}
}
```
