# Comparing Prompting and Fine-Tuning for Bias Classification

Project investigating whether a smaller language model, fine-tuned using parameter-efficient methods, can achieve classification performance comparable to a larger model using prompting alone.

## Dataset

The project uses [BRIDGE](https://huggingface.co/datasets/bridge-benchmark-2026/BRIDGE), a dataset of English excerpts from legal, government, policy, and social-media sources. Each excerpt is assigned one of three labels:

- **Harmful:** biased, discriminatory, or demeaning framing.
- **Harmless:** neutral language without harmful bias or explicit opposition to bias.
- **Antibias:** explicit opposition to bias or discrimination.

Models receive the excerpt text without source or other dataset metadata. Fixed training, validation, and test splits support consistent comparisons, with duplicate handling intended to reduce overlap between splits.

## Research Approach

The study compares zero-shot prompting, few-shot prompting, and QLoRA fine-tuning of a smaller model. It also compares the fine-tuned smaller model with a larger prompted model.

The main research questions are:

1. Does QLoRA fine-tuning improve the smaller model's classification performance over prompting alone?
2. Can the fine-tuned smaller model achieve performance comparable to the larger prompted model?
3. How do the approaches compare in inference speed, memory use, and computational cost?

Evaluation emphasizes macro-F1 and accuracy, alongside per-class performance and computational efficiency. Training and validation data support development, while the test set is reserved for final evaluation.

## Project Scope

The repository contains notebooks and supporting code for dataset inspection, cleaning, model evaluation, and experiment analysis. GPU experiments run on NCSA DeltaAI.

Results measure agreement with BRIDGE's labels. They do not establish a general measure of fairness, and differences in source material and annotation practices are considered when interpreting model performance. Dataset source attribution and applicable licenses are preserved.
