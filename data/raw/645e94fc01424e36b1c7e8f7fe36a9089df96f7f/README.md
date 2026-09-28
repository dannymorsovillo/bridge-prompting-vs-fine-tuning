---
license: other
license_name: "original-license-preserving-aggregation-with-per-example-source-licenses"
language:
  - en
task_categories:
  - text-classification
size_categories:
  - 10K<n<100K
tags:
  - benchmark
  - evaluation
  - bias-detection
  - policy-analysis
pretty_name: "BRIDGE: Bias Register-Integrated Dataset for Generalized Evaluation (Anonymous Submission)"
---

# BRIDGE: Bias Register-Integrated Dataset for Generalized Evaluation (Anonymous Submission)

## Dataset Description

BRIDGE is a three-way classification benchmark spanning formal institutional text and informal social-media text. Each example is labeled as `harmful`, `harmless`, or `antibias`. The dataset is designed to test whether models can distinguish harmful framing, explicit anti-bias language, and neutral text across different source types, eras, and registers.

This card is written for an anonymous ICLR 2027 submission and intentionally omits author and institution information.

The benchmark contains 19,420 English examples: 18,015 natural source examples and 1,405 generated antibias examples.

### Supported Tasks and Leaderboards

- **Task:** Three-way bias classification across formal and informal text
- **Metrics:** Accuracy, macro F1, precision, recall
- **Evaluation views:** Overall scores plus breakdowns by `register`, `era`, `source`, and `bias_label`
- **Leaderboard:** No public leaderboard is bundled in this repository; the dataset is intended for offline evaluation and analysis

### Languages

- English

## Dataset Structure

### Data Fields

- `sentence_id` (string): Unique identifier; original IDs are preserved and are not necessarily consecutive
- `text` (string): The excerpt to classify
- `bias_label` (string): Primary label, one of `harmful`, `harmless`, or `antibias`
- `source` (string): Canonical source family
- `register` (string): `formal` or `informal`
- `era` (string): Source-era bucket: `modern`, `contemporary`, `pre-1965`, or `1930s-1970s`
- `domain` (string): Domain associated with the annotated bias; harmless examples use `no bias`
- `is_augmented` (string): `yes` for generated examples and `no` for natural source examples

### Data Splits

This release is provided as a single benchmark set rather than a train/validation/test package.

| Split | # Examples |
|-------|-----------:|
| final benchmark | 19,420 |

### High-Level Composition

| Bias label | Count |
|------------|------:|
| harmful | 7,806 |
| harmless | 6,370 |
| antibias | 5,244 |

| Register | Count |
|----------|------:|
| formal | 14,410 |
| informal | 5,010 |

| Provenance | Count |
|------------|------:|
| Natural source examples | 18,015 |
| Generated antibias examples | 1,405 |

| Source family | Count | Original license |
|---------------|------:|------------------|
| Pile of Law | 8,097 | CC BY-SA 4.0 |
| Social Bias Frames | 5,010 | CC BY 4.0 |
| On the Books: Jim Crow Laws | 2,313 | CC BY-NC-SA 4.0 |
| Mapping Inequality: Redlining Data | 1,519 | CC BY-NC-SA 4.0 |
| GovReport Summarization | 1,134 | CC0 |
| Comparative Agendas Project (CAP) | 1,110 | CC BY-NC-SA 4.0 |
| LSC Eviction Laws Database | 237 | CC0 |

Source-family counts include both natural examples and generated examples associated with that source. Use `is_augmented` to distinguish them.

## Considerations for Using the Data

### Limitations

- The dataset mixes legal, policy, government, and social-media text, so performance may vary by genre.
- Register, source, era, topic, and genre are correlated. Differences between the natural formal and informal subsets should not be interpreted as isolating the causal effect of register.
- Labels were harmonized across multiple upstream sources, so some annotation noise and label drift are possible.
- The label distribution is not perfectly balanced.
- Generated antibias examples account for 1,405 examples (7.2%); the `is_augmented` field supports evaluation with or without these examples.
- Some examples contain offensive or harmful language.
- The dataset is best used for evaluation and robustness analysis, not as a standalone measure of fairness.

### Licensing Information

BRIDGE is a composite dataset built from multiple upstream sources and is released as a license-preserving aggregated benchmark. The benchmark contains source-derived text from datasets released under multiple licenses, including CC BY-SA 4.0, CC BY 4.0, CC BY-NC-SA 4.0, and CC0. We therefore do not apply a single overriding license to all source-derived text.

Each example includes source-family attribution through the `source` field. Source-level licenses are listed above; the dataset does not include a separate per-example license field. Source-derived text remains governed by its original source license.

Newly created BRIDGE annotations, labels, metadata, prompts, and evaluation code are released under CC BY-NC-SA 4.0 unless otherwise noted. Users are responsible for complying with the applicable upstream license terms.

## Citation

[CITATION WITHHELD FOR ANONYMOUS REVIEW]
