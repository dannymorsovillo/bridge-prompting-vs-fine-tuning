# BRIDGE research design

Agreed scope: compare prompting and QLoRA within a small model, then compare the fine-tuned small model with a larger prompted model. Model choices are provisional pending a GPU pilot.

| Condition | Starting checkpoint | Training | Demonstrations at inference |
|---|---|---|---|
| small_zero_shot | Qwen/Qwen2.5-3B-Instruct | None | None |
| small_few_shot | Qwen/Qwen2.5-3B-Instruct | None | One per class (three total) |
| small_qlora | Qwen/Qwen2.5-3B-Instruct | BRIDGE training split | None |
| large_zero_shot | Qwen/Qwen2.5-7B-Instruct | None | None |
| large_few_shot | Qwen/Qwen2.5-7B-Instruct | None | Same three training examples as small_few_shot |

Every condition receives the same task instructions and label definitions, plus the excerpt. Targets are exactly `harmful`, `harmless`, or `antibias`. Metadata is retained for auditing and subgroup evaluation, excluded from model input. All conditions use the same held-out test examples. The adapted model starts from the same instruction-tuned 3B checkpoint as its prompting baselines.

Primary metric: macro-F1 over all three classes. Secondary metrics: accuracy, per-class precision/recall/F1, confusion matrix, invalid-output rate, warmed-up inference latency, throughput, training time, and peak GPU memory. Invalid outputs count as errors, not discarded predictions. Report natural-only and generated-only evaluation views with sample counts. Record missing-class handling for subgroup macro-F1.

Choose prompt wording, demonstrations, output parsing, truncation rules, and training settings with training/validation data only. Before final testing, define an acceptable macro-F1 gap for “comparable” and the uncertainty analysis. Do not interpret a nonsignificant difference as proof of equivalence. Paired bootstrap intervals should resample duplicate groups rather than treating related excerpts as independent.

Train the small model once for the initial experiment. If resources permit, repeat key results with multiple seeds. Ordinary LoRA comparison and training-size learning curves are optional extensions.

## Prepared data

Pinned dataset: [bridge-benchmark-2026/BRIDGE](https://huggingface.co/datasets/bridge-benchmark-2026/BRIDGE/tree/645e94fc01424e36b1c7e8f7fe36a9089df96f7f).
Revision: `645e94fc01424e36b1c7e8f7fe36a9089df96f7f`.
The raw CSV and original dataset card are saved under `data/raw/<revision>/`. The card preserves the source-license table; retain it with derived data.

The actual downloaded snapshot has 19,420 rows and eight fields. It supersedes the earlier notebook's saved 19,421-row output. No models have been trained or evaluated in this preparation step.

Cleaning excludes empty normalized text and all normalized-text groups with inconsistent labels. Normalize with Unicode NFKC, case folding, and collapsed whitespace; preserve original text for model inputs. Retain the first row in the pinned CSV for consistent-label normalized duplicates, and preserve all original IDs and provenance in the cleaning audit. These are explicit exclusion rules, not corrections to ground-truth labels.

After cleaning: 18,560 examples; 813 duplicate copies removed; 47 conflicting-label rows excluded. Near-duplicates are retained but grouped using punctuation-insensitive token identity or word-trigram-set Jaccard similarity >= 0.90, including transitive links. There are 299 groups with multiple retained rows, with at most six rows in one group.

Seed 42, approximately 80/10/10, with groups kept intact and class proportions approximately preserved:

| Split | Harmful | Harmless | Antibias | Total |
|---|---:|---:|---:|---:|
| Training | 6,140 | 4,676 | 4,030 | 14,846 |
| Validation | 768 | 585 | 504 | 1,857 |
| Test | 768 | 585 | 504 | 1,857 |

This is an in-distribution split. Lexical grouping does not establish semantic-paraphrase isolation or common-document isolation; BRIDGE does not supply original document IDs. Source and label are associated, so source breakdowns matter and performance is not evidence of general fairness. Generated examples remain included and flagged. Cross-source duplicate provenance is available in the audit; a representative row alone does not capture all provenance.

Outputs in `data/processed/`:

- `train.csv`, `validation.csv`, `test.csv`: original fields plus `group_id` and `split`.
- `split_ids.csv`: saved assignment to reuse in every experiment.
- `cleaning_audit.csv`: every original row, its disposition, representative, group, and assigned split where applicable.
- `near_duplicate_pairs.csv`: detected links and similarity scores.
- `split_summary.csv`: label, source, register, era, and augmentation counts by split.
- `manifest.json`: revision, input/output checksums, settings, counts, versions, and limitations.

Rebuild offline from the project root with `.venv/bin/python scripts/prepare_bridge.py`. The script verifies the raw checksum. It deterministically rebuilds the same outputs; do not change splitting settings after beginning model selection. Run the preparation checks with `.venv/bin/python -m unittest discover -s tests -v`.

## Next execution step

Build a separate BRIDGE evaluation notebook. Use the saved training split to select three demonstrations and use validation to check zero-shot/few-shot prompts, parsing, token lengths, and GPU memory. Do not use the test set during this development. Pin model revisions and the successful software environment during the pilot. After that, add assistant-label-only QLoRA training and evaluate the frozen five conditions together.

Hardware is not yet confirmed; Colab T4 is a provisional planning assumption from the supplied notebook. Measure feasibility and cost before launching full runs. Keep inference hardware and precision comparable, and record any unavoidable differences. The 3B checkpoint uses the Qwen Research license; the 7B checkpoint uses Apache-2.0 (see their model cards).
