"""Rebuild pinned BRIDGE splits offline: .venv/bin/python scripts/prepare_bridge.py."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVISION = "645e94fc01424e36b1c7e8f7fe36a9089df96f7f"
RAW = ROOT / "data" / "raw" / REVISION / "BRIDGE_final_ICLR.csv"
RAW_SHA256 = "42447ee0b14d495bb508a3c727c7b7a64c0851b4ff4d00dd12ff17226a5192a4"
OUT = ROOT / "data" / "processed"
LABELS = ["harmful", "harmless", "antibias"]
SPLITS = ["train", "validation", "test"]
RATIOS = np.array([0.8, 0.1, 0.1])
SEED = 42
THRESHOLD = 0.9


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def near_groups(texts, threshold=THRESHOLD):
    """Connected components of token-identical or >=0.9 word-trigram Jaccard pairs.

    Global-frequency prefix filtering finds all qualifying trigram-set pairs,
    without comparing every pair. This detects lexical reuse, not paraphrases
    or all excerpts originating in the same document.
    """
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    sets, edges, token_seen = [], [], {}
    for i, text in enumerate(texts):
        tokens = tuple(re.findall(r"\w+", text))
        # Token equality catches punctuation-only variants; retain all rows.
        if tokens and tokens in token_seen:
            j = token_seen[tokens]
            union(i, j)
            edges.append((j, i, "token_identity", 1.0))
        elif tokens:
            token_seen[tokens] = i
        sets.append(set(zip(tokens, tokens[1:], tokens[2:])))
    frequencies = Counter(s for shingles in sets for s in shingles)
    postings = defaultdict(list)
    for i in sorted(range(len(sets)), key=lambda j: (len(sets[j]), j)):
        current = sets[i]
        if not current:
            continue
        ordered = sorted(current, key=lambda s: (frequencies[s], s))
        prefix = ordered[:len(current) - math.ceil(threshold * len(current)) + 1]
        candidates = set(j for s in prefix for j in postings[s])
        for j in sorted(candidates):
            if len(sets[j]) < threshold * len(current):
                continue
            similarity = len(current & sets[j]) / len(current | sets[j])
            if similarity >= threshold:
                union(i, j)
                edges.append((j, i, "trigram_jaccard", similarity))
        for shingle in prefix:
            postings[shingle].append(i)
    return [find(i) for i in range(len(texts))], edges


def assign_splits(clean):
    """Greedy grouped stratification; minimize squared class-target deviations."""
    counts = pd.crosstab(clean["group_id"], clean["bias_label"]).reindex(
        columns=LABELS, fill_value=0
    ).sort_index()
    targets = RATIOS[:, None] * counts.sum(axis=0).to_numpy()[None, :]
    actual = np.zeros_like(targets)
    rng = np.random.default_rng(SEED)
    shuffled = list(rng.permutation(len(counts)))
    order = sorted(shuffled, key=lambda i: -counts.iloc[i].sum())
    assignments = {}
    for i in order:
        vector = counts.iloc[i].to_numpy()
        change = (((actual + vector - targets) ** 2 - (actual - targets) ** 2)
                  / np.maximum(targets, 1)).sum(axis=1)
        chosen = int(np.argmin(change))
        actual[chosen] += vector
        assignments[counts.index[i]] = SPLITS[chosen]
    return clean["group_id"].map(assignments)


def prepare():
    if digest(RAW) != RAW_SHA256:
        raise ValueError("Raw CSV checksum differs from the pinned snapshot.")
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    original_columns = list(raw.columns)
    assert raw["sentence_id"].is_unique
    assert set(raw["bias_label"]) == set(LABELS)
    assert set(raw["is_augmented"]) <= {"yes", "no"}
    raw["text_key"] = raw["text"].map(normalize)
    blank = raw["text_key"].eq("")
    valid = raw.loc[~blank].copy()
    label_counts = valid.groupby("text_key")["bias_label"].nunique()
    conflicts = valid["text_key"].isin(label_counts[label_counts > 1].index)
    eligible = valid.loc[~conflicts].copy()
    # Stable representative: first row in the pinned file. Lineage retains all
    # original IDs and source/augmentation metadata, even for removed copies.
    clean = eligible.drop_duplicates("text_key", keep="first").reset_index(drop=True)
    representatives = clean.set_index("text_key")["sentence_id"]
    groups, edges = near_groups(clean["text_key"].tolist())
    clean["group_id"] = ["g_" + str(clean.iloc[g]["sentence_id"]) for g in groups]
    clean["split"] = assign_splits(clean)

    assert clean["text_key"].is_unique
    assert clean.groupby("group_id")["split"].nunique().max() == 1
    for a, b, _, _ in edges:
        assert clean.iloc[a]["split"] == clean.iloc[b]["split"]
    for split in SPLITS:
        assert set(clean.loc[clean["split"].eq(split), "bias_label"]) == set(LABELS)

    OUT.mkdir(parents=True, exist_ok=True)
    export_columns = original_columns + ["group_id", "split"]
    for split in SPLITS:
        clean.loc[clean["split"].eq(split), export_columns].to_csv(
            OUT / f"{split}.csv", index=False
        )
    clean[["sentence_id", "group_id", "split"]].to_csv(OUT / "split_ids.csv", index=False)

    lineage = raw.copy()
    lineage["status"] = "kept"
    lineage.loc[blank, "status"] = "blank_text"
    lineage.loc[lineage["text_key"].isin(label_counts[label_counts > 1].index), "status"] = "label_conflict"
    lineage["representative_id"] = lineage["text_key"].map(representatives).fillna("")
    duplicates = lineage["status"].eq("kept") & lineage["sentence_id"].ne(lineage["representative_id"])
    lineage.loc[duplicates, "status"] = "normalized_duplicate"
    lookup = clean.set_index("text_key")
    for column in ["group_id", "split"]:
        lineage[column] = lineage["text_key"].map(lookup[column]).fillna("")
    lineage.drop(columns="text_key").to_csv(OUT / "cleaning_audit.csv", index=False)
    edge_rows = [(clean.iloc[a]["sentence_id"], clean.iloc[b]["sentence_id"], kind, score)
                 for a, b, kind, score in edges]
    pd.DataFrame(edge_rows, columns=["sentence_id_a", "sentence_id_b", "method", "similarity"]).to_csv(
        OUT / "near_duplicate_pairs.csv", index=False
    )
    summary_rows = []
    for field in ["bias_label", "source", "register", "era", "is_augmented"]:
        for (split, value), count in clean.groupby(["split", field]).size().items():
            summary_rows.append((split, field, value, int(count)))
    pd.DataFrame(summary_rows, columns=["split", "field", "value", "count"]).to_csv(
        OUT / "split_summary.csv", index=False
    )
    group_sizes = clean.groupby("group_id").size()
    manifest = {
        "dataset": "bridge-benchmark-2026/BRIDGE", "revision": REVISION,
        "raw_file": str(RAW.relative_to(ROOT)), "raw_sha256": RAW_SHA256,
        "seed": SEED, "target_ratios": dict(zip(SPLITS, RATIOS.tolist())),
        "labels": LABELS, "rows_raw": len(raw), "rows_clean": len(clean),
        "cleaning_counts": {k: int(v) for k, v in lineage["status"].value_counts().items()},
        "conflicting_normalized_groups": int((label_counts > 1).sum()),
        "normalization": "NFKC, casefold, collapse whitespace; original input text preserved",
        "representative_policy": "First row in pinned CSV; all provenance retained in cleaning_audit.csv",
        "grouping": "Connected components: token identity or word-trigram-set Jaccard >= 0.9",
        "near_duplicate_threshold": THRESHOLD,
        "groups_with_multiple_rows": int((group_sizes > 1).sum()),
        "largest_group": int(group_sizes.max()),
        "split_rows": {s: int(clean["split"].eq(s).sum()) for s in SPLITS},
        "label_counts": {s: {l: int(((clean['split'] == s) & (clean['bias_label'] == l)).sum())
                             for l in LABELS} for s in SPLITS},
        "validation": "No normalized-text or detected-group overlap; all retained rows assigned once; all classes present",
        "limitations": [
            "Lexical grouping cannot guarantee paraphrase or common-document isolation; original document IDs unavailable.",
            "This is an in-distribution split, not a held-out-source experiment.",
            "Generated examples retained; report natural-only and augmented-only test metrics as separate views.",
            "Representative source metadata may hide cross-source duplicates; consult the lineage audit.",
        ],
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "script_sha256": digest(Path(__file__)),
        "output_sha256": {p.name: digest(p) for p in sorted(OUT.glob("*.csv"))},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["rows_raw", "rows_clean", "cleaning_counts", "groups_with_multiple_rows", "largest_group", "split_rows", "label_counts"]}, indent=2))
    return manifest


if __name__ == "__main__":
    prepare()
