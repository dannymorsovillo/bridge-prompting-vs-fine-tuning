"""Checks for the split pipeline's leakage-sensitive behavior."""
import itertools
import re
import unittest

import pandas as pd

from scripts.prepare_bridge import assign_splits, near_groups, normalize


class PreparationTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize('  Ｈello\nWORLD '), normalize('hello world'))

    def test_prefix_search_matches_brute_force(self):
        texts = [
            ' '.join('word' + str(i) for i in range(n))
            for n in range(3, 65)
        ]
        texts += ['short phrase', 'short, phrase!', 'unrelated example here']
        groups, edges = near_groups(texts)
        actual = {(min(a, b), max(a, b)) for a, b, kind, _ in edges
                  if kind == 'trigram_jaccard'}
        sets = []
        for text in texts:
            tokens = re.findall(r'\w+', text)
            sets.append(set(zip(tokens, tokens[1:], tokens[2:])))
        expected = set()
        for a, b in itertools.combinations(range(len(texts)), 2):
            if sets[a] and sets[b] and len(sets[a] & sets[b]) / len(sets[a] | sets[b]) >= 0.9:
                expected.add((a, b))
        self.assertEqual(actual, expected)
        self.assertEqual(groups[-3], groups[-2])
        self.assertNotEqual(groups[-3], groups[-1])

    def test_groups_preserved_with_conflicting_near_labels(self):
        rows = [(f'g{i}', label) for i in range(90)
                for label in ['harmful', 'harmless', 'antibias']]
        df = pd.DataFrame(rows, columns=['group_id', 'bias_label'])
        result = assign_splits(df)
        self.assertTrue(result.equals(assign_splits(df)))
        df['split'] = result
        self.assertEqual(df.groupby('group_id')['split'].nunique().max(), 1)
        self.assertEqual(result.value_counts().to_dict(), {'train': 216, 'validation': 27, 'test': 27})


if __name__ == '__main__':
    unittest.main()
