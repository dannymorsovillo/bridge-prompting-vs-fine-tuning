"""Exercise notebook routing without loading a model or allocating a GPU."""
import ast
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class RoutingTest(unittest.TestCase):
    def run_route(self, first_label, second_label):
        def response(label, labels):
            return {'prediction': label, 'latency_seconds': 0.1, 'input_tokens': 10,
                    **{f'{kind}_{item}': 0.5 for item in labels for kind in ['probability', 'log_score']}}
        first_labels = ['harmful', 'non_harmful']
        second_labels = ['harmless', 'antibias']
        second = Mock(return_value=response(second_label, second_labels))
        ns = dict(torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None)),
                  model=SimpleNamespace(device='cuda'), time=time,
                  stage1_labels=first_labels, stage2_labels=second_labels,
                  stage1_predict=lambda text: response(first_label, first_labels), stage2_predict=second)
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'bridge_two_stage_deltaai.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.FunctionDef) and node.name == 'predict_two_stage':
                    exec(compile(ast.Module(body=[node], type_ignores=[]), '<routing>', 'exec'), ns)
        return ns['predict_two_stage']('example'), second

    def test_harmful_skips_second_stage(self):
        result, second = self.run_route('harmful', 'antibias')
        second.assert_not_called()
        self.assertEqual(result['prediction'], 'harmful')
        self.assertIsNone(result['stage2_probability_antibias'])
        self.assertEqual(result['candidate_forward_passes'], 2)

    def test_non_harmful_uses_second_stage(self):
        for label in ['harmless', 'antibias']:
            with self.subTest(label=label):
                result, second = self.run_route('non_harmful', label)
                second.assert_called_once_with('example')
                self.assertEqual(result['prediction'], label)
                self.assertEqual(result['candidate_forward_passes'], 4)
                self.assertEqual(result['input_tokens'], 20)
