import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.bridge_helpers import LABELS, make_prompt_encoder, predict


class SharedHelpersTest(unittest.TestCase):
    def test_messages_keep_pairs_and_final_query(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: messages)
        examples = [{"text": str(i), "bias_label": label} for i, label in enumerate(LABELS)]
        messages = make_prompt_encoder(tokenizer, examples)("query")
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant', 'user'])
        self.assertEqual(messages[-1]['content'], 'query')
        self.assertEqual([messages[i]['content'] for i in [2, 4, 6]], LABELS)

    def test_candidate_alignment_and_complete_sequence_score(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Model:
            device = torch.device('cpu')
            def eval(self):
                pass
            def __call__(self, input_ids, logits_to_keep, **kwargs):
                # Each position's distribution depends on its context token.
                logits = torch.nn.functional.one_hot(input_ids, num_classes=5).float() * 2
                return SimpleNamespace(logits=logits[:, -logits_to_keep:])

        candidates = dict(zip(LABELS, [[1, 4], [2, 3, 4], [3, 4]]))
        encoder = lambda text: Batch(input_ids=torch.tensor([[0, 1]]))
        with patch('torch.cuda.synchronize'):
            result = predict('query', model=Model(), candidate_ids=candidates,
                             prompt_encoder=encoder, max_input_tokens=2)
        expected = []
        for label in LABELS:
            targets = candidates[label]
            contexts = [1] + targets[:-1]
            score = sum(torch.log_softmax(torch.nn.functional.one_hot(torch.tensor(context), 5).float() * 2, 0)[target].item()
                        for context, target in zip(contexts, targets))
            self.assertAlmostEqual(result['log_score_' + label], score, places=5)
            expected.append(score)
        probabilities = torch.softmax(torch.tensor(expected), 0)
        for label, probability in zip(LABELS, probabilities):
            self.assertAlmostEqual(result['probability_' + label], probability.item(), places=6)
        self.assertEqual(result['prediction'], LABELS[probabilities.argmax().item()])


if __name__ == '__main__':
    unittest.main()
