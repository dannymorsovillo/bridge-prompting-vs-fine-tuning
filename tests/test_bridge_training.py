import unittest
import torch
from scripts.bridge_training import make_training_example, AnswerOnlyCollator


class TrainingDataTest(unittest.TestCase):
    def test_masks_prompt_but_preserves_real_eos(self):
        encode = lambda text: {'input_ids': torch.tensor([[9, 4]])}
        item = make_training_example('text', 'antibias', prompt_encoder=encode,
                                     candidate_ids={'antibias': [3, 4]}, max_input_tokens=2)
        self.assertEqual(item['labels'], [-100, -100, 3, 4])
        longer = {'input_ids': [9, 4, 2, 3, 4], 'attention_mask': [1]*5,
                  'labels': [-100, -100, 2, 3, 4]}
        batch = AnswerOnlyCollator(4)([item, longer])
        self.assertEqual(batch['input_ids'][0].tolist(), [9, 4, 3, 4, 4])
        self.assertEqual(batch['labels'][0].tolist(), [-100, -100, 3, 4, -100])
        self.assertEqual(batch['attention_mask'][0].tolist(), [1, 1, 1, 1, 0])
        # The model's causal shift supervises first answer from last prompt position.
        self.assertEqual(batch['labels'][0, 1:].tolist(), [-100, 3, 4, -100])

    def test_overlength_is_rejected_not_truncated(self):
        with self.assertRaises(ValueError):
            make_training_example('x', 'harmful', prompt_encoder=lambda text: {'input_ids': torch.tensor([[1,2,3]])},
                                  candidate_ids={'harmful':[4]}, max_input_tokens=2)


if __name__ == '__main__':
    unittest.main()
