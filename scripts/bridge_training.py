"""Answer-only training examples and padding for BRIDGE causal-LM adapters."""
import torch


def make_training_example(text, label, *, prompt_encoder, candidate_ids, max_input_tokens):
    prompt = prompt_encoder(text)['input_ids'][0].tolist()
    if len(prompt) > max_input_tokens:
        raise ValueError('Prompt exceeds the token budget; no truncation is applied.')
    answer = list(candidate_ids[label])
    if not answer:
        raise ValueError('Answer must contain tokens.')
    ids = prompt + answer
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt) + answer}


class AnswerOnlyCollator:
    """Right-pad by position so real EOS tokens remain supervised."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        size = max(len(e['input_ids']) for e in examples)
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for e in examples:
            padding = size - len(e['input_ids'])
            batch['input_ids'].append(e['input_ids'] + [self.pad_token_id] * padding)
            batch['attention_mask'].append(e['attention_mask'] + [0] * padding)
            batch['labels'].append(e['labels'] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
