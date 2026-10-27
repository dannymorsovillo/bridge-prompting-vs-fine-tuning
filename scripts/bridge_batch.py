"""Validation for explicitly selected, completed recovery checkpoints."""
import json
from pathlib import Path


def validate_checkpoint(path):
    path = Path(path)
    for name in ['trainer_state.json', 'optimizer.pt', 'scheduler.pt',
                 'adapter_config.json', 'rng_state.pth', 'checkpoint_complete.json']:
        if not (path / name).is_file():
            raise ValueError(f'Incomplete checkpoint: {path / name} missing. Select an earlier completed checkpoint.')
    if not any((path / name).is_file() for name in ['adapter_model.safetensors', 'adapter_model.bin']):
        raise ValueError('Checkpoint adapter weights missing.')
    state = json.loads((path / 'trainer_state.json').read_text())
    marker = json.loads((path / 'checkpoint_complete.json').read_text())
    if state['global_step'] != marker['global_step']:
        raise ValueError('Checkpoint completion marker does not match Trainer state.')
    best = state.get('best_model_checkpoint')
    if best and not Path(best).is_dir():
        raise ValueError(f'Referenced best checkpoint is missing: {best}')
    return path
