"""Execute QLoRA notebook code in the active module environment (no Jupyter server)."""
import argparse
import ast
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate_notebook(path):
    notebook = json.loads(path.read_text())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']), filename=f'{path.name}:cell-{i}')
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Full checkpoint directory, e.g. checkpoints/bridge_qlora/RUN/checkpoint-100')
    parser.add_argument('--check', action='store_true', help='Validate notebook syntax only; no training/imports')
    parser.add_argument('--check-imports', action='store_true', help='Check training imports only; no model or GPU required')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    path = ROOT / 'bridge_qlora_deltaai.ipynb'
    notebook = validate_notebook(path)
    if args.check:
        print('Notebook syntax validated. No GPU or training code executed.')
        return
    from scripts.bridge_environment import check_training_imports
    check_training_imports()
    if args.check_imports:
        return
    if args.resume:
        from scripts.bridge_batch import validate_checkpoint
        checkpoint = validate_checkpoint(args.resume.resolve())
        storage_root = Path(os.environ.get('BRIDGE_STORAGE_ROOT', str(ROOT))).expanduser().resolve()
        if checkpoint.parent.parent != storage_root / 'checkpoints/bridge_qlora':
            raise ValueError('Resume a checkpoint within the configured storage root’s checkpoints/bridge_qlora/RUN folder.')
        os.environ['BRIDGE_RESUME_CHECKPOINT'] = str(checkpoint)
        os.environ['BRIDGE_RUN_ID'] = checkpoint.parent.name
    else:
        from datetime import datetime, timezone
        os.environ.pop('BRIDGE_RESUME_CHECKPOINT', None)
        os.environ['BRIDGE_RUN_ID'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = os.environ['BRIDGE_RUN_ID']
    locks = ROOT / 'logs'
    locks.mkdir(exist_ok=True)
    # Kernel releases this advisory lock on exit, including time-limit termination.
    with (locks / f'qlora-{run_id}.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f'Run ID: {run_id}', flush=True)
        scope = {'__name__': '__main__'}
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            print(f'Executing notebook cell {i}', flush=True)
            exec(compile(''.join(cell['source']), f'{path.name}:cell-{i}', 'exec'), scope)
    print('QLoRA training and final validation completed.', flush=True)


if __name__ == '__main__':
    main()
