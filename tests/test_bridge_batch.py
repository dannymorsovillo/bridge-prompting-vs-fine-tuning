import json
from pathlib import Path
import tempfile
import unittest

from scripts.bridge_batch import validate_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_requires_completed_full_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ['optimizer.pt', 'scheduler.pt', 'adapter_config.json',
                         'rng_state.pth', 'adapter_model.safetensors']:
                (path / name).touch()
            (path / 'trainer_state.json').write_text(json.dumps({'global_step': 100}))
            with self.assertRaisesRegex(ValueError, 'checkpoint_complete'):
                validate_checkpoint(path)
            (path / 'checkpoint_complete.json').write_text(json.dumps({'global_step': 100}))
            self.assertEqual(validate_checkpoint(path), path)
            (path / 'optimizer.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                validate_checkpoint(path)

    def test_rejects_wrong_step_and_missing_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ['optimizer.pt', 'scheduler.pt', 'adapter_config.json',
                         'rng_state.pth', 'adapter_model.safetensors']:
                (path / name).touch()
            (path / 'trainer_state.json').write_text(json.dumps({
                'global_step': 100, 'best_model_checkpoint': str(path / 'missing')}))
            (path / 'checkpoint_complete.json').write_text(json.dumps({'global_step': 99}))
            with self.assertRaisesRegex(ValueError, 'marker'):
                validate_checkpoint(path)
            (path / 'checkpoint_complete.json').write_text(json.dumps({'global_step': 100}))
            with self.assertRaisesRegex(ValueError, 'best checkpoint'):
                validate_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
