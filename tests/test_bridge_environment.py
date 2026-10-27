import subprocess
import sys
import unittest


class EnvironmentTests(unittest.TestCase):
    def test_optional_package_hidden_in_fresh_process(self):
        subprocess.run([sys.executable, '-c', '''
import importlib.util
from scripts.bridge_environment import disable_transformer_engine
disable_transformer_engine()
disable_transformer_engine()
assert importlib.util.find_spec('transformer_engine') is None
try:
    import transformer_engine
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Import was not blocked')
'''], check=True)

    def test_loaded_package_requires_restart(self):
        subprocess.run([sys.executable, '-c', '''
import sys, types
from scripts.bridge_environment import disable_transformer_engine
sys.modules['transformer_engine'] = types.ModuleType('transformer_engine')
try:
    disable_transformer_engine()
except RuntimeError:
    pass
else:
    raise AssertionError('Loaded package was silently replaced')
'''], check=True)
