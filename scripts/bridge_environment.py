"""Process-local workaround for DeltaAI's incompatible optional TE packages."""
import sys


def disable_transformer_engine():
    """Hide optional Transformer Engine before importing Transformers/PEFT.

    Our NF4/BF16 + SDPA training does not use Transformer Engine. A None
    module entry makes Python's find_spec return None and blocks imports;
    no packages in the shared environment are changed.
    """
    loaded = [name for name, module in sys.modules.items()
              if (name == 'transformer_engine' or name.startswith('transformer_engine.'))
              and module is not None]
    if loaded:
        raise RuntimeError('Restart Python/kernel before disabling Transformer Engine.')
    sys.modules['transformer_engine'] = None


def check_training_imports():
    disable_transformer_engine()
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.import_utils import is_te_available
    assert not is_te_available(), 'PEFT still detects Transformer Engine.'
    print('Training imports passed; optional Transformer Engine disabled. No model loaded.', flush=True)
