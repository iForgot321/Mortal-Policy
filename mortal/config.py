import toml
import os


def plain(obj):
    """`obj` with every dict as a plain dict: toml parses inline tables (`{ a = 1 }`) into a
    class local to its decoder, which pickle can't save, and trainers pickle the config into
    every checkpoint."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [plain(v) for v in obj]
    return obj


config_file = os.environ.get('MORTAL_CFG', 'config.toml')
with open(config_file, encoding='utf-8') as f:
    config = plain(toml.load(f))
