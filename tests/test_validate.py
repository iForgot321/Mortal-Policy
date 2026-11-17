"""tools/eval/validate.py: the head losses' weights for each kind of checkpoint."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('validate', os.path.join(ROOT, 'tools', 'eval', 'validate.py'))
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)


def test_a_students_file_gets_the_full_loss_config():
    # train_students.py saves only the heads' shape under [teacher] (2026-11-17: KeyError 'huber_delta')
    cfg = {'teacher': {'hidden': 256, 'detach': False}}
    assert validate.loss_config(cfg) == validate.LOSS_CFG


def test_a_teachers_own_weights_win_and_a_bare_config_gets_the_defaults():
    own = {'hidden': 256, 'detach': False, 'w_placement': 0.3, 'w_kyoku_delta': 0.7, 'w_aux': 0.1,
           'huber_delta': 2.0}
    got = validate.loss_config({'teacher': own})
    assert got == {k: own[k] for k in validate.LOSS_CFG}
    assert validate.loss_config({}) == validate.LOSS_CFG
