"""The SL teacher's trainer (mortal/train_teacher.py, mortal/teacher.py; docs/design/teacher-trainer-draft.md):
the loss and its waits mask, the heads, the EMA, the resume cursor, and a CPU smoke run that trains,
stops, resumes and finishes."""
import glob
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJ = os.path.expanduser('~/mj-data')
FILES = sorted(glob.glob(os.path.join(MJ, 'mjai', '2019', '03', '*.json.gz')))[:8]
GRP = os.path.join(ROOT, 'workdir', 'grp-v2', 'grp-v2-frozen.pth')
needs_data = pytest.mark.skipif(len(FILES) < 8 or not os.path.exists(GRP), reason='needs ~/mj-data and GRP-v2')
CFG = {'w_placement': 0.5, 'w_kyoku_delta': 0.5, 'w_aux': 0.2, 'huber_delta': 1.0}


@pytest.fixture(scope='module')
def mods():
    box_cfg = os.path.join(ROOT, 'workdir', 'box', 'configs', 'teacher-heads-quick.toml')
    os.environ.setdefault('MORTAL_CFG', box_cfg if os.path.exists(box_cfg)
                          else os.path.join(ROOT, 'experiments', 'configs', 'teacher-heads-quick.toml'))
    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    import model
    import teacher
    return model, teacher


def batch(n=6):
    torch.manual_seed(0)
    logits = torch.randn(n, 46)
    actions = torch.randint(0, 46, (n,))
    opp = torch.zeros(n, 3, 35, dtype=torch.bool)
    opp[:, 0, 0] = True  # only opponent 0 is tenpai
    opp[:, 0, 5] = True
    return logits, actions, opp


def test_waits_loss_counts_tenpai_opponents_only(mods):
    _, teacher = mods
    logits, actions, opp = batch()
    out = {'placement': torch.zeros(6, 4), 'kyoku_delta': torch.zeros(6), 'tenpai': torch.zeros(6, 3),
           'waits': torch.zeros(6, 3, 34)}
    args = (torch.ones(6), out, torch.zeros(6), torch.zeros(6, dtype=torch.long), opp, CFG)
    _, a = teacher.teacher_loss(logits, actions, *args)
    out['waits'][:, 1:] = 50.0  # wildly wrong on the non-tenpai opponents: must not count
    _, b = teacher.teacher_loss(logits, actions, *args)
    assert torch.equal(a['waits'], b['waits'])
    out['waits'][:, 0] = -50.0  # wrong on the tenpai one: must count
    _, c = teacher.teacher_loss(logits, actions, *args)
    assert c['waits'] > a['waits']


def test_policy_only_and_awr_weight(mods):
    _, teacher = mods
    logits, actions, opp = batch()
    nll = -logits.log_softmax(-1)[torch.arange(6), actions]
    total, parts = teacher.teacher_loss(logits, actions, torch.ones(6), None, None, None, None, CFG)
    assert set(parts) == {'policy'} and torch.allclose(total, nll.mean())
    w = torch.rand(6)
    total, _ = teacher.teacher_loss(logits, actions, w, None, None, None, None, CFG)
    assert torch.allclose(total, (w * nll).mean())


def test_heads_shapes_and_detach(mods):
    model, _ = mods
    phi = torch.randn(5, 1024, requires_grad=True)
    out = model.TeacherHeads()(phi)
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        'placement': (5, 4), 'kyoku_delta': (5,), 'tenpai': (5, 3), 'waits': (5, 3, 34)}
    model.TeacherHeads(detach=True)(phi)['placement'].sum().backward()
    assert phi.grad is None


def test_ema_update_and_swap(mods):
    _, teacher = mods
    p = torch.nn.Parameter(torch.zeros(3))
    ema = teacher.Ema([p], decay=0.9)
    with torch.no_grad():
        p.fill_(1.0)
    ema.update()
    assert torch.allclose(ema.shadow[0], torch.full((3,), 0.1))
    ema.swap()
    assert torch.allclose(p, torch.full((3,), 0.1))
    ema.swap()
    assert torch.allclose(p, torch.ones(3))


@needs_data
def test_resume_cursor_yields_exactly_the_unseen_batches(mods):
    _, teacher = mods

    def entries(start=None):
        ds = teacher.TeacherDataset(version=5, file_list=list(FILES), pts=[90, 50, -10, -130], file_batch_size=2,
                                    compute_advantage=False, obs_fp16=True, teacher_labels=True, masks={},
                                    enable_augmentation=True, seed=7, start_batches=start)
        return [(e[-1], e[1], float(e[4])) for e in ds]  # file batch, action, kyoku-Δ

    full = entries()
    assert sorted({b for b, *_ in full}) == list(range(8))  # 4 file batches × 2 passes
    assert entries() == full  # the order is fixed by the seed
    assert entries({0: 5}) == [e for e in full if e[0] >= 5]


@needs_data
def test_train_stop_resume_finish_on_cpu(tmp_path):
    index = tmp_path / 'index.pth'
    torch.save({'file_list': FILES[:6]}, index)
    cfg_text = open(os.path.join(ROOT, 'experiments', 'configs', 'teacher-heads-quick.toml')).read()
    cfg_text = (cfg_text.replace("'/home/iforgot/projects/Mortal-Policy/workdir/teacher/heads-quick'", repr(str(tmp_path / 'run')))
                .replace("'/home/iforgot/projects/Mortal-Policy/workdir/pilot/quick500k.pth'", repr(str(index)))
                .replace("'/home/iforgot/projects/Mortal-Policy/workdir/grp-v2/grp-v2-frozen.pth'", repr(GRP))
                .replace("device = 'cuda:0'", "device = 'cpu'").replace('enable_amp = true', 'enable_amp = false')
                .replace('enable_compile = true', 'enable_compile = false').replace('batch_size = 2048', 'batch_size = 64')
                .replace('log_every = 500', 'log_every = 5').replace('val_games = 200', 'val_games = 2')
                .replace('val_every = 10_000', 'val_every = 10').replace('num_workers = 2', 'num_workers = 1')
                .replace('file_batch_size = 20', 'file_batch_size = 2').replace('conv_channels = 192', 'conv_channels = 32')
                .replace('num_blocks = 8', 'num_blocks = 1').replace('max_steps = 158000', 'max_steps = 15')
                .replace('warm_up_steps = 2000', 'warm_up_steps = 5'))
    cfg = tmp_path / 'cfg.toml'

    def run(max_steps):
        cfg.write_text(cfg_text.replace('max_steps = 15', f'max_steps = {max_steps}'))
        r = subprocess.run([sys.executable, 'train_teacher.py'], cwd=os.path.join(ROOT, 'mortal'), capture_output=True,
                           text=True, timeout=600, env={**os.environ, 'MORTAL_CFG': str(cfg), 'MJ_DATA': MJ})
        assert r.returncode == 0, r.stderr[-3000:]
        return r.stderr

    run(15)
    st = torch.load(tmp_path / 'run' / 'state.pth', weights_only=False)
    assert st['steps'] == 15 and st['cursor']
    log = run(30)
    assert 'resumed at step 15' in log
    final = torch.load(tmp_path / 'run' / 'final.pth', weights_only=False)
    assert final['steps'] == 30 and final['heads'] is not None and 'ema_weights' in final
    timing = [json.loads(l) for l in open(tmp_path / 'run' / 'timing.jsonl')]
    assert timing[-1]['steps'] == 30 and {'data_wait_s', 'save_s', 'val_s', 'samples_per_s'} <= set(timing[-1])
