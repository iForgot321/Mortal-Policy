"""The students' trainer (mortal/train_students.py, mortal/distill.py; docs/design/distillation-draft.md,
D66, D67): the ensemble's targets, the distillation loss, and a CPU smoke run with tiny teachers that
trains, stops, resumes and finishes, writing lab-ready students."""
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJ = os.path.expanduser('~/mj-data')
GRP = os.path.join(ROOT, 'workdir', 'grp-v2', 'grp-v2-frozen.pth')
CFG = os.path.join(ROOT, 'experiments', 'configs', 'students-s1.toml')
WEIGHTS = {'w_hard': 0.1, 'w_placement': 0.5, 'w_kyoku_delta': 0.5, 'w_aux': 0.2}


def files():
    import glob
    return sorted(glob.glob(os.path.join(MJ, 'mjai', '2019', '03', '*.json.gz')))[:6]


needs_data = pytest.mark.skipif(len(files()) < 6 or not os.path.exists(GRP), reason='needs ~/mj-data and GRP-v2')


@pytest.fixture(scope='module')
def mods():
    # on a rented box, setup_box.sh's copy with the workstation root rewritten to the checkout: the first
    # module to import `config` fixes it for the whole pytest process (2026-11-11: this line without the
    # box copy failed tests/test_value.py on a box, whose workstation GRP path doesn't exist)
    box_cfg = os.path.join(ROOT, 'workdir', 'box', 'configs', 'teacher-heads-quick.toml')
    os.environ.setdefault('MORTAL_CFG', box_cfg if os.path.exists(box_cfg)
                          else os.path.join(ROOT, 'experiments', 'configs', 'teacher-heads-quick.toml'))
    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    import distill
    import model
    return distill, model


def batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(n, 46, generator=g) < 0.5
    actions = torch.randint(0, 46, (n,), generator=g)
    mask[torch.arange(n), actions] = True
    return mask, actions


def heads_out(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {'placement': torch.randn(n, 4, generator=g), 'kyoku_delta': torch.randn(n, generator=g),
            'tenpai': torch.randn(n, 3, generator=g), 'waits': torch.randn(n, 3, 34, generator=g)}


def targets_from(out, logits):
    return {'policy': logits.softmax(-1), 'placement': out['placement'].softmax(-1), 'kyoku_delta': out['kyoku_delta'],
            'tenpai': out['tenpai'].sigmoid(), 'waits': out['waits'].sigmoid()}


def test_a_student_that_matches_the_ensemble_has_zero_kl_and_head_terms(mods):
    distill, _ = mods
    mask, actions = batch()
    logits = torch.randn(8, 46).masked_fill(~mask, -torch.inf)
    out = heads_out(8)
    _, parts = distill.distill_loss(logits, out, actions, mask, targets_from(out, logits), WEIGHTS)
    for k in ('kl', 'placement', 'kyoku_delta', 'tenpai', 'waits'):
        assert parts[k].abs() < 1e-6, (k, parts[k])
    hard = -logits.log_softmax(-1)[torch.arange(8), actions].mean()
    assert torch.allclose(parts['hard'], hard)


def test_illegal_actions_never_count_and_the_gradient_is_finite(mods):
    distill, _ = mods
    mask, actions = batch()
    raw = torch.randn(8, 46)
    tgt = targets_from(heads_out(8), torch.randn(8, 46).masked_fill(~mask, -torch.inf))
    a = raw.clone().requires_grad_()
    loss_a, _ = distill.distill_loss(a.masked_fill(~mask, -torch.inf), heads_out(8, 2), actions, mask, tgt, WEIGHTS)
    loss_a.backward()
    assert torch.isfinite(a.grad).all() and (a.grad[~mask] == 0).all()
    b = raw.clone()
    b[~mask] += 100.0  # different raw logits off the mask: the masked logits are the same
    loss_b, _ = distill.distill_loss(b.masked_fill(~mask, -torch.inf), heads_out(8, 2), actions, mask, tgt, WEIGHTS)
    assert torch.allclose(loss_a, loss_b)


def test_waits_count_only_where_the_ensemble_sees_tenpai(mods):
    distill, _ = mods
    mask, actions = batch()
    logits = torch.randn(8, 46).masked_fill(~mask, -torch.inf)
    out = heads_out(8)
    tgt = targets_from(out, logits)
    tgt['tenpai'] = torch.tensor([1.0, 0.0, 0.0]).expand(8, 3).clone()
    _, a = distill.distill_loss(logits, out, actions, mask, tgt, WEIGHTS)
    out2 = {k: v.clone() for k, v in out.items()}
    out2['waits'][:, 1:] = 50.0  # wildly wrong on the opponents the ensemble sees as not tenpai
    _, b = distill.distill_loss(logits, out2, actions, mask, tgt, WEIGHTS)
    assert torch.equal(a['waits'], b['waits'])
    out2['waits'][:, 0] = 50.0
    _, c = distill.distill_loss(logits, out2, actions, mask, tgt, WEIGHTS)
    assert c['waits'] > a['waits']


def tiny_teacher(model, seed, path):
    torch.manual_seed(seed)
    brain = model.Brain(version=5, conv_channels=32, num_blocks=1, Norm='GN')
    policy, heads = model.CategoricalPolicy(), model.TeacherHeads(hidden=16)
    torch.save({'mortal': brain.state_dict(), 'policy_net': policy.state_dict(), 'heads': heads.state_dict(),
                'config': {'control': {'version': 5}, 'resnet': {'conv_channels': 32, 'num_blocks': 1},
                           'teacher': {'hidden': 16, 'detach': False}}, 'steps': 0, 'weights': 'ema'}, path)
    return brain, policy, heads


def test_the_ensemble_is_the_teachers_mean(mods, tmp_path):
    distill, model = mods
    mask, _ = batch()
    obs = torch.randn(8, 1012, 34)
    for seed in (1, 2):
        tiny_teacher(model, seed, tmp_path / f't{seed}.pth')
    t1, t2 = (distill.load_frozen_teacher(str(tmp_path / f't{s}.pth'), torch.device('cpu')) for s in (1, 2))
    one, two = distill.ensemble_targets([t1], obs, mask), distill.ensemble_targets([t2], obs, mask)
    both = distill.ensemble_targets([t1, t2], obs, mask)
    for k in one:
        assert torch.allclose(both[k], (one[k] + two[k]) / 2, atol=1e-6), k
    assert (both['policy'][~mask] == 0).all() and torch.allclose(both['policy'].sum(-1), torch.ones(8))
    assert all(not p.requires_grad for m in t1 for p in m.parameters())
    sha = distill.sha256_file(tmp_path / 't1.pth')
    distill.load_frozen_teacher(str(tmp_path / 't1.pth'), torch.device('cpu'), sha256=sha)
    with pytest.raises(SystemExit, match='does not match'):
        distill.load_frozen_teacher(str(tmp_path / 't1.pth'), torch.device('cpu'), sha256='0' * 64)


@needs_data
def test_train_stop_resume_finish_on_cpu(mods, tmp_path):
    distill, model = mods
    shas = []
    for seed in (1, 2):
        tiny_teacher(model, seed, tmp_path / f'teacher{seed}.pth')
        shas.append(distill.sha256_file(tmp_path / f'teacher{seed}.pth'))
    index = tmp_path / 'index.pth'
    torch.save({'file_list': files()}, index)
    text = open(CFG).read()
    for old, new in (
        ("'/home/iforgot/projects/Mortal-Policy/workdir/secure-box/students-s1'", repr(str(tmp_path / 'run'))),
        ("'${MJ_DATA}/index-train.pth'", repr(str(index))),
        ("'/home/iforgot/projects/Mortal-Policy/workdir/teacher/teacher-bc-s1/ema-from-state.pth'", repr(str(tmp_path / 'teacher1.pth'))),
        ("'/home/iforgot/projects/Mortal-Policy/workdir/teacher/teacher-awr-s1/ema-from-state.pth'", repr(str(tmp_path / 'teacher2.pth'))),
        ("'59d9463626125707f329355d515aa27ac2572d385906e336694047bd4b5f586f'", repr(shas[0])),
        ("'75e55855c2effa1ca52bb380f55cd6353064c43ceb92fd1330d045829a47c4b8'", repr(shas[1])),
        ("'/home/iforgot/projects/Mortal-Policy/workdir/grp-v2/grp-v2-frozen.pth'", repr(GRP)),
        ("device = 'cuda:0'", "device = 'cpu'"), ('enable_amp = true', 'enable_amp = false'),
        ('enable_compile = true', 'enable_compile = false'), ('batch_size = 2048', 'batch_size = 64'),
        ('log_every = 1000', 'log_every = 5'), ('val_games = 200', 'val_games = 2'), ('val_every = 20_000', 'val_every = 10'),
        ('num_workers = 4', 'num_workers = 1'), ('file_batch_size = 20', 'file_batch_size = 2'),
        ('conv_channels = 192', 'conv_channels = 32'), ('conv_channels = 256', 'conv_channels = 64'),
        ('num_blocks = 8', 'num_blocks = 1'), ('num_blocks = 12', 'num_blocks = 1'), ('num_blocks = 16', 'num_blocks = 2'),
        ('hidden = 256', 'hidden = 16'), ('max_steps = 1_143_000', 'max_steps = 15'), ('warm_up_steps = 5000', 'warm_up_steps = 5'),
    ):
        assert old in text, old
        text = text.replace(old, new)
    cfg = tmp_path / 'cfg.toml'

    def launch(max_steps, txt=text):
        cfg.write_text(txt.replace('max_steps = 15', f'max_steps = {max_steps}'))
        return subprocess.run([sys.executable, 'train_students.py'], cwd=os.path.join(ROOT, 'mortal'), capture_output=True,
                              text=True, timeout=900, env={**os.environ, 'MORTAL_CFG': str(cfg), 'MJ_DATA': MJ})

    def run(max_steps, txt=text):
        r = launch(max_steps, txt)
        assert r.returncode == 0, r.stderr[-3000:]
        return r.stderr

    from checkpoint import load_checkpoint
    run(15)
    st = load_checkpoint(tmp_path / 'run' / 'state.pth')
    assert st['steps'] == 15 and st['cursor'] and st['teachers'] == shas
    assert set(st['students']) == {'b8c192', 'b12c256', 'b16c256', 'control-b8c192'}
    # a teacher that isn't the configured one is refused
    bad = text.replace(repr(shas[0]), repr('0' * 64))
    r = launch(30, bad)
    assert r.returncode != 0 and 'does not match' in r.stderr
    log = run(30)
    assert 'resumed at step 15' in log
    final = load_checkpoint(tmp_path / 'run' / 'final.pth')
    assert final['steps'] == 30 and final['students'] == ['b8c192', 'b12c256', 'b16c256', 'control-b8c192']
    for name in final['students']:
        ema, raw = (load_checkpoint(tmp_path / 'run' / f'{t}-{name}.pth') for t in ('ema', 'raw'))
        assert ema['weights'] == 'ema' and raw['weights'] == 'raw'
        assert not all(torch.equal(ema['mortal'][k], raw['mortal'][k]) for k in raw['mortal'])
        assert ema['config']['student']['loss'] == ('labels' if name.startswith('control') else 'distill')
    # lab-ready: the lab builds each student from its own config
    sys.path.insert(0, ROOT)
    from lab.arena import load_engine
    assert load_engine(str(tmp_path / 'run' / 'ema-b16c256.pth'), device='cpu').version == 5
    timing = [json.loads(l) for l in open(tmp_path / 'run' / 'timing.jsonl')]
    steps = [t for t in timing if 'samples_per_s' in t]
    assert steps[-1]['steps'] == 30 and set(steps[-1]['gpu_share']) == {'teachers', *final['students']}
    vals = [t['val'] for t in timing if 'val' in t]
    assert vals and set(vals[-1]['b8c192']['ema']) >= {'policy', 'top1', 'kl_ens', 'agree_ens', 'placement', 'waits'}


@needs_data
def test_students_of_one_shape_start_from_one_initialization(mods, tmp_path):
    """The control and the distilled b8c192 differ only by their loss: at step 0 they are equal."""
    distill, model = mods
    shas = []
    for seed in (1, 2):
        tiny_teacher(model, seed, tmp_path / f'teacher{seed}.pth')
        shas.append(distill.sha256_file(tmp_path / f'teacher{seed}.pth'))
    index = tmp_path / 'index.pth'
    torch.save({'file_list': files()[:2]}, index)
    text = open(CFG).read()
    for old, new in (
        ("'/home/iforgot/projects/Mortal-Policy/workdir/secure-box/students-s1'", repr(str(tmp_path / 'run'))),
        ("'${MJ_DATA}/index-train.pth'", repr(str(index))),
        ("'/home/iforgot/projects/Mortal-Policy/workdir/teacher/teacher-bc-s1/ema-from-state.pth'", repr(str(tmp_path / 'teacher1.pth'))),
        ("'/home/iforgot/projects/Mortal-Policy/workdir/teacher/teacher-awr-s1/ema-from-state.pth'", repr(str(tmp_path / 'teacher2.pth'))),
        ("'59d9463626125707f329355d515aa27ac2572d385906e336694047bd4b5f586f'", repr(shas[0])),
        ("'75e55855c2effa1ca52bb380f55cd6353064c43ceb92fd1330d045829a47c4b8'", repr(shas[1])),
        ("'/home/iforgot/projects/Mortal-Policy/workdir/grp-v2/grp-v2-frozen.pth'", repr(GRP)),
        ("device = 'cuda:0'", "device = 'cpu'"), ('enable_amp = true', 'enable_amp = false'),
        ('enable_compile = true', 'enable_compile = false'), ('batch_size = 2048', 'batch_size = 64'),
        ('val_games = 200', 'val_games = 1'), ('num_workers = 4', 'num_workers = 1'), ('file_batch_size = 20', 'file_batch_size = 2'),
        ('conv_channels = 192', 'conv_channels = 32'), ('conv_channels = 256', 'conv_channels = 64'),
        ('num_blocks = 8', 'num_blocks = 1'), ('num_blocks = 12', 'num_blocks = 1'), ('num_blocks = 16', 'num_blocks = 2'),
        ('hidden = 256', 'hidden = 16'), ('max_steps = 1_143_000', 'max_steps = 0'), ('warm_up_steps = 5000', 'warm_up_steps = 0'),
    ):
        assert old in text, old
        text = text.replace(old, new)
    (tmp_path / 'cfg.toml').write_text(text)
    r = subprocess.run([sys.executable, 'train_students.py'], cwd=os.path.join(ROOT, 'mortal'), capture_output=True,
                       text=True, timeout=600, env={**os.environ, 'MORTAL_CFG': str(tmp_path / 'cfg.toml'), 'MJ_DATA': MJ})
    assert r.returncode == 0, r.stderr[-3000:]
    from checkpoint import load_checkpoint
    a, b = (load_checkpoint(tmp_path / 'run' / f'raw-{n}.pth') for n in ('b8c192', 'control-b8c192'))
    c = load_checkpoint(tmp_path / 'run' / 'raw-b12c256.pth')
    for k in ('mortal', 'policy_net', 'heads'):
        assert all(torch.equal(a[k][n], b[k][n]) for n in a[k])
    assert not all(torch.equal(a['policy_net'][n], c['policy_net'][n]) for n in a['policy_net'])
