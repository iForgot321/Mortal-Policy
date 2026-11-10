"""The SL teacher's trainer (mortal/train_teacher.py, mortal/teacher.py; docs/design/teacher-trainer-draft.md):
the loss and its waits mask, the heads, the EMA, the split of the files between workers, the
validation sample, the resume cursor, and a CPU smoke run that trains, stops, resumes and finishes."""
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


def test_shuffled_file_order_gives_every_worker_the_whole_window(mods):
    _, teacher = mods
    # 400 logs over 10 years, 40 a year; paths sort by date, as the split lists do
    files = [f'{y}/{m:02d}/{y}{m:02d}0100gm-00a9-0000-{i:08x}.json.gz'
             for y in range(2015, 2025) for m in (1, 7) for i in range(20)]

    def slices(order, seed=1):
        ds = teacher.TeacherDataset(version=5, file_list=files[::-1], pts=[90, 50, -10, -130], seed=seed,
                                    file_order=order)
        per = -(-len(ds.file_list) // 4)  # dataloader.worker_init_fn's contiguous slices, 4 workers
        return [ds.file_list[w * per:(w + 1) * per] for w in range(4)]

    def years(files):
        return {f[:4] for f in files}

    assert [min(years(s)) for s in slices('sorted')] == ['2015', '2017', '2020', '2022']  # one era each
    assert max(len(years(s)) for s in slices('sorted')) == 3
    shuffled = slices('shuffled')
    assert all(len(years(s)) == 10 for s in shuffled)
    assert sorted(sum(shuffled, [])) == files  # still a partition of the list
    assert slices('shuffled') == shuffled and slices('shuffled', seed=2) != shuffled  # fixed by the seed
    with pytest.raises(ValueError):
        teacher.TeacherDataset(version=5, file_list=files, pts=[90, 50, -10, -130], seed=1, file_order='random')


def test_validation_files_window_and_exclusions(mods, tmp_path):
    _, teacher = mods
    ids = [f'{y}010100gm-00a9-0000-{i:08x}' for y in range(2009, 2026) for i in range(10)]
    (tmp_path / 'splits').mkdir()
    (tmp_path / 'splits' / 'heldout_game.txt').write_text(''.join(f'{i[:4]}/{i[4:6]}/{i}.json.gz\n' for i in ids))

    def picked(games, **kw):
        files = teacher.validation_files(str(tmp_path), 'heldout_game', games, **kw)
        assert all(f.startswith(str(tmp_path / 'mjai')) for f in files)
        return [os.path.basename(f).split('.')[0] for f in files]

    assert picked(17) == ids[::10]  # every k-th log, in date order: the trainer's rule before val_start
    assert [i[:4] for i in picked(11, start='2015-01-01')] == [str(y) for y in range(2015, 2026)]
    assert [i[:4] for i in picked(6, end='2014-12-31')] == [str(y) for y in range(2009, 2015)]
    assert picked(17, exclude=[ids[0]])[0] == ids[1]
    assert len(picked(5)) == 5


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
                .replace('log_every = 500', 'log_every = 5')
                .replace('val_games = 200', "val_games = 2\nval_start = '2015-01-01'")
                .replace("file_order = 'sorted'", "file_order = 'shuffled'")
                .replace('val_every = 10_000', 'val_every = 10').replace('num_workers = 2', 'num_workers = 1')
                .replace('file_batch_size = 20', 'file_batch_size = 2').replace('conv_channels = 192', 'conv_channels = 32')
                .replace('num_blocks = 8', 'num_blocks = 1').replace('max_steps = 158000', 'max_steps = 15')
                .replace('warm_up_steps = 2000', 'warm_up_steps = 5'))
    cfg = tmp_path / 'cfg.toml'

    assert "file_order = 'shuffled'" in cfg_text and "val_start = '2015-01-01'" in cfg_text

    def launch(max_steps, text=None):
        cfg.write_text((text or cfg_text).replace('max_steps = 15', f'max_steps = {max_steps}'))
        return subprocess.run([sys.executable, 'train_teacher.py'], cwd=os.path.join(ROOT, 'mortal'), capture_output=True,
                              text=True, timeout=600, env={**os.environ, 'MORTAL_CFG': str(cfg), 'MJ_DATA': MJ})

    def run(max_steps):
        r = launch(max_steps)
        assert r.returncode == 0, r.stderr[-3000:]
        return r.stderr

    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    from checkpoint import load_checkpoint  # the trainer's states load as data
    log = run(15)
    assert '2 validation logs from heldout_game, 2015' in log
    st = load_checkpoint(tmp_path / 'run' / 'state.pth')
    assert st['steps'] == 15 and st['cursor'] and st['file_order'] == 'shuffled'
    # the cursor counts file batches of the order the run started in
    r = launch(30, cfg_text.replace("file_order = 'shuffled'", "file_order = 'sorted'"))
    assert r.returncode != 0 and "was trained in file order 'shuffled'" in r.stderr
    log = run(30)
    assert 'resumed at step 15' in log
    final = load_checkpoint(tmp_path / 'run' / 'final.pth')
    assert final['steps'] == 30 and final['heads'] is not None and 'ema_weights' in final
    timing = [json.loads(l) for l in open(tmp_path / 'run' / 'timing.jsonl')]
    assert timing[-1]['steps'] == 30 and {'data_wait_s', 'save_s', 'val_s', 'samples_per_s'} <= set(timing[-1])
