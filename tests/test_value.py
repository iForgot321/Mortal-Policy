"""Arm V's offline data path and checkpoint loading (experiments/v.md)."""
import glob
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJ = os.path.expanduser('~/mj-data')
FILES = sorted(glob.glob(os.path.join(MJ, 'mjai', '2009', '03', '*.json.gz')))[:3]
U = np.array([0.9, 0.5, -0.1, -1.3])

torch = pytest.importorskip('torch')


@pytest.fixture(scope='module')
def dataloader():
    # on a rented box, setup_box.sh's copy with the workstation root rewritten to the checkout
    box_cfg = os.path.join(ROOT, 'workdir', 'box', 'configs', 'cql-qs-a.toml')
    os.environ.setdefault('MORTAL_CFG', box_cfg if os.path.exists(box_cfg)
                          else os.path.join(ROOT, 'experiments', 'configs', 'cql-qs-a.toml'))
    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    import dataloader
    return dataloader


def test_next_ranks_stable_ties(dataloader):
    # two kyoku starts and the final scores; seats 1 and 2 tie after kyoku 0
    feature = np.zeros((2, 7))
    feature[0, 3:] = [2.5, 2.5, 2.5, 2.5]
    feature[1, 3:] = [3.3, 2.1, 2.1, 2.5]
    final = [20000, 40000, 10000, 30000]
    ranks = [dataloader.next_ranks(feature, final, p) for p in range(4)]
    assert [r[0] for r in ranks] == [0, 1, 2, 3]  # all tied: seat order
    assert [r[1] for r in ranks] == [0, 2, 3, 1]  # the tie between seats 1 and 2 goes to seat 1
    assert [r[2] for r in ranks] == [2, 0, 3, 1]


@pytest.mark.skipif(len(FILES) < 3, reason='needs converted data in ~/mj-data')
def test_value_entries(dataloader):
    from libriichi.dataset import GameplayLoader
    ds = dataloader.FileDatasetsIter(version=4, file_list=list(FILES), pts=[90, 50, -10, -130],
                                     file_batch_size=len(FILES), player_names=[],
                                     compute_advantage=False, obs_fp16=True, value_targets=True)
    it = ds.build_iter()  # loads the GRP and the fixed-unit reward calculator
    entries = list(it)
    light = GameplayLoader(version=4, oracle=False, featurize=False).load_gz_log_files(FILES)
    assert len(entries) == sum(len(g.take_actions()) for f in light for g in f)
    assert all(len(e) == 5 for e in entries)
    q = np.array([e[3] for e in entries])
    rank = np.array([e[4] for e in entries])
    # u units: a kyoku's GRP-Δu can't exceed the utility's range
    assert np.abs(q).max() <= U.max() - U.min()
    assert np.abs(q).max() > 0.01  # not standardized away, not all zero
    assert set(rank) <= {0, 1, 2, 3}
    assert ds.reward_calc.shared_stats['count'].value == 0


@pytest.mark.skipif(len(FILES) < 3, reason='needs converted data in ~/mj-data')
def test_kyoku_targets_telescope(dataloader):
    """Per seat, the kyoku targets sum to u(final rank) − E[u] at the start (the GRP
    shaping telescopes), and the rank after the last kyoku is the final rank."""
    from libriichi.dataset import GameplayLoader
    ds = dataloader.FileDatasetsIter(version=4, file_list=[], pts=[90, 50, -10, -130],
                                     compute_advantage=False, value_targets=True)
    next(ds.build_iter(), None)
    for game in GameplayLoader(version=4, oracle=False, featurize=False).load_gz_log_files(FILES[:1])[0]:
        pid = game.take_player_id()
        grp = game.take_grp()
        feature, rbp = grp.take_feature(), grp.take_rank_by_player()
        r = ds.reward_calc.calc_delta_pt(pid, feature, rbp) / dataloader.PTS_PER_U
        p0 = ds.reward_calc.calc_rank_prob(pid, feature, rbp)[0].numpy()
        assert r.sum() == pytest.approx(U[rbp[pid]] - p0 @ U, abs=1e-9)
        assert dataloader.next_ranks(feature, grp.take_final_scores(), pid)[-1] == rbp[pid]


def test_load_engine_value_checkpoint(tmp_path):
    sys.path.insert(0, ROOT)
    from lab.arena import load_engine
    from model import Brain, DQN
    torch.manual_seed(0)
    brain = Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN').eval()
    dqn = DQN(version=4).eval()
    ckpt = tmp_path / 'v.pth'
    torch.save({'mortal': brain.state_dict(), 'current_dqn': dqn.state_dict(),
                'config': {'control': {'version': 4}, 'resnet': {'conv_channels': 32, 'num_blocks': 1}}}, ckpt)
    engine = load_engine(str(ckpt), device='cpu', enable_amp=False)
    obs = np.random.default_rng(0).random((3, 1012, 34), dtype=np.float32)
    masks = np.zeros((3, 46), dtype=bool)
    masks[:, [0, 5, 37]] = True
    actions, q_out, *_ = engine.react_batch(list(obs), list(masks), None)
    with torch.inference_mode():
        q = dqn(brain(torch.as_tensor(obs)), torch.as_tensor(masks))
    assert actions == q.argmax(-1).tolist()
    assert all(a in (0, 5, 37) for a in actions)


@pytest.mark.parametrize('eps,temp,top_p', [(0.0, 1.0, 1.0), (0.3, 0.05, 1.0), (1.0, 0.5, 0.7), (1.0, 1.0, 0.0)])
def test_value_engine_samples_its_behaviour_dist(eps, temp, top_p):
    sys.path.insert(0, os.path.join(ROOT, 'mortal'))
    from engine import ValueEngine
    eng = ValueEngine(torch.nn.Identity(), torch.nn.Identity(), is_oracle=False, version=4,
                      device=torch.device('cpu'), boltzmann_epsilon=eps, boltzmann_temp=temp, top_p=top_p)
    q = torch.full((46,), -torch.inf)
    q[[3, 17, 41, 45]] = torch.tensor([0.20, 0.15, 0.10, -0.40])
    dist = eng.behaviour_dist(q[None])[0]
    assert dist.sum() == pytest.approx(1, abs=1e-6) and (dist[q == -torch.inf] == 0).all()
    nucleus = eng.nucleus(q[None])[0]
    soft = (q / temp).softmax(-1)
    if top_p <= 0:
        assert nucleus[3] == 1
    elif top_p >= 1:
        assert torch.allclose(nucleus, soft)
    else:
        # the smallest set of most likely actions whose mass reaches p (Equim's sample_top_p)
        keep = soft.sort(descending=True)
        n = int(((keep.values.cumsum(-1) - keep.values) <= top_p).sum())
        assert set(nucleus.nonzero().flatten().tolist()) == set(keep.indices[:n].tolist())
    assert dist[3] == pytest.approx(1 - eps + eps * nucleus[3].item())

    n = 50_000
    torch.manual_seed(0)
    actions, is_greedy, b = eng.sample(q.expand(n, -1))
    assert torch.equal(b, dist[actions])
    freq = torch.bincount(actions, minlength=46).double() / n
    sd = (dist.double() * (1 - dist.double()) / n).sqrt()
    assert ((freq - dist.double()).abs() <= 5 * sd + 1e-12).all()


def tiny_state(online, conv=32, blocks=1):
    from model import Brain, DQN, AuxNet
    torch.manual_seed(0)
    return {'mortal': Brain(version=4, conv_channels=conv, num_blocks=blocks, Norm='GN').state_dict(),
            'current_dqn': DQN(version=4).state_dict(), 'aux_net': AuxNet((4,)).state_dict(),
            'scaler': torch.amp.GradScaler('cpu', enabled=False).state_dict(), 'steps': 100,
            'timestamp': 0.0,
            'config': {'control': {'version': 4, 'online': online},
                       'resnet': {'conv_channels': conv, 'num_blocks': blocks}}}


def test_load_frozen_takes_either_head(tmp_path, dataloader):
    from model import CategoricalPolicy, DQN
    import player
    v = tmp_path / 'v.pth'
    torch.save(tiny_state(False), v)
    p = tmp_path / 'p.pth'
    state = tiny_state(False)
    del state['current_dqn']
    state['policy_net'] = CategoricalPolicy().state_dict()
    torch.save(state, p)
    for ckpt, kind in ((v, DQN), (p, CategoricalPolicy)):
        eng = player.load_frozen({'device': 'cpu', 'enable_compile': False, 'state_file': str(ckpt)})
        assert isinstance(eng.dqn, kind) and eng.name == 'baseline'
    assert isinstance(player.build_head('value', 4), DQN)


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def test_online_learner_loop(tmp_path, arena_run, dataloader):
    """server.py + train_value.py online on tiny nets (CPU): drains submitted 1v3 logs,
    trains on the trainee's seat without CQL, submits parameters and saves."""
    import subprocess
    import time
    from common import send_msg, recv_msg
    _, logs, _, _ = arena_run
    port = free_port()
    w = tmp_path
    cfg = f"""
[control]
version = 4
online = true
head = 'value'
state_file = '{w}/mortal.pth'
tensorboard_dir = '{w}/tb'
device = 'cpu'
enable_cudnn_benchmark = false
enable_amp = false
enable_compile = false
batch_size = 32
opt_step_every = 1
save_every = 4
submit_every = 2
[dataset]
file_batch_size = 10
reserve_ratio = 0.0
num_workers = 1
num_epochs = 1
enable_augmentation = false
augmented_first = false
obs_fp16 = true
[resnet]
conv_channels = 32
num_blocks = 1
[env]
pts = [90.0, 50.0, -10.0, -130.0]
[aux]
next_rank_weight = 0.2
[optim]
eps = 1e-8
betas = [0.9, 0.999]
weight_decay = 0.0
max_grad_norm = 0.5
[optim.scheduler]
peak = 2e-5
final = 2e-5
warm_up_steps = 0
max_steps = 0
[online.remote]
host = '127.0.0.1'
port = {port}
[online.server]
buffer_dir = '{w}/buffer'
drain_dir = '{w}/drain'
capacity = 100
force_sequential = false
[grp]
state_file = '{ROOT}/workdir/grp-v2/grp-v2-frozen.pth'
[grp.network]
hidden_size = 64
num_layers = 2
"""
    (w / 'v.toml').write_text(cfg)
    torch.save(tiny_state(False), w / 'mortal.pth')
    env = {**os.environ, 'MORTAL_CFG': str(w / 'v.toml')}
    mortal_dir = os.path.join(ROOT, 'mortal')
    server = subprocess.Popen([sys.executable, 'server.py'], cwd=mortal_dir, env=env,
                              stdout=open(w / 'server.log', 'w'), stderr=subprocess.STDOUT)
    trainer = None
    try:
        import socket
        for _ in range(100):
            try:
                socket.create_connection(('127.0.0.1', port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
        with socket.create_connection(('127.0.0.1', port)) as c:
            send_msg(c, {'type': 'submit_replay', 'param_version': 0, 'behaviour': {},
                         'logs': {os.path.basename(f): open(f, 'rb').read() for f in logs}})
        trainer = subprocess.Popen([sys.executable, 'train_value.py'], cwd=mortal_dir, env=env,
                                   stdout=open(w / 'trainer.log', 'w'), stderr=subprocess.STDOUT)
        deadline = time.time() + 240
        while time.time() < deadline and trainer.poll() is None:
            text = (w / 'trainer.log').read_text()
            if 'total steps: 104' in text and text.count('param has been submitted') >= 3:
                break
            time.sleep(1)
        text = (w / 'trainer.log').read_text()
        assert trainer.poll() is None, text[-2000:]
        assert 'drained 8 logs' in text, text[-2000:]
        assert 'total steps: 104' in text, text[-2000:]
        state = torch.load(w / 'mortal.pth', weights_only=True)
        assert state['steps'] >= 104 and state['config']['control']['online']
        with socket.create_connection(('127.0.0.1', port)) as c:
            send_msg(c, {'type': 'get_param', 'param_version': -1})
            rsp = recv_msg(c)
        assert rsp['status'] == 'ok' and 'net.weight' in rsp['dqn']
    finally:
        for p in (trainer, server):
            if p is not None:
                p.kill()
                p.wait()


def test_value_trainee_plays_1v3_and_feeds_the_value_loader(tmp_path, dataloader):
    """The actor side: ValueEngine (exploring) against a greedy DQN baseline in the arena,
    and its logs through the online loader (the trainee's seat only)."""
    from glob import glob
    from engine import MortalEngine, ValueEngine
    from libriichi.arena import OneVsThree
    from model import Brain, DQN
    torch.manual_seed(0)
    brain = Brain(version=4, conv_channels=32, num_blocks=1, Norm='GN').eval()
    dqn = DQN(version=4).eval()
    common = dict(is_oracle=False, version=4, device=torch.device('cpu'), enable_amp=False)
    env = OneVsThree(disable_progress_bar=True, log_dir=str(tmp_path / 'logs'), behaviour_dir=str(tmp_path / 'bp'))
    rankings = env.py_vs_py(
        challenger=ValueEngine(brain, dqn, name='trainee', boltzmann_epsilon=0.5, boltzmann_temp=0.05, **common),
        champion=MortalEngine(brain, dqn, name='baseline', enable_rule_based_agari_guard=True, **common),
        seed_start=(0, 0x5eed), seed_count=1)
    assert sum(rankings) == 4
    logs = sorted(glob(str(tmp_path / 'logs' / '*.json.gz')))
    assert len(logs) == 4 and len(os.listdir(tmp_path / 'bp')) == 4
    ds = dataloader.FileDatasetsIter(version=4, file_list=logs, pts=[90, 50, -10, -130], file_batch_size=4,
                                     player_names=['trainee'], compute_advantage=False, obs_fp16=True,
                                     value_targets=True)
    entries = list(ds.build_iter())
    from libriichi.dataset import GameplayLoader
    light = GameplayLoader(version=4, oracle=False, featurize=False, player_names=['trainee']).load_gz_log_files(logs)
    assert len(entries) == sum(len(g.take_actions()) for f in light for g in f) > 0
    assert all(len(f) == 1 for f in light)  # one trainee seat per log
    assert all(e[2][e[1]] for e in entries)  # every action legal under its mask
