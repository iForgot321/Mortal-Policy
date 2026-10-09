def train():
    import prelude
    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    import math
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.distributions import Categorical
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, drain_msg, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from rl_dataset import OrderedDatasetsIter, split_by_sidecar, shuffled_minibatches
    from ppo import ppo_loss
    from critic import Critic, gae_targets, explained_variance, calibration, advantage_variance_ratio
    import numpy as np
    import torch.nn.functional as F
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, CategoricalPolicy
    from libriichi.consts import obs_shape
    from config import config
    from copy import deepcopy
    from multiprocessing import Manager
    
    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    old_update_every= config['control']['old_update_every']
    test_games = config['test_play']['games']
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    entropy_weight = config['policy']['entropy_weight']
    clip_ratio = config['policy']['clip_ratio']
    dual_clip = config['policy']['dual_clip']

    # D37/D46 (05 §2.4, §3). The defaults keep upstream behaviour.
    # 'snapshot': ratios against the Old_mortal copy (upstream); 'stored': against
    # the behaviour probabilities the actors stored (needs the ordered dataset)
    ratio_reference = config['policy'].get('ratio_reference', 'snapshot')
    # per-seat ordered trajectories with their sidecars (rl_dataset.py)
    ordered = config['dataset'].get('ordered', False)
    horizon = config['policy'].get('horizon', 'K')
    # normalize the detached advantages per batch (D37)
    advantage_norm = config['policy'].get('advantage_norm', False)
    obs_fp16 = config['dataset'].get('obs_fp16', False)
    assert ratio_reference in ('snapshot', 'stored')
    stored = ratio_reference == 'stored'
    if ordered:
        assert online, 'the ordered dataset reads drained 1v3 logs and their behaviour sidecars'
        assert not enable_augmentation, 'behaviour probabilities belong to the unaugmented actions'
    if stored:
        assert ordered, "ratio_reference = 'stored' needs dataset.ordered = true"
    logging.info(f'ratio reference: {ratio_reference}, ordered: {ordered}, horizon: {horizon}, advantage norm: {advantage_norm}')

    # K1 critic V(h, i) (05 §4, arm P1): GAE advantages from a separate network
    # with the opponents' hands as extra input. warmup = B0: the policy stays
    # frozen (the actors keep its parameters) and only the critic trains.
    critic_cfg = config.get('critic', {})
    use_critic = critic_cfg.get('enabled', False)
    critic_warmup = critic_cfg.get('warmup', False)
    if use_critic:
        assert stored, "the critic path needs ratio_reference = 'stored'"
        critic_gamma = critic_cfg.get('gamma', 1.0)
        critic_lambda = critic_cfg.get('gae_lambda', 0.95)
        critic_max_grad_norm = critic_cfg.get('max_grad_norm', 0)
        logging.info(f'critic: on, warmup (B0): {critic_warmup}, gamma: {critic_gamma}, lambda: {critic_lambda}')
    else:
        assert not critic_warmup, 'critic.warmup needs critic.enabled'

    # seeds the init (the critic's too), the file order and (through torch's
    # RNG) the DataLoader worker seeds; unseeded runs keep upstream behaviour
    if (seed := config['control'].get('seed')) is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logging.info(f'seed: {seed}')

    mortal = Brain(version=version, **config['resnet'], Norm="GN").to(device)
    policy_net = CategoricalPolicy().to(device)
    all_models = (mortal, policy_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    Old_mortal = deepcopy(mortal)
    Old_policy_net = deepcopy(policy_net)

    if use_critic:
        critic = Critic(
            conv_channels = critic_cfg['conv_channels'],
            num_blocks = critic_cfg['num_blocks'],
            version = version,
        ).to(device)
        if enable_compile:
            critic.compile()
        critic_optimizer = optim.AdamW(
            critic.parameters(),
            lr = 1,
            weight_decay = critic_cfg['optim'].get('weight_decay', 0.0),
            betas = betas,
            eps = eps,
        )
        critic_scheduler = LinearWarmUpCosineAnnealingLR(critic_optimizer, **critic_cfg['optim']['scheduler'])
        logging.info(f'critic params: {parameter_count(critic):,}')
    
    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy_net):,}')

    decay_params = []
    no_decay_params = []
    for model in (mortal, policy_net):
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    manager = Manager()
    shared_stats = {
        'count': manager.Value('i', 0),
        'mean': manager.Value('d', 0.0),
        'M2': manager.Value('d', 0.0),
        'lock': manager.Lock()
            }
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=False, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        Old_mortal.load_state_dict(state['mortal'])
        policy_net.load_state_dict(state['policy_net'])
        Old_policy_net.load_state_dict(state['policy_net'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            if 'shared_stats' in state and state['shared_stats'] is not None:
                shared_stats['count'].value = state['shared_stats']['count']
                shared_stats['mean'].value = state['shared_stats']['mean']
                shared_stats['M2'].value = float(state['shared_stats']['variance'] * state['shared_stats']['count']) 
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']
        if use_critic and 'critic' in state:
            critic.load_state_dict(state['critic'])
            critic_optimizer.load_state_dict(state['critic_optimizer'])
            critic_scheduler.load_state_dict(state['critic_scheduler'])
            logging.info('loaded the critic')
       
    optimizer.zero_grad(set_to_none=True)

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    # parameter versions this process has submitted; with the server's version
    # at drain time this gives each sample's policy lag
    submitted = 0

    def submit(is_idle):
        nonlocal submitted
        submit_param(mortal, policy_net, is_idle=is_idle)
        submitted += 1
        logging.info('param has been submitted')

    if online:
        submit(is_idle=True)

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'important_ratio': 0,
        'entropy': 0,
        'loss': 0,
        'stats': 0,
    }
    # 'stored' path diagnostics, summed over ppo_batches batches
    ppo_stats = {}
    ppo_batches = 0
    lag_max = 0
    # critic diagnostics: (value, Monte-Carlo return) of every decision before
    # the critic trained on it, and each chunk's advantage variance ratio
    critic_diag = {'value': [], 'mc_return': [], 'adv_var_ratio': []}
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx
        nonlocal stats
        nonlocal Old_mortal
        nonlocal Old_policy_net

        player_names = []
        if online:
            player_names = ['trainee']
            if ordered:
                rsp = drain_msg()
                dirname = rsp['drain_dir']
                behaviour_dir = rsp['behaviour_drain_dir']
                param_versions = rsp['param_versions']
                # the server's version at drain, less what this process had submitted
                version_base = rsp['param_version'] - submitted
            else:
                dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        if ordered:
            # never train on a log without its behaviour probabilities (D46)
            file_list, no_behaviour = split_by_sidecar(file_list, behaviour_dir)
            if no_behaviour:
                logging.warning(f'skipping {len(no_behaviour):,} drained logs without a behaviour sidecar')
            writer.add_scalar('data/logs_without_behaviour', len(no_behaviour), steps)
            file_data = OrderedDatasetsIter(
                version = version,
                file_list = file_list,
                behaviour_dir = behaviour_dir,
                pts = pts,
                grp_state_file = config['grp']['state_file'],
                grp_network = config['grp']['network'],
                horizon = horizon,
                targets = 'gae' if use_critic else 'mc',
                player_names = player_names,
                file_batch_size = file_batch_size,
                num_epochs = num_epochs,
                obs_fp16 = obs_fp16,
                versions = param_versions,
            )
        else:
            file_data = FileDatasetsIter(
                version = version,
                file_list = file_list,
                pts = pts,
                shared_stats=shared_stats,
                file_batch_size = file_batch_size,
                reserve_ratio = reserve_ratio,
                player_names = player_names,
                num_epochs = num_epochs,
                enable_augmentation = enable_augmentation,
                augmented_first = augmented_first,
            )
        data_loader = iter(DataLoader(
            dataset = file_data,
            # the critic path gets whole-trajectory chunks and batches them itself
            batch_size = None if use_critic else batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
            persistent_workers=True
        ))

        remaining = []
        remaining_bs = 0
        first_batch = True
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, advantage, behaviour_prob=None, param_version=None, priv=None, returns=None):
            nonlocal steps
            nonlocal idx
            nonlocal pb
            nonlocal Old_mortal
            nonlocal Old_policy_net
            nonlocal ppo_batches
            nonlocal lag_max
            nonlocal first_batch

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            advantage = advantage.to(dtype=torch.float32, device=device)
            assert masks[range(batch_size), actions].all()

            if stored:
                behaviour_prob = behaviour_prob.to(dtype=torch.float32, device=device)
                if critic_warmup:
                    loss = 0
                else:
                    with torch.autocast(device.type, enabled=enable_amp):
                        logits = policy_net.logits(mortal(obs), masks)
                    loss, ppo_batch = ppo_loss(
                        logits, actions, advantage, behaviour_prob,
                        clip_ratio = clip_ratio,
                        dual_clip = dual_clip,
                        entropy_weight = entropy_weight,
                        advantage_norm = advantage_norm,
                    )
                if use_critic:
                    priv = priv.to(dtype=torch.float32, device=device)
                    returns = returns.to(dtype=torch.float32, device=device)
                    with torch.autocast(device.type, enabled=enable_amp):
                        value = critic(obs, priv)
                    critic_loss = F.mse_loss(value.float(), returns)
                    # disjoint parameters: one backward serves both networks
                    loss = loss + critic_loss
            else:
                if advantage_norm:
                    advantage = (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8)
                with torch.no_grad():
                    with torch.autocast(device.type, enabled=enable_amp):
                        old_dist = Categorical(probs=Old_policy_net(Old_mortal(obs),masks))
                        old_log_prob = old_dist.log_prob(actions)

                with torch.autocast(device.type, enabled=enable_amp):
                    dist = Categorical(probs=policy_net(mortal(obs),masks))
                    new_log_prob = dist.log_prob(actions)
                    ratio = (new_log_prob - old_log_prob).exp()
                    loss1 = ratio * advantage
                    loss2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
                    min_loss = torch.min(loss1, loss2)
                    
                    clip_loss = torch.where(
                    advantage < 0,
                    torch.max(min_loss , dual_clip * advantage),
                    min_loss
                    )
                    entropy = dist.entropy().view(-1, 1)
                    entropy_loss = entropy * entropy_weight

                    loss =  -(clip_loss + entropy_loss).mean()
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                if stored:
                    if not critic_warmup:
                        stats['important_ratio'] += ppo_batch['ratio_mean']
                        stats['entropy'] += ppo_batch['entropy']
                        for k, v in ppo_batch.items():
                            ppo_stats[k] = ppo_stats.get(k, 0) + v
                    if use_critic:
                        ppo_stats['critic_loss'] = ppo_stats.get('critic_loss', 0) + critic_loss
                    ppo_batches += 1
                    known = param_version[param_version >= 0]
                    if len(known) > 0:
                        lag = (version_base + submitted - known).float()
                        ppo_stats['lag_mean'] = ppo_stats.get('lag_mean', 0) + lag.mean()
                        lag_max = max(lag_max, int(lag.max()))
                    if first_batch and not critic_warmup:
                        # 05 §8: ~1 when the actors played the current weights
                        # (AMP noise ~1e-3 aside)
                        first_batch = False
                        logging.info(
                            f'first minibatch of this drain: ratio mean {ppo_batch["ratio_mean"]:.6f}, '
                            f'p99 {ppo_batch["ratio_p99"]:.6f}, excluded {ppo_batch["excluded_frac"]:.4%}'
                        )
                        writer.add_scalar('ppo/first_batch_ratio_mean', ppo_batch['ratio_mean'], steps)
                else:
                    stats['important_ratio'] += ratio.mean()
                    stats['entropy'] += entropy.mean()
                stats['loss'] += loss

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                # B0 leaves the policy alone (and GradScaler refuses a step without grads)
                if not critic_warmup:
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                        clip_grad_norm_(params, max_grad_norm)
                    scaler.step(optimizer)
                if use_critic:
                    if critic_max_grad_norm > 0:
                        scaler.unscale_(critic_optimizer)
                        clip_grad_norm_(critic.parameters(), critic_max_grad_norm)
                    scaler.step(critic_optimizer)
                    critic_optimizer.zero_grad(set_to_none=True)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if use_critic:
                critic_scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit(is_idle=False)

            if steps % save_every == 0:
                pb.close()

                writer.add_scalar('important_ratio/ratio', stats['important_ratio'] / save_every, steps)
                writer.add_scalar('entropy/entropy', stats['entropy'] / save_every, steps)
                writer.add_scalar('loss', stats['loss'] / save_every, steps)
                if not online:
                    pass
                if ppo_batches > 0:
                    for k, v in ppo_stats.items():
                        writer.add_scalar('critic/loss' if k == 'critic_loss' else f'ppo/{k}', v / ppo_batches, steps)
                    writer.add_scalar('ppo/lag_max', lag_max, steps)
                    ppo_stats.clear()
                    ppo_batches = 0
                    lag_max = 0
                if critic_diag['value']:
                    # 05 §4: B0 runs until calibration is monotone and EV has plateaued
                    values = np.concatenate(critic_diag['value'])
                    mc_returns = np.concatenate(critic_diag['mc_return'])
                    ev = explained_variance(values, mc_returns)
                    mean_v, mean_t, monotone = calibration(values, mc_returns)
                    writer.add_scalar('critic/explained_variance', ev, steps)
                    writer.add_scalar('critic/calibration_monotone', int(monotone), steps)
                    writer.add_scalars('critic/calibration_target', {f'bin{i}': t for i, t in enumerate(mean_t)}, steps)
                    writer.add_scalars('critic/calibration_value', {f'bin{i}': v for i, v in enumerate(mean_v)}, steps)
                    writer.add_scalar('critic/adv_var_ratio', np.nanmean(critic_diag['adv_var_ratio']), steps)
                    logging.info(f'critic: explained variance {ev:.4f}, calibration monotone: {monotone}')
                    for v in critic_diag.values():
                        v.clear()
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')
                stats_dict = save_shared_stats(shared_stats, steps, writer)
                state = {
                    'mortal': mortal.state_dict(),
                    'policy_net': policy_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                    'shared_stats': stats_dict,
                }
                if use_critic:
                    state['critic'] = critic.state_dict()
                    state['critic_optimizer'] = critic_optimizer.state_dict()
                    state['critic_scheduler'] = critic_scheduler.state_dict()
                # atomic, so a crash or a fleet pull never sees a torn checkpoint
                torch.save(state, state_file + '.tmp')
                os.replace(state_file + '.tmp', state_file)

                if online and steps % submit_every != 0:
                    submit(is_idle=False)

                if not stored and steps % old_update_every == 0:
                    Old_mortal = deepcopy(mortal)
                    Old_policy_net = deepcopy(policy_net)
    
                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, policy_net, device)
                    mortal.train()
                    policy_net.train()
                    
                    
                    avg_pt = stat.avg_pt([90, 45, 0, -135]) # for display only, never used in training
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        def critic_chunks():
            # values and GAE on whole trajectories, before shuffling (05 §3.3)
            for chunk in data_loader:
                flat = gae_targets(chunk, critic, device=device, gamma=critic_gamma, lam=critic_lambda, enable_amp=enable_amp)
                critic_diag['value'].append(flat['value'].numpy())
                critic_diag['mc_return'].append(flat['mc_return'].numpy())
                critic_diag['adv_var_ratio'].append(advantage_variance_ratio(flat['advantage'].numpy(), flat['group'].numpy()))
                yield flat

        if use_critic:
            keys = ('obs', 'action', 'mask', 'advantage', 'behaviour_prob', 'param_version', 'priv', 'returns')
            for mb in shuffled_minibatches(critic_chunks(), batch_size, keys):
                train_batch(*(mb[k] for k in keys))
        else:
            # (obs, actions, masks, advantage), plus (behaviour_prob, param_version)
            # from the ordered dataset
            for batch in data_loader:
                bs = batch[0].shape[0]
                if bs != batch_size:
                    remaining.append(batch)
                    remaining_bs += bs
                    continue
                train_batch(*batch)

            remaining_batches = remaining_bs // batch_size
            if remaining_batches > 0:
                columns = [torch.cat(c, dim=0) for c in zip(*remaining)]

                start = 0
                end = batch_size
                while end <= remaining_bs:
                    train_batch(*(c[start:end] for c in columns))
                    start = end
                    end += batch_size
        pb.close()

        if online:
            submit(is_idle=True)

    def save_shared_stats(shared_stats, steps, writer):
  
        count = shared_stats['count'].value
        mean = shared_stats['mean'].value
        m2 = shared_stats['M2'].value

        if count == 0:
            return {'count': 0, 'mean': 0, 'variance': 0}
        
        if count > 0:
       
            variance = m2 / count if count > 1 else 0
            std_dev = math.sqrt(variance)
            writer.add_scalar('stats/count', count, steps)
            writer.add_scalar('stats/mean', mean, steps)
            writer.add_scalar('stats/std_dev', std_dev, steps)
            return {
            'count': count,
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev
            }
       
  

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break
    

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
