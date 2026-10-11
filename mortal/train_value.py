"""Arm V, offline: Equim's value-based Mortal (`train.py`, removed upstream in
a24399b) on this fork's data path (05 §2.1, 04 §7: the quick CQL model).

Port of a24399b^:mortal/train.py, offline mode. Kept: Brain + dueling DQN +
next-rank AuxNet in one AdamW, Q target = the decision's kyoku GRP-Δu (MC,
γ = 1), loss 0.5·MSE + min_q_weight·CQL(logsumexp Q − Q(a)) + next_rank_weight·CE.
Changed, as in the pilot quick students (`train_offline_phase1.py`):
- rewards in fixed u units, never standardized (05 §3.2, D37);
- GroupNorm, `control.seed`, Q6/Q8 masks, fp16 obs storage;
- no test_play: evaluation is the lab's job.
Online value learning (no CQL) is not ported yet; `control.online` must be false.

Run from mortal/:  MORTAL_CFG=../experiments/configs/cql-qs-a.toml python train_value.py
"""


def train():
    import prelude

    import logging
    import random
    import torch
    import numpy as np
    from os import path
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import parameter_count, tqdm
    from dataloader import FileDatasetsIter, worker_init_fn, load_masks
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']
    if config['control']['online']:
        raise SystemExit('train_value.py: online value learning is not ported yet')

    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    assert save_every % opt_step_every == 0

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

    # seeds the init, the file order and (through torch's RNG) the
    # DataLoader worker seeds
    if (seed := config['control'].get('seed')) is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logging.info(f'seed: {seed}')

    mortal = Brain(version=version, **config['resnet'], Norm="GN").to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    decay_params = []
    no_decay_params = []
    for model in all_models:
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

    steps = 0
    state_file = config['control']['state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        steps = state['steps']

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    masks = None
    if masks_file := config['dataset'].get('masks_file'):
        masks = load_masks(masks_file)
        logging.info(f'loaded masks for {len(masks):,} games')
    file_list = torch.load(config['dataset']['file_index'], weights_only=True)['file_list']
    logging.info(f'file list size: {len(file_list):,}')
    logging.info(f'total steps: {steps:,}')
    if steps > 0:
        # a resumed run would see a fresh shuffle of the whole subset, not the
        # rest of its epoch; the pilots never resumed either
        logging.warning('resuming mid-epoch: the remaining data is not the unseen rest')

    if num_workers > 1:
        random.shuffle(file_list)
    file_data = FileDatasetsIter(
        version = version,
        file_list = file_list,
        pts = pts,
        file_batch_size = file_batch_size,
        reserve_ratio = reserve_ratio,
        player_names = [],
        num_epochs = num_epochs,
        enable_augmentation = enable_augmentation,
        augmented_first = augmented_first,
        masks = masks,
        compute_advantage = False,
        obs_fp16 = config['dataset'].get('obs_fp16', False),
        value_targets = True,
    )
    data_loader = iter(DataLoader(
        dataset = file_data,
        batch_size = batch_size,
        drop_last = False,
        num_workers = num_workers,
        pin_memory = True,
        worker_init_fn = worker_init_fn,
        persistent_workers = True,
    ))

    pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

    def train_batch(obs, actions, masks, q_target, player_ranks):
        nonlocal steps
        nonlocal idx
        nonlocal pb

        obs = obs.to(dtype=torch.float32, device=device)
        actions = actions.to(dtype=torch.int64, device=device)
        masks = masks.to(dtype=torch.bool, device=device)
        q_target = q_target.to(dtype=torch.float32, device=device)
        player_ranks = player_ranks.to(dtype=torch.int64, device=device)
        assert masks[range(batch_size), actions].all()

        with torch.autocast(device.type, enabled=enable_amp):
            phi = mortal(obs)
            q_out = dqn(phi, masks)
            q = q_out[range(batch_size), actions]
            dqn_loss = 0.5 * mse(q, q_target)
            cql_loss = q_out.logsumexp(-1).mean() - q.mean()

            next_rank_logits, = aux_net(phi)
            next_rank_loss = ce(next_rank_logits, player_ranks)

            loss = sum((
                dqn_loss,
                cql_loss * min_q_weight,
                next_rank_loss * next_rank_weight,
            ))
        scaler.scale(loss / opt_step_every).backward()

        with torch.inference_mode():
            stats['dqn_loss'] += dqn_loss
            stats['cql_loss'] += cql_loss
            stats['next_rank_loss'] += next_rank_loss
            all_q[idx] = q
            all_q_target[idx] = q_target

        steps += 1
        idx += 1
        if idx % opt_step_every == 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                clip_grad_norm_(params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        pb.update(1)

        if steps % save_every == 0:
            pb.close()

            # downsample to reduce tensorboard event size
            all_q_1d = all_q.cpu().numpy().flatten()[::128]
            all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

            writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
            writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
            writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
            writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
            writer.add_histogram('q_predicted', all_q_1d, steps)
            writer.add_histogram('q_target', all_q_target_1d, steps)
            writer.flush()
            logging.info(
                f'total steps: {steps:,}, dqn {stats["dqn_loss"] / save_every:.4f}, '
                f'cql {stats["cql_loss"] / save_every:.4f}, '
                f'next rank {stats["next_rank_loss"] / save_every:.4f}'
            )

            for k in stats:
                stats[k] = 0
            idx = 0

            state = {
                'mortal': mortal.state_dict(),
                'current_dqn': dqn.state_dict(),
                'aux_net': aux_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'steps': steps,
                'timestamp': datetime.now().timestamp(),
                'config': config,
            }
            torch.save(state, state_file)
            pb = tqdm(total=save_every, desc='TRAIN')

    remaining = []
    remaining_bs = 0
    for batch in data_loader:
        bs = batch[0].shape[0]
        if bs != batch_size:
            remaining.append(batch)
            remaining_bs += bs
            continue
        train_batch(*batch)

    if remaining_bs >= batch_size:
        cols = [torch.cat(col, dim=0) for col in zip(*remaining)]
        for start in range(0, remaining_bs - batch_size + 1, batch_size):
            train_batch(*(col[start:start + batch_size] for col in cols))
    pb.close()
    logging.info(f'data exhausted at step {steps:,}')


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
