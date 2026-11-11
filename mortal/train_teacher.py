"""Train an SL teacher (04 §5; docs/design/teacher-trainer-draft.md): the policy (BC or AWR) and,
with [teacher] heads = true, the auxiliary heads (placement, kyoku-Δ, opponent tenpai and waits).

train_offline_phase1.py's loop, plus:
- the heads and 04 §5's weighted loss (mortal/teacher.py);
- EMA weights (04 §5: 0.999), saved beside the raw ones;
- a fixed validation set from heldout_game, every val_every steps, for the raw and the EMA weights
  ([dataset] val_start = 'YYYY-MM-DD' keeps it inside the train window; heldout_time is scored on the
  workstation, tools/eval/validate.py);
- checkpoints every save_minutes (and on SIGTERM), atomically, under control.run_dir;
- a resume that doesn't re-see data (TeacherDataset's file-batch cursor);
- the file list shuffled by the seed before the loader's workers split it ([dataset] file_order,
  default 'shuffled'; 'sorted' is the teachers' order, one era per worker);
- no test play (evaluation is the lab's job), so no baseline checkpoint is loaded;
- the loop's free speed-ups (the bench_loop.py investigation): the label-mask check on the CPU,
  the next batch copied on a side stream while this one trains, AdamW(fused=True);
- every log_every steps a timing line (data wait, saving, validation, samples/s, the main process's RSS) in
  run_dir/timing.jsonl and TensorBoard.

Paths in the config may use ${MJ_DATA} (the data directory: mjai/, splits/, the box's file indexes).
Writes run_dir/state.pth while training and run_dir/final.pth (raw + EMA weights, read-only) at the end.

    cd mortal && MORTAL_CFG=../experiments/configs/teacher-heads-quick.toml python train_teacher.py
"""

def train():
    import prelude
    import json
    import logging
    import math
    import os
    import random
    import signal
    import time
    from itertools import chain
    import numpy as np
    import torch
    from torch import optim
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from multiprocessing import Manager
    from config import config
    from dataloader import worker_init_fn, load_masks
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, CategoricalPolicy, TeacherHeads
    from checkpoint import load_checkpoint
    from teacher import (TeacherDataset, teacher_loss, Ema, ema_weights, param_groups, validation_files,
                         validate as run_validation)

    ctl, ds, tcfg, opt_cfg = config['control'], config['dataset'], config['teacher'], config['optim']
    ev = os.path.expandvars
    version = ctl['version']
    seed = ctl['seed']
    batch_size = ctl['batch_size']
    run_dir = ev(ctl['run_dir'])
    os.makedirs(run_dir, exist_ok=True)
    # one trainer per run directory: a second one would overwrite the first's checkpoints
    import fcntl
    lock = open(os.path.join(run_dir, 'trainer.lock'), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'another trainer holds {run_dir}/trainer.lock')
    state_file = os.path.join(run_dir, 'state.pth')
    objective = config['policy']['objective']
    assert objective in ('awr', 'bc'), objective
    awr_beta, awr_clip = config['policy']['awr_beta'], config['policy']['awr_clip']
    use_heads = tcfg['heads']
    max_steps = opt_cfg['scheduler']['max_steps']
    file_order = ds.get('file_order', 'shuffled')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(ctl['device'])
    torch.backends.cudnn.benchmark = ctl['enable_cudnn_benchmark']
    enable_amp = ctl['enable_amp']

    mortal = Brain(version=version, **config['resnet'], Norm='GN').to(device)
    policy = CategoricalPolicy().to(device)
    heads = TeacherHeads(hidden=tcfg['hidden'], detach=tcfg['detach']).to(device) if use_heads else None
    models = [m for m in (mortal, policy, heads) if m is not None]
    logging.info(f'version {version}, trunk {config["resnet"]}, heads {use_heads} (detach {tcfg["detach"]}), '
                 f'objective {objective}, params {sum(p.numel() for m in models for p in m.parameters()):,}')

    # weight decay on conv and linear weights only (04 §5)
    decay, no_decay = param_groups(models)
    params = decay + no_decay
    optimizer = optim.AdamW([{'params': decay, 'weight_decay': opt_cfg['weight_decay']}, {'params': no_decay}],
                            lr=1, weight_decay=0, betas=opt_cfg['betas'], eps=opt_cfg['eps'], fused=device.type == 'cuda')
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **opt_cfg['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    ema = Ema(params, opt_cfg['ema_decay'])

    manager = Manager()
    shared_stats = {'count': manager.Value('i', 0), 'mean': manager.Value('d', 0.0),
                    'M2': manager.Value('d', 0.0), 'lock': manager.Lock()}
    steps, cursor, start_batches = 0, {}, {}
    if os.path.exists(state_file):
        st = load_checkpoint(state_file, map_location=device)
        for m, k in ((mortal, 'mortal'), (policy, 'policy_net'), (heads, 'heads')):
            if m is not None:
                m.load_state_dict(st[k])
        optimizer.load_state_dict(st['optimizer'])
        scheduler.load_state_dict(st['scheduler'])
        scaler.load_state_dict(st['scaler'])
        ema.load_state_dict(st['ema'])
        for k in ('count', 'mean', 'M2'):
            shared_stats[k].value = st['shared_stats'][k]
        steps = st['steps']
        # the cursor counts file batches of one order (states from before the key are 'sorted')
        if (saved_order := st.get('file_order', 'sorted')) != file_order:
            raise SystemExit(f"{state_file} was trained in file order '{saved_order}', the config says "
                             f"'{file_order}': set [dataset] file_order = '{saved_order}' to resume it")
        start_batches = {int(w): b for w, b in st['cursor'].items()}
        cursor = {w: b - 1 for w, b in start_batches.items()}  # the last file batch trained on
        logging.info(f'resumed at step {steps:,}; file-batch cursor {cursor}')
    if ctl['enable_compile']:
        for m in models:
            m.compile()

    def save(path, final=False):
        t = time.time()
        st = {
            'mortal': mortal.state_dict(), 'policy_net': policy.state_dict(),
            'heads': heads.state_dict() if heads is not None else None,
            'steps': steps, 'presentations': steps * batch_size, 'config': config, 'timestamp': time.time(),
            'file_order': file_order,
            'shared_stats': {k: shared_stats[k].value for k in ('count', 'mean', 'M2')},
        }
        if final:
            # copies: state_dict() is the live tensors, which hold the raw weights again when saved
            st['ema_weights'] = ema_weights(ema, {'mortal': mortal, 'policy_net': policy, 'heads': heads})
        else:
            st.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                      ema=ema.state_dict(), cursor={str(w): b + 1 for w, b in cursor.items()})
        tmp = path + '.tmp'
        torch.save(st, tmp)
        os.replace(tmp, path)
        return time.time() - t

    masks = load_masks(ev(ds['masks_file'])) if ds.get('masks_file') else {}
    file_list = sorted(torch.load(ev(ds['file_index']), weights_only=True)['file_list'])
    logging.info(f'{len(file_list):,} training files (file order: {file_order}); {len(masks):,} masked games')
    dataset = TeacherDataset(
        version=version, file_list=file_list, pts=config['env']['pts'], shared_stats=shared_stats,
        file_batch_size=ds['file_batch_size'], num_epochs=ds['num_epochs'],
        enable_augmentation=ds['enable_augmentation'], augmented_first=ds['augmented_first'],
        masks=masks, compute_advantage=objective == 'awr', obs_fp16=True, teacher_labels=True,
        seed=seed, start_batches=start_batches, file_order=file_order,
    )
    loader = iter(DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True,
                             num_workers=ds['num_workers'], pin_memory=True,
                             worker_init_fn=worker_init_fn, persistent_workers=True))

    # a fixed validation set: every k-th log of the split, from val_start on if it is set
    val_files = validation_files(ev(ds['data_root']), ds['val_split'], ds['val_games'], start=ds.get('val_start'))
    val_days = sorted(os.path.basename(f)[:8] for f in val_files)
    logging.info(f'{len(val_files)} validation logs from {ds["val_split"]}, {val_days[0]} to {val_days[-1]}')

    def validate():
        return run_validation(mortal, policy, heads, val_files, version=version, pts=config['env']['pts'],
                              masks=masks, loss_cfg=tcfg, device=device, batch_size=batch_size,
                              enable_amp=enable_amp,
                              # few files at a time: validation runs in the main process, beside the training
                              file_batch_size=ds.get('val_file_batch_size', 4))

    writer = SummaryWriter(os.path.join(run_dir, 'tb'))
    timing_log = open(os.path.join(run_dir, 'timing.jsonl'), 'a')
    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    copy_stream = torch.cuda.Stream() if device.type == 'cuda' else None
    data_wait = 0.0

    def fetch():
        """The next batch: checked on the CPU, then copied on the side stream."""
        nonlocal data_wait
        t = time.perf_counter()
        try:
            b = next(loader)
        except StopIteration:
            return None
        data_wait += time.perf_counter() - t
        obs, actions, mask, adv, kd, rank, opp, worker, batch_no = b
        assert mask[torch.arange(len(actions)), actions].all()
        w = int(worker[0])
        cursor[w] = max(cursor.get(w, -1), int(batch_no.max()))
        tensors = (obs, actions, mask, adv, kd, rank, opp)
        if copy_stream is None:
            return tuple(x.to(device) for x in tensors), None
        with torch.cuda.stream(copy_stream):
            g = tuple(x.to(device, non_blocking=True) for x in tensors)
            done = torch.cuda.Event()
            done.record(copy_stream)
        return g, done

    acc = {}
    t_interval, save_time, val_time = time.time(), 0.0, 0.0
    last_save = time.time()
    nxt = fetch()
    logging.info(f'training from step {steps:,} to {max_steps:,}')
    while nxt is not None and steps < max_steps and not stop['flag']:
        (obs, actions, mask, adv, kd, rank, opp), done = nxt
        if done is not None:
            torch.cuda.current_stream().wait_event(done)
            for x in (obs, actions, mask, adv, kd, rank, opp):
                x.record_stream(torch.cuda.current_stream())
        nxt = fetch()

        with torch.autocast(device.type, enabled=enable_amp):
            phi = mortal(obs.float())
            logits = policy.logits(phi, mask).float()
            out = heads(phi) if heads is not None else None
        if objective == 'awr':
            weight = torch.exp(adv.float() / awr_beta)
            if awr_clip is not None:
                weight = weight.clamp(max=awr_clip)
        else:
            weight = torch.ones_like(kd)
        loss, parts = teacher_loss(logits, actions, weight, out, kd, rank, opp, tcfg)
        scaler.scale(loss).backward()
        if opt_cfg['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(params, opt_cfg['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        ema.update()
        steps += 1
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        acc['weight'] = acc.get('weight', 0.0) + weight.mean().detach()

        if steps % ds['val_every'] == 0:
            t = time.time()
            for tag, swap in (('raw', False), ('ema', True)):
                if swap:
                    ema.swap()
                for k, v in validate().items():
                    writer.add_scalar(f'val_{tag}/{k}', v, steps)
                if swap:
                    ema.swap()
            val_time += time.time() - t
        if time.time() - last_save >= ctl['save_minutes'] * 60:
            save_time += save(state_file)
            last_save = time.time()
        if steps % ctl['log_every'] == 0:
            n = ctl['log_every']
            means = {k: (v / n).item() for k, v in acc.items()}  # the one sync per interval
            wall = time.time() - t_interval
            with open('/proc/self/statm') as f:
                rss_gb = int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**30
            rec = {'steps': steps, 'presentations': steps * batch_size, 'wall_s': round(wall, 2),
                   'rss_main_gb': round(rss_gb, 2),
                   'samples_per_s': round(n * batch_size / wall), 'data_wait_s': round(data_wait, 2),
                   'save_s': round(save_time, 2), 'val_s': round(val_time, 2),
                   'lr': scheduler.get_last_lr()[0], **{f'loss_{k}': round(v, 5) for k, v in means.items()}}
            timing_log.write(json.dumps(rec) + '\n')
            timing_log.flush()
            for k, v in means.items():
                writer.add_scalar(f'train/{k}', v, steps)
            writer.add_scalar('timing/samples_per_s', rec['samples_per_s'], steps)
            writer.add_scalar('timing/data_wait_frac', data_wait / wall, steps)
            writer.add_scalar('lr', rec['lr'], steps)
            logging.info(f'step {steps:,}: {rec["samples_per_s"]:,}/s, data wait {data_wait / wall:.1%}, '
                         f'policy {means["policy"]:.4f}' + (f', placement {means["placement"]:.4f}' if heads is not None else ''))
            acc, data_wait, save_time, val_time = {}, 0.0, 0.0, 0.0
            t_interval = time.time()

    save(state_file)
    if stop['flag']:
        logging.info(f'stopped by SIGTERM at step {steps:,}; state saved')
        return
    final = os.path.join(run_dir, 'final.pth')
    save(final, final=True)
    os.chmod(final, 0o444)
    for tag, swap in (('raw', False), ('ema', True)):
        if swap:
            ema.swap()
        logging.info(f'final validation ({tag}): {validate()}')
        if swap:
            ema.swap()
    logging.info(f'done at step {steps:,} ({steps * batch_size:,} presentations): {final}')

if __name__ == '__main__':
    train()
