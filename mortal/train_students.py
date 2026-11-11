"""Distil the student candidates (04 §6; docs/design/distillation-draft.md; D66, D67) in one stream.

Each batch from the train window is featurized once and run through the frozen teachers (forward
only, mortal/distill.py: their ensemble's policy and head targets). Every student then trains on
it with its own optimizer, scheduler, scaler and EMA:
- `loss = 'distill'`: KL to the ensemble + 0.1 × the human action's CE + the heads toward the
  ensemble's outputs (distill.distill_loss);
- `loss = 'labels'`: D66's control, the teachers' own loss on the true labels (teacher.teacher_loss,
  BC), no teacher term.
Students of the same shape start from the same initialization, so the control and the distilled
b8c192 differ only by their loss.

train_teacher.py's mechanics: the seeded, resumable file order (TeacherDataset; [dataset]
file_order); checkpoints every save_minutes and on SIGTERM, atomically; one trainer per run
directory; the side-stream prefetch; validation every val_every steps for every student, raw and
EMA (distill.validate_students: the human move, the heads against the labels, the KL to the
ensemble and the argmax agreement with it); a timing line every log_every steps, with each
part's share of the step (the teachers, each student).

The teachers are named in [teachers] with their sha256, and the trainer refuses a mismatch.
Paths may use ${MJ_DATA}. Writes run_dir/state.pth while training; at the end, per student,
run_dir/ema-<name>.pth and run_dir/raw-<name>.pth (lab-ready: `mortal`, `policy_net`, `heads` and
a config naming its shape), and run_dir/final.pth, the end marker; all read-only.

    cd mortal && MORTAL_CFG=../experiments/configs/students-s1.toml python train_students.py
"""


def train():
    import prelude  # noqa: F401
    import fcntl
    import json
    import logging
    import os
    import random
    import signal
    import time
    from multiprocessing import Manager
    import numpy as np
    import torch
    from torch import optim
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from config import config
    from checkpoint import load_checkpoint
    from dataloader import worker_init_fn, load_masks
    from distill import Student, distill_loss, ensemble_targets, load_frozen_teacher, validate_students
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, CategoricalPolicy, TeacherHeads
    from teacher import TeacherDataset, teacher_loss, Ema, ema_weights, param_groups, validation_files

    ctl, ds, opt_cfg = config['control'], config['dataset'], config['optim']
    dcfg, tcfg = config['distill'], config['teacher']
    ev = os.path.expandvars
    version, seed, batch_size = ctl['version'], ctl['seed'], ctl['batch_size']
    run_dir = ev(ctl['run_dir'])
    os.makedirs(run_dir, exist_ok=True)
    lock = open(os.path.join(run_dir, 'trainer.lock'), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'another trainer holds {run_dir}/trainer.lock')
    state_file = os.path.join(run_dir, 'state.pth')
    max_steps = opt_cfg['scheduler']['max_steps']
    file_order = ds.get('file_order', 'shuffled')

    random.seed(seed)
    np.random.seed(seed)
    device = torch.device(ctl['device'])
    torch.backends.cudnn.benchmark = ctl['enable_cudnn_benchmark']
    enable_amp = ctl['enable_amp']

    # the frozen teachers (D67: the real EMA weights), checked against the config's hashes
    tfiles, tsums = config['teachers']['files'], config['teachers']['sha256']
    assert len(tfiles) == len(tsums) and tfiles, 'one sha256 per teacher file'
    teachers = [load_frozen_teacher(ev(p), device, sha256=h) for p, h in zip(tfiles, tsums)]
    for p, h in zip(tfiles, tsums):
        logging.info(f'teacher {p} ({h[:16]}…)')

    students = []
    for name, sc in config['students'].items():
        kind = sc.get('loss', 'distill')
        assert kind in ('distill', 'labels'), (name, kind)
        # one initialization per shape: the control and the distilled student of its shape match
        torch.manual_seed(seed * 1_000_003 + sc['conv_channels'] * 1_000 + sc['num_blocks'])
        brain = Brain(version=version, conv_channels=sc['conv_channels'], num_blocks=sc['num_blocks'], Norm='GN').to(device)
        policy = CategoricalPolicy().to(device)
        heads = TeacherHeads(hidden=tcfg['hidden'], detach=tcfg['detach']).to(device)
        decay, no_decay = param_groups([brain, policy, heads])
        params = decay + no_decay
        optimizer = optim.AdamW([{'params': decay, 'weight_decay': opt_cfg['weight_decay']}, {'params': no_decay}],
                                lr=1, weight_decay=0, betas=opt_cfg['betas'], eps=opt_cfg['eps'],
                                fused=device.type == 'cuda')
        scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **opt_cfg['scheduler'])
        students.append(Student(name, kind, brain, policy, heads, optimizer, scheduler,
                                GradScaler(device.type, enabled=enable_amp), Ema(params, opt_cfg['ema_decay']), params))
        logging.info(f'student {name}: b{sc["num_blocks"]}c{sc["conv_channels"]}, loss {kind}, '
                     f'params {sum(p.numel() for p in params):,}')
    torch.manual_seed(seed)

    manager = Manager()
    shared_stats = {'count': manager.Value('i', 0), 'mean': manager.Value('d', 0.0),
                    'M2': manager.Value('d', 0.0), 'lock': manager.Lock()}
    steps, cursor, start_batches = 0, {}, {}
    if os.path.exists(state_file):
        st = load_checkpoint(state_file, map_location=device)
        if (saved_order := st.get('file_order')) != file_order:
            raise SystemExit(f"{state_file} was trained in file order '{saved_order}', the config says "
                             f"'{file_order}': set [dataset] file_order = '{saved_order}' to resume it")
        if set(st['students']) != {s.name for s in students}:
            raise SystemExit(f'{state_file} holds students {sorted(st["students"])}, the config {[s.name for s in students]}')
        if st['teachers'] != list(tsums):
            raise SystemExit(f'{state_file} was trained with other teachers')
        for s in students:
            ss = st['students'][s.name]
            for k, m in s.modules.items():
                m.load_state_dict(ss[k])
            s.optimizer.load_state_dict(ss['optimizer'])
            s.scheduler.load_state_dict(ss['scheduler'])
            s.scaler.load_state_dict(ss['scaler'])
            s.ema.load_state_dict(ss['ema'])
        steps = st['steps']
        start_batches = {int(w): b for w, b in st['cursor'].items()}
        cursor = {w: b - 1 for w, b in start_batches.items()}
        logging.info(f'resumed at step {steps:,}; file-batch cursor {cursor}')
    if ctl['enable_compile']:
        for t in teachers:
            for m in t:
                m.compile()
        for s in students:
            for m in s.modules.values():
                m.compile()

    def student_config(s):
        """The config a lab-ready student file carries: its own shape (lab/arena.py, tools/eval)."""
        sc = config['students'][s.name]
        return {'control': {'version': version}, 'resnet': {'conv_channels': sc['conv_channels'],
                'num_blocks': sc['num_blocks'], 'norm': 'GN'}, 'teacher': {'hidden': tcfg['hidden'], 'detach': tcfg['detach']},
                'student': {'name': s.name, 'loss': s.kind}, 'run': config}

    def save(path):
        t = time.time()
        st = {
            'students': {s.name: {**{k: m.state_dict() for k, m in s.modules.items()},
                                  'optimizer': s.optimizer.state_dict(), 'scheduler': s.scheduler.state_dict(),
                                  'scaler': s.scaler.state_dict(), 'ema': s.ema.state_dict()} for s in students},
            'steps': steps, 'presentations': steps * batch_size, 'config': config, 'timestamp': time.time(),
            'file_order': file_order, 'teachers': list(tsums), 'cursor': {str(w): b + 1 for w, b in cursor.items()},
        }
        torch.save(st, path + '.tmp')
        os.replace(path + '.tmp', path)
        return time.time() - t

    def save_final():
        for s in students:
            for tag, weights in (('ema', ema_weights(s.ema, s.modules)),
                                 ('raw', {k: {n: t.detach().clone() for n, t in m.state_dict().items()}
                                          for k, m in s.modules.items()})):
                path = os.path.join(run_dir, f'{tag}-{s.name}.pth')
                torch.save({**weights, 'config': student_config(s), 'steps': steps, 'presentations': steps * batch_size,
                            'weights': tag, 'teachers': list(tsums)}, path + '.tmp')
                os.replace(path + '.tmp', path)
                os.chmod(path, 0o444)
        final = os.path.join(run_dir, 'final.pth')
        torch.save({'students': [s.name for s in students], 'steps': steps, 'presentations': steps * batch_size,
                    'config': config, 'teachers': list(tsums)}, final + '.tmp')
        os.replace(final + '.tmp', final)
        os.chmod(final, 0o444)
        return final

    masks = load_masks(ev(ds['masks_file'])) if ds.get('masks_file') else {}
    file_list = sorted(torch.load(ev(ds['file_index']), weights_only=True)['file_list'])
    logging.info(f'{len(file_list):,} training files (file order: {file_order}); {len(masks):,} masked games')
    dataset = TeacherDataset(
        version=version, file_list=file_list, pts=config['env']['pts'], shared_stats=shared_stats,
        file_batch_size=ds['file_batch_size'], num_epochs=ds['num_epochs'],
        enable_augmentation=ds['enable_augmentation'], augmented_first=ds['augmented_first'],
        # the labels: the control's targets, and every student's validation against them
        masks=masks, compute_advantage=False, obs_fp16=True, teacher_labels=True,
        seed=seed, start_batches=start_batches, file_order=file_order,
    )
    loader = iter(DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True,
                             num_workers=ds['num_workers'], pin_memory=True,
                             worker_init_fn=worker_init_fn, persistent_workers=True))

    val_files = validation_files(ev(ds['data_root']), ds['val_split'], ds['val_games'], start=ds.get('val_start'))
    val_days = sorted(os.path.basename(f)[:8] for f in val_files)
    logging.info(f'{len(val_files)} validation logs from {ds["val_split"]}, {val_days[0]} to {val_days[-1]}')

    def validate():
        return validate_students(students, teachers, val_files, version=version, pts=config['env']['pts'],
                                 masks=masks, loss_cfg=tcfg, device=device, batch_size=batch_size,
                                 enable_amp=enable_amp, file_batch_size=ds.get('val_file_batch_size', 4))

    def log_validation(res):
        writer.add_scalar('val/ensemble_policy', res['ensemble']['policy'], steps)
        writer.add_scalar('val/ensemble_top1', res['ensemble']['top1'], steps)
        for s in students:
            for tag, d in res[s.name].items():
                for k, v in d.items():
                    writer.add_scalar(f'val_{tag}/{s.name}/{k}', v, steps)
        rec = {'steps': steps, 'val': res}
        timing_log.write(json.dumps(rec) + '\n')
        timing_log.flush()
        logging.info(f'validation at step {steps:,}: ensemble top-1 {res["ensemble"]["top1"]:.4f}; '
                     + ', '.join(f'{s.name} top-1 {res[s.name]["ema"]["top1"]:.4f} (EMA), KL {res[s.name]["ema"]["kl_ens"]:.4f}'
                                 for s in students))

    writer = SummaryWriter(os.path.join(run_dir, 'tb'))
    timing_log = open(os.path.join(run_dir, 'timing.jsonl'), 'a')
    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    cuda = device.type == 'cuda'
    copy_stream = torch.cuda.Stream() if cuda else None
    data_wait = 0.0

    def fetch():
        nonlocal data_wait
        t = time.perf_counter()
        try:
            b = next(loader)
        except StopIteration:
            return None
        data_wait += time.perf_counter() - t
        obs, actions, mask, _, kd, rank, opp, worker, batch_no = b
        assert mask[torch.arange(len(actions)), actions].all()
        w = int(worker[0])
        cursor[w] = max(cursor.get(w, -1), int(batch_no.max()))
        tensors = (obs, actions, mask, kd, rank, opp)
        if copy_stream is None:
            return tuple(x.to(device) for x in tensors), None
        with torch.cuda.stream(copy_stream):
            g = tuple(x.to(device, non_blocking=True) for x in tensors)
            done = torch.cuda.Event()
            done.record(copy_stream)
        return g, done

    # per-part GPU time (the teachers, each student), summed over a log interval
    marks = []
    part_ms = {}

    def mark():
        if cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def elapsed(a, b):
        return a.elapsed_time(b) if cuda else (b - a) * 1000

    acc = {}
    t_interval, save_time, val_time = time.time(), 0.0, 0.0
    last_save = time.time()
    names = ['teachers'] + [s.name for s in students]
    nxt = fetch()
    logging.info(f'training from step {steps:,} to {max_steps:,}')
    while nxt is not None and steps < max_steps and not stop['flag']:
        (obs, actions, mask, kd, rank, opp), done = nxt
        if done is not None:
            torch.cuda.current_stream().wait_event(done)
            for x in (obs, actions, mask, kd, rank, opp):
                x.record_stream(torch.cuda.current_stream())
        nxt = fetch()
        obs = obs.float()

        ev_marks = [mark()]
        with torch.autocast(device.type, enabled=enable_amp):
            tgt = ensemble_targets(teachers, obs, mask)
        ev_marks.append(mark())
        for s in students:
            with torch.autocast(device.type, enabled=enable_amp):
                logits, out = s.forward(obs, mask)
            if s.kind == 'distill':
                loss, parts = distill_loss(logits, out, actions, mask, tgt, dcfg)
            else:
                loss, parts = teacher_loss(logits, actions, torch.ones_like(kd), out, kd, rank, opp, tcfg)
            s.scaler.scale(loss).backward()
            if opt_cfg['max_grad_norm'] > 0:
                s.scaler.unscale_(s.optimizer)
                clip_grad_norm_(s.params, opt_cfg['max_grad_norm'])
            s.scaler.step(s.optimizer)
            s.scaler.update()
            s.optimizer.zero_grad(set_to_none=True)
            s.scheduler.step()
            s.ema.update()
            for k, v in parts.items():
                acc[f'{s.name}/{k}'] = acc.get(f'{s.name}/{k}', 0.0) + v
            ev_marks.append(mark())
        marks.append(ev_marks)
        steps += 1

        if steps % ds['val_every'] == 0:
            t = time.time()
            log_validation(validate())
            val_time += time.time() - t
        if time.time() - last_save >= ctl['save_minutes'] * 60:
            save_time += save(state_file)
            last_save = time.time()
        if steps % ctl['log_every'] == 0:
            n = ctl['log_every']
            means = {k: (v / n).item() for k, v in acc.items()}  # the one sync per interval
            for ms in marks:
                for name, a, b in zip(names, ms[:-1], ms[1:]):
                    part_ms[name] = part_ms.get(name, 0.0) + elapsed(a, b)
            gpu_total = sum(part_ms.values()) or 1.0
            wall = time.time() - t_interval
            with open('/proc/self/statm') as f:
                rss_gb = int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**30
            rec = {'steps': steps, 'presentations': steps * batch_size, 'wall_s': round(wall, 2),
                   'rss_main_gb': round(rss_gb, 2), 'samples_per_s': round(n * batch_size / wall),
                   'data_wait_s': round(data_wait, 2), 'save_s': round(save_time, 2), 'val_s': round(val_time, 2),
                   'lr': students[0].scheduler.get_last_lr()[0],
                   'gpu_share': {k: round(v / gpu_total, 4) for k, v in part_ms.items()},
                   'gpu_ms_per_step': round(gpu_total / n, 2),
                   **{f'loss_{k}': round(v, 5) for k, v in means.items()}}
            if cuda:
                rec['gpu_mem_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            timing_log.write(json.dumps(rec) + '\n')
            timing_log.flush()
            for k, v in means.items():
                writer.add_scalar(f'train/{k}', v, steps)
            for k, v in rec['gpu_share'].items():
                writer.add_scalar(f'timing/gpu_share/{k}', v, steps)
            writer.add_scalar('timing/samples_per_s', rec['samples_per_s'], steps)
            writer.add_scalar('timing/data_wait_frac', data_wait / wall, steps)
            writer.add_scalar('lr', rec['lr'], steps)
            logging.info(f'step {steps:,}: {rec["samples_per_s"]:,}/s, data wait {data_wait / wall:.1%}, '
                         + ', '.join(f'{s.name} {means.get(f"{s.name}/kl", means.get(f"{s.name}/policy", float("nan"))):.4f}'
                                     for s in students))
            acc, marks, part_ms = {}, [], {}
            data_wait, save_time, val_time = 0.0, 0.0, 0.0
            t_interval = time.time()

    save(state_file)
    if stop['flag']:
        logging.info(f'stopped by SIGTERM at step {steps:,}; state saved')
        return
    final = save_final()
    log_validation(validate())
    logging.info(f'done at step {steps:,} ({steps * batch_size:,} presentations): {final}')


if __name__ == '__main__':
    train()
