"""Pieces of the SL teacher's trainer (train_teacher.py; 04 §5, docs/design/teacher-trainer-draft.md)
kept apart so tests and tools can use them: the resumable dataset, the loss, the EMA and the
validation pass. (Importing it needs MORTAL_CFG, as dataloader.py does.)"""
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info
from dataloader import FileDatasetsIter
from libriichi.dataset import GameplayLoader

# how the file list is split between the loader's workers (TeacherDataset)
FILE_ORDERS = ('shuffled', 'sorted')

class TeacherDataset(FileDatasetsIter):
    """FileDatasetsIter in an order fixed by the seed, with every entry tagged by its worker and
    file batch, so a run resumed after a lost box skips what it has seen.

    Each worker walks its slice of the file list once per pass (augmented and not), in an
    order fixed by (seed, pass, worker), in batches of `file_batch_size` files numbered across
    passes. `start_batches[worker]` file batches are skipped: a resumed run passes one more than
    the last batch it trained on, so at most the unseen rest of one file batch per worker is lost.
    reserve_ratio is not supported (always 0).

    `file_order` fixes the list that `dataloader.worker_init_fn` cuts into one contiguous slice per
    worker. Log paths sort by date, and a batch holds one worker's entries only, so:
    - 'shuffled' (default): the sorted list shuffled once by the seed. Every worker's slice, and so
      every batch, spans the whole window.
    - 'sorted': the teachers' order (teacher-{bc,awr}-s1, heads-quick). Each worker gets one era
      (≈ 2.6 years with 4 workers on the train window), every batch is one era, and the run's last
      steps lack the era of whichever worker runs out first. Kept so those runs can be reproduced.
    The resume cursor counts file batches of one order: a run must resume in the order it started."""
    def __init__(self, *args, seed, start_batches=None, file_order='shuffled', **kwargs):
        super().__init__(*args, **kwargs)
        if file_order not in FILE_ORDERS:
            raise ValueError(f'file_order must be one of {FILE_ORDERS}, got {file_order!r}')
        self.seed = seed
        self.start_batches = start_batches or {}
        self.file_order = file_order
        self.file_list = sorted(self.file_list)
        if file_order == 'shuffled':
            random.Random(seed * 1_000_003 + 500_009).shuffle(self.file_list)
        self.n_pass = 0
        self.n_batch = 0

    def load_files(self, augmented):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        rng = random.Random(self.seed * 1_000_003 + self.n_pass * 1_009 + worker)
        self.n_pass += 1
        files = sorted(self.file_list)
        rng.shuffle(files)
        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
            opponent_labels = self.teacher_labels,
        )
        skip = self.start_batches.get(worker, 0)
        for start_idx in range(0, len(files), self.file_batch_size):
            batch_no = self.n_batch
            self.n_batch += 1
            if batch_no < skip:
                continue
            self.buffer = []
            self.populate_buffer(files[start_idx:start_idx + self.file_batch_size])
            # its own seed per file batch, so skipping batches leaves the later ones' order unchanged
            random.Random(self.seed * 1_000_003 + batch_no * 7_919 + 17).shuffle(self.buffer)
            for entry in self.buffer:
                yield entry + [worker, batch_no]
        self.buffer = []

def teacher_loss(logits, actions, weight, out, kyoku_delta, final_rank, opponent_labels, cfg):
    """04 §5's loss: L_policy + w_placement·L_placement + w_kyoku_delta·L_kyokuΔ + w_aux·L_aux.

    `logits` are the policy's masked raw logits (fp32, D37), `weight` the per-sample AWR weight (1 for
    BC). `out` is TeacherHeads' output (None: policy only). L_aux is the tenpai BCE plus the waits
    BCE over tenpai opponents only. Returns the total and its parts (detached)."""
    logp = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
    policy = -(weight * logp).mean()
    parts = {'policy': policy.detach()}
    if out is None:
        return policy, parts
    placement = F.cross_entropy(out['placement'].float(), final_rank)
    kd = F.smooth_l1_loss(out['kyoku_delta'].float(), kyoku_delta, beta=cfg['huber_delta'])
    tenpai_t = opponent_labels[:, :, 0].float()
    tenpai = F.binary_cross_entropy_with_logits(out['tenpai'].float(), tenpai_t)
    waits_all = F.binary_cross_entropy_with_logits(
        out['waits'].float(), opponent_labels[:, :, 1:].float(), reduction='none')
    waits = (waits_all * tenpai_t[..., None]).sum() / (tenpai_t.sum() * 34).clamp(min=1)
    total = (policy + cfg['w_placement'] * placement + cfg['w_kyoku_delta'] * kd
             + cfg['w_aux'] * (tenpai + waits))
    parts.update(placement=placement.detach(), kyoku_delta=kd.detach(), tenpai=tenpai.detach(), waits=waits.detach())
    return total, parts

def validation_files(data_root, split, games, start=None, end=None, exclude=()):
    """A fixed validation sample: every k-th log of `data_root/splits/<split>.txt` in path (date)
    order, at most `games` of them, as paths under `data_root/mjai`.

    `start` and `end` ('YYYY-MM-DD', inclusive) keep only the logs played in that range. heldout_game
    covers 2009 on, while the train window starts in 2015: without `start`, about a quarter of the
    sample lies outside the window the model trains on. `exclude` holds log ids to leave out."""
    def day(s):
        return s.replace('-', '')
    with open(os.path.join(data_root, 'splits', split + '.txt')) as f:
        rels = sorted(l.strip() for l in f if l.strip())
    ids = [os.path.basename(r).split('.', 1)[0] for r in rels]
    exclude = set(exclude)
    rels = [r for r, i in zip(rels, ids)
            if (start is None or i[:8] >= day(start)) and (end is None or i[:8] <= day(end)) and i not in exclude]
    return [os.path.join(data_root, 'mjai', r) for r in rels[:: max(1, len(rels) // games)][:games]]

@torch.no_grad()
def validate(mortal, policy, heads, files, *, version, pts, masks, loss_cfg, device, batch_size, enable_amp,
             file_batch_size=4):
    """Mean losses and policy top-1 over every seat's decisions in `files`, for the weights the
    modules hold (`heads` may be None). Runs in the calling process, `file_batch_size` logs at a
    time: the trainer validates beside its loader workers."""
    models = [m for m in (mortal, policy, heads) if m is not None]
    training = [m.training for m in models]
    for m in models:
        m.eval()
    vds = FileDatasetsIter(version=version, file_list=list(files), pts=pts, file_batch_size=file_batch_size,
                           masks=masks, compute_advantage=False, obs_fp16=True, teacher_labels=True)
    sums, n, top1 = {}, 0, 0
    for obs, actions, mask, _, kd, rank, opp in DataLoader(vds, batch_size=batch_size, num_workers=0):
        obs, actions, mask = obs.to(device).float(), actions.to(device), mask.to(device)
        kd, rank, opp = kd.to(device), rank.to(device), opp.to(device)
        with torch.autocast(device.type, enabled=enable_amp):
            phi = mortal(obs)
            logits = policy.logits(phi, mask).float()
            out = heads(phi) if heads is not None else None
        _, parts = teacher_loss(logits, actions, torch.ones_like(kd), out, kd, rank, opp, loss_cfg)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v.item() * len(actions)
        top1 += (logits.argmax(-1) == actions).sum().item()
        n += len(actions)
    for m, t in zip(models, training):
        m.train(t)
    return {**{k: v / n for k, v in sums.items()}, 'top1': top1 / n, 'samples': n}

class Ema:
    """An exponential moving average of parameters (04 §5: 0.999), kept in fp32 on their device."""
    def __init__(self, params, decay):
        self.params = list(params)
        self.decay = decay
        self.shadow = [p.detach().clone().float() for p in self.params]

    @torch.no_grad()
    def update(self):
        torch._foreach_lerp_(self.shadow, [p.detach().float() for p in self.params], 1 - self.decay)

    def state_dict(self):
        return {'decay': self.decay, 'shadow': [t.cpu() for t in self.shadow]}

    def load_state_dict(self, state):
        self.decay = state['decay']
        for t, s in zip(self.shadow, state['shadow']):
            t.copy_(s)

    @torch.no_grad()
    def swap(self):
        """Exchange the live and the EMA weights (call twice to swap back)."""
        for p, s in zip(self.params, self.shadow):
            tmp = p.detach().clone()
            p.copy_(s)
            s.copy_(tmp)
