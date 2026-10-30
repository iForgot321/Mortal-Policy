"""Pieces of the SL teacher's trainer (train_teacher.py; 04 §5, docs/design/teacher-trainer-draft.md)
kept apart so tests can use them: the resumable dataset, the loss and the EMA. (Importing it needs
MORTAL_CFG, as dataloader.py does.)"""
import random
import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info
from dataloader import FileDatasetsIter
from libriichi.dataset import GameplayLoader

class TeacherDataset(FileDatasetsIter):
    """FileDatasetsIter in an order fixed by the seed, with every entry tagged by its worker and
    file batch, so a run resumed after a lost box skips what it has seen.

    Each worker walks its slice of the (sorted) file list once per pass (augmented and not), in an
    order fixed by (seed, pass, worker), in batches of `file_batch_size` files numbered across
    passes. `start_batches[worker]` file batches are skipped: a resumed run passes one more than
    the last batch it trained on, so at most the unseen rest of one file batch per worker is lost.
    reserve_ratio is not supported (always 0)."""
    def __init__(self, *args, seed, start_batches=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.start_batches = start_batches or {}
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
