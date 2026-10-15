//! S-1: belief particles (`06` §2, D38).
//!
//! A particle is a whole latent history of the kyoku, every starting hand and
//! every wall slot, sampled from what one seat has seen. Replaying the kyoku's
//! public events through it must give back that seat's view.
//!
//! The proposal is uniform (v1): the opponents' unknown slots are dealt in
//! order from the tiles the seat hasn't seen, each with probability
//! proportional to its count among the choices that keep the public events
//! possible. Under the uniform prior over deals the importance weight is
//! then the product, over those slots, of the share of the unseen tiles that
//! were possible choices. The opponents' action likelihoods (D38) multiply
//! that weight and are computed elsewhere.

use super::replay::visible;
use crate::algo::shanten;
use crate::arena::Board;
use crate::mjai::Event;
use crate::must_tile;
use crate::tile::Tile;
use crate::tu8;

use anyhow::{Context, Result, bail, ensure};
use rand::prelude::*;

/// Tile kinds with the red fives apart: ids 0–36.
const KINDS: usize = 37;

/// A slot of the latent history, indexed as in `Board`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Haipai(u8, u8),
    Yama(u8),
    Rinshan(u8),
    Dora(u8),
    Ura(u8),
}

/// A tile an opponent must have held: one of its entries (its starting tiles,
/// then its draws) below `bound` holds a tile of `kind`.
#[derive(Clone, Copy, Debug)]
struct Need {
    kind: u8,
    bound: usize,
}

/// What `seat` knows of a kyoku's latent history at a root: the slots whose
/// tiles it has seen, and what the public events require of each opponent.
pub struct Layout {
    seat: u8,
    /// `start_kyoku`'s kyoku, honba, kyotaku and scores; no tiles.
    head: Board,
    known: Vec<(Slot, Tile)>,
    /// Per seat, its entries in order: 13 starting tiles, then each draw.
    entries: [Vec<Slot>; 4],
    needs: [Vec<Need>; 4],
    /// A draw the player then discarded from hand: in the arena, a discard of
    /// the tile just drawn is always marked tsumogiri, so this draw isn't the
    /// discarded tile.
    not: Vec<(Slot, Tile)>,
    /// Per seat, every tile that left its concealed hand (discards, called
    /// and kanned tiles), with how many entries it had then.
    exits: [Vec<(Tile, usize)>; 4],
    /// Per seat that declared riichi, how many entries it had at the riichi
    /// discard: its hand then must be tenpai.
    riichi_at: [Option<usize>; 4],
    yama_drawn: usize,
    rinshan_drawn: usize,
    dora_revealed: usize,
}

fn full_counts() -> [u8; KINDS] {
    let mut counts = [4; KINDS];
    for five in [tu8!(5m), tu8!(5p), tu8!(5s)] {
        counts[five as usize] = 3;
    }
    for aka in [tu8!(5mr), tu8!(5pr), tu8!(5sr)] {
        counts[aka as usize] = 1;
    }
    counts
}

impl Layout {
    /// The layout from the kyoku's events before the root, as `seat` saw
    /// them. Hidden tiles in `events` are ignored, so full logs can be passed.
    pub fn new(events: &[Event], seat: u8) -> Result<Self> {
        let first = events.first().context("empty kyoku")?;
        let Event::StartKyoku {
            bakaze,
            kyoku,
            honba,
            kyotaku,
            scores,
            dora_marker,
            tehais,
            ..
        } = visible(first, seat)
        else {
            bail!("a kyoku starts with start_kyoku");
        };
        let mut layout = Self {
            seat,
            head: Board {
                kyoku: (bakaze.as_u8() - tu8!(E)) * 4 + kyoku - 1,
                honba,
                kyotaku,
                scores,
                ..Default::default()
            },
            known: vec![(Slot::Dora(4), dora_marker)],
            entries: Default::default(),
            needs: Default::default(),
            not: vec![],
            exits: Default::default(),
            riichi_at: [None; 4],
            yama_drawn: 0,
            rinshan_drawn: 0,
            dora_revealed: 1,
        };
        for (s, entries) in layout.entries.iter_mut().enumerate() {
            entries.extend((0..13).map(|i| Slot::Haipai(s as u8, i)));
        }
        for (i, &tile) in tehais[seat as usize].iter().enumerate() {
            layout.known.push((Slot::Haipai(seat, i as u8), tile));
        }

        // the draw each player holds since its last tsumo, until it discards
        let mut current: [Option<usize>; 4] = [None; 4];
        let mut rinshan_next = false;
        let mut riichi_next = [false; 4];
        for ev in &events[1..] {
            match visible(ev, seat) {
                Event::Tsumo { actor, pai } => {
                    let slot = if rinshan_next {
                        layout.rinshan_drawn += 1;
                        Slot::Rinshan(4 - layout.rinshan_drawn as u8)
                    } else {
                        layout.yama_drawn += 1;
                        Slot::Yama(70 - layout.yama_drawn as u8)
                    };
                    rinshan_next = false;
                    let a = actor as usize;
                    if actor == seat {
                        layout.known.push((slot, pai));
                    }
                    current[a] = Some(layout.entries[a].len());
                    layout.entries[a].push(slot);
                }
                Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                } if actor != seat => {
                    let a = actor as usize;
                    let drawn = current[a].take();
                    let held = layout.entries[a].len();
                    layout.exits[a].push((pai, held));
                    if std::mem::take(&mut riichi_next[a]) {
                        layout.riichi_at[a] = Some(held);
                    }
                    if tsumogiri {
                        let i = drawn.context("tsumogiri without a draw")?;
                        layout.known.push((layout.entries[a][i], pai));
                    } else {
                        let bound = drawn.unwrap_or(layout.entries[a].len());
                        layout.need(a, pai, bound);
                        if let Some(i) = drawn {
                            layout.not.push((layout.entries[a][i], pai));
                        }
                    }
                }
                Event::Chi {
                    actor, consumed, ..
                }
                | Event::Pon {
                    actor, consumed, ..
                } => {
                    current[actor as usize] = None;
                    if actor != seat {
                        layout.need_all(actor as usize, &consumed);
                    }
                }
                Event::Daiminkan {
                    actor, consumed, ..
                } => {
                    current[actor as usize] = None;
                    rinshan_next = true;
                    if actor != seat {
                        layout.need_all(actor as usize, &consumed);
                    }
                }
                Event::Ankan { actor, consumed } => {
                    current[actor as usize] = None;
                    rinshan_next = true;
                    if actor != seat {
                        layout.need_all(actor as usize, &consumed);
                    }
                }
                Event::Kakan { actor, pai, .. } => {
                    current[actor as usize] = None;
                    rinshan_next = true;
                    if actor != seat {
                        layout.need_all(actor as usize, &[pai]);
                    }
                }
                Event::Dora { dora_marker } => {
                    layout.dora_revealed += 1;
                    let slot = Slot::Dora(5 - layout.dora_revealed as u8);
                    layout.known.push((slot, dora_marker));
                }
                Event::Reach { actor } if actor != seat => riichi_next[actor as usize] = true,
                Event::Dahai { .. } | Event::Reach { .. } | Event::ReachAccepted { .. } => (),
                ev => bail!("unexpected event before a root: {ev:?}"),
            }
        }
        Ok(layout)
    }

    fn need(&mut self, seat: usize, tile: Tile, bound: usize) {
        self.needs[seat].push(Need {
            kind: tile.as_u8(),
            bound,
        });
    }

    /// Tiles a call or kan took from the hand: needs and exits at once.
    fn need_all(&mut self, seat: usize, tiles: &[Tile]) {
        let bound = self.entries[seat].len();
        for &t in tiles {
            self.need(seat, t, bound);
            self.exits[seat].push((t, bound));
        }
    }

    /// Samples one latent history. Returns the board and the log of its
    /// importance weight under the uniform prior (the module's doc). The
    /// board still has to replay the seat's view (`replay_seen_by`): riichi
    /// needs a tenpai hand, which this sampler doesn't enforce.
    pub fn sample(&self, rng: &mut impl Rng) -> Result<(Board, f64)> {
        let mut pool = full_counts();
        let mut tiles: Vec<(Slot, Tile)> = Vec::with_capacity(136);
        for &(slot, tile) in &self.known {
            let n = &mut pool[tile.as_usize()];
            ensure!(*n > 0, "the seat has seen more {tile} than exist");
            *n -= 1;
            tiles.push((slot, tile));
        }
        let is_known = |slot| self.known.iter().any(|&(s, _)| s == slot);

        let mut needs = self.needs.clone();
        let mut owed = [0_u8; KINDS];
        for n in needs.iter().flatten() {
            owed[n.kind as usize] += 1;
        }
        let mut log_w = 0.;

        for (s, entries) in self.entries.iter().enumerate() {
            if s == self.seat as usize {
                continue;
            }
            let open: Vec<usize> = (0..entries.len())
                .filter(|&i| !is_known(entries[i]))
                .collect();
            let excl: Vec<Option<u8>> = entries
                .iter()
                .map(|&slot| {
                    self.not
                        .iter()
                        .find(|&&(x, _)| x == slot)
                        .map(|&(_, t)| t.as_u8())
                })
                .collect();
            for (k, &i) in open.iter().enumerate() {
                let slot = entries[i];
                let later = &open[k + 1..];
                let uncovered_ok = can_meet(later, &excl, &needs[s], None);

                let mut weights = [0_u32; KINDS];
                for kind in 0..KINDS {
                    if pool[kind] == 0 || excl[i] == Some(kind as u8) {
                        continue;
                    }
                    let cover = earliest(&needs[s], kind as u8, i);
                    let spare = pool[kind] > owed[kind] - u8::from(cover.is_some());
                    let ok = match cover {
                        None => uncovered_ok,
                        Some(c) => can_meet(later, &excl, &needs[s], Some(c)),
                    };
                    if spare && ok {
                        weights[kind] = pool[kind] as u32;
                    }
                }

                let total: u32 = pool.iter().map(|&n| n as u32).sum();
                let feasible: u32 = weights.iter().sum();
                ensure!(feasible > 0, "no tile fits slot {slot:?}");
                log_w += (f64::from(feasible) / f64::from(total)).ln();
                let mut x = rng.random_range(0..feasible);
                let kind = weights
                    .iter()
                    .position(|&w| {
                        if x < w {
                            return true;
                        }
                        x -= w;
                        false
                    })
                    .unwrap();

                pool[kind] -= 1;
                if let Some(c) = earliest(&needs[s], kind as u8, i) {
                    needs[s].swap_remove(c);
                    owed[kind] -= 1;
                }
                tiles.push((slot, must_tile!(kind)));
            }
            ensure!(needs[s].is_empty(), "seat {s}'s needs left unmet");
        }

        // the undealt slots: the rest of the pool, uniformly
        let mut rest: Vec<Tile> = (0..KINDS)
            .flat_map(|k| std::iter::repeat_n(must_tile!(k), pool[k] as usize))
            .collect();
        rest.shuffle(rng);
        let undealt = (0..70 - self.yama_drawn as u8)
            .map(Slot::Yama)
            .chain((0..4 - self.rinshan_drawn as u8).map(Slot::Rinshan))
            .chain((0..5 - self.dora_revealed as u8).map(Slot::Dora))
            .chain((0..5).map(Slot::Ura));
        let mut placed = 0;
        for (slot, tile) in undealt.zip(&rest) {
            tiles.push((slot, *tile));
            placed += 1;
        }
        ensure!(
            placed == rest.len() && tiles.len() == 136,
            "slots and tiles don't match: {} tiles, {} undealt slots filled of {}",
            tiles.len(),
            placed,
            rest.len(),
        );

        Ok((self.board(&tiles), log_w))
    }

    fn board(&self, tiles: &[(Slot, Tile)]) -> Board {
        let mut board = Board {
            yama: vec![Tile::default(); 70],
            rinshan: vec![Tile::default(); 4],
            dora_indicators: vec![Tile::default(); 5],
            ura_indicators: vec![Tile::default(); 5],
            ..self.head.clone()
        };
        for &(slot, tile) in tiles {
            match slot {
                Slot::Haipai(s, i) => board.haipai[s as usize][i as usize] = tile,
                Slot::Yama(i) => board.yama[i as usize] = tile,
                Slot::Rinshan(i) => board.rinshan[i as usize] = tile,
                Slot::Dora(i) => board.dora_indicators[i as usize] = tile,
                Slot::Ura(i) => board.ura_indicators[i as usize] = tile,
            }
        }
        board
    }
}

/// The tile of `slot` in `board`.
fn slot_tile(board: &Board, slot: Slot) -> Tile {
    match slot {
        Slot::Haipai(s, i) => board.haipai[s as usize][i as usize],
        Slot::Yama(i) => board.yama[i as usize],
        Slot::Rinshan(i) => board.rinshan[i as usize],
        Slot::Dora(i) => board.dora_indicators[i as usize],
        Slot::Ura(i) => board.ura_indicators[i as usize],
    }
}

/// An opponent's entry: a tile the seat has seen, or open slot `u`.
#[derive(Clone, Copy)]
enum Entry {
    Seen(Tile),
    Open(usize),
}

const UNDEALT: usize = 4;

/// A layout compiled for the swap chain: a particle is the tiles of the open
/// slots, the opponents' unseen entries first, then the undealt slots.
struct Open<'a> {
    layout: &'a Layout,
    slots: Vec<Slot>,
    /// The opponent whose entry each open slot is, or `UNDEALT`.
    owner: Vec<usize>,
    not: Vec<Option<Tile>>,
    entries: [Vec<Entry>; 4],
    /// Open slots before this one are opponents' entries.
    in_hands: usize,
}

impl<'a> Open<'a> {
    fn new(layout: &'a Layout) -> Self {
        let seen = |slot| {
            layout
                .known
                .iter()
                .find(|&&(x, _)| x == slot)
                .map(|&(_, t)| t)
        };
        let mut open = Self {
            layout,
            slots: vec![],
            owner: vec![],
            not: vec![],
            entries: Default::default(),
            in_hands: 0,
        };
        for (s, entries) in layout.entries.iter().enumerate() {
            if s == layout.seat as usize {
                continue;
            }
            for &slot in entries {
                let entry = match seen(slot) {
                    Some(t) => Entry::Seen(t),
                    None => {
                        open.slots.push(slot);
                        open.owner.push(s);
                        open.not.push(
                            layout
                                .not
                                .iter()
                                .find(|&&(x, _)| x == slot)
                                .map(|&(_, t)| t),
                        );
                        Entry::Open(open.slots.len() - 1)
                    }
                };
                open.entries[s].push(entry);
            }
        }
        open.in_hands = open.slots.len();
        let undealt = (0..70 - layout.yama_drawn as u8)
            .map(Slot::Yama)
            .chain((0..4 - layout.rinshan_drawn as u8).map(Slot::Rinshan))
            .chain((0..5 - layout.dora_revealed as u8).map(Slot::Dora))
            .chain((0..5).map(Slot::Ura));
        for slot in undealt {
            open.slots.push(slot);
            open.owner.push(UNDEALT);
            open.not.push(None);
        }
        open
    }

    const fn tile(e: Entry, tiles: &[Tile]) -> Tile {
        match e {
            Entry::Seen(t) => t,
            Entry::Open(u) => tiles[u],
        }
    }

    /// Opponent `s`'s needs are met in time and no excluded draw holds its
    /// excluded tile.
    fn fits(&self, s: usize, tiles: &[Tile]) -> bool {
        let mut needs = self.layout.needs[s].clone();
        for (i, &e) in self.entries[s].iter().enumerate() {
            let Entry::Open(u) = e else {
                continue;
            };
            let t = tiles[u];
            if self.not[u] == Some(t) {
                return false;
            }
            if let Some(c) = earliest(&needs, t.as_u8(), i) {
                needs.swap_remove(c);
            }
        }
        needs.is_empty()
    }

    /// Shanten of opponent `s`'s hand after its riichi discard; 0 without
    /// riichi. Assumes `fits`.
    fn riichi_shanten(&self, s: usize, tiles: &[Tile]) -> i8 {
        let Some(at) = self.layout.riichi_at[s] else {
            return 0;
        };
        let mut hand = [0_i16; 34];
        for &e in &self.entries[s][..at] {
            hand[Self::tile(e, tiles).deaka().as_usize()] += 1;
        }
        for &(t, when) in &self.layout.exits[s] {
            if when <= at {
                hand[t.deaka().as_usize()] -= 1;
            }
        }
        let mut counts = [0_u8; 34];
        for (c, &h) in counts.iter_mut().zip(&hand) {
            if h < 0 {
                return i8::MAX;
            }
            *c = h as u8;
        }
        let len: u8 = counts.iter().sum();
        shanten::calc_all(&counts, len / 3)
    }

    /// A random open slot of opponent `s` that held a tile at its riichi.
    fn riichi_slot(&self, s: usize, rng: &mut impl Rng) -> Option<usize> {
        let at = self.layout.riichi_at[s]?;
        let open: Vec<usize> = self.entries[s][..at]
            .iter()
            .filter_map(|&e| match e {
                Entry::Open(u) => Some(u),
                Entry::Seen(_) => None,
            })
            .collect();
        (!open.is_empty()).then(|| open[rng.random_range(0..open.len())])
    }

    fn energy(&self, tiles: &[Tile]) -> i32 {
        (0..4)
            .map(|s| i32::from(self.riichi_shanten(s, tiles).max(0)))
            .sum()
    }

    /// Swaps two open slots, the first in a hand, the second anywhere: a
    /// proposal that depends only on the pair, so it is symmetric.
    fn propose(&self, rng: &mut impl Rng) -> (usize, usize) {
        let u = rng.random_range(0..self.in_hands);
        let mut v = rng.random_range(0..self.slots.len() - 1);
        if v >= u {
            v += 1;
        }
        (u, v)
    }

    fn owners(&self, u: usize, v: usize) -> impl Iterator<Item = usize> {
        let (a, b) = (self.owner[u], self.owner[v]);
        [a, b]
            .into_iter()
            .enumerate()
            .filter(move |&(i, o)| o != UNDEALT && (i == 0 || o != a))
            .map(|(_, o)| o)
    }

    fn board(&self, tiles: &[Tile]) -> Board {
        let all: Vec<(Slot, Tile)> = self
            .layout
            .known
            .iter()
            .copied()
            .chain(self.slots.iter().copied().zip(tiles.iter().copied()))
            .collect();
        self.layout.board(&all)
    }
}

impl Layout {
    /// `n` particles from a Metropolis chain of tile swaps between open slots
    /// (D38's rejuvenation moves). A swap that breaks a constraint is
    /// rejected; the proposal is symmetric, so the chain's stationary
    /// distribution is uniform over the latent histories that fit the layout
    /// and give every riichi player a tenpai hand.
    ///
    /// The chain starts from `sample`, repaired towards tenpai riichi hands by
    /// swaps that never raise the total shanten (at most `max_starts` tries),
    /// then runs `burn_in` swaps and `thin` swaps between particles. The
    /// undealt slots of each particle are shuffled, an exact step for them.
    /// Each board still has to replay the seat's view (`replay_seen_by`).
    #[cfg(test)]
    pub fn chain(
        &self,
        n: usize,
        burn_in: usize,
        thin: usize,
        max_starts: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<Board>> {
        let mut out = Vec::with_capacity(n);
        self.chain_until(n, n, burn_in, thin, max_starts, rng, |b| {
            out.push(b);
            true
        })?;
        Ok(out)
    }

    /// `chain`, offering each particle to `keep` until it has kept `n` or
    /// `max_draws` were offered. Returns how many were offered.
    #[allow(clippy::too_many_arguments)]
    pub fn chain_until(
        &self,
        n: usize,
        max_draws: usize,
        burn_in: usize,
        thin: usize,
        max_starts: usize,
        rng: &mut impl Rng,
        mut keep: impl FnMut(Board) -> bool,
    ) -> Result<usize> {
        let open = Open::new(self);
        let mut tiles = self.start(&open, max_starts, rng)?;
        let ok = |tiles: &[Tile], u, v| {
            open.owners(u, v)
                .all(|s| open.fits(s, tiles) && open.riichi_shanten(s, tiles) <= 0)
        };
        let step = |tiles: &mut Vec<Tile>, rng: &mut _| {
            let (u, v) = open.propose(rng);
            if tiles[u] != tiles[v] {
                tiles.swap(u, v);
                if !ok(tiles, u, v) {
                    tiles.swap(u, v);
                }
            }
        };
        for _ in 0..burn_in {
            step(&mut tiles, rng);
        }
        let (mut kept, mut drawn) = (0, 0);
        while kept < n && drawn < max_draws {
            for _ in 0..thin {
                step(&mut tiles, rng);
            }
            let mut particle = tiles.clone();
            particle[open.in_hands..].shuffle(rng);
            drawn += 1;
            kept += usize::from(keep(open.board(&particle)));
        }
        Ok(drawn)
    }

    /// A valid start: a `sample`, then greedy repair of the riichi hands.
    /// Each repair step tries `TRIES` swaps of a tile in a riichi hand that
    /// is short of tenpai and keeps the best one if it doesn't raise the
    /// total shanten. Only the start is biased; the burn-in forgets it.
    fn start(&self, open: &Open<'_>, max_starts: usize, rng: &mut impl Rng) -> Result<Vec<Tile>> {
        const REPAIR_STEPS: usize = 2_000;
        const TRIES: usize = 32;
        let mut last = None;
        for _ in 0..max_starts {
            let (board, _) = match self.sample(rng) {
                Ok(x) => x,
                Err(e) => {
                    last = Some(e);
                    continue;
                }
            };
            let mut tiles: Vec<Tile> = open.slots.iter().map(|&s| slot_tile(&board, s)).collect();
            let mut energy = open.energy(&tiles);
            for _ in 0..REPAIR_STEPS {
                if energy == 0 {
                    return Ok(tiles);
                }
                let short: Vec<usize> = (0..4)
                    .filter(|&s| open.riichi_shanten(s, &tiles) > 0)
                    .collect();
                let mut best = None;
                for _ in 0..TRIES {
                    let s = short[rng.random_range(0..short.len())];
                    let Some(u) = open.riichi_slot(s, rng) else {
                        break;
                    };
                    let v = rng.random_range(0..open.slots.len());
                    if u == v || tiles[u] == tiles[v] {
                        continue;
                    }
                    tiles.swap(u, v);
                    if open.owners(u, v).all(|o| open.fits(o, &tiles)) {
                        let e = open.energy(&tiles);
                        if best.is_none_or(|(_, _, b)| e < b) {
                            best = Some((u, v, e));
                        }
                    }
                    tiles.swap(u, v);
                }
                if let Some((u, v, e)) = best
                    && e <= energy
                {
                    tiles.swap(u, v);
                    energy = e;
                }
            }
            if energy == 0 {
                return Ok(tiles);
            }
            last = Some(anyhow::anyhow!(
                "riichi hands stayed {energy} shanten short"
            ));
        }
        let why = last.map_or_else(String::new, |e| format!(": {e}"));
        bail!("no start in {max_starts} tries{why}")
    }
}

impl Layout {
    /// A latent history whose concealed hands at the root are `hands`
    /// (counts by tile id, red fives apart; the deciding seat's own is
    /// ignored). Each opponent's open entries get its hand and the tiles it
    /// must have held: a bipartite matching meets every need in time and
    /// keeps excluded draws clear, then `shuffle` swaps within each
    /// opponent's entries spread the arrangement. A riichi hand must be
    /// tenpai, or there is no such history. The undealt slots get the rest
    /// of the tiles in random order. The board still has to replay the seat's
    /// view (`replay_seen_by`).
    pub fn complete(
        &self,
        hands: &[[u8; KINDS]; 4],
        shuffle: usize,
        rng: &mut impl Rng,
    ) -> Result<Board> {
        let open = Open::new(self);
        let mut pool = full_counts();
        let mut take = |kind: usize, what: &str| -> Result<()> {
            ensure!(
                pool[kind] > 0,
                "more {} than exist ({what})",
                must_tile!(kind)
            );
            pool[kind] -= 1;
            Ok(())
        };
        for &(_, t) in &self.known {
            take(t.as_usize(), "seen")?;
        }
        let mut tiles = vec![Tile::default(); open.slots.len()];
        for s in (0..4).filter(|&s| s != self.seat as usize) {
            let slots: Vec<(usize, usize)> = open.entries[s]
                .iter()
                .enumerate()
                .filter_map(|(i, &e)| match e {
                    Entry::Open(u) => Some((i, u)),
                    Entry::Seen(_) => None,
                })
                .collect();
            // what goes in them: the needs, then the hand (no deadline)
            let mut items: Vec<Need> = self.needs[s].clone();
            for (kind, &n) in hands[s].iter().enumerate() {
                items.extend((0..n).map(|_| Need {
                    kind: kind as u8,
                    bound: usize::MAX,
                }));
            }
            ensure!(
                items.len() == slots.len(),
                "seat {s}: {} tiles for {} open entries",
                items.len(),
                slots.len(),
            );
            for item in &items {
                take(item.kind as usize, "a hand or a need")?;
            }
            items.shuffle(rng);
            let holder = match_items(&items, &slots, &open.not)
                .with_context(|| format!("seat {s}: the hand doesn't fit its history"))?;
            for (k, &(_, u)) in slots.iter().enumerate() {
                tiles[u] = must_tile!(items[holder[k]].kind);
            }
        }
        for s in (0..4).filter(|&s| s != self.seat as usize) {
            ensure!(
                open.riichi_shanten(s, &tiles) <= 0,
                "seat {s}'s hand isn't tenpai at its riichi",
            );
        }

        // spread each opponent's arrangement, keeping it consistent
        let hand_slots: Vec<Vec<usize>> = (0..4)
            .map(|s| {
                open.entries[s]
                    .iter()
                    .filter_map(|&e| match e {
                        Entry::Open(u) => Some(u),
                        Entry::Seen(_) => None,
                    })
                    .collect()
            })
            .collect();
        for _ in 0..shuffle {
            let s = open.owner[rng.random_range(0..open.in_hands)];
            let us = &hand_slots[s];
            let (u, v) = (
                us[rng.random_range(0..us.len())],
                us[rng.random_range(0..us.len())],
            );
            if tiles[u] == tiles[v] {
                continue;
            }
            tiles.swap(u, v);
            if !(open.fits(s, &tiles) && open.riichi_shanten(s, &tiles) <= 0) {
                tiles.swap(u, v);
            }
        }

        let mut rest: Vec<Tile> = (0..KINDS)
            .flat_map(|k| std::iter::repeat_n(must_tile!(k), pool[k] as usize))
            .collect();
        ensure!(
            rest.len() == open.slots.len() - open.in_hands,
            "{} tiles left for {} undealt slots",
            rest.len(),
            open.slots.len() - open.in_hands,
        );
        rest.shuffle(rng);
        tiles[open.in_hands..].copy_from_slice(&rest);
        Ok(open.board(&tiles))
    }
}

/// A bipartite matching of `items` to open entries `slots` (`(entry index,
/// open slot)`, in time order): item `n` can go to an entry below its bound
/// whose slot doesn't exclude its kind. Returns, per slot, the item in it.
fn match_items(
    items: &[Need],
    slots: &[(usize, usize)],
    not: &[Option<Tile>],
) -> Result<Vec<usize>> {
    fn augment(
        n: usize,
        items: &[Need],
        slots: &[(usize, usize)],
        not: &[Option<Tile>],
        holder: &mut [usize],
        seen: &mut [bool],
    ) -> bool {
        let need = items[n];
        for (k, &(i, u)) in slots.iter().enumerate() {
            if i >= need.bound {
                break;
            }
            if seen[k] || not[u].is_some_and(|t| t.as_u8() == need.kind) {
                continue;
            }
            seen[k] = true;
            if holder[k] == usize::MAX || augment(holder[k], items, slots, not, holder, seen) {
                holder[k] = n;
                return true;
            }
        }
        false
    }
    let mut holder = vec![usize::MAX; slots.len()];
    // the tightest deadlines first
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&n| items[n].bound);
    for n in order {
        let mut seen = vec![false; slots.len()];
        ensure!(
            augment(n, items, slots, not, &mut holder, &mut seen),
            "no entry for {}",
            must_tile!(items[n].kind),
        );
    }
    Ok(holder)
}

#[cfg(test)]
impl Layout {
    /// Whether `board` fits the layout: the seen tiles where they were seen,
    /// every opponent's needs met by its own entries in time, and no excluded
    /// draw. The true history must fit; a test of how the layout reads the
    /// public events.
    pub fn admits(&self, board: &Board) -> Result<()> {
        let tile = |slot| slot_tile(board, slot);
        for &(slot, t) in &self.known {
            ensure!(tile(slot) == t, "{slot:?} holds {}, seen {t}", tile(slot));
        }
        for &(slot, t) in &self.not {
            ensure!(tile(slot) != t, "{slot:?} holds the excluded {t}");
        }
        for (s, entries) in self.entries.iter().enumerate() {
            if s == self.seat as usize {
                continue;
            }
            let mut needs = self.needs[s].clone();
            for (i, &slot) in entries.iter().enumerate() {
                if self.known.iter().any(|&(x, _)| x == slot) {
                    continue;
                }
                if let Some(c) = earliest(&needs, tile(slot).as_u8(), i) {
                    needs.swap_remove(c);
                }
            }
            ensure!(needs.is_empty(), "seat {s}'s needs unmet: {needs:?}");
        }
        Ok(())
    }
}

/// The need of `kind` with the earliest bound that entry `at` can meet.
fn earliest(needs: &[Need], kind: u8, at: usize) -> Option<usize> {
    needs
        .iter()
        .enumerate()
        .filter(|&(_, n)| n.kind == kind && n.bound > at)
        .min_by_key(|&(_, n)| n.bound)
        .map(|(j, _)| j)
}

/// Whether the open entries `later` (sorted) can still meet `needs`, all
/// but `skip`, where entry `i` can't hold kind `excl[i]`: a bipartite
/// matching of needs to entries below their bounds. Hall's condition on the
/// bounds alone is checked first, as a quick rejection.
fn can_meet(later: &[usize], excl: &[Option<u8>], needs: &[Need], skip: Option<usize>) -> bool {
    let live: Vec<Need> = needs
        .iter()
        .enumerate()
        .filter(|&(j, _)| Some(j) != skip)
        .map(|(_, &n)| n)
        .collect();
    if !hall(later, live.iter().map(|n| n.bound)) {
        return false;
    }
    let mut holder = vec![usize::MAX; later.len()];
    (0..live.len()).all(|n| {
        let mut seen = vec![false; later.len()];
        augment(n, &live, later, excl, &mut holder, &mut seen)
    })
}

/// Kuhn's augmenting path for need `n`; `holder[k]` is the need entry
/// `later[k]` meets.
fn augment(
    n: usize,
    live: &[Need],
    later: &[usize],
    excl: &[Option<u8>],
    holder: &mut [usize],
    seen: &mut [bool],
) -> bool {
    let need = live[n];
    for (k, &i) in later.iter().enumerate() {
        if i >= need.bound {
            break;
        }
        if seen[k] || excl[i] == Some(need.kind) {
            continue;
        }
        seen[k] = true;
        if holder[k] == usize::MAX || augment(holder[k], live, later, excl, holder, seen) {
            holder[k] = n;
            return true;
        }
    }
    false
}

/// Hall's condition for meeting needs with the open entries `slots` (sorted):
/// the k-th earliest bound must have at least k open entries below it.
fn hall(slots: &[usize], bounds: impl Iterator<Item = usize>) -> bool {
    let mut bounds: Vec<usize> = bounds.collect();
    bounds.sort_unstable();
    let mut below = 0;
    for (k, &b) in bounds.iter().enumerate() {
        while below < slots.len() && slots[below] < b {
            below += 1;
        }
        if below <= k {
            return false;
        }
    }
    true
}
