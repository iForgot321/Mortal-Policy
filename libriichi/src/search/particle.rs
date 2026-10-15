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

    fn need_all(&mut self, seat: usize, tiles: &[Tile]) {
        let bound = self.entries[seat].len();
        for &t in tiles {
            self.need(seat, t, bound);
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
            for (k, &i) in open.iter().enumerate() {
                let slot = entries[i];
                let later = &open[k + 1..];
                let not = self.not.iter().find(|&&(x, _)| x == slot).map(|&(_, t)| t);
                let uncovered_ok = hall(later, needs[s].iter().map(|n| n.bound));

                let mut weights = [0_u32; KINDS];
                for kind in 0..KINDS {
                    if pool[kind] == 0 || not.is_some_and(|t| t.as_usize() == kind) {
                        continue;
                    }
                    let cover = earliest(&needs[s], kind as u8, i);
                    let spare = pool[kind] > owed[kind] - u8::from(cover.is_some());
                    let ok = match cover {
                        None => uncovered_ok,
                        Some(c) => hall(
                            later,
                            needs[s]
                                .iter()
                                .enumerate()
                                .filter(|&(j, _)| j != c)
                                .map(|(_, n)| n.bound),
                        ),
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

#[cfg(test)]
impl Layout {
    /// Whether `board` fits the layout: the seen tiles where they were seen,
    /// every opponent's needs met by its own entries in time, and no excluded
    /// draw. The true history must fit; a test of how the layout reads the
    /// public events.
    pub fn admits(&self, board: &Board) -> Result<()> {
        let tile = |slot| match slot {
            Slot::Haipai(s, i) => board.haipai[s as usize][i as usize],
            Slot::Yama(i) => board.yama[i as usize],
            Slot::Rinshan(i) => board.rinshan[i as usize],
            Slot::Dora(i) => board.dora_indicators[i as usize],
            Slot::Ura(i) => board.ura_indicators[i as usize],
        };
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
