//! The belief stage's worlds (`06` §0, §2): particles sampled from what the
//! deciding seat has seen. Nothing here reads a seed or a hidden tile.

use super::particle::Layout;
use super::replay::{replay_seen_by, root_stop, visible};
use super::{Outcome, branch, parse, play_out};
use crate::arena::{Board, BoardState};
use crate::mjai::Event;

use anyhow::{Context, Result, bail};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

/// `(events, until)`: a kyoku's events from `start_kyoku` as mjai JSON lines,
/// and the index of the action decided at the root. The other seats' hidden
/// tiles in the lines are never read.
type BeliefRoot = (Vec<String>, usize);

pub struct Particle {
    pub board: Board,
    /// Replayed to the root.
    pub state: BoardState,
}

pub struct RootBelief {
    pub seat: u8,
    pub particles: Vec<Particle>,
    /// Particles the chain gave, valid or not.
    pub drawn: usize,
    /// Why particles failed, or why the chain couldn't start: the first few.
    pub errors: Vec<String>,
}

/// How the chain runs (`Layout::chain`).
#[derive(Clone, Copy)]
pub struct ChainParams {
    pub burn_in: usize,
    pub thin: usize,
    pub max_starts: usize,
}

/// The seat that decides at `events[until]`.
pub fn root_seat(events: &[Event], until: usize) -> Result<u8> {
    match events.get(until) {
        Some(Event::Dahai { actor, .. } | Event::Reach { actor }) => Ok(*actor),
        ev => bail!("a root decides a discard or riichi, not {ev:?}"),
    }
}

/// `n` particles from the deciding seat's view, keeping those that replay
/// it to the root. A root whose chain can't start gets none.
pub fn sample_root(
    events: &[Event],
    until: usize,
    n: usize,
    params: ChainParams,
    rng: &mut ChaCha12Rng,
) -> Result<RootBelief> {
    let seat = root_seat(events, until)?;
    let seen: Vec<Event> = events.iter().map(|e| visible(e, seat)).collect();
    let layout = Layout::new(&seen[..root_stop(&seen, until)], seat)?;
    let mut errors = vec![];
    let mut particles = Vec::with_capacity(n);
    // a riichi player's later ankan must keep its waits, which only the
    // replay checks: draw up to 4n to make up for those
    let drawn = layout
        .chain_until(
            n,
            4 * n,
            params.burn_in,
            params.thin,
            params.max_starts,
            rng,
            |board| match replay_seen_by(board.clone(), &seen, until, seat) {
                Ok(state) => {
                    particles.push(Particle { board, state });
                    true
                }
                Err(e) => {
                    if errors.len() < 4 {
                        errors.push(format!("{e:#}"));
                    }
                    false
                }
            },
        )
        .unwrap_or_else(|e| {
            errors.push(format!("{e:#}"));
            0
        });
    Ok(RootBelief {
        seat,
        particles,
        drawn,
        errors,
    })
}

/// The rng of root `i` under `seed`.
pub fn root_rng(seed: u64, i: usize) -> ChaCha12Rng {
    ChaCha12Rng::seed_from_u64(seed ^ (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

/// Particles for each root, from what its deciding seat has seen (S-1).
#[pyclass]
pub struct Beliefs {
    roots: Vec<RootBelief>,
}

#[pymethods]
impl Beliefs {
    #[new]
    #[pyo3(signature = (roots, n, *, seed, burn_in = 2000, thin = 200, max_starts = 20))]
    fn new(
        roots: Vec<BeliefRoot>,
        n: usize,
        seed: u64,
        burn_in: usize,
        thin: usize,
        max_starts: usize,
        py: Python<'_>,
    ) -> Result<Self> {
        let params = ChainParams {
            burn_in,
            thin,
            max_starts,
        };
        let roots = py.allow_threads(move || {
            roots
                .par_iter()
                .enumerate()
                .map(|(i, (lines, until))| {
                    let events = parse(lines)?;
                    sample_root(&events, *until, n, params, &mut root_rng(seed, i))
                        .with_context(|| format!("root {i} (decision at event {until})"))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(Self { roots })
    }

    /// The deciding seat of each root.
    fn seats(&self) -> Vec<usize> {
        self.roots.iter().map(|r| r.seat as usize).collect()
    }

    /// Per root, why particles failed or the chain couldn't start.
    fn errors(&self) -> Vec<Vec<String>> {
        self.roots.iter().map(|r| r.errors.clone()).collect()
    }

    /// `(valid, drawn)` particles per root.
    fn counts(&self) -> Vec<(usize, usize)> {
        self.roots
            .iter()
            .map(|r| (r.particles.len(), r.drawn))
            .collect()
    }

    /// A particle's latent history as mjai tiles: the four starting hands,
    /// then the yama, rinshan, dora and ura indicators, as `Board` holds them.
    fn tiles(&self, root: usize, particle: usize) -> Result<Vec<Vec<String>>> {
        let b = &self
            .roots
            .get(root)
            .and_then(|r| r.particles.get(particle))
            .context("no such particle")?
            .board;
        let strs = |ts: &[crate::tile::Tile]| ts.iter().map(|t| t.to_string()).collect();
        let mut out: Vec<Vec<String>> = b.haipai.iter().map(|h| strs(h)).collect();
        for pile in [&b.yama, &b.rinshan, &b.dora_indicators, &b.ura_indicators] {
            out.push(strs(pile));
        }
        Ok(out)
    }

    /// Plays each job `(root, particle, action)` to the end of its kyoku with
    /// `engine` at every seat, from the particle's history.
    #[pyo3(signature = (engine, jobs, *, keep_logs = false))]
    fn rollouts(
        &self,
        engine: PyObject,
        jobs: Vec<(usize, usize, u8)>,
        keep_logs: bool,
        py: Python<'_>,
    ) -> Result<Vec<Outcome>> {
        py.allow_threads(move || {
            let branches = jobs
                .iter()
                .map(|&(root, particle, action)| {
                    let p = self
                        .roots
                        .get(root)
                        .and_then(|r| r.particles.get(particle))
                        .with_context(|| format!("no particle {particle} at root {root}"))?;
                    branch(p.state.clone(), action)
                })
                .collect::<Result<Vec<_>>>()?;
            play_out(engine, branches, keep_logs)
        })
    }
}
