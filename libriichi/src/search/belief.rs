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
    /// Log importance weight of the uniform proposal (`Layout::sample`).
    pub log_w: f64,
}

pub struct RootBelief {
    pub seat: u8,
    pub particles: Vec<Particle>,
    /// Samples drawn, accepted or not.
    pub tries: usize,
}

/// The seat that decides at `events[until]`.
pub fn root_seat(events: &[Event], until: usize) -> Result<u8> {
    match events.get(until) {
        Some(Event::Dahai { actor, .. } | Event::Reach { actor }) => Ok(*actor),
        ev => bail!("a root decides a discard or riichi, not {ev:?}"),
    }
}

/// Up to `n` particles that replay the deciding seat's view to the root,
/// from at most `max_tries` samples.
pub fn sample_root(
    events: &[Event],
    until: usize,
    n: usize,
    max_tries: usize,
    rng: &mut ChaCha12Rng,
) -> Result<RootBelief> {
    let seat = root_seat(events, until)?;
    let seen: Vec<Event> = events.iter().map(|e| visible(e, seat)).collect();
    let layout = Layout::new(&seen[..root_stop(&seen, until)], seat)?;
    let mut particles = Vec::with_capacity(n);
    let mut tries = 0;
    while particles.len() < n && tries < max_tries {
        tries += 1;
        // a dead end (no tile fits a slot) counts as a rejection
        let Ok((board, log_w)) = layout.sample(rng) else {
            continue;
        };
        if let Ok(state) = replay_seen_by(board.clone(), &seen, until, seat) {
            particles.push(Particle {
                board,
                state,
                log_w,
            });
        }
    }
    Ok(RootBelief {
        seat,
        particles,
        tries,
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
    #[pyo3(signature = (roots, n, *, seed, max_tries))]
    fn new(
        roots: Vec<BeliefRoot>,
        n: usize,
        seed: u64,
        max_tries: usize,
        py: Python<'_>,
    ) -> Result<Self> {
        let roots = py.allow_threads(move || {
            roots
                .par_iter()
                .enumerate()
                .map(|(i, (lines, until))| {
                    let events = parse(lines)?;
                    sample_root(&events, *until, n, max_tries, &mut root_rng(seed, i))
                        .with_context(|| format!("root {i} (decision at event {until})"))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(Self { roots })
    }

    /// The deciding seat of each root.
    fn seats(&self) -> Vec<u8> {
        self.roots.iter().map(|r| r.seat).collect()
    }

    /// `(accepted, tries)` per root.
    fn counts(&self) -> Vec<(usize, usize)> {
        self.roots
            .iter()
            .map(|r| (r.particles.len(), r.tries))
            .collect()
    }

    /// The proposal's log importance weights, per root and particle.
    fn log_weights(&self) -> Vec<Vec<f64>> {
        self.roots
            .iter()
            .map(|r| r.particles.iter().map(|p| p.log_w).collect())
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
