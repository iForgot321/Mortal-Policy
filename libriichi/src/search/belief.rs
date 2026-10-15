//! The belief stage's worlds (`06` §0, §2): particles sampled from what the
//! deciding seat has seen. Nothing here reads a seed or a hidden tile.

use super::likelihood::{Asked, asked};
use super::particle::Layout;
use super::replay::{replay_seen_by, root_stop, visible};
use super::{Outcome, branch, parse, play_out};
use crate::arena::{Board, BoardState};
use crate::mjai::Event;

use anyhow::{Context, Result, bail};
use numpy::{PyArray1, PyArray2};
use pyo3::intern;
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
    /// Log of the opponents' action likelihood (`Beliefs::weigh`); 0 before.
    pub log_lik: f64,
}

pub struct RootBelief {
    pub seat: u8,
    /// The kyoku as the seat saw it, and the root's decision index.
    pub seen: Vec<Event>,
    pub until: usize,
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
                    particles.push(Particle {
                        board,
                        state,
                        log_lik: 0.,
                    });
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
        seen,
        until,
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

    /// Weighs each particle by the blueprint's likelihood of what the log
    /// shows the opponents doing (D38): the sum, over their decisions, of
    /// the log of the probability `engine` gives the actions that agree with
    /// the log, at least `floor`. `engine` must return probabilities from
    /// `react_batch` (a policy engine). Roots go `group` at a time to bound
    /// memory, observations `batch` at a time.
    #[pyo3(signature = (engine, *, version = 4, batch = 1024, group = 16, floor = 1e-8))]
    fn weigh(
        &mut self,
        engine: PyObject,
        version: u32,
        batch: usize,
        group: usize,
        floor: f64,
        py: Python<'_>,
    ) -> Result<()> {
        py.allow_threads(|| {
            for roots in self.roots.chunks_mut(group.max(1)) {
                weigh_group(&engine, roots, version, batch.max(1), floor)?;
            }
            Ok(())
        })
    }

    /// Each seat's shanten at the root, per root and particle.
    fn shanten(&self) -> Vec<Vec<[i8; 4]>> {
        self.roots
            .iter()
            .map(|r| {
                r.particles
                    .iter()
                    .map(|p| {
                        let states = p.state.agent_context().player_states;
                        std::array::from_fn(|s| states[s].shanten())
                    })
                    .collect()
            })
            .collect()
    }

    /// The log likelihoods `weigh` gave, per root and particle.
    fn log_likelihoods(&self) -> Vec<Vec<f64>> {
        self.roots
            .iter()
            .map(|r| r.particles.iter().map(|p| p.log_lik).collect())
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

fn weigh_group(
    engine: &PyObject,
    roots: &mut [RootBelief],
    version: u32,
    batch: usize,
    floor: f64,
) -> Result<()> {
    let histories: Vec<History<'_>> = roots
        .iter()
        .flat_map(|r| {
            r.particles
                .iter()
                .map(move |p| (p.board.clone(), r.seen.as_slice(), r.until, r.seat))
        })
        .collect();
    let mut lls = log_likelihoods(engine, &histories, version, batch, floor)?.into_iter();
    for root in roots.iter_mut() {
        for p in &mut root.particles {
            p.log_lik = lls.next().context("one likelihood per particle")?;
        }
    }
    Ok(())
}

/// A latent history to weigh: the board, the kyoku as `seat` saw it, and
/// the root's decision index.
pub type History<'a> = (Board, &'a [Event], usize, u8);

/// The opponents' log action likelihood of each history (`Beliefs::weigh`).
pub fn log_likelihoods(
    engine: &PyObject,
    histories: &[History<'_>],
    version: u32,
    batch: usize,
    floor: f64,
) -> Result<Vec<f64>> {
    // every informative opponent decision: (history, decision)
    let asked: Vec<(usize, Asked)> = histories
        .par_iter()
        .enumerate()
        .map(|(h, (board, seen, until, seat))| {
            asked(board.clone(), seen, *until, *seat)
                .map(|list| list.into_iter().map(|a| (h, a)).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    let mut out = vec![0.; histories.len()];
    for chunk in asked.chunks(batch) {
        let encoded: Vec<_> = chunk
            .par_iter()
            .map(|(_, a)| a.state.encode_obs(version, false))
            .collect();
        let probs: Vec<Vec<f32>> = Python::with_gil(|py| {
            let (obs, masks): (Vec<_>, Vec<_>) = encoded
                .into_iter()
                .map(|(o, m)| {
                    (
                        PyArray2::from_owned_array(py, o),
                        PyArray1::from_owned_array(py, m),
                    )
                })
                .unzip();
            let ret = engine
                .bind(py)
                .call_method1(intern!(py, "react_batch"), (obs, masks, py.None()))
                .context("failed to execute `react_batch` on Python engine")?;
            ret.get_item(1)?
                .extract()
                .context("`react_batch` must return the probabilities second")
        })?;
        for ((h, a), pr) in chunk.iter().zip(probs) {
            let agreeing: f64 = pr
                .iter()
                .zip(&a.agree)
                .filter(|&(_, &ok)| ok)
                .map(|(&x, _)| f64::from(x))
                .sum();
            out[*h] += agreeing.max(floor).ln();
        }
    }
    Ok(out)
}
