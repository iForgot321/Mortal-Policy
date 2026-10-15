//! Decision-time search (`06`): replay a kyoku from a latent history, branch
//! at a decision and roll every branch out to the end of the kyoku.
//!
//! The pilot's oracle stage (`06` §0) builds the latent history from the arena
//! seed. That reads the true hidden state, so it is a lab-only diagnostic and
//! never runs in a deployed bot.

mod belief;
mod particle;
mod replay;
mod rollout;
#[cfg(test)]
mod test;

pub use replay::{board_from_seed, decision_seat, replay, root_reaction};
pub use rollout::{Branch, redeal_undealt, rollout};

use crate::agent::new_py_agent;
use crate::arena::BoardState;
use crate::arena::game::{BatchGame, Index};
use crate::mjai::{Event, EventExt};
use crate::py_helper::add_submodule;
use std::fs::{self, File};
use std::io;
use std::path::Path;

use anyhow::{Context, Result};
use flate2::Compression;
use flate2::read::GzEncoder;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde_json as json;

/// `(seed, events, until)`: a self-play kyoku's arena seed, its events from
/// `start_kyoku` as mjai JSON lines, and the index of the event decided at the
/// root (`replay`'s `until`).
type Root = ((u64, u64), Vec<String>, usize);

/// `(root, action, redeal)`: the root's index, the action taken there (a
/// discard or riichi), and `None` to keep the true walls or the seed that
/// deals the undealt tiles again (`redeal_undealt`).
type Job = (usize, u8, Option<u64>);

/// A rollout's kyoku result, and its whole kyoku log when asked for.
#[pyclass(get_all)]
pub struct Outcome {
    /// Counts from 0 (E1); the seat `kyoku % 4` is the dealer.
    pub kyoku: u8,
    pub can_renchan: bool,
    pub has_hora: bool,
    pub has_abortive_ryukyoku: bool,
    pub kyotaku_left: u8,
    /// After the kyoku's deltas and riichi deposits.
    pub scores: [i32; 4],
    pub log: Option<Vec<String>>,
}

fn parse(lines: &[String]) -> Result<Vec<Event>> {
    Ok(lines
        .iter()
        .map(|l| json::from_str(l))
        .collect::<Result<_, _>>()?)
}

fn replay_roots(roots: &[Root]) -> Result<Vec<BoardState>> {
    roots
        .par_iter()
        .enumerate()
        .map(|(i, (seed, lines, until))| {
            let events = parse(lines)?;
            let start = events.first().context("empty kyoku")?;
            replay(board_from_seed(*seed, start)?, &events, *until)
                .with_context(|| format!("root {i} (seed {seed:?}, decision at event {until})"))
        })
        .collect()
}

/// A root's board taking `action`, the deciding seat's discard or riichi.
fn branch(state: BoardState, action: u8) -> Result<Branch> {
    let seat = decision_seat(&state)? as usize;
    let player = &state.agent_context().player_states[seat];
    let ev = root_reaction(player, action)?;
    // the board's own check panics describing some states (see `replay`)
    player
        .validate_reaction(&ev)
        .with_context(|| format!("seat {seat} cannot {ev:?} at this root"))?;
    let mut reactions: [EventExt; 4] = Default::default();
    reactions[seat] = EventExt::no_meta(ev);
    Ok(Branch { state, reactions })
}

/// Rolls the branches out with `engine` at every seat.
fn play_out(engine: PyObject, branches: Vec<Branch>, keep_logs: bool) -> Result<Vec<Outcome>> {
    let player_ids: Vec<u8> = (0..4).cycle().take(4 * branches.len()).collect();
    let mut agent = new_py_agent(engine, &player_ids)?;
    rollout(agent.as_mut(), branches)?
        .into_iter()
        .map(|mut state| {
            let r = state.end();
            let log = keep_logs
                .then(|| state.take_log().iter().map(json::to_string).collect())
                .transpose()?;
            Ok(Outcome {
                kyoku: r.kyoku,
                can_renchan: r.can_renchan,
                has_hora: r.has_hora,
                has_abortive_ryukyoku: r.has_abortive_ryukyoku,
                kyotaku_left: r.kyotaku_left,
                scores: r.scores,
                log,
            })
        })
        .collect()
}

/// Plays each job's branch to the end of its kyoku with `engine` at every
/// seat, from the true latent history of its root (lab-only, `06` §0).
#[pyfunction]
#[pyo3(signature = (engine, roots, jobs, *, keep_logs = false))]
fn rollouts(
    engine: PyObject,
    roots: Vec<Root>,
    jobs: Vec<Job>,
    keep_logs: bool,
    py: Python<'_>,
) -> Result<Vec<Outcome>> {
    py.allow_threads(move || {
        let replayed = replay_roots(&roots)?;
        let branches = jobs
            .iter()
            .map(|&(root, action, redeal)| {
                let mut state = replayed
                    .get(root)
                    .with_context(|| format!("no root {root}"))?
                    .clone();
                if let Some(seed) = redeal {
                    redeal_undealt(&mut state, &mut ChaCha12Rng::seed_from_u64(seed));
                }
                branch(state, action)
            })
            .collect::<Result<Vec<_>>>()?;
        play_out(engine, branches, keep_logs)
    })
}

/// `(seat, obs, mask)` of the player deciding at each root, encoded as the
/// arena encodes it for an agent.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn root_observations<'py>(
    roots: Vec<Root>,
    version: u32,
    py: Python<'py>,
) -> Result<Vec<(u8, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<bool>>)>> {
    let encoded = py.allow_threads(move || {
        replay_roots(&roots)?
            .par_iter()
            .map(|state| {
                let seat = decision_seat(state)?;
                let (obs, mask) =
                    state.agent_context().player_states[seat as usize].encode_obs(version, false);
                Ok((seat, obs, mask))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    Ok(encoded
        .into_iter()
        .map(|(seat, obs, mask)| {
            (
                seat,
                PyArray2::from_owned_array(py, obs),
                PyArray1::from_owned_array(py, mask),
            )
        })
        .collect())
}

/// Hanchans with `engine` at all four seats, one per nonce in
/// `[seed_start.0, seed_start.0 + seed_count)` with key `seed_start.1`. Writes
/// each log to `log_dir/{nonce}_{key}.json.gz` and returns the final scores.
/// The 1v3 arena would play every seed four times, identically when the four
/// seats are one deterministic engine.
#[pyfunction]
fn self_play(
    engine: PyObject,
    seed_start: (u64, u64),
    seed_count: u64,
    log_dir: String,
    py: Python<'_>,
) -> Result<Vec<[i32; 4]>> {
    py.allow_threads(move || {
        fs::create_dir_all(&log_dir)?;
        let (nonce, key) = seed_start;
        let seeds: Vec<_> = (nonce..nonce + seed_count).map(|n| (n, key)).collect();
        let player_ids: Vec<u8> = (0..4).cycle().take(4 * seeds.len()).collect();
        let mut agents = [new_py_agent(engine, &player_ids)?];
        let indexes: Vec<_> = (0..seeds.len())
            .map(|g| {
                std::array::from_fn(|s| Index {
                    agent_idx: 0,
                    player_id_idx: 4 * g + s,
                })
            })
            .collect();
        let results = BatchGame::tenhou_hanchan(true).run(&mut agents, &indexes, &seeds)?;
        results.par_iter().try_for_each(|r| {
            let (n, k) = r.seed;
            let path = Path::new(&log_dir).join(format!("{n}_{k}.json.gz"));
            let log = r.dump_json_log()?;
            let mut gz = GzEncoder::new(log.as_bytes(), Compression::best());
            io::copy(&mut gz, &mut File::create(path)?)?;
            anyhow::Ok(())
        })?;
        Ok(results.iter().map(|r| r.scores).collect())
    })
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "search")?;
    m.add_class::<Outcome>()?;
    m.add_class::<belief::Beliefs>()?;
    m.add_function(wrap_pyfunction!(rollouts, &m)?)?;
    m.add_function(wrap_pyfunction!(root_observations, &m)?)?;
    m.add_function(wrap_pyfunction!(self_play, &m)?)?;
    add_submodule(py, prefix, super_mod, &m)
}
