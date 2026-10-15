//! Training examples for the learned hand model (D58): at the decisions of
//! every seat in a full self-play log, that seat's observation and the true
//! concealed hands of the three others. Reads hidden tiles: lab-only, like
//! training data.

use crate::mjai::Event;
use crate::state::PlayerState;
use std::fs::File;
use std::io::Read;

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use ndarray::prelude::*;
use numpy::{PyArray2, PyArray3};
use pyo3::prelude::*;
use serde_json as json;

/// One example: the deciding seat's observation, and for each other seat in
/// turn order after it, its concealed tile counts (34 kinds) and red fives.
pub struct HandExample {
    pub obs: Array2<f32>,
    pub hands: [[u8; 34]; 3],
    pub akas: [[bool; 3]; 3],
    /// Copies of each kind the deciding seat hasn't seen.
    pub unseen: [u8; 34],
}

/// The examples of one game's events: every `stride`-th decision, starting
/// at the `offset`-th.
pub fn examples(
    events: &[Event],
    version: u32,
    stride: usize,
    offset: usize,
) -> Result<Vec<HandExample>> {
    let mut out = vec![];
    let mut states: Option<[PlayerState; 4]> = None;
    let mut n = 0;
    for ev in events {
        match ev {
            Event::StartKyoku { .. } => {
                states = Some(std::array::from_fn(|s| PlayerState::new(s as u8)));
            }
            // the board doesn't broadcast these, and they end the kyoku
            Event::Hora { .. } | Event::Ryukyoku { .. } | Event::EndKyoku => {
                states = None;
                continue;
            }
            Event::StartGame { .. } | Event::EndGame => continue,
            _ => (),
        }
        let Some(states) = states.as_mut() else {
            continue;
        };
        for s in states.iter_mut() {
            s.update(ev).context("the log doesn't replay")?;
        }
        for (seat, state) in states.iter().enumerate() {
            if !state.last_cans().can_act() {
                continue;
            }
            n += 1;
            if (n - 1) % stride != offset % stride {
                continue;
            }
            let others: [&PlayerState; 3] = std::array::from_fn(|i| &states[(seat + 1 + i) % 4]);
            out.push(HandExample {
                obs: state.encode_obs(version, false).0,
                hands: others.map(PlayerState::tehai),
                akas: others.map(PlayerState::akas_in_hand),
                unseen: state.tiles_seen().map(|seen| 4 - seen),
            });
        }
    }
    Ok(out)
}

/// `(obs, hands, akas, unseen)` from a self-play log: `obs` is (N, C, 34)
/// float32, `hands` (N, 3, 34) uint8 counts, `akas` (N, 9) bool, the red 5m,
/// 5p and 5s of each of the three others, and `unseen` (N, 34) uint8, the
/// copies of each kind the deciding seat hasn't seen.
#[pyfunction]
#[pyo3(signature = (path, *, version = 4, stride = 1, offset = 0))]
#[allow(clippy::type_complexity)]
pub fn hand_examples<'py>(
    path: &str,
    version: u32,
    stride: usize,
    offset: usize,
    py: Python<'py>,
) -> Result<(
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<u8>>,
)> {
    let path = path.to_owned();
    let exs = py.allow_threads(move || {
        let mut raw = String::new();
        GzDecoder::new(File::open(&path)?).read_to_string(&mut raw)?;
        let events = raw
            .lines()
            .map(json::from_str)
            .collect::<Result<Vec<Event>, _>>()
            .with_context(|| format!("{path}: not an mjai log"))?;
        examples(&events, version, stride.max(1), offset)
            .with_context(|| format!("{path}: no examples"))
    })?;

    let n = exs.len();
    let shape = exs.first().map_or((0, 34), |e| e.obs.dim());
    let mut obs = Array3::<f32>::zeros((n, shape.0, shape.1));
    let mut hands = Array3::<u8>::zeros((n, 3, 34));
    let mut akas = Array2::<bool>::from_elem((n, 9), false);
    let mut unseen = Array2::<u8>::zeros((n, 34));
    for (i, e) in exs.into_iter().enumerate() {
        obs.slice_mut(s![i, .., ..]).assign(&e.obs);
        unseen
            .slice_mut(s![i, ..])
            .assign(&ArrayView1::from(&e.unseen[..]));
        for (j, (h, a)) in e.hands.iter().zip(&e.akas).enumerate() {
            hands
                .slice_mut(s![i, j, ..])
                .assign(&ArrayView1::from(&h[..]));
            for (k, &x) in a.iter().enumerate() {
                akas[[i, 3 * j + k]] = x;
            }
        }
    }
    Ok((
        PyArray3::from_owned_array(py, obs),
        PyArray3::from_owned_array(py, hands),
        PyArray2::from_owned_array(py, akas),
        PyArray2::from_owned_array(py, unseen),
    ))
}
