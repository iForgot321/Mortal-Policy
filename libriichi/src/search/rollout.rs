//! Rollouts from a branched root to the end of its kyoku (`06` §4).

use crate::agent::BatchAgent;
use crate::arena::{BoardState, Poll};
use crate::mjai::EventExt;
use crate::tile::Tile;
use std::mem;

use anyhow::Result;
use rand::prelude::*;

/// A kyoku to play out: the board after a forced replay, and the reactions it
/// waits for (the candidate action at the root).
pub struct Branch {
    pub state: BoardState,
    pub reactions: [EventExt; 4],
}

/// Deals the undealt tiles again in a uniformly random order, keeping the size
/// of every pile: the rest of the yama, the rinshan, the unrevealed dora
/// indicators and the ura indicators. The hands, past draws and revealed
/// indicators stay. This keeps what has happened and resamples only the future
/// wall, which no inference can know.
pub fn redeal_undealt(state: &mut BoardState, rng: &mut impl Rng) {
    let board = state.undealt_mut();
    let mut tiles: Vec<Tile> = [
        &board.yama,
        &board.rinshan,
        &board.dora_indicators,
        &board.ura_indicators,
    ]
    .into_iter()
    .flatten()
    .copied()
    .collect();
    tiles.shuffle(rng);

    let mut rest = tiles.as_slice();
    for pile in [
        &mut board.yama,
        &mut board.rinshan,
        &mut board.dora_indicators,
        &mut board.ura_indicators,
    ] {
        let (head, tail) = rest.split_at(pile.len());
        pile.copy_from_slice(head);
        rest = tail;
    }
}

/// Plays every branch to the end of its kyoku with `agent` at all four seats:
/// branch `i`, seat `s` is the agent's index `4 * i + s`. Returns the boards
/// at `end_kyoku`, in branch order.
pub fn rollout(agent: &mut dyn BatchAgent, branches: Vec<Branch>) -> Result<Vec<BoardState>> {
    let (mut states, mut reactions): (Vec<_>, Vec<_>) =
        branches.into_iter().map(|b| (b.state, b.reactions)).unzip();
    let mut active: Vec<usize> = (0..states.len()).collect();

    while !active.is_empty() {
        let mut in_game = Vec::with_capacity(active.len());
        for i in active {
            let state = &mut states[i];
            if matches!(state.poll(mem::take(&mut reactions[i]))?, Poll::End) {
                continue;
            }
            let ctx = state.agent_context();
            for (seat, s) in ctx.player_states.iter().enumerate() {
                if s.last_cans().can_act() {
                    agent.set_scene(4 * i + seat, ctx.log, s, None)?;
                }
            }
            in_game.push(i);
        }

        for &i in &in_game {
            let ctx = states[i].agent_context();
            for (seat, s) in ctx.player_states.iter().enumerate() {
                if s.last_cans().can_act() {
                    reactions[i][seat] = agent.get_reaction(4 * i + seat, ctx.log, s, None)?;
                }
            }
        }
        active = in_game;
    }

    Ok(states)
}
