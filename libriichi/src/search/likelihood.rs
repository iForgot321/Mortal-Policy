//! The opponents' action likelihoods that weight a belief particle (D38):
//! the blueprint's probability of what the log shows each opponent doing,
//! given that opponent's private observation in the particle's history.

use super::replay::replay_watched;
use crate::arena::Board;
use crate::chi_type::ChiType;
use crate::consts::ACTION_SPACE;
use crate::mjai::Event;
use crate::state::PlayerState;

use anyhow::Result;

/// An opponent's decision in a particle's history: its state when asked, and
/// the actions that agree with the log.
pub struct Asked {
    pub state: PlayerState,
    pub agree: [bool; ACTION_SPACE],
}

/// The opponents' decisions in `board`'s history up to the root, as `seat`
/// saw the kyoku. A decision every possible action agrees with is left out:
/// its factor is 1.
pub fn asked(board: Board, seen: &[Event], until: usize, seat: u8) -> Result<Vec<Asked>> {
    let mut out = vec![];
    replay_watched(board, seen, until, Some(seat), |states, next| {
        for (o, state) in states.iter().enumerate() {
            if o == seat as usize || !state.last_cans().can_act() {
                continue;
            }
            let agree = agreeing(o as u8, next);
            if informative(state, &agree) {
                out.push(Asked {
                    state: state.clone(),
                    agree,
                });
            }
        }
    })?;
    Ok(out)
}

/// The actions of seat `o` that agree with the logged `next` events. A claim
/// that lost to a higher-priority claim never reaches the log, so every
/// action weaker than the logged claim agrees.
fn agreeing(o: u8, next: &[Event]) -> [bool; ACTION_SPACE] {
    let mut agree = [false; ACTION_SPACE];
    let next: Vec<&Event> = next
        .iter()
        .skip_while(|e| matches!(e, Event::Dora { .. } | Event::ReachAccepted { .. }))
        .collect();
    let all_but = |agree: &mut [bool; ACTION_SPACE], not: &[usize]| {
        agree.fill(true);
        for &a in not {
            agree[a] = false;
        }
    };
    match next.first() {
        Some(Event::Hora { .. }) => {
            let rons = next.iter().take_while(|e| matches!(e, Event::Hora { .. }));
            let mut rons = rons.map(|e| e.actor());
            if rons.any(|a| a == Some(o)) {
                agree[43] = true;
            } else {
                all_but(&mut agree, &[43]);
            }
        }
        Some(ev) if ev.actor() == Some(o) => {
            if let Some(a) = label(ev) {
                agree[a] = true;
            }
        }
        Some(Event::Pon { .. } | Event::Daiminkan { .. } | Event::Chi { .. }) => {
            all_but(&mut agree, &[41, 42, 43]);
        }
        // nobody claimed
        _ => agree[45] = true,
    }
    agree
}

/// The action index of a logged action.
fn label(ev: &Event) -> Option<usize> {
    Some(match *ev {
        Event::Dahai { pai, .. } => pai.as_usize(),
        Event::Reach { .. } => 37,
        Event::Chi { pai, consumed, .. } => match ChiType::new(consumed, pai) {
            ChiType::Low => 38,
            ChiType::Mid => 39,
            ChiType::High => 40,
        },
        Event::Pon { .. } => 41,
        Event::Daiminkan { .. } | Event::Ankan { .. } | Event::Kakan { .. } => 42,
        Event::Hora { .. } => 43,
        Event::Ryukyoku { .. } => 44,
        _ => return None,
    })
}

/// Whether some action the player could take disagrees with the log.
fn informative(state: &PlayerState, agree: &[bool; ACTION_SPACE]) -> bool {
    let cans = state.last_cans();
    let discards = cans.can_discard.then(|| state.discard_candidates_aka());
    let legal = |a: usize| match a {
        0..=36 => discards.is_some_and(|d| d[a]),
        37 => cans.can_riichi,
        38 => cans.can_chi_low,
        39 => cans.can_chi_mid,
        40 => cans.can_chi_high,
        41 => cans.can_pon,
        42 => cans.can_daiminkan || cans.can_ankan || cans.can_kakan,
        43 => cans.can_agari(),
        44 => cans.can_ryukyoku,
        // a pass, on someone else's tile
        _ => !cans.can_discard,
    };
    (0..ACTION_SPACE).any(|a| legal(a) && !agree[a])
}
