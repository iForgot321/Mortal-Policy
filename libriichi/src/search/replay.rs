//! Forced replay of a kyoku's public events from a latent history (`06` §3).

use crate::arena::{Board, BoardState, Poll};
use crate::mjai::{Event, EventExt};
use crate::must_tile;
use crate::state::PlayerState;
use crate::tu8;

use anyhow::{Context, Result, bail, ensure};

/// The board that `start_kyoku` describes, dealt from the arena seed: the true
/// latent history of a self-play kyoku. Lab-only, like everything that reads a
/// seed (`06` §2: inference search never reads the true hidden state).
pub fn board_from_seed(seed: (u64, u64), start_kyoku: &Event) -> Result<Board> {
    let Event::StartKyoku {
        bakaze,
        kyoku,
        honba,
        kyotaku,
        scores,
        ..
    } = *start_kyoku
    else {
        bail!("a kyoku starts with start_kyoku, not {start_kyoku:?}");
    };
    let mut board = Board {
        kyoku: (bakaze.as_u8() - tu8!(E)) * 4 + kyoku - 1,
        honba,
        kyotaku,
        scores,
        ..Default::default()
    };
    board.init_from_seed(seed);
    Ok(board)
}

/// Plays `board` through the kyoku's logged `events` (`start_kyoku` first),
/// forcing every reaction from the log, and stops where the board waits for
/// the reaction that makes it emit `events[until]`, the action decided at the
/// root. With `until` equal to `events.len()`, the replay runs through
/// `end_kyoku`.
///
/// Every event the board emits must equal the logged one, so this also checks
/// that the latent history is consistent with the log (`06` §2–3).
pub fn replay(board: Board, events: &[Event], until: usize) -> Result<BoardState> {
    ensure!(
        until <= events.len(),
        "cannot replay to event {until} of {}",
        events.len(),
    );
    // An open kan's dora is revealed in the step of the discard after it.
    let mut until = until;
    if matches!(events.get(until), Some(Event::Dahai { .. })) {
        while until > 0 && matches!(events[until - 1], Event::Dora { .. }) {
            until -= 1;
        }
    }
    let mut state = board.into_state();
    let mut reactions: [EventExt; 4] = Default::default();
    let mut checked = 0;
    loop {
        let poll = state.poll(reactions)?;
        let log = state.agent_context().log;
        ensure!(
            log.len() <= events.len(),
            "the board emitted {} events, more than the {} logged",
            log.len(),
            events.len(),
        );
        for (i, (got, want)) in log.iter().zip(events).enumerate().skip(checked) {
            ensure!(
                got.event == *want,
                "event {i} differs from the log: the board emitted {:?}, the log has {want:?}",
                got.event,
            );
        }
        checked = log.len();

        if matches!(poll, Poll::End) {
            ensure!(
                checked == until && until == events.len(),
                "the kyoku ended at event {checked}, before event {until}",
            );
            return Ok(state);
        }
        if checked == until {
            return Ok(state);
        }
        ensure!(
            checked < until,
            "the replay passed event {until}: the board is at {checked}",
        );
        reactions = forced_reactions(state.agent_context().player_states, &events[checked..])?;
    }
}

/// The reactions that make the board emit the logged `next` events. Claims
/// that lost to a higher-priority claim never reach the log, and replacing
/// them with a pass changes nothing the board emits.
fn forced_reactions(states: &[PlayerState; 4], next: &[Event]) -> Result<[EventExt; 4]> {
    let mut reactions: [EventExt; 4] = Default::default();
    // The board logs these before the action that triggered them: an open
    // kan's dora before the discard after it, and a riichi's acceptance
    // before a call on the riichi discard.
    let pos = next
        .iter()
        .position(|e| !matches!(e, Event::Dora { .. } | Event::ReachAccepted { .. }))
        .context("the log ends while the board waits")?;
    let next = &next[pos..];
    let first = &next[0];
    match *first {
        Event::Dahai { actor, .. }
        | Event::Reach { actor }
        | Event::Chi { actor, .. }
        | Event::Pon { actor, .. }
        | Event::Daiminkan { actor, .. }
        | Event::Ankan { actor, .. }
        | Event::Kakan { actor, .. } => {
            reactions[actor as usize] = EventExt::no_meta(first.clone());
        }
        Event::Hora { .. } => {
            // every ron of a multi-ron, in the order the board logs them
            for ev in next {
                let Event::Hora { actor, target, .. } = *ev else {
                    break;
                };
                let hora = Event::Hora {
                    actor,
                    target,
                    deltas: None,
                    ura_markers: None,
                };
                reactions[actor as usize] = EventExt::no_meta(hora);
            }
        }
        Event::Ryukyoku { .. } => {
            // 九種九牌 is the only draw a player declares; the board emits the
            // others itself once everyone passes
            if let Some(seat) = states.iter().position(|s| s.last_cans().can_ryukyoku) {
                reactions[seat] = EventExt::no_meta(Event::Ryukyoku { deltas: None });
            }
        }
        // tsumo, dora, reach_accepted, end_kyoku: everyone passed
        _ => (),
    }
    Ok(reactions)
}

/// The seat that decides at a replayed root. Search branches only on a
/// player's own turn, where nobody else can act.
pub fn decision_seat(state: &BoardState) -> Result<u8> {
    let mut seats = state
        .agent_context()
        .player_states
        .iter()
        .enumerate()
        .filter(|(_, s)| s.last_cans().can_act());
    match (seats.next(), seats.next()) {
        (Some((seat, s)), None) if s.last_cans().can_discard => Ok(seat as u8),
        _ => bail!("the root is not a single player's discard decision"),
    }
}

/// The event for a candidate root action: a discard (0–36) or riichi (37).
/// The pilot's class, riichi vs dama, needs nothing else (`06` §0).
pub fn root_reaction(state: &PlayerState, action: u8) -> Result<Event> {
    let actor = state.player_id();
    match action {
        0..=36 => {
            let pai = must_tile!(action);
            Ok(Event::Dahai {
                actor,
                pai,
                tsumogiri: state.last_self_tsumo() == Some(pai),
            })
        }
        37 => Ok(Event::Reach { actor }),
        _ => bail!("search branches on a discard or riichi, not action {action}"),
    }
}
