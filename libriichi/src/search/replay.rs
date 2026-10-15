//! Forced replay of a kyoku's public events from a latent history (`06` §3).

use crate::arena::{Board, BoardState, Poll};
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::{must_tile, t, tu8};

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

/// What `seat` sees of `ev`: the other players' starting hands and draws are
/// hidden.
pub fn visible(ev: &Event, seat: u8) -> Event {
    let mut ev = ev.clone();
    match &mut ev {
        Event::StartKyoku { tehais, .. } => {
            for (s, tehai) in tehais.iter_mut().enumerate() {
                if s != seat as usize {
                    *tehai = [t!(?); 13];
                }
            }
        }
        Event::Tsumo { actor, pai } if *actor != seat => *pai = t!(?),
        _ => (),
    }
    ev
}

/// Where the board waits for the action logged at `events[until]`: an open
/// kan's dora is revealed in the step of the discard after it, so a root at
/// that discard starts before the dora.
pub fn root_stop(events: &[Event], until: usize) -> usize {
    let mut stop = until;
    if matches!(events.get(until), Some(Event::Dahai { .. })) {
        while stop > 0 && matches!(events[stop - 1], Event::Dora { .. }) {
            stop -= 1;
        }
    }
    stop
}

/// Plays `board` through the kyoku's logged `events` (`start_kyoku` first),
/// forcing every reaction from the log, and stops where the board waits for
/// the reaction that makes it emit `events[until]`, the action decided at the
/// root (`root_stop`). With `until` equal to `events.len()`, the replay runs
/// through `end_kyoku`.
///
/// Every event the board emits must equal the logged one, so this also checks
/// that the latent history is consistent with the log (`06` §2–3).
pub fn replay(board: Board, events: &[Event], until: usize) -> Result<BoardState> {
    replay_as(board, events, until, None)
}

/// `replay` as `seat` sees the kyoku: the board's events and the logged ones
/// are compared after hiding the other players' hands and draws (`visible`).
/// A particle is valid when this succeeds (`06` §2).
pub fn replay_seen_by(
    board: Board,
    events: &[Event],
    until: usize,
    seat: u8,
) -> Result<BoardState> {
    replay_as(board, events, until, Some(seat))
}

fn replay_as(board: Board, events: &[Event], until: usize, seat: Option<u8>) -> Result<BoardState> {
    replay_watched(board, events, until, seat, |_, _| ())
}

/// `replay`, or `replay_seen_by` with `seat`, calling `watch` wherever the
/// board waits for reactions before the root: with the players' states and
/// the logged events from there on.
pub fn replay_watched(
    board: Board,
    events: &[Event],
    until: usize,
    seat: Option<u8>,
    mut watch: impl FnMut(&[PlayerState; 4], &[Event]),
) -> Result<BoardState> {
    ensure!(
        until <= events.len(),
        "cannot replay to event {until} of {}",
        events.len(),
    );
    let until = root_stop(events, until);
    let view = |ev: &Event| seat.map_or_else(|| ev.clone(), |s| visible(ev, s));
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
            let (got, want) = (view(&got.event), view(want));
            ensure!(
                got == want,
                "event {i} differs from the log: the board emitted {got:?}, the log has {want:?}",
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
        let states = state.agent_context().player_states;
        watch(states, &events[checked..]);
        reactions = forced_reactions(states, &events[checked..])?;
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
            // Arena agents mark a discard of the tile just drawn as tsumogiri,
            // so a discard from hand isn't the drawn tile.
            if let Event::Dahai {
                pai,
                tsumogiri: false,
                ..
            } = *first
            {
                ensure!(
                    states[actor as usize].last_self_tsumo() != Some(pai),
                    "seat {actor} discarded {pai} from hand, but it had just drawn it",
                );
            }
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
    // Checked here so the board never sees an illegal reaction: its own error
    // path describes the state with `brief_info`, which can panic on a
    // particle's hand.
    for (seat, r) in reactions.iter().enumerate() {
        states[seat]
            .validate_reaction(&r.event)
            .with_context(|| format!("seat {seat} cannot {:?}", r.event))?;
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
