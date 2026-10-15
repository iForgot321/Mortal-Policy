use super::*;
use crate::agent::{BatchAgent, ShantenGreedy};
use crate::arena::Board;
use crate::arena::GameResult;
use crate::arena::game::{BatchGame, Index};
use crate::tile::Tile;

/// Self-play hanchans of `ShantenGreedy`, which riichis whenever it can and
/// takes every win, so the logs have riichi, ron, tsumo and draws.
fn games(seeds: &[(u64, u64)]) -> Vec<GameResult> {
    let player_ids: Vec<u8> = (0..4).cycle().take(4 * seeds.len()).collect();
    let mut agents = [Box::new(ShantenGreedy::new_batched(&player_ids).unwrap()) as _];
    let indexes: Vec<_> = (0..seeds.len())
        .map(|g| {
            std::array::from_fn(|s| Index {
                agent_idx: 0,
                player_id_idx: 4 * g + s,
            })
        })
        .collect();
    BatchGame::tenhou_hanchan(true)
        .run(&mut agents, &indexes, seeds)
        .unwrap()
}

/// Every kyoku's events and arena seed.
fn kyokus(seeds: &[(u64, u64)]) -> Vec<((u64, u64), Vec<Event>)> {
    games(seeds)
        .into_iter()
        .flat_map(|g| {
            g.game_log
                .into_iter()
                .map(move |log| (g.seed, log.into_iter().map(|e| e.event).collect()))
        })
        .collect()
}

const SEEDS: [(u64, u64); 2] = [(1009, 0x5ea7c4), (1021, 0x5ea7c4)];

/// Kyokus of a small random network's self-play (`tests/data/search`), saved
/// because it calls and kans freely, which no Rust agent does.
fn fixture_kyokus() -> Vec<((u64, u64), Vec<Event>)> {
    use flate2::read::GzDecoder;
    use std::io::Read;
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/data/search");
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    let mut out = vec![];
    for path in paths {
        let mut raw = String::new();
        GzDecoder::new(std::fs::File::open(path).unwrap())
            .read_to_string(&mut raw)
            .unwrap();
        let events: Vec<Event> = raw
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let Event::StartGame {
            seed: Some(seed), ..
        } = events[0]
        else {
            panic!("no seed");
        };
        let mut kyoku = vec![];
        for ev in events {
            if matches!(ev, Event::StartKyoku { .. }) {
                kyoku.clear();
            }
            let end = matches!(ev, Event::EndKyoku);
            kyoku.push(ev);
            if end {
                out.push((seed, std::mem::take(&mut kyoku)));
            }
        }
    }
    out
}

/// Rule-based kyokus (riichi, wins) and the fixture's (calls, kans).
fn all_kyokus() -> Vec<((u64, u64), Vec<Event>)> {
    let mut all = kyokus(&SEEDS);
    all.extend(fixture_kyokus());
    all
}

#[test]
fn every_kyoku_replays_from_its_seed() {
    let kyokus = all_kyokus();
    assert!(kyokus.len() >= 40);
    for (seed, events) in &kyokus {
        let board = board_from_seed(*seed, &events[0]).unwrap();
        replay(board, events, events.len()).unwrap();
    }
}

#[test]
fn a_wrong_seed_fails_at_start_kyoku() {
    let (seed, events) = &kyokus(&SEEDS[..1])[0];
    let board = board_from_seed((seed.0 + 1, seed.1), &events[0]).unwrap();
    let Err(err) = replay(board, events, events.len()) else {
        panic!("a wrong seed replayed");
    };
    assert!(err.to_string().starts_with("event 0 differs"), "{err}");
}

/// The roots of `events`: each discard or riichi on a player's own turn.
fn roots(events: &[Event]) -> Vec<(usize, u8)> {
    events
        .iter()
        .enumerate()
        .filter_map(|(i, ev)| match *ev {
            Event::Dahai { pai, .. } => Some((i, pai.as_u8())),
            Event::Reach { .. } => Some((i, 37)),
            _ => None,
        })
        .collect()
}

/// S-2's test (`06` §3): branching with the logged action from the true latent
/// history reproduces the logged rest of the kyoku.
#[test]
fn the_logged_action_reproduces_the_kyoku() {
    let mut branches = vec![];
    let mut want = vec![];
    for (seed, events) in kyokus(&SEEDS) {
        for (until, action) in roots(&events).into_iter().step_by(5) {
            let state = replay(board_from_seed(seed, &events[0]).unwrap(), &events, until).unwrap();
            let seat = decision_seat(&state).unwrap() as usize;
            let ev = root_reaction(&state.agent_context().player_states[seat], action).unwrap();
            assert_eq!(ev, events[until]);
            let mut reactions: [EventExt; 4] = Default::default();
            reactions[seat] = EventExt::no_meta(ev);
            branches.push(Branch { state, reactions });
            want.push(events.clone());
        }
    }
    assert!(branches.len() > 100);

    let player_ids: Vec<u8> = (0..4).cycle().take(4 * branches.len()).collect();
    let mut agent = ShantenGreedy::new_batched(&player_ids).unwrap();
    let ends = rollout(&mut agent as &mut dyn BatchAgent, branches).unwrap();
    for (mut state, events) in ends.into_iter().zip(want) {
        let got: Vec<_> = state.take_log().into_iter().map(|e| e.event).collect();
        assert_eq!(got, events);
    }
}

fn piles(state: &mut BoardState) -> [Vec<Tile>; 4] {
    let b = state.undealt_mut();
    [
        b.yama.clone(),
        b.rinshan.clone(),
        b.dora_indicators.clone(),
        b.ura_indicators.clone(),
    ]
}

fn sorted(piles: &[Vec<Tile>; 4]) -> Vec<u8> {
    let mut all: Vec<_> = piles.iter().flatten().map(|t| t.as_u8()).collect();
    all.sort_unstable();
    all
}

#[test]
fn redeal_keeps_what_was_dealt() {
    let (seed, events) = kyokus(&SEEDS[..1]).swap_remove(0);
    let (until, action) = roots(&events)[10];
    let root = replay(board_from_seed(seed, &events[0]).unwrap(), &events, until).unwrap();

    let mut a = root.clone();
    let mut b = root.clone();
    redeal_undealt(&mut a, &mut ChaCha12Rng::seed_from_u64(1));
    redeal_undealt(&mut b, &mut ChaCha12Rng::seed_from_u64(2));
    let before = piles(&mut root.clone());
    let (after_a, after_b) = (piles(&mut a), piles(&mut b));
    for (x, y) in before.iter().zip(&after_a) {
        assert_eq!(x.len(), y.len());
    }
    assert_eq!(sorted(&before), sorted(&after_a));
    assert_ne!(after_a, before);
    assert_ne!(after_a, after_b);

    // a redealt branch still starts from the logged history and plays legally
    let seat = decision_seat(&a).unwrap() as usize;
    let ev = root_reaction(&a.agent_context().player_states[seat], action).unwrap();
    let mut reactions: [EventExt; 4] = Default::default();
    reactions[seat] = EventExt::no_meta(ev);
    let mut agent = ShantenGreedy::new_batched(&[0, 1, 2, 3]).unwrap();
    let mut end = rollout(
        &mut agent as &mut dyn BatchAgent,
        vec![Branch {
            state: a,
            reactions,
        }],
    )
    .unwrap();
    let log: Vec<_> = end[0].take_log().into_iter().map(|e| e.event).collect();
    assert_eq!(log[..until], events[..until]);
    assert!(matches!(log.last(), Some(Event::EndKyoku)));
}

#[test]
fn root_reaction_takes_discards_and_riichi_only() {
    let (seed, events) = kyokus(&SEEDS[..1]).swap_remove(0);
    let (until, _) = roots(&events)[0];
    let state = replay(board_from_seed(seed, &events[0]).unwrap(), &events, until).unwrap();
    let seat = decision_seat(&state).unwrap() as usize;
    let s = &state.agent_context().player_states[seat];
    assert!(root_reaction(s, 37).is_ok());
    root_reaction(s, 43).unwrap_err();
}

mod particles {
    use super::super::belief::{root_rng, sample_root};
    use super::super::particle::Layout;
    use super::super::replay::{root_stop, visible};
    use super::*;

    /// Roots of every kyoku with the true board at each, a sample of them.
    fn true_roots(every: usize) -> Vec<(Board, Vec<Event>, usize)> {
        let mut out = vec![];
        for (seed, events) in all_kyokus() {
            for (until, _) in roots(&events).into_iter().step_by(every) {
                out.push((
                    board_from_seed(seed, &events[0]).unwrap(),
                    events.clone(),
                    until,
                ));
            }
        }
        out
    }

    fn counts(b: &Board) -> Vec<u8> {
        let mut all: Vec<_> = b
            .haipai
            .iter()
            .flatten()
            .chain(&b.yama)
            .chain(&b.rinshan)
            .chain(&b.dora_indicators)
            .chain(&b.ura_indicators)
            .map(|t| t.as_u8())
            .collect();
        all.sort_unstable();
        all
    }

    /// The layout reads the public events right: the true history fits it.
    #[test]
    fn the_truth_fits_the_layout() {
        for (board, events, until) in true_roots(1) {
            let seat = super::super::belief::root_seat(&events, until).unwrap();
            let layout = Layout::new(&events[..root_stop(&events, until)], seat).unwrap();
            layout.admits(&board).unwrap();
        }
    }

    /// 06 §2's test: every accepted particle replays to the root
    /// observation, and conserves the tiles.
    #[test]
    fn particles_give_back_the_root_observation() {
        let mut accepted = 0;
        for (i, (board, events, until)) in true_roots(9).into_iter().enumerate() {
            let full = counts(&board);
            let truth = replay(board, &events, until).unwrap();
            let seat = decision_seat(&truth).unwrap() as usize;
            let want = truth.agent_context().player_states[seat].encode_obs(4, false);
            let belief = sample_root(&events, until, 4, 400, &mut root_rng(7, i)).unwrap();
            assert_eq!(belief.seat as usize, seat);
            for p in &belief.particles {
                assert_eq!(counts(&p.board), full);
                let got = p.state.agent_context().player_states[seat].encode_obs(4, false);
                assert_eq!(got, want);
            }
            accepted += belief.particles.len();
        }
        assert!(accepted > 100, "{accepted}");
    }

    /// Leaks are hard failures (06 §2): the sampler gives the same particles
    /// from the full log and from the seat's view of it.
    #[test]
    fn the_sampler_reads_no_hidden_tile() {
        for (i, (_, events, until)) in true_roots(13).into_iter().enumerate() {
            let seat = super::super::belief::root_seat(&events, until).unwrap();
            let seen: Vec<_> = events.iter().map(|e| visible(e, seat)).collect();
            let a = sample_root(&events, until, 2, 100, &mut root_rng(3, i)).unwrap();
            let b = sample_root(&seen, until, 2, 100, &mut root_rng(3, i)).unwrap();
            assert_eq!(a.tries, b.tries);
            for (x, y) in a.particles.iter().zip(&b.particles) {
                assert_eq!(counts(&x.board), counts(&y.board));
                assert_eq!(x.board.haipai, y.board.haipai);
                assert_eq!(x.board.yama, y.board.yama);
                assert!((x.log_w - y.log_w).abs() < 1e-12);
            }
        }
    }
}
