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
    use super::super::belief::{ChainParams, root_rng, sample_root};

    const PARAMS: ChainParams = ChainParams {
        burn_in: 500,
        thin: 20,
        max_starts: 20,
    };
    use super::super::particle::Layout;
    use super::super::replay::{replay_seen_by, root_stop, visible};
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
            let belief = sample_root(&events, until, 4, PARAMS, &mut root_rng(7, i)).unwrap();
            assert_eq!(belief.seat as usize, seat);
            assert_eq!(belief.particles.len(), belief.drawn);
            for p in &belief.particles {
                assert_eq!(counts(&p.board), full);
                let got = p.state.agent_context().player_states[seat].encode_obs(4, false);
                assert_eq!(got, want);
            }
            accepted += belief.particles.len();
        }
        assert!(accepted > 200, "{accepted}");
    }

    /// Leaks are hard failures (06 §2): the sampler gives the same particles
    /// from the full log and from the seat's view of it.
    #[test]
    fn the_sampler_reads_no_hidden_tile() {
        for (i, (_, events, until)) in true_roots(13).into_iter().enumerate() {
            let seat = super::super::belief::root_seat(&events, until).unwrap();
            let seen: Vec<_> = events.iter().map(|e| visible(e, seat)).collect();
            let a = sample_root(&events, until, 2, PARAMS, &mut root_rng(3, i)).unwrap();
            let b = sample_root(&seen, until, 2, PARAMS, &mut root_rng(3, i)).unwrap();
            assert_eq!(a.particles.len(), b.particles.len());
            for (x, y) in a.particles.iter().zip(&b.particles) {
                assert_eq!(x.board.haipai, y.board.haipai);
                assert_eq!(x.board.yama, y.board.yama);
                assert_eq!(x.board.ura_indicators, y.board.ura_indicators);
            }
        }
    }

    /// Weighted mean and squared standard error of the count of each tile
    /// kind in each seat's concealed hand at the root. `n_eff` is the
    /// sample size the error assumes: the ESS for importance weights.
    fn hand_stats(states: &[(BoardState, f64)]) -> [[(f64, f64); 34]; 4] {
        let lw: Vec<f64> = states.iter().map(|&(_, w)| w).collect();
        let n_eff = ess(&lw);
        let max = lw.iter().copied().fold(f64::MIN, f64::max);
        let total: f64 = lw.iter().map(|w| (w - max).exp()).sum();
        let mut out = [[(0., 0.); 34]; 4];
        for (s, row) in out.iter_mut().enumerate() {
            for (k, cell) in row.iter_mut().enumerate() {
                let (mut m, mut m2) = (0., 0.);
                for (state, w) in states {
                    let w = (w - max).exp() / total;
                    let x = f64::from(state.agent_context().player_states[s].tehai()[k]);
                    m += w * x;
                    m2 += w * x * x;
                }
                *cell = (m, (m2 - m * m).max(0.) / n_eff);
            }
        }
        out
    }

    fn ess(log_w: &[f64]) -> f64 {
        let max = log_w.iter().copied().fold(f64::MIN, f64::max);
        let w: Vec<f64> = log_w.iter().map(|x| (x - max).exp()).collect();
        let (s, s2): (f64, f64) = (w.iter().sum(), w.iter().map(|x| x * x).sum());
        s * s / s2
    }

    /// The chain's stationary distribution is the uniform prior over
    /// consistent histories. Early in a kyoku the uniform proposal's
    /// importance weights are reliable, and the two must agree on what each
    /// opponent holds: every difference within its standard error.
    #[test]
    fn the_chain_agrees_with_importance_sampling() {
        let (mut z2, mut n, mut worst) = (0., 0, 0_f64);
        let mut roots_used = 0;
        for (i, (_, events, until)) in true_roots(1).into_iter().enumerate() {
            let turn = events[..until]
                .iter()
                .filter(|e| matches!(e, Event::Tsumo { .. }))
                .count();
            if !(4..=6).contains(&turn) || i % 3 != 0 || roots_used == 6 {
                continue;
            }
            let seat = super::super::belief::root_seat(&events, until).unwrap();
            let seen: Vec<_> = events.iter().map(|e| visible(e, seat)).collect();
            let layout = Layout::new(&seen[..root_stop(&seen, until)], seat).unwrap();
            let replay_ok = |b: Board| replay_seen_by(b, &seen, until, seat).ok();

            let mut rng = root_rng(11, i);
            let sis: Vec<_> = (0..12_000)
                .filter_map(|_| layout.sample(&mut rng).ok())
                .filter_map(|(b, lw)| replay_ok(b).map(|st| (st, lw)))
                .collect();
            let lw: Vec<f64> = sis.iter().map(|&(_, w)| w).collect();
            if ess(&lw) < 400. {
                continue;
            }
            let chain: Vec<_> = layout
                .chain(3000, 2000, 200, 20, &mut rng)
                .unwrap()
                .into_iter()
                .filter_map(|b| replay_ok(b).map(|st| (st, 0.)))
                .collect();
            let (a, b) = (hand_stats(&sis), hand_stats(&chain));
            for s in (0..4).filter(|&s| s != seat as usize) {
                for k in 0..34 {
                    let ((ma, va), (mb, vb)) = (a[s][k], b[s][k]);
                    if va + vb > 0. {
                        let z = (ma - mb) / (va + vb).sqrt();
                        z2 += z * z;
                        n += 1;
                        worst = worst.max(z.abs());
                    }
                }
            }
            roots_used += 1;
            eprintln!("root {i}: turn {turn}, ESS {:.0}", ess(&lw));
        }
        let mean_z2 = z2 / f64::from(n);
        eprintln!("{roots_used} roots, {n} comparisons: mean z² {mean_z2:.2}, max |z| {worst:.2}");
        assert!(roots_used >= 3, "{roots_used}");
        assert!(
            mean_z2 < 1.5 && worst < 5.,
            "mean z² {mean_z2}, max |z| {worst}"
        );
    }
}
