use super::{Agent, BatchifiedAgent, InvisibleState};
use crate::algo::shanten;
use crate::mjai::{Event, EventExt};
use crate::must_tile;
use crate::state::PlayerState;
use crate::tile::Tile;
use std::cmp::Reverse;

use anyhow::{Context, Result};

/// `ShantenGreedy` is the frozen rule-based floor of the evaluation lab. It
/// discards to minimize shanten, then to maximize ukeire (the number of unseen
/// tiles that would lower shanten), never calls or kans, declares riichi
/// whenever it is legal, and always takes a win.
///
/// Changing its behavior invalidates every comparison against it.
pub struct ShantenGreedy(pub u8);

impl ShantenGreedy {
    pub fn new_batched(player_ids: &[u8]) -> Result<BatchifiedAgent<Self>> {
        BatchifiedAgent::new(|id| Ok(Self(id)), player_ids)
    }

    fn choose_discard(state: &PlayerState) -> Result<Tile> {
        let tehai = state.tehai();
        let tiles_seen = state.tiles_seen();
        let len_div3 = state.tehai_len_div3();

        state
            .discard_candidates_aka()
            .iter()
            .enumerate()
            .filter(|&(_, &legal)| legal)
            .map(|(tid, _)| {
                let pai = must_tile!(tid);
                let mut hand = tehai;
                hand[pai.deaka().as_usize()] -= 1;
                let shanten = shanten::calc_all(&hand, len_div3);
                let ukeire = Self::ukeire(&mut hand, len_div3, shanten, &tiles_seen);
                (
                    (shanten, Reverse(ukeire), Self::discard_priority(pai), tid),
                    pai,
                )
            })
            .min_by_key(|&(key, _)| key)
            .map(|(_, pai)| pai)
            .context("no legal discard")
    }

    /// Unseen copies of every tile kind that lowers the shanten of a 3n+1
    /// hand. `tiles_seen` already includes the player's own hand.
    fn ukeire(hand: &mut [u8; 34], len_div3: u8, shanten: i8, tiles_seen: &[u8; 34]) -> u8 {
        let mut ukeire = 0;
        for tid in 0..34 {
            let left = 4 - tiles_seen[tid];
            if left == 0 {
                continue;
            }
            hand[tid] += 1;
            if shanten::calc_all(hand, len_div3) < shanten {
                ukeire += left;
            }
            hand[tid] -= 1;
        }
        ukeire
    }

    /// Tie-break among discards of equal shanten and ukeire: honors first,
    /// then from the outside of each suit inwards, keeping aka fives.
    fn discard_priority(pai: Tile) -> u8 {
        let kind = pai.deaka().as_u8();
        let base = if kind >= 27 {
            0
        } else {
            let num = kind % 9 + 1;
            num.min(10 - num)
        };
        if pai.is_aka() { base + 10 } else { base }
    }
}

impl Agent for ShantenGreedy {
    fn name(&self) -> String {
        "shanten-greedy".to_owned()
    }

    fn react(
        &mut self,
        _: &[EventExt],
        state: &PlayerState,
        _: Option<InvisibleState>,
    ) -> Result<EventExt> {
        let actor = self.0;
        let cans = state.last_cans();

        let ev = if cans.can_agari() {
            Event::Hora {
                actor,
                target: cans.target_actor,
                deltas: None,
                ura_markers: None,
            }
        } else if cans.can_riichi {
            Event::Reach { actor }
        } else if cans.can_discard {
            let pai = Self::choose_discard(state)?;
            let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
            Event::Dahai {
                actor,
                pai,
                tsumogiri,
            }
        } else {
            Event::None
        };
        Ok(EventExt::no_meta(ev))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::agent::{BatchAgent, Tsumogiri};
    use crate::arena::game::{BatchGame, Index};

    #[test]
    fn beats_tsumogiri() {
        // One shanten-greedy seat rotating through all four seats against
        // three tsumogiri bots, which never win. The board rejects any illegal
        // reaction, so finishing the games also checks legality.
        let sets = 16;
        let games = sets * 4;
        let greedy_ids: Vec<u8> = (0..4).cycle().take(games).collect();
        let tsumogiri_ids: Vec<u8> = (0..games)
            .flat_map(|i| (0..4).filter(move |&s| s != (i % 4) as u8))
            .collect();
        let mut agents = [
            Box::new(ShantenGreedy::new_batched(&greedy_ids).unwrap()) as Box<dyn BatchAgent>,
            Box::new(Tsumogiri::new_batched(&tsumogiri_ids).unwrap()) as _,
        ];

        let mut tsumogiri_idx = 0;
        let indexes: Vec<[Index; 4]> = (0..games)
            .map(|i| {
                std::array::from_fn(|seat| {
                    if seat == i % 4 {
                        Index {
                            agent_idx: 0,
                            player_id_idx: i,
                        }
                    } else {
                        tsumogiri_idx += 1;
                        Index {
                            agent_idx: 1,
                            player_id_idx: tsumogiri_idx - 1,
                        }
                    }
                })
            })
            .collect();
        let seeds: Vec<_> = (0..sets as u64)
            .flat_map(|seed| std::iter::repeat_n((seed, 0x5eed), 4))
            .collect();

        let results = BatchGame::tenhou_hanchan(true)
            .run(&mut agents, &indexes, &seeds)
            .unwrap();
        let firsts = results
            .iter()
            .enumerate()
            .filter(|(i, r)| r.rankings().rank_by_player[i % 4] == 0)
            .count();
        assert!(firsts * 10 >= games * 9, "only {firsts}/{games} firsts");
    }
}
