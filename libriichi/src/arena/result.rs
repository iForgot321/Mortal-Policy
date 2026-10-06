use crate::mjai::{Event, EventExt};
use crate::rankings::Rankings;

use anyhow::Result;
use serde::Serialize;
use serde_json as json;

#[derive(Debug, Clone)]
pub struct KyokuResult {
    pub kyoku: u8,
    // pub honba: u8,
    pub can_renchan: bool,
    pub has_hora: bool,
    pub has_abortive_ryukyoku: bool,
    pub kyotaku_left: u8,
    pub scores: [i32; 4],
}

#[derive(Debug, Clone, Default)]
pub struct GameResult {
    pub names: [String; 4],
    pub scores: [i32; 4],
    pub seed: (u64, u64),
    pub game_log: Vec<Vec<EventExt>>,
    /// Empty unless the game was run with `record_behaviour`.
    pub behaviour: Vec<BehaviourRecord>,
}

/// One reaction of one seat, with the behaviour probability of what it
/// emitted (05 §3.4, D46). Passes and overruled claims are included, although
/// the mjai log never shows them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BehaviourRecord {
    /// Index of the event decided on in the dumped log, where `start_game` is
    /// 0. Same as `Gameplay.event_idx`.
    pub event_idx: u32,
    pub actor: u8,
    /// The second decision of a kan: which tile.
    pub kan_select: bool,
    /// Action index, or the tile id for a kan select.
    pub label: u8,
    /// `None` when the emitted action was not drawn from the policy.
    pub prob: Option<f32>,
    /// Whether the board executed the reaction. A pass counts as applied when
    /// nobody claimed. False for a claim overruled by a higher-priority one
    /// and for every reaction voided by an abortive draw.
    pub applied: bool,
}

/// The sidecar layout: one JSON object of equal-length columns.
#[derive(Serialize, Default)]
struct BehaviourColumns {
    event_idx: Vec<u32>,
    actor: Vec<u8>,
    kan_select: Vec<bool>,
    label: Vec<u8>,
    prob: Vec<Option<f32>>,
    applied: Vec<bool>,
}

impl GameResult {
    #[inline]
    pub fn rankings(&self) -> Rankings {
        Rankings::new(self.scores)
    }

    pub fn dump_json_log(&self) -> Result<String> {
        let mut v = vec![];

        let start_game = Event::StartGame {
            names: self.names.clone(),
            seed: Some(self.seed),
        };
        json::to_writer(&mut v, &start_game)?;
        v.push(b'\n');

        for ev in self.game_log.iter().flatten() {
            json::to_writer(&mut v, ev)?;
            v.push(b'\n');
        }

        json::to_writer(&mut v, &Event::EndGame)?;
        v.push(b'\n');

        Ok(String::from_utf8(v)?)
    }

    /// The behaviour sidecar of this game (D46).
    pub fn dump_behaviour_json(&self) -> Result<String> {
        let mut cols = BehaviourColumns::default();
        for r in &self.behaviour {
            cols.event_idx.push(r.event_idx);
            cols.actor.push(r.actor);
            cols.kan_select.push(r.kan_select);
            cols.label.push(r.label);
            cols.prob.push(r.prob);
            cols.applied.push(r.applied);
        }
        Ok(json::to_string(&cols)?)
    }
}
