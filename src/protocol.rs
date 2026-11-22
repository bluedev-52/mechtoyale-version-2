use crate::math::{V2, V3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    Hello(Hello),
    Obs(Box<Obs>),
    End,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct Hello {
    pub protocol: String,
    pub slot: u32,
    pub seed: u64,
    pub map: MapRef,
    pub config: Config,
}
#[derive(Debug, Deserialize)]
pub struct MapRef {
    pub dir: String,
}
#[derive(Debug, Deserialize)]
pub struct Config(pub Value);
impl Config {
    pub fn get(&self, section: &str, key: &str) -> f64 {
        self.0[section][key].as_f64().unwrap_or_default()
    }
    pub fn weapon(&self, kind: &str, key: &str) -> f64 {
        self.0["weapons"][kind][key].as_f64().unwrap_or_default()
    }
    pub fn dt(&self) -> f64 {
        self.get("match", "decide_every") / self.get("match", "tick_hz").max(1.0)
    }
}

#[derive(Debug, Deserialize)]
pub struct Obs {
    pub tick: u32,
    pub t: f64,
    pub alive: u32,
    #[serde(rename = "self")]
    pub own: Own,
    #[serde(default)]
    pub mechs: Vec<Entity>,
    #[serde(default)]
    pub projectiles: Vec<Projectile>,
    #[serde(default)]
    pub crates: Vec<Loot>,
    #[serde(default)]
    pub cubes: Vec<Loot>,
    #[serde(default)]
    pub pickups: Vec<Loot>,
    #[serde(default)]
    pub beams: Vec<Beam>,
    #[serde(default)]
    pub sounds: Vec<Sound>,
    #[serde(default)]
    pub events: Vec<Event>,
    #[serde(default)]
    pub drops: Vec<Drop>,
    pub zone: Zone,
    pub offer: Option<Offer>,
}

#[derive(Debug, Deserialize)]
pub struct Own {
    pub pos: V3,
    pub vel: V3,
    pub aim: [f64; 2],
    pub hp: f64,
    pub max_hp: f64,
    pub energy: f64,
    pub max_energy: f64,
    pub grounded: bool,
    pub cubes: u32,
    pub in_zone: bool,
    #[serde(default)]
    pub weapons: Vec<Option<Weapon>>,
    #[serde(default)]
    pub upgrades: Vec<String>,
    #[serde(default)]
    pub stats: Value,
}
impl Own {
    pub fn stat(&self, config: &Config, key: &str) -> f64 {
        self.stats[key]
            .as_f64()
            .unwrap_or_else(|| config.get("mech", key))
    }
    pub fn weapon(&self, slot: usize) -> Option<&Weapon> {
        self.weapons.get(slot)?.as_ref()
    }
    pub fn has(&self, kind: &str) -> bool {
        self.weapons.iter().flatten().any(|w| w.kind == kind)
    }
    pub fn size(&self, config: &Config, key: &str) -> f64 {
        self.stats[key]
            .as_f64()
            .unwrap_or_else(|| config.get("mech", key) * size_scale(config, self.cubes))
    }
}

pub fn size_scale(config: &Config, cubes: u32) -> f64 {
    1.0 + config.get("cubes", "size_per_cube") * f64::from(cubes)
}

#[derive(Debug, Deserialize)]
pub struct Weapon {
    pub kind: String,
    #[serde(default)]
    pub cooldown: f64,
    pub ammo: Option<u32>,
    pub reload: Option<f64>,
    pub charge: Option<f64>,
    #[serde(default)]
    pub stats: Value,
}
impl Weapon {
    pub fn stat(&self, config: &Config, key: &str) -> f64 {
        self.stats[key]
            .as_f64()
            .unwrap_or_else(|| config.weapon(&self.kind, key))
    }
    pub fn charge(&self) -> f64 {
        self.charge.unwrap_or(0.0)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Entity {
    pub id: u32,
    pub kind: String,
    pub pos: V3,
    #[serde(default)]
    pub vel: V3,
    #[serde(default)]
    pub aim: [f64; 2],
    pub hp: Option<f64>,
    pub max_hp: Option<f64>,
    #[serde(default)]
    pub cubes: u32,
    #[serde(default)]
    pub weapons: Vec<String>,
    #[serde(default)]
    pub charging: bool,
}

#[derive(Debug, Deserialize)]
pub struct Projectile {
    pub kind: String,
    pub pos: V3,
    pub vel: V3,
}
#[derive(Debug, Deserialize)]
pub struct Loot {
    pub id: u32,
    pub pos: V3,
    pub hp: Option<f64>,
    #[serde(default)]
    pub kind: String,
}
#[derive(Debug, Deserialize)]
pub struct Beam {
    pub by: Option<u32>,
    pub kind: String,
    pub path: Vec<V3>,
    pub age: f64,
}
#[derive(Debug, Deserialize)]
pub struct Sound {
    pub kind: String,
    pub bearing: Option<f64>,
    pub dist: String,
    pub age: f64,
}
#[derive(Debug, Deserialize)]
pub struct Event {
    #[serde(rename = "type")]
    pub kind: String,
    pub bearing: Option<f64>,
    pub by: Option<u32>,
    pub target: Option<u32>,
    pub weapon: Option<String>,
}
#[derive(Debug, Deserialize)]
pub struct Drop {
    pub pos: V3,
    pub radius: f64,
    pub lands_in: f64,
    pub capture_s: f64,
    pub progress: f64,
    pub holder: Option<u32>,
}
#[derive(Debug, Deserialize)]
pub struct Offer {
    pub choices: Vec<Choice>,
}
#[derive(Debug, Deserialize)]
pub struct Choice {
    pub id: String,
}
#[derive(Debug, Deserialize)]
pub struct Zone {
    pub center: V2,
    pub radius: f64,
    pub next_center: V2,
    pub next_radius: f64,
    pub shrink_start: f64,
    pub shrink_end: f64,
    pub dps: f64,
}
impl Zone {
    pub fn future(&self, now: f64, ahead: f64) -> (V2, f64) {
        let remaining = self.shrink_end - now.max(self.shrink_start);
        let progress = if remaining > 0.0 {
            ((now + ahead - now.max(self.shrink_start)) / remaining).clamp(0.0, 1.0)
        } else {
            1.0
        };
        (
            self.center + (self.next_center - self.center) * progress,
            self.radius + (self.next_radius - self.radius) * progress,
        )
    }
}

#[derive(Serialize)]
pub struct Action {
    pub tick: u32,
    #[serde(rename = "move")]
    pub movement: [f64; 2],
    pub aim: [f64; 2],
    pub fire: [bool; 2],
    pub reload: bool,
    pub jump: bool,
    pub boost: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pick: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub take: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<Value>,
}
impl Action {
    pub fn new(obs: &Obs) -> Self {
        Self {
            tick: obs.tick,
            movement: [0.0; 2],
            aim: obs.own.aim,
            fire: [false; 2],
            reload: false,
            jump: false,
            boost: false,
            pick: None,
            take: None,
            drop: None,
            debug: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tolerates_protocol_extensions_and_nullable_fields() {
        let msg = json!({
            "type": "obs", "tick": 42, "t": 2.1, "alive": 16, "extension": {},
            "self": {
                "pos": [0, 0, 0], "vel": [0, 0, 0], "aim": [0, 0], "hp": 1, "max_hp": 1,
                "energy": 0, "max_energy": 1, "grounded": true, "cubes": 0, "in_zone": true,
                "weapons": [{"kind": "gun", "cooldown": 0, "reload": null, "extra": true}, null]
            },
            "mechs": [{"id": 1, "kind": "mech", "pos": [10, 0, 0], "hp": null}],
            "events": [{"type": "damaged", "by": null, "bearing": null, "rear": null}],
            "drops": [{"pos": [50,0,0], "radius": 8, "holder": null, "lands_in": 0,
                       "capture_s": 4, "progress": 0}],
            "zone": {"center": [0,0], "radius": 50, "next_center": [4,6], "next_radius": 25,
                     "shrink_start": 2, "shrink_end": 4, "dps": 20},
            "offer": null
        });
        let Message::Obs(obs) = serde_json::from_value(msg).unwrap() else {
            panic!("expected obs");
        };
        assert!(obs.own.weapon(0).is_some());
        assert!(obs.own.weapon(1).is_none());
        assert!(obs.mechs[0].hp.is_none());
        assert!(obs.events[0].bearing.is_none());
        assert!(obs.drops[0].holder.is_none());
        assert!(obs.offer.is_none());
        assert!(obs.sounds.is_empty());
        assert_eq!(serde_json::to_value(Action::new(&obs)).unwrap()["tick"], 42);
    }

    #[test]
    fn interpolates_from_the_current_circle_mid_shrink() {
        let zone = Zone {
            center: V2(10.0, 0.0),
            radius: 80.0,
            next_center: V2(30.0, 0.0),
            next_radius: 40.0,
            shrink_start: 20.0,
            shrink_end: 40.0,
            dps: 20.0,
        };
        assert_eq!(zone.future(30.0, 5.0), (V2(20.0, 0.0), 60.0));
        assert_eq!(zone.future(30.0, 10.0), (V2(30.0, 0.0), 40.0));
    }
}
