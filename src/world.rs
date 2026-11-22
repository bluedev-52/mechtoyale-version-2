use crate::map::Arena;
use crate::math::{V2, V3};
use crate::protocol::{size_scale, Config, Entity, Obs};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Unknown,
    Squad,
    Ffa,
}

pub struct Track {
    pub entity: Entity,
    pub last_seen: f64,
    pub charge_s: f64,
    observed: bool,
}
impl Track {
    pub fn predicted(&self, now: f64) -> V3 {
        self.entity.pos + self.entity.vel * (now - self.last_seen).clamp(0.0, 1.5)
    }
    pub fn observed(&self) -> bool {
        self.observed
    }
    pub fn mech(&self) -> bool {
        self.entity.kind == "mech"
    }
    pub fn hp_ratio(&self) -> f64 {
        self.entity
            .hp
            .zip(self.entity.max_hp)
            .map_or(1.0, |(hp, max)| hp / max.max(1.0))
    }
    pub fn has(&self, kind: &str) -> bool {
        self.entity.weapons.iter().any(|w| w == kind)
    }
}

struct Heard {
    pos: V2,
    t: f64,
    risk: f64,
    width: f64,
}

pub struct World {
    pub tracks: BTreeMap<u32, Track>,
    pub mode: Mode,
    pub ffa_confidence: f64,
    pub regen_ready_in: f64,
    pub bearing: Option<(f64, f64)>,
    last_combat_t: f64,
    own_kills: u32,
    last_alive: Option<u32>,
    heard: VecDeque<Heard>,
}
impl Default for World {
    fn default() -> Self {
        Self {
            tracks: BTreeMap::new(),
            mode: Mode::Unknown,
            ffa_confidence: 0.5,
            regen_ready_in: 0.0,
            bearing: None,
            last_combat_t: f64::NEG_INFINITY,
            own_kills: 0,
            last_alive: None,
            heard: VecDeque::new(),
        }
    }
}

impl World {
    pub fn update(&mut self, obs: &Obs, cfg: &Config, slot: u32, map: &Arena) {
        let kills = obs
            .events
            .iter()
            .filter(|event| event.kind == "kill")
            .count() as u32;
        self.own_kills += kills;
        for track in self.tracks.values_mut() {
            track.observed = false;
        }
        for entity in &obs.mechs {
            let entry = self.tracks.entry(entity.id).or_insert_with(|| Track {
                entity: entity.clone(),
                last_seen: obs.t,
                charge_s: 0.0,
                observed: true,
            });
            let dt = obs.t - entry.last_seen;
            let residual = entity.pos - entry.predicted(obs.t);
            let velocity = if dt > 0.0 && dt < 1.0 {
                let observed_velocity = entry.entity.vel + residual * (0.35 / dt);
                entity.vel * 0.65 + observed_velocity * 0.35
            } else {
                entity.vel
            };
            let position = if (0.0..0.4).contains(&dt) && residual.norm() < 8.0 {
                entry.predicted(obs.t) + residual * 0.65
            } else {
                entity.pos
            };
            entry.charge_s = if entity.charging && entry.entity.charging {
                entry.charge_s + dt
            } else {
                0.0
            };
            entry.entity.clone_from(entity);
            entry.entity.pos = position;
            entry.entity.vel = velocity;
            entry.last_seen = obs.t;
            entry.observed = true;
        }

        let mut inter_enemy_fire = false;
        for beam in &obs.beams {
            let Some(shooter) = beam.by.filter(|&id| id != slot) else {
                continue;
            };
            let Some(&origin) = beam.path.first() else {
                continue;
            };
            let direction = beam
                .path
                .get(1)
                .map_or(V3::default(), |p| (*p - origin).unit());
            for &point in beam.path.iter().skip(1) {
                let own_distance = (point.xy() - obs.own.pos.xy()).norm();
                if own_distance <= obs.own.size(cfg, "radius") + 4.0 {
                    continue;
                }
                inter_enemy_fire |= self.tracks.values().any(|target| {
                    target.mech()
                        && target.entity.id != shooter
                        && (target
                            .predicted(obs.t)
                            .up(target_height(cfg, &target.entity) * 0.5)
                            - point)
                            .norm()
                            < target_radius(cfg, &target.entity) + 0.75
                });
            }
            let entry = self.tracks.entry(shooter).or_insert_with(|| Track {
                entity: Entity {
                    id: shooter,
                    kind: "mech".into(),
                    pos: origin,
                    vel: V3::default(),
                    aim: direction.aim(),
                    hp: None,
                    max_hp: None,
                    cubes: 0,
                    weapons: vec![],
                    charging: false,
                },
                last_seen: f64::NEG_INFINITY,
                observed: false,
                charge_s: 0.0,
            });
            if !entry.has(&beam.kind) {
                entry.entity.weapons.push(beam.kind.clone());
            }
            if !entry.observed && obs.t - beam.age >= entry.last_seen {
                let scale = size_scale(cfg, entry.entity.cubes);
                let offset = direction * (cfg.get("mech", "radius") * scale + 0.5);
                let feet = (origin - offset).up(-cfg.get("mech", "muzzle_height") * scale);
                entry.entity.pos = feet.up((map.floor_below(feet) - feet.2).max(0.0));
                entry.last_seen = obs.t - beam.age;
                entry.entity.charging = false;
            }
        }

        for event in &obs.events {
            match event.kind.as_str() {
                "damaged" => {
                    self.last_combat_t = obs.t;
                    if let Some(yaw) = event.bearing {
                        self.bearing = Some((yaw, obs.t));
                    }
                    if let (Some(track), Some(weapon)) = (
                        event.by.and_then(|id| self.tracks.get_mut(&id)),
                        event.weapon.as_ref(),
                    ) {
                        if !track.has(weapon) {
                            track.entity.weapons.push(weapon.clone());
                        }
                    }
                }
                "hit" => self.last_combat_t = obs.t,
                "kill" => {
                    if let Some(id) = event.target {
                        self.tracks.remove(&id);
                    }
                }
                _ => {}
            }
        }
        if !obs.own.in_zone {
            self.last_combat_t = obs.t;
        }
        self.regen_ready_in =
            (obs.own.stat(cfg, "regen_delay") - (obs.t - self.last_combat_t)).max(0.0);
        self.tracks
            .retain(|_, track| obs.t - track.last_seen <= cfg.get("vision", "memory_s") + 1.5);

        for sound in &obs.sounds {
            if matches!(sound.kind.as_str(), "crate" | "drop" | "death") {
                continue;
            }
            let Some(yaw) = sound.bearing else { continue };
            let near = cfg.get("sound", "near");
            let mid = cfg.get("sound", "mid");
            let range = cfg.0["sound"]["range"][&sound.kind]
                .as_f64()
                .unwrap_or(mid * 2.0);
            let (low, high) = match sound.dist.as_str() {
                "near" => (0.0, near),
                "mid" => (near, mid),
                "far" => (mid, range.max(mid)),
                _ => continue,
            };
            let distance = (low + high.min(range).max(low)) * 0.5;
            let sigma_yaw =
                (cfg.get("sound", "bearing_noise_deg") * obs.own.stat(cfg, "hearing")).to_radians();
            let width = (distance * sigma_yaw).hypot((high - low) * 0.25).max(8.0);
            let risk = if sound.kind == "npc_gun" { 0.4 } else { 1.0 };
            self.heard.push_back(Heard {
                pos: obs.own.pos.xy() + V2::at(yaw) * distance,
                t: obs.t - sound.age,
                risk,
                width,
            });
        }
        self.heard.retain(|sound| obs.t - sound.t < 12.0);
        self.heard.drain(..self.heard.len().saturating_sub(96));

        let early_dps = cfg.0["zone"]["phases"][1]["dps"]
            .as_f64()
            .unwrap_or(obs.zone.dps);
        if inter_enemy_fire || self.mechanical_crossfire(obs, cfg) {
            self.ffa_confidence = 1.0;
        } else if self
            .last_alive
            .is_some_and(|last| last.saturating_sub(obs.alive) > kills)
            && obs.zone.dps <= early_dps
        {
            self.ffa_confidence += (1.0 - self.ffa_confidence) * 0.8;
        }
        let starting_alive = cfg.get("match", "mechs") as u32;
        let converging = self
            .tracks
            .values()
            .filter(|track| {
                track.mech()
                    && track.observed
                    && track
                        .entity
                        .vel
                        .xy()
                        .dot((obs.own.pos.xy() - track.entity.pos.xy()).unit())
                        > 2.0
            })
            .count();
        if self.ffa_confidence < 0.8
            && obs.t > 35.0
            && obs.alive == starting_alive.saturating_sub(self.own_kills)
            && converging >= 2
        {
            self.ffa_confidence = 0.05;
        }
        self.mode = if self.ffa_confidence >= 0.8 {
            Mode::Ffa
        } else if self.ffa_confidence <= 0.2 {
            Mode::Squad
        } else {
            Mode::Unknown
        };
        self.last_alive = Some(obs.alive);
    }

    fn mechanical_crossfire(&self, obs: &Obs, cfg: &Config) -> bool {
        obs.projectiles
            .iter()
            .filter(|p| p.kind == "gun")
            .any(|shot| {
                let direction = shot.vel.unit();
                let muzzle_distance = |track: &Track| {
                    track
                        .entity
                        .pos
                        .up(cfg.get("mech", "muzzle_height") * size_scale(cfg, track.entity.cubes))
                };
                let self_offset = obs.own.pos.up(obs.own.size(cfg, "height") * 0.5) - shot.pos;
                if (self_offset - direction * self_offset.dot(direction)).norm()
                    < obs.own.size(cfg, "height") + 5.0
                {
                    return false;
                }
                let mut shooter = false;
                let mut target = false;
                for track in self.tracks.values().filter(|t| t.mech() && t.observed) {
                    let offset = muzzle_distance(track) - shot.pos;
                    let along = offset.dot(direction);
                    if along < 0.0
                        && along > -cfg.weapon("gun", "range")
                        && (offset - direction * along).norm() < 6.0
                        && V3::dir(track.entity.aim).dot(direction) > 0.985
                    {
                        shooter = true;
                    }
                    if along > 8.0
                        && along < cfg.weapon("gun", "range") * 0.7
                        && (offset - direction * along).norm() < target_radius(cfg, &track.entity)
                    {
                        target = true;
                    }
                }
                shooter && target
            })
    }

    pub fn heat(&self, pos: V2, now: f64) -> f64 {
        self.heard
            .iter()
            .map(|sound| {
                let distance = (sound.pos - pos).norm() / sound.width;
                sound.risk * (-(now - sound.t).max(0.0) / 4.0 - 0.5 * distance * distance).exp()
            })
            .sum()
    }

    pub fn danger(&self, pos: V2, now: f64) -> f64 {
        self.tracks
            .values()
            .filter(|t| t.mech())
            .map(|track| {
                let d = (track.predicted(now).xy() - pos).norm();
                let proximity = (1.0 - d / 160.0).max(0.0);
                proximity * proximity * (1.0 + f64::from(track.entity.cubes) * 0.05)
            })
            .sum::<f64>()
            + self.heat(pos, now).min(4.0) * 0.18
    }
}

pub fn target_height(cfg: &Config, entity: &Entity) -> f64 {
    if entity.kind == "npc" {
        cfg.get("npc", "height")
    } else {
        cfg.get("mech", "height") * size_scale(cfg, entity.cubes)
    }
}

pub fn target_radius(cfg: &Config, entity: &Entity) -> f64 {
    if entity.kind == "npc" {
        cfg.get("npc", "radius")
    } else {
        cfg.get("mech", "radius") * size_scale(cfg, entity.cubes)
    }
}
