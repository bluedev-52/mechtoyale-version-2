use crate::aim;
use crate::ballistics::{self, Shot, Target};
use crate::map::Arena;
use crate::math::{wrap_pi, SplitMix64, V2, V3};
use crate::nav::Navigator;
use crate::protocol::{Action, Config, Hello, Loot, Obs, Own};
use crate::world::{target_height, target_radius, Mode, Track, World};
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    Zone,
    Regen,
    Drop,
    Roof,
    Farm,
    Engage,
}

#[derive(Clone, Copy)]
struct Plan {
    state: State,
    goal: V3,
    score: f64,
    target: Option<Target>,
    loot: Option<(u32, Option<usize>)>,
}
impl Plan {
    fn new(state: State, goal: V3, score: f64) -> Self {
        Self {
            state,
            goal,
            score,
            target: None,
            loot: None,
        }
    }
}

pub struct Bot {
    cfg: Config,
    slot: u32,
    map: Arena,
    world: World,
    nav: Navigator,
    state: State,
    scout_yaw: f64,
    laser_cover: Option<V3>,
    return_to_cover_until: f64,
    debug_remaining: usize,
}

struct Threat<'a> {
    track: &'a Track,
    distance: f64,
    yaw: f64,
    clear: bool,
    distracted: bool,
}

impl Bot {
    pub fn new(hello: Hello) -> std::io::Result<Self> {
        let map = Arena::load(Path::new(&hello.map.dir))?;
        let mut rng = SplitMix64(hello.seed);
        let scout_yaw = (rng.next() >> 11) as f64 / (1_u64 << 53) as f64 * std::f64::consts::TAU;
        let debug_remaining = if std::env::var("DEBUG_DRAW").as_deref() == Ok("1") {
            hello.config.get("debug", "max_match_bytes") as usize
        } else {
            0
        };
        Ok(Self {
            cfg: hello.config,
            slot: hello.slot,
            map,
            world: World::default(),
            nav: Navigator::default(),
            state: State::Farm,
            scout_yaw,
            laser_cover: None,
            return_to_cover_until: 0.0,
            debug_remaining,
        })
    }

    pub fn decide(&mut self, obs: &Obs) -> Action {
        let start = Instant::now();
        let budget = self.cfg.get("bot", "tick_budget_ms");
        let deadline = start + Duration::from_secs_f64((budget * 0.4).clamp(1.0, 18.0) / 1000.0);
        self.world.update(obs, &self.cfg, self.slot, &self.map);
        let threats = Self::threats(obs, &self.cfg, &self.world, &self.map);
        let plan = self.plan(obs, &threats);
        let defensive = matches!(plan.state, State::Zone | State::Regen);
        let mut action = Action::new(obs);
        action.pick = pick_upgrade(obs, &self.cfg);
        (action.take, action.drop) = self.take(obs, &threats, plan.loot);
        let shots = if plan.state == State::Regen {
            Vec::new()
        } else {
            self.shots(obs, &threats, plan.target, deadline)
        };
        let shot = shots.iter().max_by(|a, b| a.score.total_cmp(&b.score));
        let mut goal = plan.goal;
        let mut peeking = false;
        if !defensive {
            if let Some(cover) = self
                .laser_cover
                .filter(|_| obs.t < self.return_to_cover_until)
            {
                goal = cover;
            }
            if let Some((shot, laser)) = shot.and_then(|s| {
                obs.own
                    .weapon(s.slot)
                    .filter(|w| w.kind == "laser")
                    .map(|w| (s, w))
            }) {
                if !shot.pull {
                    if let Some(cover) = self.laser_cover {
                        goal = cover;
                        self.return_to_cover_until = obs.t + 1.0;
                    }
                } else if laser.charge() <= 0.0 {
                    self.laser_cover = Some(obs.own.pos);
                } else if laser.charge() >= laser.stat(&self.cfg, "min_charge") {
                    let muzzle = obs.own.pos.up(obs.own.size(&self.cfg, "muzzle_height"));
                    if !self.map.clear(muzzle, shot.target) {
                        if let Some(peek) = self.peek(obs, shot.target) {
                            goal = peek;
                            peeking = true;
                        }
                    }
                }
            } else if plan.state == State::Engage
                && shot.is_none()
                && !obs.own.has("gun")
                && !obs.own.has("mortar")
                && obs.t >= self.return_to_cover_until
            {
                // The laser needs cover to start charging. If there is none nearby,
                // the engagement goal keeps closing toward lightning range.
                let laser = obs.own.weapons.iter().flatten().find(|w| w.kind == "laser");
                if let (Some(target), Some(laser)) = (plan.target, laser) {
                    let (center, radius) = obs.zone.future(obs.t, 10.0);
                    let muzzle_height = obs.own.size(&self.cfg, "muzzle_height");
                    let mid_height = obs.own.size(&self.cfg, "height") * 0.5;
                    let barrel = obs.own.size(&self.cfg, "radius") + 0.5;
                    let range = laser.stat(&self.cfg, "range") + barrel;
                    let can_charge = |pos: V3| {
                        let muzzle = pos.up(muzzle_height);
                        (target.pos - muzzle).norm() <= range
                            && ballistics::laser_has_cover(
                                &self.map,
                                muzzle,
                                barrel,
                                pos.up(mid_height),
                                target.pos,
                            )
                    };
                    if let Some(cover) =
                        self.cover(obs, &threats, center, radius * 0.97, false, can_charge)
                    {
                        if (cover - obs.own.pos).norm() > 2.0 {
                            goal = cover;
                        }
                    }
                }
            }
        }
        let avoid = !peeking && (defensive || self.world.mode == Mode::Ffa);
        let mut movement = self.nav.steer(
            obs,
            &self.cfg,
            &self.map,
            &self.world,
            goal,
            defensive || plan.state == State::Engage,
            avoid,
            deadline,
        );
        let stealth = self.world.mode == Mode::Ffa
            && obs.alive <= 6
            && obs.own.in_zone
            && !defensive
            && threats.iter().all(|threat| threat.distance >= 60.0)
            && shot.is_none();
        if stealth && !movement.jump && obs.own.grounded {
            let limit = (self.cfg.get("sound", "step_speed") - 0.3).max(0.0)
                / obs.own.stat(&self.cfg, "walk_speed").max(1.0);
            movement.direction = movement.direction.unit() * movement.direction.norm().min(limit);
            movement.boost = false;
        }
        action.movement = [movement.direction.0, movement.direction.1];
        action.jump = movement.jump;
        action.boost = movement.boost;
        action.aim = aim::idle(obs, &self.cfg, &self.world, movement.direction, defensive);
        if let Some(best) = shot {
            if !defensive
                || best.pull
                || obs.own.weapon(best.slot).is_some_and(|w| w.charge() > 0.0)
            {
                action.aim = best.aim;
                action.fire[best.slot] = best.pull;
                for other in &shots {
                    if other.pull
                        && other.slot != best.slot
                        && wrap_pi(other.aim[0] - best.aim[0]).abs() < 0.06
                        && (other.aim[1] - best.aim[1]).abs() < 0.06
                    {
                        action.fire[other.slot] = true;
                    }
                }
            }
        }
        // A held laser fires on release under mechfight/1. Keep holding a charge that has no
        // shot until it can be aimed, unless retreating for regeneration.
        if plan.state != State::Regen {
            for slot in 0..2 {
                if obs
                    .own
                    .weapon(slot)
                    .is_some_and(|w| w.kind == "laser" && w.charge() > 0.0)
                {
                    let releasing = shot.is_some_and(|s| s.slot == slot && !s.pull);
                    action.fire[slot] = !releasing;
                }
            }
        }
        action.reload = obs.own.weapons.iter().flatten().any(|w| {
            w.kind == "gun"
                && w.ammo
                    .is_some_and(|ammo| f64::from(ammo) <= w.stat(&self.cfg, "mag") / 3.0)
                && w.reload.unwrap_or_default() <= 0.0
        }) && threats
            .iter()
            .all(|t| !t.clear || t.distance > self.cfg.weapon("gun", "range"))
            && plan.target.is_none();
        if self.debug_remaining > 0 {
            let label: String = format!("{:?} {:?}", self.world.mode, plan.state)
                .chars()
                .take(self.cfg.get("debug", "max_text_chars") as usize)
                .collect();
            let mut draw = serde_json::json!({"label": label});
            if self.cfg.get("debug", "max_lines") >= 1.0 {
                let p = obs.own.pos;
                draw["lines"] = serde_json::json!([{
                    "a": [p.0, p.1, p.2 + 0.5], "b": [goal.0, goal.1, goal.2 + 0.5],
                    "color": "#40c0ff"
                }]);
            }
            let bytes = draw.to_string().len();
            if bytes <= self.debug_remaining && bytes <= self.cfg.get("debug", "max_bytes") as usize
            {
                self.debug_remaining -= bytes;
                action.debug = Some(draw);
            }
        }
        self.state = plan.state;
        action
    }

    fn threats<'a>(obs: &Obs, cfg: &Config, world: &'a World, map: &Arena) -> Vec<Threat<'a>> {
        let own = &obs.own;
        let eye = own.pos.up(own.size(cfg, "muzzle_height"));
        let mut threats: Vec<_> = world
            .tracks
            .values()
            .filter(|t| t.mech())
            .map(|track| {
                let pos = track.predicted(obs.t);
                let delta = (pos - own.pos).xy();
                let height = target_height(cfg, &track.entity);
                Threat {
                    track,
                    distance: delta.norm(),
                    yaw: delta.yaw(),
                    clear: map.clear(eye, pos.up(height * 0.5)),
                    distracted: wrap_pi(track.entity.aim[0] - delta.yaw() - std::f64::consts::PI)
                        .abs()
                        > (cfg.get("mech", "front_arc_deg") * 0.5).to_radians(),
                }
            })
            .collect();
        threats.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        threats
    }

    fn target(&self, track: &Track, t: f64, priority: f64) -> Target {
        Target {
            pos: track
                .predicted(t)
                .up(target_height(&self.cfg, &track.entity) * 0.5),
            vel: track.entity.vel,
            radius: target_radius(&self.cfg, &track.entity),
            hp: track.entity.hp,
            has_mortar: track.has("mortar"),
            charging: track.entity.charging,
            priority,
            speculative: !track.observed(),
        }
    }

    fn shots(
        &self,
        obs: &Obs,
        threats: &[Threat],
        task_target: Option<Target>,
        deadline: Instant,
    ) -> Vec<Shot> {
        let mut targets = Vec::with_capacity(6);
        targets.extend(threats.iter().take(5).map(|threat| {
            let isolation = if self.world.mode == Mode::Squad {
                8.0
            } else {
                0.0
            };
            let priority = 30.0
                + (1.0 - threat.track.hp_ratio()) * 32.0
                + if threat.distracted { 8.0 } else { 0.0 }
                + if threat.clear { 5.0 } else { 0.0 }
                + if threat.track.charge_s > 0.1 {
                    7.0
                } else {
                    0.0
                }
                - threat.distance * 0.10
                + isolation;
            self.target(threat.track, obs.t, priority)
        }));
        targets.extend(task_target);
        targets.sort_by(|a, b| b.priority.total_cmp(&a.priority));
        let mut shots = Vec::new();
        for target in targets.into_iter().take(5) {
            if Instant::now() >= deadline {
                break;
            }
            for slot in 0..2 {
                if let Some(w) = obs.own.weapon(slot) {
                    if let Some(shot) =
                        ballistics::solve(slot, w, target, obs, &self.cfg, &self.map)
                    {
                        shots.push(shot);
                    }
                }
            }
        }
        shots
    }

    fn ground_goal(&self, p: V2, obs: &Obs) -> V3 {
        let radius = obs.own.size(&self.cfg, "radius") + 0.2;
        let height = obs.own.size(&self.cfg, "height");
        let p = self.map.bounds.inset(p, radius);
        for dist in [0.0, 6.0, 12.0, 21.0] {
            for i in 0..if dist == 0.0 { 1 } else { 8 } {
                let offset = V2::at(f64::from(i) * std::f64::consts::TAU / 8.0) * dist;
                let xy = self.map.bounds.inset(p + offset, radius);
                let z = self.map.ground(xy);
                if self
                    .map
                    .surfaces(xy)
                    .first()
                    .is_some_and(|s| s.clearance >= height)
                {
                    return xy.z(z);
                }
            }
        }
        p.z(self.map.ground(p))
    }

    fn cover(
        &self,
        obs: &Obs,
        threats: &[Threat],
        center: V2,
        radius: f64,
        roof: bool,
        allowed: impl Fn(V3) -> bool,
    ) -> Option<V3> {
        let own = &obs.own;
        let eye = own.size(&self.cfg, "eye_height");
        let origin = own.pos;
        let mut best: Option<(V3, f64)> = None;
        for dist in [0.0, 12.0, 24.0, 39.0, 52.0] {
            for i in 0..if dist == 0.0 { 1 } else { 12 } {
                let xy = origin.xy()
                    + V2::at(self.scout_yaw + f64::from(i) * std::f64::consts::TAU / 12.0) * dist;
                if (xy - center).norm() >= radius
                    || self.map.bounds.inset(xy, own.size(&self.cfg, "radius")) != xy
                {
                    continue;
                }
                let ground = self.map.ground(xy);
                for surface in self.map.surfaces(xy) {
                    if surface.clearance < own.size(&self.cfg, "height") {
                        continue;
                    }
                    let z = surface.z;
                    let rise = (z - origin.2).max(0.0);
                    let climb_budget = (own.energy - 15.0).max(0.0)
                        * own.stat(&self.cfg, "max_rise_speed")
                        / own.stat(&self.cfg, "jet_drain").max(1.0);
                    if rise > 2.0 && rise > climb_budget {
                        continue;
                    }
                    if roof && !(12.0..=48.0).contains(&(z - ground)) {
                        continue;
                    }
                    let p = xy.z(z);
                    if !allowed(p) {
                        continue;
                    }
                    let exposure: f64 = threats
                        .iter()
                        .take(4)
                        .filter(|threat| {
                            self.map.clear(
                                p.up(eye * 0.7),
                                threat
                                    .track
                                    .predicted(obs.t)
                                    .up(target_height(&self.cfg, &threat.track.entity) * 0.8),
                            )
                        })
                        .map(|threat| 1.6 - (threat.distance / 250.0).min(1.0))
                        .sum();
                    let edge = roof
                        && (0..8).any(|i| {
                            let offset = V2::at(f64::from(i) * std::f64::consts::TAU / 8.0) * 3.0;
                            self.map.floor_below((xy + offset).z(z + 0.25)) < z - 3.0
                        });
                    if edge {
                        continue;
                    }
                    let score = exposure * 24.0
                        + self.world.danger(xy, obs.t) * 12.0
                        + dist * 0.26
                        + rise * 0.30;
                    if best.is_none_or(|(_, old)| score < old) {
                        best = Some((p, score));
                    }
                }
            }
        }
        best.map(|(p, _)| p)
    }

    fn peek(&self, obs: &Obs, target: V3) -> Option<V3> {
        let own = &obs.own;
        let forward = (target.xy() - own.pos.xy()).unit();
        let side = forward.perp();
        let eye = own.size(&self.cfg, "muzzle_height");
        for dist in [4.0, 7.0, 11.0, 15.0] {
            for dir in [
                side,
                -side,
                (forward + side).unit(),
                (forward - side).unit(),
                forward,
            ] {
                let xy = own.pos.xy() + dir * dist;
                let z = self
                    .map
                    .floor_below(xy.z(own.pos.2 + own.stat(&self.cfg, "step_up")));
                let p = xy.z(z);
                if (z - own.pos.2).abs() < 2.0
                    && (xy - obs.zone.center).norm() < obs.zone.radius
                    && self.map.clear(own.pos.up(eye), p.up(eye))
                    && self.map.clear(p.up(eye), target)
                {
                    return Some(p);
                }
            }
        }
        None
    }

    fn plan(&self, obs: &Obs, threats: &[Threat]) -> Plan {
        let own = &obs.own;
        let zone = &obs.zone;
        let pos = own.pos.xy();
        let hp = own.hp / own.max_hp.max(1.0);
        let squad = self.world.mode == Mode::Squad;
        let ffa = self.world.mode == Mode::Ffa;
        let walk = own.stat(&self.cfg, "walk_speed").max(1.0);
        let d_next = (pos - zone.next_center).norm();
        let to_inner = (d_next - zone.next_radius * 0.7).max(0.0);
        let to_next_edge = (d_next - zone.next_radius * 0.9).max(0.0);
        let (soon_center, soon_radius) = zone.future(obs.t, to_next_edge / walk + 7.0);
        if !own.in_zone
            || (to_inner > 8.0
                && ((pos - soon_center).norm() > soon_radius * 0.98
                    || (to_next_edge > 2.0
                        && to_next_edge / walk + 7.0 >= zone.shrink_start - obs.t)))
        {
            let anchor =
                zone.next_center + (pos - zone.next_center).unit() * (zone.next_radius * 0.66);
            let goal = self
                .cover(
                    obs,
                    threats,
                    zone.next_center,
                    zone.next_radius * 0.72,
                    false,
                    |_| true,
                )
                .unwrap_or_else(|| self.ground_goal(anchor, obs));
            return Plan::new(State::Zone, goal, 100.0);
        }
        let finish = threats.iter().any(|threat| {
            threat.clear
                && threat.distance < 50.0
                && threat.track.entity.hp.is_some_and(|hp| hp < 85.0)
        });
        if hp < 0.22
            || (!finish
                && (hp < if squad { 0.58 } else { 0.48 }
                    || self.state == State::Regen && hp < 0.82))
        {
            let away = threats
                .first()
                .map_or(V2::at(self.scout_yaw), |threat| -V2::at(threat.yaw));
            let fallback = if threats.is_empty() && self.world.regen_ready_in == 0.0 {
                own.pos
            } else {
                self.ground_goal(
                    pos + (away * 28.0 + (zone.next_center - pos).unit() * 18.0),
                    obs,
                )
            };
            let goal = self
                .cover(obs, threats, soon_center, soon_radius * 0.94, false, |_| {
                    true
                })
                .unwrap_or(fallback);
            return Plan::new(State::Regen, goal, 91.0);
        }

        let (safe_center, safe_radius) = zone.future(obs.t, 20.0);
        let safe = |p: V3| (p.xy() - safe_center).norm() < safe_radius * 0.98;
        let choose = |a: Plan, b: Plan| {
            let score = |plan: Plan| plan.score + if plan.state == self.state { 12.0 } else { 0.0 };
            if score(b) > score(a) {
                b
            } else {
                a
            }
        };
        let explore = zone.next_center
            + V2::at(self.scout_yaw + (obs.t / 18.0).floor() * 1.3)
                * (zone.next_radius * 0.45).min(130.0);
        let mut best = Plan::new(State::Farm, self.ground_goal(explore, obs), 15.0);
        let cap = self.cfg.get("cubes", "stat_cap") as u32;
        let next_upgrade = self.cfg.0["cubes"]["upgrade_thresholds"]
            .as_array()
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|v| v.as_u64() == Some(u64::from(own.cubes) + 1))
            });
        let farm = if own.cubes < cap.min(6) {
            76.0
        } else if own.cubes < cap {
            53.0
        } else if next_upgrade {
            40.0
        } else {
            10.0
        };
        let travel = |p: V3| (p.xy() - pos).norm() + (p.2 - own.pos.2).abs() * 2.0;
        let chance = |p: V3| self.world.danger(p.xy(), obs.t) * if ffa { 9.0 } else { 4.0 };
        for cube in obs.cubes.iter().filter(|c| safe(c.pos)) {
            let score = farm + 15.0 - travel(cube.pos) * 0.28 - chance(cube.pos);
            best = choose(best, Plan::new(State::Farm, cube.pos, score));
        }
        for item in obs.pickups.iter().filter(|item| safe(item.pos)) {
            if let Some((value, slot)) = self.loadout(obs, threats, item) {
                let mut plan = Plan::new(
                    State::Farm,
                    item.pos,
                    69.0 + value * 0.18 - travel(item.pos) * 0.29 - chance(item.pos),
                );
                plan.loot = Some((item.id, slot));
                best = choose(best, plan);
            }
        }
        for loot in obs.crates.iter().filter(|c| safe(c.pos)) {
            let mut plan = Plan::new(
                State::Farm,
                loot.pos,
                farm - travel(loot.pos) * 0.32 - chance(loot.pos),
            );
            plan.target = Some(Target {
                pos: loot.pos.up(self.cfg.get("crates", "height") * 0.5),
                vel: V3::default(),
                hp: loot.hp,
                radius: self.cfg.get("crates", "radius"),
                priority: 26.0,
                has_mortar: false,
                charging: false,
                speculative: false,
            });
            best = choose(best, plan);
        }
        for npc in self
            .world
            .tracks
            .values()
            .filter(|t| !t.mech() && t.observed())
        {
            let p = npc.predicted(obs.t);
            if safe(p)
                && threats
                    .iter()
                    .all(|threat| threat.distance > if ffa { 85.0 } else { 55.0 })
            {
                let mut plan = Plan::new(State::Farm, p, farm + 5.0 - travel(p) * 0.32 - chance(p));
                plan.target = Some(self.target(npc, obs.t, 28.0));
                best = choose(best, plan);
            }
        }
        for drop in &obs.drops {
            let rivals = threats
                .iter()
                .filter(|t| {
                    (t.track.predicted(obs.t).xy() - drop.pos.xy()).norm() < drop.radius + 34.0
                })
                .count();
            let time = travel(drop.pos) / walk;
            if safe(drop.pos)
                && time <= drop.lands_in + 6.0
                && rivals < 2
                && drop.lands_in <= time + 8.0
            {
                let capture = drop.holder == Some(self.slot);
                let goal = if capture && (drop.pos.xy() - pos).norm() < drop.radius * 0.75 {
                    own.pos
                } else {
                    drop.pos
                };
                let progress = if capture {
                    drop.progress / drop.capture_s.max(0.1) * 8.0
                } else {
                    0.0
                };
                best = choose(
                    best,
                    Plan::new(
                        State::Drop,
                        goal,
                        79.0 + progress + (1.0 - hp) * 7.0 - time - rivals as f64 * 11.0,
                    ),
                );
            }
        }

        let nearby = threats.iter().filter(|t| t.distance < 125.0).count();
        let sandwich = threats.iter().take(nearby).any(|a| {
            threats
                .iter()
                .take(nearby)
                .any(|b| wrap_pi(a.yaw - b.yaw).abs() > 110_f64.to_radians())
        });
        if (own.cubes >= 4 || own.has("laser") || own.has("mortar"))
            && !sandwich
            && (best.score < 74.0 || self.state == State::Roof)
        {
            if let Some(point) = self.cover(
                obs,
                threats,
                zone.next_center,
                zone.next_radius * 0.9,
                true,
                |_| true,
            ) {
                best = choose(
                    best,
                    Plan::new(State::Roof, point, if squad { 72.0 } else { 64.0 }),
                );
            }
        }
        for threat in threats.iter().take(4) {
            let track = threat.track;
            let opponent = track.predicted(obs.t);
            let isolated = threats
                .iter()
                .filter(|other| {
                    other.track.entity.id != track.entity.id
                        && (other.track.predicted(obs.t).xy() - opponent.xy()).norm() < 90.0
                })
                .count()
                == 0;
            let finish = track.hp_ratio() < 0.55;
            let score = if squad && isolated && obs.t < 200.0 {
                85.0
            } else if ffa && (finish || threat.distracted) {
                80.0
            } else if !ffa && isolated && threat.distance < 105.0 {
                75.0
            } else {
                48.0
            };
            let lightning = own.weapons.iter().flatten().find(|w| w.kind == "lightning");
            let no_ranged_fallback = threat.clear && !own.has("gun") && !own.has("mortar");
            let optimal = if let Some(lightning) = lightning
                .filter(|_| track.has("laser") || obs.zone.radius <= 65.0 || no_ranged_fallback)
            {
                lightning.stat(&self.cfg, "range") * 0.76
            } else if own.has("laser") || own.has("mortar") {
                105.0
            } else {
                67.0
            };
            let forward = V2::at(threat.yaw);
            let dist_error = threat.distance - optimal;
            let flank = if self.scout_yaw.sin() > 0.0 {
                forward.perp()
            } else {
                -forward.perp()
            };
            let desired = pos
                + forward * dist_error.clamp(-25.0, 35.0)
                + flank * if threat.clear { 14.0 } else { 3.0 };
            let goal = self.ground_goal(desired, obs);
            if !safe(goal) {
                continue;
            }
            let score = score + if finish { 9.0 } else { 0.0 }
                - if sandwich && !squad { 40.0 } else { 0.0 }
                - (threat.distance - 85.0).max(0.0) * 0.14
                - if !threat.clear { 8.0 } else { 0.0 };
            let mut plan = Plan::new(State::Engage, goal, score);
            plan.target = Some(self.target(track, obs.t, 55.0));
            best = choose(best, plan);
        }
        if ffa && sandwich && best.state != State::Drop {
            if let Some(cover) =
                self.cover(obs, threats, safe_center, safe_radius * 0.94, false, |_| {
                    true
                })
            {
                best = Plan::new(State::Farm, cover, 86.0);
            }
        }
        best
    }

    fn loadout(&self, obs: &Obs, threats: &[Threat], item: &Loot) -> Option<(f64, Option<usize>)> {
        let own = &obs.own;
        if own.has(&item.kind)
            || !matches!(item.kind.as_str(), "gun" | "mortar" | "laser" | "lightning")
        {
            return None;
        }
        let laser_pressure = threats
            .iter()
            .filter(|t| t.track.observed() && t.track.has("laser"))
            .count()
            >= 2;
        let value = |a: &str, b: &str| -> f64 {
            let carries = |kind| a == kind || b == kind;
            let nominal = match (
                obs.zone.radius > 140.0,
                carries("laser"),
                carries("lightning"),
                carries("mortar"),
            ) {
                (true, true, true, _) => 100.0,
                (true, true, _, true) => 88.0,
                (true, true, _, _) => 80.0,
                (true, _, true, true) => 78.0,
                (true, _, true, _) => 70.0,
                (true, _, _, true) => 65.0,
                (false, _, true, true) => 100.0,
                (false, true, true, _) => 95.0,
                (false, _, true, _) => 80.0,
                (false, _, _, true) => 75.0,
                _ => 65.0,
            };
            nominal
                - if laser_pressure && carries("mortar") {
                    25.0
                } else {
                    0.0
                }
        };
        let (Some(a), Some(b)) = (own.weapon(0), own.weapon(1)) else {
            return Some((35.0, None));
        };
        if threats.iter().any(|t| t.clear && t.distance < 35.0) {
            return None;
        }
        let current = value(&a.kind, &b.kind);
        [(a, &b.kind), (b, &a.kind)]
            .into_iter()
            .enumerate()
            .filter(|(_, (remove, _))| remove.kind != "laser" || remove.charge() <= 0.0)
            .map(|(slot, (_, other))| (value(&item.kind, other) - current, Some(slot)))
            .filter(|(gain, _)| *gain >= 15.0)
            .max_by(|a, b| a.0.total_cmp(&b.0))
    }

    fn take(
        &self,
        obs: &Obs,
        threats: &[Threat],
        planned: Option<(u32, Option<usize>)>,
    ) -> (Option<u32>, Option<usize>) {
        let radius = self.cfg.get("loot", "pickup_radius");
        obs.pickups
            .iter()
            .filter(|item| {
                (item.pos.xy() - obs.own.pos.xy()).norm() <= radius
                    && (item.pos.2 - obs.own.pos.2).abs() <= 4.0
            })
            .filter_map(|item| {
                let (value, slot) = self.loadout(obs, threats, item)?;
                let score = value
                    + if planned.is_some_and(|(id, _)| id == item.id) {
                        15.0
                    } else {
                        0.0
                    };
                Some((item.id, slot, score))
            })
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .map_or((None, None), |(id, slot, _)| (Some(id), slot))
    }
}

fn pick_upgrade(obs: &Obs, cfg: &Config) -> Option<usize> {
    let choices = &obs.offer.as_ref()?.choices;
    let score = |id: &str, own: &Own| {
        let Some(upgrade) = cfg.0["upgrades"]
            .as_array()
            .and_then(|defs| defs.iter().find(|def| def["id"].as_str() == Some(id)))
        else {
            return 0;
        };
        upgrade["effects"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|e| e["path"].as_str())
            .map(|path| match path {
                "mech.armor" => 106,
                "mech.hp" | "mech.walk_speed" | "mech.boost_speed" | "mech.energy"
                | "mech.energy_regen" => 100,
                "mech.regen_delay" | "mech.regen_per_s" => 90,
                "mech.near_range_bonus" if !own.upgrades.iter().any(|u| u == id) => 85,
                "mech.turn_rate" | "mech.jet_drain" => 55,
                _ if path.starts_with("weapons.laser.") && own.has("laser") => 80,
                _ if path.starts_with("weapons.lightning.") && own.has("lightning") => 80,
                _ if path.starts_with("weapons.mortar.") && own.has("mortar") => 55,
                _ if path.starts_with("weapons.gun.") && own.has("gun") => 35,
                _ => 0,
            })
            .max()
            .unwrap_or_default()
    };
    choices
        .iter()
        .take(3)
        .enumerate()
        .max_by_key(|(idx, c)| (score(&c.id, &obs.own), std::cmp::Reverse(*idx)))
        .map(|(idx, _)| idx)
}
