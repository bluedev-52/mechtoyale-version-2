use crate::map::{nav_cell, nav_point, Arena, Cell, NAV_STEP};
use crate::math::{V2, V3};
use crate::protocol::{Config, Obs};
use crate::world::World;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Node {
    cell: Cell,
    layer: usize,
}

struct Frontier {
    node: Node,
    remaining: f64,
    cost: f64,
}
impl Frontier {
    fn total(&self) -> f64 {
        self.cost + self.remaining
    }
}
impl PartialEq for Frontier {
    fn eq(&self, b: &Self) -> bool {
        self.total().total_cmp(&b.total()).is_eq()
    }
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Frontier {
    fn cmp(&self, b: &Self) -> Ordering {
        b.total()
            .total_cmp(&self.total())
            .then_with(|| b.remaining.total_cmp(&self.remaining))
    }
}

#[derive(Default)]
pub struct Movement {
    pub direction: V2,
    pub jump: bool,
    pub boost: bool,
}

#[derive(Default)]
pub struct Navigator {
    path: VecDeque<V3>,
    goal: Option<V3>,
    replan_at: u32,
    recent: VecDeque<V2>,
    last_move: V2,
    stalled_in_air: usize,
    climbing: bool,
}

struct Mobility {
    height: f64,
    radius: f64,
    step: f64,
    slope: f64,
    speed: f64,
    jump: f64,
    gravity: f64,
    accel: f64,
    rise_speed: f64,
    drain: f64,
    energy: f64,
}

impl Mobility {
    fn new(obs: &Obs, cfg: &Config) -> Self {
        let s = |k| obs.own.stat(cfg, k);
        Self {
            height: obs.own.size(cfg, "height"),
            radius: obs.own.size(cfg, "radius"),
            step: s("step_up"),
            slope: s("max_slope_deg").to_radians().tan(),
            speed: s("walk_speed").max(1.0),
            jump: s("jump_speed"),
            gravity: s("gravity"),
            accel: s("jet_accel"),
            rise_speed: s("max_rise_speed").max(0.1),
            drain: s("jet_drain"),
            energy: obs.own.energy,
        }
    }
    fn climb_cost(&self, rise: f64) -> f64 {
        let speed = self.rise_speed;
        let initial = self.jump.min(speed);
        let accel = self.accel - self.gravity;
        if accel <= 0.0 {
            return f64::INFINITY;
        }
        let accel_time = (speed - initial) / accel;
        let accel_distance = (initial + speed) * 0.5 * accel_time;
        let time = if rise < accel_distance {
            ((initial * initial + 2.0 * accel * rise).sqrt() - initial) / accel
        } else {
            accel_time + (rise - accel_distance) / speed
        };
        self.drain * time
    }
}

fn standing_clear(map: &Arena, a: V3, b: V3, m: &Mobility) -> bool {
    let head = m.height - 0.05;
    let side = (b.xy() - a.xy()).unit().perp() * (m.radius * 0.8);
    [V2::default(), side, -side].into_iter().all(|offset| {
        let a = a + offset.z(0.0);
        let b = b + offset.z(0.0);
        map.clear(a.up(m.step + 0.05), b.up(m.step + 0.05)) && map.clear(a.up(head), b.up(head))
    })
}

fn passable(map: &Arena, a: V3, b: V3, m: &Mobility) -> Option<f64> {
    let rise = b.2 - a.2;
    let dist = (b.xy() - a.xy()).norm();
    if rise > m.step || rise > dist * m.slope {
        let energy = m.climb_cost(rise + 0.4);
        return (rise <= 55.0
            && energy + 15.0 <= m.energy
            && map.clear(a.up(m.height), a.up(m.height + rise))
            && standing_clear(map, a.up(rise + 0.4), b.up(0.4), m))
        .then_some(energy);
    }
    let edge = b.up((-rise).max(0.0));
    standing_clear(map, a, edge, m).then_some(0.0)
}

fn direct_walk(map: &Arena, start: V3, goal: V3, m: &Mobility) -> bool {
    let delta = goal.xy() - start.xy();
    if delta.norm() > 80.0 {
        return false;
    }
    let n = (delta.norm() / NAV_STEP).ceil().max(1.0) as usize;
    let mut from = start;
    for i in 1..=n {
        let at = start.xy() + delta * (i as f64 / n as f64);
        let Some(next) = map
            .surfaces(at)
            .into_iter()
            .rev()
            .find(|surface| surface.clearance >= m.height && surface.z <= from.2 + m.step)
        else {
            return false;
        };
        let to = at.z(next.z);
        if passable(map, from, to, m) != Some(0.0) {
            return false;
        }
        from = to;
    }
    (from.2 - goal.2).abs() < 2.0
}

fn nearest_layer(map: &Arena, pos: V3, height: f64) -> Option<Node> {
    let key = nav_cell(pos.xy());
    map.layers(key)
        .iter()
        .enumerate()
        .filter(|(_, s)| s.clearance >= height)
        .min_by(|(_, a), (_, b)| (a.z - pos.2).abs().total_cmp(&(b.z - pos.2).abs()))
        .map(|(layer, _)| Node { cell: key, layer })
}

#[allow(clippy::too_many_arguments)]
fn search(
    map: &Arena,
    obs: &Obs,
    world: &World,
    goal: V3,
    mobility: &Mobility,
    avoid: bool,
    cfg: &Config,
    deadline: Instant,
) -> VecDeque<V3> {
    let Some(start) = nearest_layer(map, obs.own.pos, mobility.height) else {
        return VecDeque::new();
    };
    let target = map.bounds.inset(goal.xy(), mobility.radius);
    let point = |node: Node| nav_point(node.cell).z(map.layers(node.cell)[node.layer].z);
    let heuristic = |node: Node| {
        (point(node).xy() - target).norm() / mobility.speed + (point(node).2 - goal.2).abs() * 0.06
    };
    let mut open = BinaryHeap::from([Frontier {
        node: start,
        cost: 0.0,
        remaining: heuristic(start),
    }]);
    let mut visited = HashMap::from([(start, (0.0, start))]);
    let mut best = start;
    let mut expanded = 0;
    let mut penalties = HashMap::new();
    let (zone_center, zone_radius) = obs.zone.future(obs.t, 5.0);
    while let Some(Frontier {
        node,
        cost,
        remaining,
    }) = open.pop()
    {
        if cost > visited[&node].0 {
            continue;
        }
        if remaining < heuristic(best) {
            best = node;
        }
        if expanded >= 1800
            || (expanded % 8 == 0 && Instant::now() >= deadline)
            || ((point(node).xy() - target).norm() <= NAV_STEP
                && (point(node).2 - goal.2).abs() < 2.0)
        {
            break;
        }
        expanded += 1;
        let a = point(node);
        for (dx, dy) in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ] {
            let cell = (node.cell.0 + dx, node.cell.1 + dy);
            let xy = nav_point(cell);
            if map.bounds.inset(xy, mobility.radius) != xy {
                continue;
            }
            for (layer, next) in map.layers(cell).iter().enumerate() {
                if next.clearance < mobility.height {
                    continue;
                }
                let b = xy.z(next.z);
                let Some(energy) = passable(map, a, b, mobility) else {
                    continue;
                };
                let next_node = Node { cell, layer };
                let penalty = *penalties.entry(cell).or_insert_with(|| {
                    let zone_cost = ((xy - zone_center).norm() - zone_radius).max(0.0) * 0.02;
                    let danger = if avoid {
                        world.danger(xy, obs.t) * 0.65
                    } else {
                        0.0
                    };
                    zone_cost * (obs.zone.dps / obs.own.max_hp.max(1.0)) + danger
                });
                let travel = (b.xy() - a.xy()).norm() / mobility.speed;
                let cover = if avoid
                    && world
                        .tracks
                        .values()
                        .filter(|t| t.mech())
                        .take(2)
                        .any(|track| {
                            !map.clear(
                                track.predicted(obs.t).up(cfg.get("mech", "muzzle_height")),
                                b.up(mobility.height * 0.5),
                            )
                        }) {
                    0.12
                } else {
                    0.0
                };
                let cost = cost + travel * (1.0 + penalty - cover) + 0.06 * energy;
                if visited.get(&next_node).is_some_and(|&(old, _)| old <= cost) {
                    continue;
                }
                visited.insert(next_node, (cost, node));
                open.push(Frontier {
                    node: next_node,
                    cost,
                    remaining: heuristic(next_node),
                });
            }
        }
    }
    let mut route = VecDeque::new();
    while best != start {
        route.push_front(point(best));
        best = visited[&best].1;
    }
    route
}

impl Navigator {
    #[allow(clippy::too_many_arguments)]
    pub fn steer(
        &mut self,
        obs: &Obs,
        cfg: &Config,
        map: &Arena,
        world: &World,
        goal: V3,
        rush: bool,
        avoid: bool,
        deadline: Instant,
    ) -> Movement {
        let m = Mobility::new(obs, cfg);
        let own = &obs.own;
        let pos = own.pos;
        let moved_goal = self.goal.is_none_or(|last| (goal - last).norm() > 12.0);
        if !self.climbing && (moved_goal || obs.tick >= self.replan_at) {
            let search_deadline = deadline.min(Instant::now() + Duration::from_micros(1_500));
            self.goal = Some(goal);
            self.replan_at = obs
                .tick
                .saturating_add((cfg.get("match", "decide_every") as u32).max(1) * 5);
            let needs_cover = avoid
                && world.tracks.values().any(|track| {
                    track.mech() && (track.predicted(obs.t) - pos).norm2() < 75.0_f64.powi(2)
                });
            let path = if !needs_cover && direct_walk(map, pos, goal, &m) {
                self.path.clear();
                VecDeque::new()
            } else {
                search(map, obs, world, goal, &m, avoid, cfg, search_deadline)
            };
            if moved_goal || !path.is_empty() {
                self.path = path;
            }
        }
        while self
            .path
            .front()
            .is_some_and(|p| (p.xy() - pos.xy()).norm() < 1.7 && (p.2 - pos.2).abs() < 1.6)
        {
            self.path.pop_front();
        }
        let waypoint = self.path.front().copied().unwrap_or(goal);
        let delta = waypoint.xy() - pos.xy();
        let mut movement = Movement {
            direction: delta.unit() * (delta.norm() / 3.0).min(1.0),
            boost: rush && own.grounded && own.energy > 0.15 * own.max_energy && delta.norm() > 5.0,
            jump: false,
        };
        let rise = waypoint.2 - pos.2;
        let floor = map.floor_below(pos);
        if ((rise > m.step || self.climbing && rise > -0.4) && delta.norm() < 10.0)
            && (self.climbing || m.climb_cost(waypoint.2 - floor + 0.4) + 15.0 <= m.energy)
        {
            movement.jump = own.grounded;
            movement.boost = true;
            self.climbing = true;
            if !own.grounded && own.vel.2 <= 0.2 {
                self.stalled_in_air += 1;
            } else {
                self.stalled_in_air = 0;
            }
            if self.stalled_in_air >= 3 {
                movement.direction = -delta.unit();
                movement.boost = false;
                self.stalled_in_air = 0;
                self.climbing = false;
            }
        } else {
            self.climbing = false;
            self.stalled_in_air = 0;
        }

        self.recent.push_back(pos.xy());
        if self.recent.len() > 6 {
            self.recent.pop_front();
        }
        if own.grounded
            && !self.climbing
            && self.recent.len() == 6
            && self.last_move.norm() > 0.5
            && self
                .recent
                .front()
                .is_some_and(|p| (*p - pos.xy()).norm() < 0.8)
        {
            let side = movement.direction.perp().unit();
            let escape = |d: V2| {
                map.clear(
                    pos.up(m.height * 0.5),
                    pos.up(m.height * 0.5) + (d * 6.0).z(0.0),
                )
            };
            movement.direction = if escape(side) { side } else { -side };
            self.replan_at = obs.tick;
            self.recent.clear();
        }

        if let Some(dodge) = evade(obs, cfg, map, world, movement.direction) {
            let probe = pos.up(m.height * 0.5) + (dodge * 5.0).z(0.0);
            if (map.bounds.inset(probe.xy(), m.radius) - probe.xy()).norm() < 0.1
                && map.clear(pos.up(m.height * 0.5), probe)
            {
                movement.direction = dodge;
                movement.boost = own.energy > 10.0 && (own.grounded || self.climbing);
            }
        }
        self.last_move = movement.direction;
        movement
    }
}

fn evade(obs: &Obs, cfg: &Config, map: &Arena, world: &World, movement: V2) -> Option<V2> {
    let own = &obs.own;
    let radius = own.size(cfg, "radius");
    let center = own.pos.up(own.size(cfg, "height") * 0.5);
    let mut lateral = V2::default();
    let mut emergency = None;
    for shot in &obs.projectiles {
        if shot.kind == "mortar" {
            let gravity = cfg.weapon("mortar", "gravity");
            if gravity <= 0.0 {
                continue;
            }
            let mut t = 0.0;
            for _ in 0..3 {
                let ahead = (shot.pos + shot.vel * t).up(-0.5 * gravity * t * t);
                let surface = map.floor_below(ahead);
                let discriminant = shot.vel.2 * shot.vel.2 + 2.0 * gravity * (shot.pos.2 - surface);
                if discriminant < 0.0 {
                    break;
                }
                t = (shot.vel.2 + discriminant.sqrt()) / gravity;
            }
            if t > 0.0 && t < 1.25 {
                let impact = (shot.pos + shot.vel * t).xy();
                let ahead = own.pos.xy() + own.vel.xy() * t;
                let distance = (ahead - impact).norm();
                if distance < cfg.weapon("mortar", "splash_radius") + radius {
                    let away = (ahead - impact).unit();
                    if away.norm2() > 0.0 {
                        emergency = Some(away);
                    }
                }
            }
        } else if matches!(shot.kind.as_str(), "gun" | "npc" | "npc_gun") {
            let velocity = shot.vel - own.vel;
            let relative = center - shot.pos;
            let t = relative.dot(velocity) / velocity.norm2().max(1e-6);
            if t <= 0.0 || t > 0.55 {
                continue;
            }
            let closest = relative - velocity * t;
            if closest.xy().norm() < radius + 1.8
                && closest.2.abs() < own.size(cfg, "height") * 0.5 + 1.0
            {
                let side = shot.vel.xy().perp().unit();
                let sign = if closest.xy().dot(side).abs() > 0.2 {
                    closest.xy().dot(side)
                } else {
                    movement.dot(side)
                };
                lateral = lateral + side * 1.4 * sign.signum();
            }
        }
    }
    for track in world
        .tracks
        .values()
        .filter(|t| t.mech() && t.entity.charging && t.observed())
    {
        let direction = V3::dir(track.entity.aim);
        let offset = center
            - track
                .entity
                .pos
                .up(crate::world::target_height(cfg, &track.entity) * 0.78);
        if offset.dot(direction) > 0.0
            && offset.norm() <= cfg.weapon("laser", "range")
            && (offset - direction * offset.dot(direction)).norm() < radius + 2.0
            && map.clear(center, center - offset)
        {
            let side = direction.xy().perp().unit();
            let sign = if offset.xy().dot(side).abs() > 0.2 {
                offset.xy().dot(side)
            } else {
                movement.dot(side)
            };
            lateral = lateral + side * 2.0 * sign.signum();
        }
    }
    emergency.or_else(|| (lateral.norm2() > 0.01).then(|| (movement + lateral).unit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn charged_jet_energy_increases_with_roof_height() {
        let m = Mobility {
            height: 9.0,
            radius: 2.5,
            step: 1.2,
            slope: 1.2,
            speed: 11.7,
            jump: 11.0,
            gravity: 22.0,
            accel: 34.0,
            rise_speed: 10.0,
            drain: 15.0,
            energy: 100.0,
        };
        assert!(m.climb_cost(35.0) + 15.0 < m.energy);
        assert!(m.climb_cost(60.0) + 15.0 > m.energy);
    }
}
