use crate::map::Arena;
use crate::math::{wrap_pi, V3};
use crate::protocol::{Config, Obs, Weapon};

#[derive(Clone, Copy)]
pub struct Target {
    pub pos: V3,
    pub vel: V3,
    pub radius: f64,
    pub hp: Option<f64>,
    pub charging: bool,
    pub has_mortar: bool,
    pub priority: f64,
    pub speculative: bool,
}

pub struct Shot {
    pub slot: usize,
    pub target: V3,
    pub aim: [f64; 2],
    pub pull: bool,
    pub score: f64,
}

fn intercept(offset: V3, velocity: V3, speed: f64) -> Option<f64> {
    let a = velocity.norm2() - speed * speed;
    let b = 2.0 * offset.dot(velocity);
    let c = offset.norm2();
    if a.abs() < 1e-9 {
        let t = -c / b;
        return (t.is_finite() && t > 0.0).then_some(t);
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let q = -0.5 * (b + discriminant.sqrt().copysign(b));
    [q / a, c / q]
        .into_iter()
        .filter(|t| t.is_finite() && *t > 0.0)
        .min_by(f64::total_cmp)
}

fn mortar_angles(offset: V3, speed: f64, gravity: f64, high: bool) -> Option<(V3, f64)> {
    let d = offset.xy().norm();
    if d < 0.1 || speed <= 0.0 || gravity <= 0.0 {
        return None;
    }
    let v2 = speed * speed;
    let discriminant = v2 * v2 - gravity * (gravity * d * d + 2.0 * offset.2 * v2);
    if discriminant < 0.0 {
        return None;
    }
    let numerator = v2 + discriminant.sqrt();
    let tangent = if high {
        numerator / (gravity * d)
    } else {
        (gravity * d * d + 2.0 * offset.2 * v2) / (d * numerator)
    };
    let pitch = tangent.atan();
    let velocity = (offset.xy().unit() * (speed * pitch.cos())).z(speed * pitch.sin());
    Some((velocity, d / velocity.xy().norm()))
}

fn arc_clear(
    map: &Arena,
    start: V3,
    velocity: V3,
    gravity: f64,
    time: f64,
    segments: usize,
) -> bool {
    let mut last = start;
    for i in 1..=segments {
        let t = time * i as f64 / segments as f64;
        let p = (start + velocity * t).up(-0.5 * gravity * t * t);
        if !map.clear(last, p) {
            return false;
        }
        last = p;
    }
    true
}

fn aligned(actual: V3, desired: V3, radius: f64, distance: f64) -> bool {
    actual.dot(desired) > (radius / distance.max(0.1)).atan().cos()
}

pub fn solve(
    slot: usize,
    weapon: &Weapon,
    target: Target,
    obs: &Obs,
    cfg: &Config,
    map: &Arena,
) -> Option<Shot> {
    let own = &obs.own;
    let dim = |key| own.size(cfg, key);
    let stat = |key| weapon.stat(cfg, key);
    let speed = stat("speed");
    let range = stat("range");
    let actual = V3::dir(own.aim);
    let muzzle = own.pos.up(if weapon.kind == "mortar" {
        dim("height") + 0.5
    } else {
        dim("muzzle_height")
    });
    let barrel = dim("radius") + 0.5;
    let start = muzzle + actual * barrel;
    let barrel_clear = map.clear(muzzle, start);
    let offset = target.pos - muzzle;
    let distance = offset.norm();
    let ready = weapon.cooldown <= cfg.dt() * 0.5;
    let damage_scale = 1.0
        + cfg.get("cubes", "damage_per_cube")
            * f64::from(own.cubes).min(cfg.get("cubes", "stat_cap"));
    let damage = stat("damage") * damage_scale;
    let min_pitch = own.stat(cfg, "min_pitch");
    let max_pitch = own.stat(cfg, "max_pitch");
    let facing = |direction: V3| {
        let [yaw, pitch] = direction.aim();
        [yaw, pitch.clamp(min_pitch, max_pitch)]
    };

    match weapon.kind.as_str() {
        "gun" if !target.speculative && speed > 0.0 => {
            let against_wall = !map.clear(target.pos, target.pos + target.vel.unit() * 4.0);
            let velocity = target.vel * if against_wall { 0.55 } else { 0.90 };
            let initial = muzzle + offset.unit() * barrel;
            let time = intercept(target.pos - initial, velocity, speed)?;
            if time > range / speed {
                return None;
            }
            let impact = target.pos + velocity * time;
            if !map.clear(initial, impact) {
                return None;
            }
            let desired = (impact - muzzle).unit();
            let aim = facing(desired);
            let pull = ready
                && barrel_clear
                && weapon.reload.unwrap_or_default() <= 0.0
                && weapon.ammo.unwrap_or_default() > 0
                && aligned(actual, desired, target.radius, distance)
                && map.clear(
                    start,
                    start + actual * (impact - start).dot(actual).max(0.0),
                );
            Some(Shot {
                slot,
                target: target.pos,
                aim,
                pull,
                score: target.priority + if pull { 18.0 } else { 0.0 }
                    - if weapon.ammo == Some(0) || weapon.reload.unwrap_or_default() > 0.0 {
                        28.0
                    } else {
                        0.0
                    },
            })
        }
        "mortar" => {
            if !ready {
                return None;
            }
            let splash = stat("splash_radius");
            if distance < splash + 2.5 || distance > speed * stat("ttl") + barrel {
                return None;
            }
            let gravity = stat("gravity");
            for (high, steps) in [(false, 12), (true, 16)] {
                let mut predicted = target.pos;
                let mut origin = muzzle;
                let mut solution = None;
                for _ in 0..3 {
                    solution = mortar_angles(predicted - origin, speed, gravity, high);
                    let Some((velocity, t)) = solution else {
                        break;
                    };
                    origin = muzzle + velocity.unit() * barrel;
                    predicted = target.pos + target.vel * (0.75 * t);
                }
                let Some((velocity, time)) = solution else {
                    continue;
                };
                let direction = velocity.unit();
                let pitch = direction.aim()[1];
                if time > stat("ttl")
                    || pitch < min_pitch
                    || pitch > max_pitch
                    || (predicted.xy() - own.pos.xy()).norm() < splash + 2.5
                    || !map.clear(muzzle, origin)
                    || !arc_clear(map, origin, velocity, gravity, time, steps)
                {
                    continue;
                }
                let actual_impact =
                    (start + actual * (speed * time)).up(-0.5 * gravity * time * time);
                let pull = ready
                    && barrel_clear
                    && (actual_impact - predicted).norm() < target.radius.max(splash * 0.65)
                    && arc_clear(map, start, actual * speed, gravity, time, steps);
                return Some(Shot {
                    slot,
                    target: target.pos,
                    aim: facing(direction),
                    pull,
                    score: target.priority + if pull { 31.0 } else { 10.0 },
                });
            }
            None
        }
        "laser" if distance <= range + barrel && (ready || weapon.charge() > 0.0) => {
            let desired = offset.unit();
            let visible = map.clear(muzzle + desired * barrel, target.pos);
            let charge = weapon.charge();
            let full = stat("charge").max(0.01);
            let min = stat("min_charge");
            let cookoff = if target.has_mortar {
                1.0 + stat("cookoff")
            } else {
                1.0
            };
            let estimated = damage
                * cookoff
                * (stat("min_frac") + (1.0 - stat("min_frac")) * (charge / full).min(1.0));
            let lethal = target.hp.is_some_and(|hp| estimated >= hp);
            let release = visible
                && barrel_clear
                && aligned(actual, desired, target.radius * 0.9, distance)
                && map.clear(
                    start,
                    start + actual * (target.pos - start).dot(actual).max(0.0),
                )
                && charge >= min
                && (charge >= full.min(0.65) || lethal);
            let covered = !visible && !map.clear(target.pos, own.pos.up(dim("height") * 0.5));
            if charge <= 0.0 && !covered && !lethal {
                return None;
            }
            Some(Shot {
                slot,
                target: target.pos,
                aim: facing(desired),
                pull: !release,
                score: target.priority * cookoff + if charge > 0.0 { 35.0 } else { 10.0 },
            })
        }
        "lightning"
            if distance <= range && !target.speculative && map.clear(muzzle, target.pos) =>
        {
            let cost = stat("energy");
            let lethal = target.hp.is_some_and(|hp| hp <= damage);
            let cone = (stat("cone_deg") * 0.5).to_radians();
            let aim = facing(offset.unit());
            let pull = ready
                && own.energy >= cost
                && (lethal || own.energy >= 12.0)
                && wrap_pi(aim[0] - own.aim[0]).abs() <= cone
                && barrel_clear;
            Some(Shot {
                slot,
                target: target.pos,
                aim,
                pull,
                score: target.priority
                    + if target.charging {
                        60.0
                    } else if pull {
                        24.0
                    } else {
                        9.0
                    },
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gun_interception_closes_the_distance() {
        let offset = V3(110.0, 25.0, -8.0);
        let velocity = V3(-10.0, 18.0, 2.0);
        let t = intercept(offset, velocity, 110.0).unwrap();
        assert!(((offset + velocity * t).norm() - 110.0 * t).abs() < 1e-9);
        assert!(intercept(offset, offset, 1.0).is_none());
    }
    #[test]
    fn both_mortar_branches_reach_an_elevated_target() {
        let target = V3(150.0, 20.0, 10.0);
        let (low, tl) = mortar_angles(target, 72.0, 22.0, false).unwrap();
        let (high, th) = mortar_angles(target, 72.0, 22.0, true).unwrap();
        assert!(tl < th);
        for (v, t) in [(low, tl), (high, th)] {
            assert!(((v * t).up(-11.0 * t * t) - target).norm() < 1e-9);
        }
        assert!(mortar_angles(target * 3.0, 72.0, 22.0, false).is_none());
    }
}
