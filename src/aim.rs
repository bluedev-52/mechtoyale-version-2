use crate::math::{wrap_pi, V2};
use crate::protocol::{Config, Obs};
use crate::world::World;

pub fn idle(obs: &Obs, cfg: &Config, world: &World, movement: V2, defensive: bool) -> [f64; 2] {
    if let Some((yaw, t)) = world.bearing.filter(|_| defensive) {
        if obs.t - t < 2.0 {
            return [yaw, 0.0];
        }
    }
    let pos = obs.own.pos;
    let near = world
        .tracks
        .values()
        .filter(|t| t.mech())
        .map(|t| t.predicted(obs.t) - pos)
        .min_by(|a, b| a.norm2().total_cmp(&b.norm2()));
    if let Some(enemy) =
        near.filter(|enemy| defensive || enemy.norm() < cfg.get("vision", "near_range"))
    {
        return enemy.aim();
    }
    let forward = if movement.norm2() > 0.01 {
        movement.yaw()
    } else {
        obs.own.aim[0]
    };
    let phase = (obs.t / 1.5).rem_euclid(2.0);
    let offset = 95_f64.to_radians() * (1.0 - 2.0 * (phase - 1.0).abs());
    [wrap_pi(forward + offset), 0.0]
}
