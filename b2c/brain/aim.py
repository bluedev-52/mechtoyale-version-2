"""Aiming helpers: yaw/pitch to a point, linear lead for straight shots, mortar elevation."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_pitch(frm: Sequence[float], to: Sequence[float]) -> Tuple[float, float]:
    dx, dy, dz = to[0] - frm[0], to[1] - frm[1], to[2] - frm[2]
    return math.atan2(dy, dx), math.atan2(dz, max(1e-6, math.hypot(dx, dy)))


def lead(shooter: Sequence[float], target: Sequence[float], vel: Sequence[float], speed: float,
         frac: float = 1.0) -> Tuple[float, float, float]:
    """Point to aim at so a projectile of `speed` meets a target moving at constant `vel`."""
    px, py, pz = target[0] - shooter[0], target[1] - shooter[1], target[2] - shooter[2]
    vx, vy, vz = vel[0] * frac, vel[1] * frac, vel[2] * frac
    a = vx * vx + vy * vy + vz * vz - speed * speed
    b = 2 * (px * vx + py * vy + pz * vz)
    c = px * px + py * py + pz * pz
    t = math.sqrt(c) / speed
    if abs(a) > 1e-9:
        disc = b * b - 4 * a * c
        if disc >= 0:
            r = math.sqrt(disc)
            ts = [x for x in ((-b - r) / (2 * a), (-b + r) / (2 * a)) if x > 0]
            if ts:
                t = min(ts)
    return target[0] + vx * t, target[1] + vy * t, target[2] + vz * t


def mortar_pitch(dist: float, dz: float, speed: float, g: float, high: bool = False) -> Optional[float]:
    """Launch pitch for a shell to land `dist` metres away and `dz` higher, or None if out of range."""
    v2 = speed * speed
    disc = v2 * v2 - g * (g * dist * dist + 2 * dz * v2)
    if disc < 0 or dist < 1e-3:
        return None
    r = math.sqrt(disc)
    tan = (v2 + r) / (g * dist) if high else (v2 - r) / (g * dist)
    return math.atan(tan)
