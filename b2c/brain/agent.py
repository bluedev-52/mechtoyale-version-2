"""Minimum Risk Movement + Utility Scoring bot for mechfight/1 (practice-squad calibration).

Every tick the bot samples a bounded set of reachable destinations P_t (three rings of 12
directions over a 0.4-1.4 s horizon, plus tactical anchors toward cover, the zone and loot),
vetoes illegal or immediately lethal ones with -1,000,000 and scores the rest with

    U(p) = - w_zone f_zone - w_cross a(h) f_cross - w_aoe f_aoe - w_trap f_trap - w_kin f_kin
           + w_cover a(h) f_cover + w_dist f_dist + w_loot b(h) f_loot + w_space f_space

with every f in [0, 1] and phase weights summing to 100. The practice squad (15 house bots that
share vision, never fight each other and hunt us) gets the document's defensive variant: +5 on
w_cross and w_cover paid by w_loot and w_dist in the first two phases, every enemy treated as
aiming at us (psi = 1.5), and a(h) active from h < 0.50.

Targets are scored with S(e) = u_hit + u_finish (1 - h_e) + u_duel [aims at me] - u_obst and only
switched when another one beats the current by 10%.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from brain.aim import lead, mortar_pitch, wrap, yaw_pitch
from brain.crossfire import Crossfire
from brain.nav import NavField, NavGrid
from brain.world import World

# Practice-squad variant of the document's weight table (sum of magnitudes = 100 per phase).
PHASE_W = {
    "early": dict(zone=14, cross=27, aoe=10, trap=12, kin=6, cover=19, dist=1, loot=5, space=6),
    "mid": dict(zone=18, cross=25, aoe=12, trap=11, kin=7, cover=19, dist=3, loot=1, space=4),
    "final": dict(zone=20, cross=14, aoe=14, trap=9, kin=12, cover=11, dist=16, loot=2, space=2),
}
# Offensive profile: weight moved from crossfire/cover/escape to weapon range and loot so the bot
# closes on hunters, drones and crates (every kill is a mech that dies before us).
PHASE_W_OFF = {
    "early": dict(zone=14, cross=16, aoe=10, trap=8, kin=6, cover=10, dist=18, loot=12, space=6),
    "mid": dict(zone=18, cross=15, aoe=11, trap=7, kin=7, cover=10, dist=20, loot=8, space=4),
    "final": dict(zone=20, cross=14, aoe=14, trap=9, kin=12, cover=11, dist=16, loot=2, space=2),
}

DEFAULTS = {
    "style": "offense",      # "offense" hunts kills, drones and crates; "defense" is the document's practice variant
    "farm_r": 110.0,         # offense: engage idle drones within this range
    "crate_r": 70.0,         # offense: shoot crates within this range when nothing else is in sight
    "chase_s": 5.0,          # offense: keep closing on a mech seen this recently
    "fire_min_off": 0.2,     # offense: trigger threshold on S_target
    "budget_s": 0.022,       # internal compute budget per tick (document: 20-25 ms)
    "alpha_h": 0.50,         # a(h) activates below this HP fraction (practice variant)
    "gamma": 1.25,           # a(h) = 1 + gamma (1 - h)
    "psi": 1.5,              # every squad member is hunting us
    "k_cross": 0.5,          # f_cross normalisation (per-enemy pressure before the [0, 1] clip)
    "eps": 1.0,
    "decay_s": 3.0,          # memory decay of last-known enemy positions
    "los_keep": 0.15,        # f_cross fraction kept when a building blocks the line
    "los_keep_mortar": 0.5,
    "d_loot": 40.0,
    "b_loot": 10.0,
    "u_hit": 1.0,
    "u_finish": 0.8,
    "u_duel": 1.0,
    "u_obst": 1.5,
    "u_counter": 0.5,
    "zone_horizon_s": 3.0,   # f_zone projects the shrinking circle this far ahead
    "silent_move": 0.42,     # |move| <= 5.0 / 11.7 suppresses step sounds
    "regen_hp": 0.65,        # below this, skip speculative long shots that reset regen
    "hysteresis": 1.10,
    "fire_min": 0.35,        # minimum S_target to pull the trigger
    "energy_reserve": 30.0,
    "safe_frac": 0.72,
    "gun_ropt": 115.0,
    "w_dodge": 12.0,
    "w_commit": 20.0,
    "dodge_boost_r": 0.0,
    "w_arc": 0.0,
    "loot_r": 200.0,
    "loot_mem": 1,
    "mortar_ok": 0,
    "swap_mortar": 0,
    "aim_tol": 2.2,
    "w_keep": 0.0,           # sniper: penalty for spots inside keep_r of any squad mech
    "keep_r": 120.0,          # gun fires when the aim error is within atan(aim_tol / d)
    "ambush": 0,             # hide from far hunters (squad forgets us after ~8 s unseen), fight them up close
    "amb_r": 70.0,           # a squad mech in sight inside this range ends the hiding
    "amb_cov": 2.5,          # cover weight multiplier while hiding
    "amb_dist": 0.2,         # range-keeping weight multiplier while hiding
    "w_hold": 0.0,           # bonus for staying put on a covered spot while hiding
    "duel": 0,               # one hunter in play: close in, keep it under fire and finish it
    "duel_r": 60.0,          # gun range while dueling
    "duel_w": 2.0,           # range-keeping weight multiplier while dueling
    "duel_cov": 0.5,         # cover/crossfire weight multiplier while dueling
    "duel_hp": 0.5,          # only duel above this HP fraction
    "duel_n": 1,             # duel while at most this many squad mechs were seen in the last 5 s
    "duel_boost": 1,         # boost to close the gap while dueling
    "surv": 0,               # survival: drift to the next zone centre, fight only when engaged
    "w_center": 15.0,        # survival: weight of closeness to the next zone centre
    "surv_dist": 0.3,        # survival: range-keeping weight multiplier
    "surv_fire_r": 100.0,    # survival: hold fire beyond this range unless that mech shot us in the last 8 s
    "no_gun": 0,             # with a laser or lightning, trade the gun for the other one
    "upg_weapon": 0,         # take laser/lightning upgrades before survival ones
    "dual": 0,               # fire the gun alongside a charging laser or the lightning
    "laser_hold": 0.75,      # laser charge (s) to release at on an exposed target
    "cube_grab": 1,          # walk straight onto visible cubes within cube_grab_r (3.5 m pickup radius)
    "cube_grab_r": 40.0,
    "drop_grab": 0,          # walk onto a landed, unheld supply drop (3 cubes + full HP) within drop_grab_r
    "drop_grab_r": 250.0,
    "grab": 1,               # walk straight onto a wanted weapon pickup (rings never land within 4 m)
    "grab_r": 80.0,          # grab pickups this close
    "w_grab": 40.0,          # utility of closing on the grab point
    "grab_safe": 0,          # only grab when no gunner is within 100 m
    "take_any": 1,           # take the weapon even with a shooter aiming at us
    "amb_urg": 0.4,          # stop hiding once zone urgency reaches this
    "amb_fire_r": 999.0,     # while hiding, hold fire beyond this range unless shot recently
}

# (1) universal survival: armor/HP/regen, mobility and energy; (2) sensors; (3) main weapon.
UPGRADE_ORDER = ["armor", "plating", "nanites", "servos", "capacitor", "reactor", "jets", "sensors",
                 "laser_charge", "laser_damage", "laser_pierce", "gun_rate", "gun_damage", "gun_mag",
                 "gun_velocity", "gyros", "lightning_efficiency", "lightning_range", "lightning_chain",
                 "mortar_reload", "mortar_velocity", "mortar_splash"]

COMBAT_SOUNDS = {"gun": 0.6, "laser": 1.0, "laser_charge": 1.0, "lightning": 0.8, "mortar": 0.7,
                 "explosion": 0.4, "npc_gun": 0.0}
MOVE_SOUNDS = {"step": 1.0, "boost": 0.8, "jet": 0.8, "reload": 0.8}
BAND_DIST = {"near": 35.0, "mid": 110.0, "far": 240.0}
EST_DIST = {"laser": 150.0, "gun": 110.0, "lightning": 30.0, "mortar": 150.0}
NDIR = 16
RING_DIRS = 12
RINGS = (5.0, 10.0, 16.0)
# r_opt(w) and sigma_w for f_dist (lightning 18-36 m, laser 95-220 m, mortar 65-170 m, gun 40-110 m);
# a reload or laser cooldown adds R_COOL to back out of the exchange
R_OPT = {"lightning": (27.0, 9.0), "gun": (75.0, 35.0), "laser": (157.0, 60.0), "mortar": (117.0, 50.0)}
R_COOL = 40.0
# f_cross per weapon: (lethality, gate range, D_ref)
WEAPON_THREAT = {"gun": (0.65, 235.0, 230.0), "laser": (1.0, 295.0, 200.0), "lightning": (1.5, 55.0, 45.0),
                 "mortar": (0.6, 240.0, 240.0), "npc": (0.3, 95.0, 60.0)}
LASER_CHARGING_DREF = 280.0
UPGRADE_STEPS = (1, 3, 6, 10, 15, 21, 28)


def _params() -> dict:
    p = dict(DEFAULTS)
    raw = os.environ.get("AGENT_PARAMS")
    if raw:
        try:
            p.update(json.loads(raw))
        except ValueError:
            pass
    return p


class Threat:
    __slots__ = ("id", "pos", "vel", "aim", "weapons", "charging", "hp", "max_hp", "seen", "est", "conf",
                 "cubes")

    def __init__(self, tid: int):
        self.id = tid
        self.cubes = 0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.aim: Optional[float] = None
        self.weapons: set = {"gun"}
        self.charging = False
        self.hp = 1400.0
        self.max_hp = 1400.0
        self.seen = -1e9
        self.est = True
        self.conf = 0.0


def pid_gone(pid, entry, seen, x: float, y: float, t: float) -> bool:
    """A remembered pickup is gone once we stand near enough to see it (45 m proximity) and it
    is missing, or after 120 s."""
    if pid in seen:
        return False
    q = entry[1]
    return math.dist(q[:2], (x, y)) < 35.0 or t - entry[2] > 120.0


def weapon_danger(kinds, d: float) -> float:
    """threat(e): how dangerous the enemy's arsenal is at distance d, in [0, 1]."""
    if "npc" in kinds:
        return 0.3 if d <= 50 else 0.12 if d <= 95 else 0.0
    s = 0.0
    if "laser" in kinds and d <= 295:
        s = max(s, min(1.0, (300.0 - d) / 130.0))
    if "gun" in kinds and d <= 235:
        s = max(s, 0.65 * (1.15 - d / 235))
    if "lightning" in kinds and d <= 55:
        s = 1.0
    if "mortar" in kinds and 12 <= d <= 240:
        s = max(s, 0.6)
    return min(1.0, s)


class Agent:
    name = "survivor"

    def __init__(self, hello: dict):
        self.P = _params()
        self.cfg = hello.get("config", {})
        self.rng = random.Random(hello.get("seed", 0))
        mp = hello.get("map", {})
        self.world = World(mp.get("dir") or "/arena", mp.get("files"))
        self.mech = self.cfg.get("mech", {})
        self.nav = NavGrid(self.world, self.mech)
        self.escape = self._escape_map()
        self.field: Optional[NavField] = None
        self.field_key = None
        self.pending: Optional[Tuple[tuple, NavField]] = None
        self.cover: Optional[NavField] = None
        self.cover_key = None
        self.threats: Dict[int, Threat] = {}
        self.blobs: List[Tuple[float, float, float, str, float]] = []
        self.attackers: Dict[int, float] = {}
        self.last_hit_t = -1e9
        self.last_dmg_bearing: Optional[Tuple[float, float]] = None
        self.target_id: Optional[int] = None
        self.target_s = 0.0
        self.dest: Optional[Tuple[float, float]] = None
        self.heading: Optional[Tuple[float, float]] = None
        self.straight_s = 0.0
        self.still_s = 0.0
        self.last_pos: Optional[Tuple[float, float]] = None
        self.stuck = 0.0
        self.unstick_until = -1.0
        self.unstick_dir = (1.0, 0.0)
        self.scan_yaw: Optional[float] = None
        self.scan_t = 0.0
        self.look_age = np.zeros(NDIR)
        self.mode = "early"
        self.t = 0.0
        self.offense = self.P.get("style") == "offense"
        self.crates: List[dict] = []
        self.rear_hits: List[Tuple[float, float, Optional[int]]] = []
        self.flip = 1.0
        self.flip_t = 0.0
        self.loot_mem: Dict[int, Tuple[str, Tuple[float, float, float], float]] = {}
        self.commit_dir: Optional[Tuple[float, float]] = None
        self.commit_until = -1.0
        self.crossfire = Crossfire(self.cfg)

    def _escape_map(self) -> np.ndarray:
        """Fraction of the 8 directions with at least two walkable steps (8 m) from each cell."""
        nav = self.nav
        from brain.nav import DIRS8, shift
        esc = np.zeros(nav.walk.shape, np.float32)
        for d, (dj, di, _) in enumerate(DIRS8):
            ok1 = np.isfinite(nav.cost_full[d])
            ok2 = shift(ok1, dj, di)
            esc += (ok1 & ok2).astype(np.float32)
        return esc / 8.0

    # ================================================================== main
    def act(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        self.deadline = t0 + self.P["budget_s"]
        self.t = float(obs["t"])
        me = obs["self"]
        self.me = me
        self.pos = np.array(me["pos"], dtype=float)
        self.zone = obs["zone"]
        self.alive = int(obs.get("alive", 16))
        self.crates = obs.get("crates", [])
        a: dict = {}
        if obs.get("offer"):
            a["pick"] = self.pick_upgrade(obs["offer"])
        self.update_zone(obs)
        self.update_threats(obs)
        self.crossfire.observe(obs)
        self.mode = self.phase()
        locks = self.laser_locks(obs)
        incoming = self.incoming(obs)
        target = self.select_target()
        move, boost, jump = self.plan_move(obs, locks, incoming, target)
        boost, jump = self.movement_budget(move, boost, jump, locks, incoming)
        aim, fire = self.plan_aim_fire(locks, target)
        a.update(self.plan_take(obs, target))
        # proactive reload only with <= 4 rounds and nobody close (reload is heard at 60 m)
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        close = any(math.dist(th.pos[:2], self.pos[:2]) < 60 for th in self.enemies() if th.id < 100)
        if target is None and not close and gun is not None and gun.get("reload", 0) <= 0 \
                and gun.get("ammo", 12) <= 4:
            a["reload"] = True
        a.update(move=[round(move[0], 4), round(move[1], 4)], boost=boost, jump=jump,
                 aim=[round(aim[0], 4), round(aim[1], 4)], fire=fire)
        if self.crossfire.sample is not None and os.environ.get("MECH_LOCAL_METRICS") == "1":
            a["debug"] = {"crossfire": self.crossfire.sample}
        return a

    def movement_budget(self, move, boost, jump, locks, incoming):
        """Hold mobility energy for urgent displacement or verified cover runs."""
        me = self.me
        energy, cap = me["energy"], me["max_energy"]
        stats = me["stats"]
        reserve = cap * 0.45
        min_step = stats["boost_drain"] * 0.1
        original = (boost, jump)
        sample = self.crossfire.sample or {}
        active = False
        if jump or not me["grounded"]:
            # A short jet can help clear a low ledge during unstick. Open-air ascent
            # uses the reserve without shortening the route to ground cover.
            tx, ty = self.pos[:2] + 4.0 * np.asarray(move)
            z = float(self.pos[2])
            top = self.world.roof_top(tx, ty)
            blocked = not self.nav.clear_line(*self.pos[:2], tx, ty)
            ledge = (self.t < self.unstick_until and blocked
                     and z + stats["step_up"] < top <= z + 6.0
                     and not self.world.solid_point(tx, ty, top + stats["height"] * 0.5))
            boost = boost and ledge and energy >= reserve + stats["jet_drain"] * 0.3
            jump = jump and self.t < self.unstick_until and blocked
        else:
            moving = math.hypot(*move) >= 0.3
            urgent_zone = not me.get("in_zone", True) or self.zone_urgency() >= 0.9
            urgent_hit = any(
                (kind == "mortar" and eta <= 0.7
                 and math.dist(point[:2], self.pos[:2]) <= self.cfg["weapons"]["mortar"]["splash_radius"] + 10)
                or (kind == "gun" and 0 <= eta <= 0.2
                    and abs((self.pos[0] - point[0]) * velocity[1]
                            - (self.pos[1] - point[1]) * velocity[0])
                    <= (stats["radius"] + 3.0) * max(1.0, math.hypot(*velocity[:2])))
                for kind, point, velocity, eta in incoming
            )
            urgent = moving and (urgent_zone or urgent_hit or any(not lock.est for lock in locks))
            cover_run = moving and sample.get("cover_run", False)
            active = cover_run and energy >= 0.16 * cap + min_step
            boost = active or (boost and urgent and energy >= 0.16 * cap + min_step)
            if not boost and sample.get("danger") and moving and energy >= reserve + min_step:
                boost = original[0]
        if (boost, jump) != original or active:
            sample.update(energy_change=(boost, jump) != original, energy_defense=active)
            self.crossfire.sample = sample
        return bool(boost), bool(jump)

    def phase(self) -> str:
        """Phase from mechs alive; the squad rarely loses members, so a closing zone also
        advances the phase."""
        n, r = self.alive, float(self.zone.get("radius", 1e9))
        if n <= 4 or r <= 40:
            return "final"
        if n <= 10 or r <= 120:
            return "mid"
        return "early"

    # ================================================================== zone
    def safe_radius(self, nr: float) -> float:
        return max(4.0, nr * self.P["safe_frac"])

    def update_zone(self, obs: dict) -> None:
        z = obs["zone"]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0], 1), round(nc[1], 1), round(nr, 1))
        if self.field_key != key and (self.pending is None or self.pending[0] != key):
            goal = self.nav.circle_mask(nc[0], nc[1], self.safe_radius(nr))
            if not (goal & self.nav.walk).any():
                goal = self.nav.circle_mask(nc[0], nc[1], max(12.0, nr))
            self.pending = (key, NavField(self.nav, goal))
        if self.pending is not None:
            budget = 0.006 if self.field is not None else 0.010
            if self.pending[1].run(budget_s=budget):
                self.field_key, self.field = self.pending
                self.pending = None

    def _cells(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nav = self.nav
        I = np.clip(np.rint((xs - nav.w.minx) / nav.w.step - nav.k // 2) / nav.k, 0, nav.NX - 1).astype(np.intp)
        J = np.clip(np.rint((ys - nav.w.miny) / nav.w.step - nav.k // 2) / nav.k, 0, nav.NY - 1).astype(np.intp)
        return J, I

    def _field_dists(self, f: Optional[NavField], xs: np.ndarray, ys: np.ndarray, fallback) -> np.ndarray:
        if f is None:
            return np.broadcast_to(np.asarray(fallback, dtype=float), xs.shape).copy()
        J, I = self._cells(xs, ys)
        dj, di = J - f.j0, I - f.i0
        ok = (dj >= 0) & (dj < f.D.shape[0]) & (di >= 0) & (di < f.D.shape[1])
        out = np.full(len(xs), np.inf)
        out[ok] = f.D[dj[ok], di[ok]]
        cx, cy = self.nav.center_batch(J, I)
        return np.where(np.isfinite(out), out + np.hypot(cx - xs, cy - ys), fallback)

    def field_dist(self, x: float, y: float) -> float:
        z = self.zone
        fb = max(0.0, math.dist((x, y), z["next_center"]) - self.safe_radius(z["next_radius"]))
        return float(self._field_dists(self.field, np.array([x]), np.array([y]), fb)[0])

    def time_to_edge(self, x: float, y: float) -> float:
        """Seconds until the zone edge passes (x, y) in the current phase (inf if it will not)."""
        z = self.zone
        c, r = z["center"], z["radius"]
        nc, nr = z["next_center"], z["next_radius"]
        if math.dist((x, y), c) > r:
            return 0.0
        if math.dist((x, y), nc) <= nr:
            return math.inf
        t0 = max(self.t, z["shrink_start"])
        span = max(1e-3, z["shrink_end"] - t0)
        lo, hi = 0.0, 1.0
        for _ in range(14):
            f = (lo + hi) / 2
            cx, cy = c[0] + (nc[0] - c[0]) * f, c[1] + (nc[1] - c[1]) * f
            if math.hypot(x - cx, y - cy) <= r + (nr - r) * f:
                lo = f
            else:
                hi = f
        return (t0 - self.t) + lo * span

    def zone_urgency(self) -> float:
        x, y = self.pos[0], self.pos[1]
        z = self.zone
        if math.dist((x, y), z["center"]) > z["radius"]:
            return 1.0
        if math.dist((x, y), z["next_center"]) <= self.safe_radius(z["next_radius"]):
            return 0.0
        need = self.field_dist(x, y) / (self.mech.get("walk_speed", 11.7) * 0.8) + 10.0
        left = self.time_to_edge(x, y)
        if left == math.inf:
            return 0.15 if z["shrink_start"] - self.t > need + 20 else 0.5
        slack = left - need
        if slack <= 0:
            return 1.0
        return max(0.15, min(1.0, 1.0 - slack / 30.0))

    # ================================================================== threats
    def update_threats(self, obs: dict) -> None:
        t = self.t
        vis = {}
        for m in obs["mechs"]:
            npc = m.get("kind") != "mech"
            th = self.threats.get(m["id"]) or Threat(m["id"])
            th.pos = np.array(m["pos"], dtype=float)
            th.vel = np.array(m.get("vel", (0, 0, 0)), dtype=float)
            th.aim = m["aim"][0] if m.get("aim") else None
            th.weapons = {"npc"} if npc else set(m.get("weapons") or ["gun"])
            th.charging = bool(m.get("charging"))
            th.hp = float(m.get("hp", th.hp))
            th.max_hp = float(m.get("max_hp", th.max_hp))
            th.cubes = int(m.get("cubes", th.cubes) or 0)
            th.seen, th.est, th.conf = t, False, 1.0
            self.threats[m["id"]] = th
            vis[m["id"]] = th
        for b in obs.get("beams", []):
            by = b.get("by")
            if by is None or by in vis or by >= 100:
                continue
            th = self.threats.get(by) or Threat(by)
            p0 = b["path"][0]
            th.pos = np.array([p0[0], p0[1], p0[2] - self.mech.get("muzzle_height", 7.0)])
            th.weapons.add(b["kind"])
            th.seen, th.est, th.conf, th.charging = t, True, 0.8, False
            self.threats[by] = th
        for e in obs.get("events", []):
            if e.get("type") == "damaged" and e.get("weapon") != "zone":
                self.last_hit_t = t
                if e.get("bearing") is not None:
                    self.last_dmg_bearing = (e["bearing"], t)
                    if e.get("rear"):
                        self.rear_hits = [r for r in self.rear_hits if t - r[1] < 2.0]
                        self.rear_hits.append((e["bearing"], t, e.get("by")))
                by = e.get("by")
                if by is not None:
                    self.attackers[by] = t
                if by is not None and by not in vis and e.get("bearing") is not None:
                    th = self.threats.get(by)
                    if th is None or t - th.seen > 1.0:
                        th = th or Threat(by)
                        d = 30.0 if by >= 100 else EST_DIST.get(e.get("weapon"), 100.0)
                        th.pos = self.pos + np.array([d * math.cos(e["bearing"]), d * math.sin(e["bearing"]), 0.0])
                        th.est, th.conf, th.seen = True, 0.6, t
                        th.aim = wrap(e["bearing"] + math.pi)
                    th.weapons.add("npc" if by >= 100 else e.get("weapon", "gun"))
                    self.threats[by] = th
        for th in list(self.threats.values()):
            age = t - th.seen
            if age > (6.0 if not th.est else 8.0):
                del self.threats[th.id]
            elif th.id not in vis:
                th.charging = False
        for d in obs.get("drops", []):
            h = d.get("holder")
            if h is not None and h != self.me["id"] and h not in vis:
                th = self.threats.get(h) or Threat(h)
                th.pos = np.array(d["pos"], dtype=float)
                th.seen, th.est, th.conf = t, True, 0.7
                self.threats[h] = th
        self.blobs = [b for b in self.blobs if t - b[2] < 6.0]
        for s in obs.get("sounds", []):
            k = s.get("kind")
            w = COMBAT_SOUNDS.get(k, MOVE_SOUNDS.get(k, 0.0))
            if w <= 0 or s.get("bearing") is None:
                continue
            d = 25.0 if k in MOVE_SOUNDS else BAND_DIST.get(s.get("dist"), 200.0)
            x = self.pos[0] + d * math.cos(s["bearing"])
            y = self.pos[1] + d * math.sin(s["bearing"])
            self.blobs.append((x, y, t - s.get("age", 0.0), k, w))

    def enemies(self) -> List[Threat]:
        """E_t: squad mechs we know of, plus NPC drones that are shooting at us."""
        out = []
        for th in self.threats.values():
            if th.id < 0:
                continue
            if th.id >= 100 and self.t - self.attackers.get(th.id, -1e9) > 3.0:
                continue
            out.append(th)
        return out

    def prey(self) -> List[Threat]:
        """Fire candidates: the enemies, plus (offense) idle drones in range for their 2 cubes."""
        out = self.enemies()
        if self.offense:
            ids = {th.id for th in out}
            for th in self.threats.values():
                if th.id >= 100 and th.id not in ids and not th.est and self.t - th.seen < 0.3 \
                        and math.dist(th.pos[:2], self.pos[:2]) <= self.P["farm_r"]:
                    out.append(th)
        return out

    def chase_target(self) -> Optional[Threat]:
        """Offense: the nearest squad mech seen in the last chase_s seconds (tracked through cover)."""
        best, bd = None, 1e9
        for th in self.threats.values():
            if 0 <= th.id < 100 and not th.est and self.t - th.seen < self.P["chase_s"]:
                d = math.dist(th.pos[:2], self.pos[:2])
                if d < bd and d < 230:
                    best, bd = th, d
        return best

    def aims_at_me(self, th: Threat, memory: float = 1.5) -> bool:
        if self.t - self.attackers.get(th.id, -1e9) < memory:
            return True
        if th.est or th.aim is None or self.t - th.seen > 0.3:
            return False
        d = math.dist(th.pos[:2], self.pos[:2])
        b = math.atan2(self.pos[1] - th.pos[1], self.pos[0] - th.pos[0])
        return abs(wrap(th.aim - b)) < math.atan2(6.0, max(d, 1.0)) + 0.06

    def laser_locks(self, obs: dict) -> List[Threat]:
        out = [th for th in self.threats.values() if th.charging and not th.est and self.aims_at_me(th)]
        for s in obs.get("sounds", []):
            if s.get("kind") == "laser_charge" and s.get("dist") in ("near", "mid") and s.get("bearing") is not None:
                th = Threat(-1)
                d = BAND_DIST[s["dist"]]
                th.pos = self.pos + np.array([d * math.cos(s["bearing"]), d * math.sin(s["bearing"]), 0.0])
                th.weapons, th.charging, th.est, th.conf = {"laser"}, True, True, 0.5
                out.append(th)
        return out

    def incoming(self, obs: dict) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
        """Enemy projectiles that might reach us: (kind, pos, vel, time)."""
        out = []
        g = (self.cfg.get("weapons") or {}).get("mortar", {}).get("gravity", 22.0)
        for p in obs.get("projectiles", []):
            pos = np.array(p["pos"], dtype=float)
            vel = np.array(p["vel"], dtype=float)
            rel = self.pos - pos
            if p.get("kind") == "mortar":
                q, v = pos.copy(), vel.copy()
                for k in range(1, 80):
                    v[2] -= g * 0.1
                    q = q + v * 0.1
                    if q[2] <= self.world.roof_top(q[0], q[1]) + 0.5:
                        if math.dist(q[:2], self.pos[:2]) < 40:
                            out.append(("mortar", q, vel, k * 0.1))
                        break
            else:
                sp = float(np.linalg.norm(vel[:2])) or 1.0
                along = float(rel[:2] @ vel[:2]) / sp
                if along < -5 or along > sp * 0.8:
                    continue
                perp = abs(float(rel[0] * vel[1] - rel[1] * vel[0])) / sp
                if perp < 12:
                    out.append((p.get("kind", "gun"), pos, vel, along / sp))
        return out

    def update_cover(self, srcs: List[Threat]) -> None:
        """Tactical anchor: local field (~70 m) of walking distance to cells shielded from srcs."""
        if not srcs:
            self.cover = None
            return
        x, y = float(self.pos[0]), float(self.pos[1])
        p = srcs[0].pos
        if self.cover is not None and self.cover_key is not None:
            kx, ky, px, py, kt = self.cover_key
            if self.t - kt < 0.5 and math.hypot(x - kx, y - ky) < 4 and math.hypot(p[0] - px, p[1] - py) < 8:
                return
        nav = self.nav
        J, I = nav.cell(x, y)
        R = 18
        j0, j1 = max(0, J - R), min(nav.NY, J + R + 1)
        i0, i1 = max(0, I - R), min(nav.NX, I + R + 1)
        X = nav.cx[i0:i1][None, :]
        Y = nav.cy[j0:j1][:, None]
        covered = nav.walk[j0:j1, i0:i1].copy()
        rf = nav.ray_free[:, j0:j1, i0:i1]
        for th in srcs:
            dx, dy = th.pos[0] - X, th.pos[1] - Y
            idx = np.rint(np.arctan2(dy, dx) * (NDIR / (2 * math.pi))).astype(np.int64) % NDIR
            free = np.take_along_axis(rf, idx[None], 0)[0]
            covered &= free < np.hypot(dx, dy) - nav.res
        f = NavField(nav, covered, window=(j0, j1, i0, i1))
        f.run(max_iters=40)
        self.cover = f
        self.cover_key = (x, y, float(p[0]), float(p[1]), self.t)

    # ================================================================== target selection
    def _hit_utility(self, d: float, kinds: List[Optional[str]]) -> float:
        me = self.me
        u = 0.0
        if "lightning" in kinds and d <= 44 and me["energy"] >= 3:
            u = max(u, 1.0)
        if "gun" in kinds:
            u = max(u, max(0.0, (230.0 - d) / 230.0))
        if "laser" in kinds and d <= 280:
            u = max(u, 0.9)
        if "mortar" in kinds and 20 <= d <= 235:
            u = max(u, 0.5)
        return u

    def _shot_clear(self, muzzle: np.ndarray, th: Threat) -> bool:
        """u_obst test: a shot from our muzzle reaches the target's feet, torso or head."""
        hs = (0.8, 1.75, 2.6) if th.id >= 100 else (2.0, 4.5, 7.5)
        B = np.array([th.pos + np.array([0.0, 0.0, h]) for h in hs])
        A = np.repeat(muzzle[None], len(hs), axis=0)
        return bool(self.world.los_batch(A, B, step=2.0).any())

    def select_target(self) -> Optional[Tuple[Threat, float, bool]]:
        """S(e) = u_hit + u_finish (1 - h_e) + u_duel [aims at me] - u_obst, 10% hysteresis.
        Against the squad a member that hit us in the last 8 s is still hunting us."""
        P = self.P
        kinds = [w["kind"] if w else None for w in self.me["weapons"]]
        mh = self.me["stats"].get("muzzle_height", self.mech.get("muzzle_height", 7.0))
        muzzle = self.pos + np.array([0.0, 0.0, mh])
        scored: Dict[int, Tuple[float, Threat, bool]] = {}
        any_mech = any(0 <= th.id < 100 and not th.est and self.t - th.seen < 0.3 for th in self.threats.values())
        for th in self.prey():
            if th.est or self.t - th.seen > 0.3:
                continue
            d = math.dist(th.pos[:2], self.pos[:2])
            hit = self._hit_utility(d, kinds)
            if hit <= 0:
                continue
            clear = self._shot_clear(muzzle, th)
            counter = 0.0
            if "laser" in kinds and "mortar" in th.weapons and d <= 280:
                counter = 1.0                      # cook-off +60%
            elif "lightning" in kinds and th.charging and d <= 44:
                counter = 1.6                      # short cancels the charge
            elif "mortar" in kinds and "lightning" in th.weapons and 25 <= d <= 235:
                counter = 0.6                      # concussion keeps the arc out of reach
            lob = "mortar" in kinds and 25 <= d <= 235
            obst = 0.0 if clear else (0.3 if lob else 1.0)
            s = P["u_hit"] * hit + P["u_finish"] * (1.0 - th.hp / max(1.0, th.max_hp)) \
                + P["u_counter"] * counter + P["u_duel"] * self.aims_at_me(th, memory=8.0) \
                - P["u_obst"] * obst
            if th.id >= 100 and any_mech:
                s -= 0.4  # a squad mech in sight outranks a drone
            scored[th.id] = (s, th, clear)
        if not scored:
            self.target_id = None
            return None
        best_id = max(scored, key=lambda k: scored[k][0])
        cur = scored.get(self.target_id) if self.target_id is not None else None
        if cur is not None and scored[best_id][0] <= P["hysteresis"] * cur[0]:
            best_id = self.target_id
        self.target_id = best_id
        s, th, clear = scored[best_id]
        self.target_s = s
        return th, s, clear

    # ================================================================== movement
    def candidates(self) -> List[Tuple[float, float]]:
        x, y = float(self.pos[0]), float(self.pos[1])
        pts = [(x, y)]
        off = self.rng.uniform(0, 2 * math.pi / RING_DIRS)
        for r in RINGS:
            for k in range(RING_DIRS):
                a = off + 2 * math.pi * k / RING_DIRS
                pts.append((x + r * math.cos(a), y + r * math.sin(a)))
        for f in (self.field, self.cover):
            if f is not None:
                d = f.steer(x, y)
                if d is not None:
                    pts.append((x + d[0] * 12.0, y + d[1] * 12.0))
        if self.dest is not None and 1.0 < math.dist(self.dest, (x, y)) < 20.0:
            pts.append(self.dest)
        return pts

    def pressure(self, th: Threat, d: np.ndarray) -> np.ndarray:
        """threat(e) / ((d / D_ref(e))^2 + eps), maximised over the enemy's weapons; lethality is
        scaled by its cubes (1 + 0.04 min(cubes, 10)) and gated by each weapon's reach."""
        P = self.P
        out = np.zeros_like(d)
        cube = 1.0 + 0.04 * min(th.cubes, 10)
        for k in th.weapons:
            spec = WEAPON_THREAT.get(k)
            if spec is None:
                continue
            leth, gate, dref = spec
            if k == "laser" and th.charging:
                dref, leth = LASER_CHARGING_DREF, leth * 1.5
            lo = 12.0 if k == "mortar" else 0.0
            v = leth * cube / ((d / dref) ** 2 + P["eps"])
            out = np.maximum(out, np.where((d <= gate) & (d >= lo), v, 0.0))
        return P["k_cross"] * out

    def f_cross_at(self, X: np.ndarray, Y: np.ndarray, srcs, los: Optional[np.ndarray]):
        """f_cross = clip(sum_e threat psi rho_rear decay / ((d / D_ref)^2 + eps)); a prism on the
        line keeps only a fraction (more for mortar carriers), trees discount nothing."""
        P = self.P
        n = len(X)
        press = np.zeros((len(srcs), n))
        # rho_rear: 1.5 when e would sit outside the front arc our torso can reach this tick
        yaw = float(self.me["aim"][0])
        reach = math.pi / 2 + float(self.me["stats"].get("turn_rate", 4.0)) * 0.1
        for s, th in enumerate(srcs):
            d = np.hypot(X - th.pos[0], Y - th.pos[1])
            psi = P["psi"] if th.id < 100 else 1.0
            decay = math.exp(-max(0.0, self.t - th.seen) / P["decay_s"]) * th.conf
            to_e = np.arctan2(th.pos[1] - Y, th.pos[0] - X)
            off = np.abs((to_e - yaw + np.pi) % (2 * np.pi) - np.pi)
            rho = np.where(off > reach, 1.5, 1.0)
            v = self.pressure(th, d) * psi * rho * decay
            if los is not None:
                keep = P["los_keep_mortar"] if "mortar" in th.weapons else P["los_keep"]
                v = v * np.where(los[s], 1.0, keep)
            press[s] = v
        return np.clip(press.sum(0), 0.0, 1.0), press

    def plan_move(self, obs: dict, locks, incoming, target):
        P = self.P
        me = self.me
        x, y, z = self.pos
        hp = me["hp"] / max(1.0, me["max_hp"])
        W = {k: v * P.get("wm_" + k, 1.0) for k, v in (PHASE_W_OFF if self.offense else PHASE_W)[self.mode].items()}
        alpha = 1.0 + P["gamma"] * (1.0 - hp) if hp < P["alpha_h"] else 1.0
        beta = hp
        on_roof = z > self.world.ground_at(x, y) + 2.5

        # stuck detection
        if self.last_pos is not None and self.dest is not None and math.dist(self.dest, (x, y)) > 2.0 \
                and math.dist(self.last_pos, (x, y)) < 0.25 and me.get("grounded", True):
            self.stuck += 0.1
        else:
            self.stuck = max(0.0, self.stuck - 0.05)
        moved = self.last_pos is not None and math.dist(self.last_pos, (x, y)) > 0.3
        self.still_s = 0.0 if moved else self.still_s + 0.1
        self.last_pos = (x, y)
        if self.stuck > 1.2:
            a = self.rng.uniform(0, 2 * math.pi)
            self.unstick_dir = (math.cos(a), math.sin(a))
            self.unstick_until = self.t + 0.8
            self.stuck = 0.0
        if self.t < self.unstick_until:
            return self.unstick_dir, me["energy"] > 20, bool(me.get("grounded"))

        enemies = self.enemies()
        dmeE = {th.id: math.dist(th.pos[:2], (x, y)) for th in enemies}
        pnow = {th.id: float(self.pressure(th, np.array([dmeE[th.id]]))[0]) * th.conf for th in enemies}
        srcs = [th for th in sorted(enemies, key=lambda th: -pnow[th.id]) if pnow[th.id] > 0.005][:4]
        self.hiding = bool(P["ambush"]) and self.t - self.last_hit_t > 2.0 and me.get("in_zone", True) \
            and self.zone_urgency() < P["amb_urg"] and not any(
            th.id < 100 and not th.est and self.t - th.seen < 0.5 and dmeE[th.id] < P["amb_r"] for th in enemies)
        fresh = [th for th in enemies if th.id < 100 and not th.est and self.t - th.seen < 5.0]
        self.dueling = bool(P["duel"]) and 0 < len(fresh) <= P["duel_n"] and hp > P["duel_hp"] \
            and me.get("in_zone", True) and self.zone_urgency() < 0.5
        duel_th = min(fresh, key=lambda th: dmeE[th.id]) if self.dueling else None
        if self.dueling:
            W["dist"] *= P["duel_w"]
            W["cover"] *= P["duel_cov"]
            W["cross"] *= P["duel_cov"]
        if P["surv"]:
            W["dist"] *= P["surv_dist"]
        if self.hiding:
            W["cover"] *= P["amb_cov"]
            W["dist"] *= P["amb_dist"]
        if time.perf_counter() < self.deadline:
            self.update_cover([th for th in srcs if th.conf >= 0.5][:2])

        self.grab_pt = None
        if P["grab"]:
            kinds0 = [w["kind"] if w else None for w in me["weapons"]]
            pref0 = self.weapon_pref(kinds0)
            busy = P["grab_safe"] and any(th.id < 100 and not th.est and self.t - th.seen < 1.0
                                          and dmeE[th.id] < 100.0 for th in enemies)
            if pref0 and not busy:
                best_g = None
                for pk in obs.get("pickups", []):
                    if pk["kind"] in kinds0 or pk["kind"] not in pref0 or abs(pk["pos"][2] - z) > 3.5:
                        continue
                    dg = math.dist(pk["pos"][:2], (x, y))
                    if dg < P["grab_r"] and (best_g is None or dg < best_g[0]):
                        best_g = (dg, (float(pk["pos"][0]), float(pk["pos"][1])))
                if best_g is not None:
                    self.grab_pt = best_g[1]
        if self.grab_pt is None and P["drop_grab"]:
            best_d = None
            for dr in obs.get("drops", []):
                q = dr["pos"]
                dd = math.dist(q[:2], (x, y))
                if dr.get("holder") not in (None, me["id"]) or abs(q[2] - z) > 3.5 \
                        or float(dr.get("lands_in", 99.0)) > dd / 11.7:
                    continue
                if dd < P["drop_grab_r"] and self.time_to_edge(q[0], q[1]) > 10.0 \
                        and (best_d is None or dd < best_d[0]):
                    best_d = (dd, (float(q[0]), float(q[1])))
            if best_d is not None:
                self.grab_pt = best_d[1]
        if self.grab_pt is None and P["cube_grab"] and int(me.get("cubes", 0)) < 10:
            best_c = None
            for c in obs.get("cubes", []):
                if abs(c["pos"][2] - z) > 3.5:
                    continue
                dc = math.dist(c["pos"][:2], (x, y))
                if dc < P["cube_grab_r"] and (best_c is None or dc < best_c[0]):
                    best_c = (dc, (float(c["pos"][0]), float(c["pos"][1])))
            if best_c is not None:
                self.grab_pt = best_c[1]
        pts = self.candidates()
        if self.grab_pt is not None:
            gx, gy = self.grab_pt
            dg0 = math.dist(self.grab_pt, (x, y))
            if dg0 > 16.0:
                pts.append((x + (gx - x) / dg0 * 16.0, y + (gy - y) / dg0 * 16.0))
            else:
                pts.append(self.grab_pt)
        n = len(pts)
        C = np.array(pts)
        X, Y = C[:, 0], C[:, 1]
        dist = np.hypot(X - x, Y - y)
        ux = np.where(dist > 0.1, (X - x) / np.maximum(dist, 1e-6), 0.0)
        uy = np.where(dist > 0.1, (Y - y) / np.maximum(dist, 1e-6), 0.0)
        Jc, Ic = self._cells(X, Y)

        # ---------------------------------------------------------- hard filter
        veto = np.zeros(n, dtype=bool)
        if not on_roof:
            for k in range(1, n):
                veto[k] = not self.nav.clear_line(x, y, X[k], Y[k])
        zn = self.zone
        zc, zr = zn["center"], float(zn["radius"])
        nc, nr = zn["next_center"], float(zn["next_radius"])
        dc = np.hypot(X - zc[0], Y - zc[1])
        dn = np.hypot(X - nc[0], Y - nc[1])
        sr = self.safe_radius(nr)
        fd = self._field_dists(self.field, X, Y, np.maximum(0.0, dn - sr))
        urg = self.zone_urgency()
        inside = dc <= zr
        if (inside & ~veto).any():
            veto |= ~inside
        if not me.get("in_zone", True) or urg >= 1.0:
            better = fd < fd[0] - 1.0
            if (better & ~veto).any():
                veto |= ~better

        # ---------------------------------------------------------- f_zone
        # project the circle zone_horizon_s ahead by linear interpolation of the shrink window
        t_s, t_e = float(zn["shrink_start"]), float(zn["shrink_end"])
        tp = self.t + P["zone_horizon_s"]
        frac = 0.0
        if tp > t_s and t_e > max(self.t, t_s):
            frac = min(1.0, (tp - max(self.t, t_s)) / (t_e - max(self.t, t_s)))
        cpx, cpy = zc[0] + (nc[0] - zc[0]) * frac, zc[1] + (nc[1] - zc[1]) * frac
        rp = zr + (nr - zr) * frac
        m = min(40.0, max(8.0, 0.2 * rp))
        dp = np.hypot(X - cpx, Y - cpy)
        sev = 0.5 + 0.5 * min(1.0, float(zn.get("dps", 20)) / 260.0)
        fz_proj = np.minimum(1.0, np.maximum(0.0, (dp - (rp - m)) / m) ** 2 * sev)
        if hp < 0.9:
            fz_proj = np.minimum(1.0, fz_proj + 0.3 * (dc > zr))  # stepping out also burns the 5 s regen window
        fz_next = 0.6 * np.minimum(1.0, (fd / 40.0) ** 2) + 0.4 * np.minimum(1.0, fd / 200.0)
        f_zone = np.maximum(fz_proj, urg * fz_next)

        # ---------------------------------------------------------- f_cross and f_cover
        los = None
        if srcs and time.perf_counter() < self.deadline:
            gz = np.full(n, z) if on_roof else np.array([self.world.ground_at(px, py) for px, py in pts])
            eh = self.mech.get("eye_height", 8.0)
            A = np.repeat([th.pos + np.array([0.0, 0.0, eh]) for th in srcs], n, axis=0)
            B = np.stack([np.tile(X, len(srcs)), np.tile(Y, len(srcs)), np.tile(gz + 5.5, len(srcs))], axis=1)
            los = self.world.los_batch(A, B, step=2.0).reshape(len(srcs), n)
        f_cross, press = self.f_cross_at(X, Y, srcs, los)
        # solid prisms only: trees let bullets and lasers through
        static_cover = 1.0 - self.nav.openness[Jc, Ic]
        if srcs and los is not None:
            wts = press.max(1) + 1e-6
            blocked = (~los).astype(float)
            f_cover = 0.8 * (wts @ blocked) / wts.sum() + 0.2 * static_cover
        else:
            f_cover = 0.5 * static_cover
        kinds = [w["kind"] if w else None for w in me["weapons"]]
        if ("mortar" in kinds or "laser" in kinds) and not on_roof:
            g0 = self.world.ground_at(x, y)
            gz2 = np.array([self.world.ground_at(px, py) for px, py in pts])
            f_cover = np.clip(f_cover + 0.15 * np.clip((gz2 - g0) / 5.0, 0.0, 1.0), 0.0, 1.0)

        # ---------------------------------------------------------- f_aoe
        f_aoe = np.zeros(n)
        spd = self.mech.get("boost_speed", 19.5) if me["energy"] > 10 else self.mech.get("walk_speed", 11.7)
        r = me["stats"].get("radius", 2.5)
        for kind, ppos, pvel, tt in incoming:
            if kind == "mortar":
                reach_d = np.minimum(dist, spd * max(0.0, min(tt, 3.0) - 0.15))
                dd = np.hypot(x + ux * reach_d - ppos[0], y + uy * reach_d - ppos[1])
                R = (self.cfg.get("weapons") or {}).get("mortar", {}).get("splash_radius", 7.0) + r + 2.0
                f_aoe = np.maximum(f_aoe, np.clip(2.0 - dd / R, 0.0, 1.0))
            else:
                best = np.full(n, 1e9)
                for tau in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
                    bx, by = ppos[0] + pvel[0] * tau, ppos[1] + pvel[1] * tau
                    mv = np.minimum(dist, spd * max(0.0, tau - 0.1))
                    best = np.minimum(best, np.hypot(x + ux * mv - bx, y + uy * mv - by))
                f_aoe = np.maximum(f_aoe, (1.0 if kind == "gun" else 0.5) * np.clip(1.0 - (best - r) / 2.0, 0.0, 1.0))
        for th in locks:
            idx = next((s for s, sth in enumerate(srcs) if sth is th), None)
            exposed = los[idx].astype(float) if (los is not None and idx is not None) else np.ones(n)
            bx, by = x - th.pos[0], y - th.pos[1]
            L = math.hypot(bx, by) or 1.0
            lat = np.abs((X - th.pos[0]) * by - (Y - th.pos[1]) * bx) / L
            f_aoe = np.maximum(f_aoe, exposed * (0.5 + 0.5 * np.clip(1.0 - lat / 8.0, 0.0, 1.0)))
        for bx, by, bt, k, w in self.blobs:
            if k in COMBAT_SOUNDS and self.t - bt < 3.0:
                # bearing error ~7 deg: the blob widens with its distance from us
                spread = 25.0 + 0.12 * math.hypot(bx - x, by - y)
                db = np.hypot(X - bx, Y - by)
                f_aoe = np.maximum(f_aoe, 0.5 * w * np.clip(1.0 - db / spread, 0.0, 1.0))

        # ---------------------------------------------------------- f_trap
        near = any(d < 90.0 for d in dmeE.values())
        # no energy for jump + boost (or one concussion away from it) makes walls inescapable
        short_e = 1.0 + 0.5 * (me["energy"] < 45)
        f_trap = np.clip((1.0 - self.escape[Jc, Ic]) * (1.0 if near else 0.3) * short_e, 0.0, 1.0)

        # ---------------------------------------------------------- f_kin
        vel = np.array(me.get("vel", (0.0, 0.0, 0.0))[:2], dtype=float)
        vs = float(np.hypot(*vel))
        shooter_near = any(self.aims_at_me(th) and dmeE[th.id] < 150 for th in enemies) or bool(locks)
        turn = np.zeros(n)
        if vs > 2.0 and shooter_near:
            cosang = (ux * vel[0] + uy * vel[1]) / vs
            turn = np.clip((-cosang) / 1.0, 0.0, 1.0) * (dist > 0.1)
        still = np.zeros(n)
        if shooter_near or self.t - self.last_hit_t < 3.0:
            still[dist < 1.5] = 1.0
        elif self.still_s > 3.0:
            still[dist < 1.5] = min(1.0, (self.still_s - 3.0) / 5.0)
        repeat = np.zeros(n)
        if self.heading is not None and self.straight_s > 1.5 and shooter_near:
            repeat = np.clip((ux * self.heading[0] + uy * self.heading[1] - 0.9) / 0.1, 0.0, 1.0)
        f_kin = np.clip(0.45 * turn + 0.4 * still + 0.15 * repeat, 0.0, 1.0)

        # ---------------------------------------------------------- f_dodge
        # a bullet needs d / 252 s to arrive: moving across the line of fire by more than the
        # hitbox (2.5 m + 0.3 m) in that time makes an unled shot miss
        f_dodge = np.zeros(n)
        gunners = [th for th in enemies if th.id < 100 and not th.est and self.t - th.seen < 1.0
                   and "gun" in th.weapons and 15.0 < dmeE[th.id] < 235.0]
        if gunners and P["w_dodge"] > 0:
            acc, wsum = np.zeros(n), 0.0
            sp = float(self.mech.get("walk_speed", 11.7))
            for th in gunners:
                d = dmeE[th.id]
                bx, by = (x - th.pos[0]) / d, (y - th.pos[1]) / d
                lat = np.abs(-ux * by + uy * bx)
                need = 2.8 / (d / 252.0 * sp)
                wt = 1.0 / (1.0 + (d / 120.0) ** 2)
                acc += wt * np.clip(lat / max(need, 0.05), 0.0, 1.0) * (dist > 0.1)
                wsum += wt
            f_dodge = acc / wsum
        # lightning (240 dps, 30 deg cone) cannot be dodged inside 45 m: keep carriers outside it
        f_arc = np.zeros(n)
        if P["w_arc"] > 0:
            for th in enemies:
                if th.id < 100 and "lightning" in th.weapons and self.t - th.seen < 3.0:
                    dl = np.hypot(X - th.pos[0], Y - th.pos[1])
                    f_arc = np.maximum(f_arc, np.clip((62.0 - dl) / 20.0, 0.0, 1.0))

        # sniper spacing: fight from range, never let a hunter inside keep_r
        f_keep = np.zeros(n)
        if P["w_keep"] > 0:
            for th in enemies:
                if th.id < 100 and self.t - th.seen < 4.0:
                    dk = np.hypot(X - th.pos[0], Y - th.pos[1])
                    f_keep = f_keep + np.clip((P["keep_r"] - dk) / P["keep_r"], 0.0, 1.0)
        # ---------------------------------------------------------- f_dist
        f_dist = np.zeros(n)
        if duel_th is not None and (target is None or target[0].id >= 100):
            target = (duel_th, 0.0, False)
        if target is None and self.offense:
            ch = self.chase_target()
            if ch is not None:
                target = (ch, 0.0, False)
        if target is not None:
            th = target[0]
            ready = [w for w in me["weapons"] if w]
            best_k = None
            for k in ("lightning", "laser", "mortar", "gun"):
                if k in kinds and (k != "lightning" or me["energy"] >= 15):
                    best_k = k
                    break
            if best_k is not None:
                ropt, sig = R_OPT[best_k]
                if best_k == "gun":
                    ropt = P["duel_r"] if self.dueling else P["gun_ropt"]
                w = next(w for w in ready if w["kind"] == best_k)
                if w.get("reload", 0) > 0 or (best_k == "laser" and w.get("cooldown", 0) > 0.3):
                    ropt += R_COOL
                if "lightning" in th.weapons and best_k != "lightning":
                    ropt = max(ropt, 60.0)  # stay outside their 45 m arc
                dt = np.hypot(X - th.pos[0], Y - th.pos[1])
                f_dist = np.exp(-((dt - ropt) ** 2) / (2 * sig * sig))
                if self.offense and not th.est and self.t - th.seen < 0.5:
                    # strafe across the line of fire, flipping side every 0.5-0.9 s, so bullets
                    # aimed at us miss while we keep shooting
                    if self.t > self.flip_t:
                        self.flip = -self.flip
                        self.flip_t = self.t + self.rng.uniform(0.5, 0.9)
                    bx, by = x - th.pos[0], y - th.pos[1]
                    Lb = math.hypot(bx, by) or 1.0
                    px_, py_ = -by / Lb * self.flip, bx / Lb * self.flip
                    side = np.clip(ux * px_ + uy * py_, 0.0, 1.0) * (dist > 0.1)
                    f_dist = 0.6 * f_dist + 0.4 * side

        # ---------------------------------------------------------- f_loot
        f_loot = np.zeros(n)
        objs = self.loot_objects(obs, kinds)
        if objs:
            O = np.array([(o[0], o[1]) for o in objs])
            oc, _ = self.f_cross_at(O[:, 0], O[:, 1], srcs, None)
            for (ox, oy, need), c in zip(objs, oc):
                do = np.hypot(X - ox, Y - oy)
                v = need * (0.5 + np.arctan((P["d_loot"] - do) / P["b_loot"]) / math.pi) * (1.0 - c)
                f_loot = np.maximum(f_loot, v)
        best_loot = max((o[2] for o in objs), default=0.0)

        # ---------------------------------------------------------- f_space
        mechs = [th for th in enemies if th.id < 100]
        if mechs:
            M = np.array([th.pos[:2] for th in mechs])
            dmin = np.min(np.hypot(X[:, None] - M[None, :, 0], Y[:, None] - M[None, :, 1]), axis=1)
            f_vor = np.clip(dmin / 150.0, 0.0, 1.0)
        else:
            f_vor = np.full(n, 0.5)
        f_per = np.where((self.alive > 10) & (dn <= sr), np.clip(dn / max(1.0, 0.6 * nr), 0.0, 1.0), 0.0)
        ang = (np.rint(np.arctan2(uy, ux) * (NDIR / (2 * math.pi))).astype(np.int64)) % NDIR
        stale = np.minimum(self.t - self.look_age[ang], 8.0) / 8.0 * (dist > 0.1)
        f_space = 0.5 * f_vor + 0.3 * f_per + 0.2 * stale

        U = (- W["zone"] * f_zone - W["cross"] * alpha * f_cross - W["aoe"] * f_aoe
             - W["trap"] * f_trap - W["kin"] * f_kin + W["cover"] * alpha * f_cover
             + W["dist"] * f_dist + W["loot"] * beta * f_loot + W["space"] * f_space
             + P["w_dodge"] * f_dodge - P["w_arc"] * f_arc - P["w_keep"] * f_keep)
        if self.grab_pt is not None:
            dgc = np.hypot(X - self.grab_pt[0], Y - self.grab_pt[1])
            U = U + P["w_grab"] * np.clip(1.0 - dgc / max(math.dist(self.grab_pt, (x, y)), 1.0), 0.0, 1.0)
        if P["surv"]:
            U = U + P["w_center"] * np.clip(1.0 - dn / max(nr, 1.0), 0.0, 1.0)
        if self.hiding and P["w_hold"] > 0:
            U = U + P["w_hold"] * f_cover * (dist < 1.5) + W["kin"] * 0.4 * still
        # under fire, hold a direction for 0.6-1.1 s: flipping every tick cancels our speed
        # (58.5 m/s^2 takes 0.4 s to reverse 11.7 m/s), so even unled shots keep landing
        committed = gunners and self.commit_dir is not None and self.t < self.commit_until
        if committed and P["w_commit"] > 0:
            U = U + P["w_commit"] * np.clip(ux * self.commit_dir[0] + uy * self.commit_dir[1], 0.0, 1.0)
        U[veto] = -1_000_000.0

        best = int(np.argmax(U))
        if time.perf_counter() < self.deadline:
            best = self.crossfire.select(self, obs, pts, U, veto, best, f_aoe)
        tx, ty = pts[best]
        self.dest = (tx, ty)
        if on_roof and best == 0:
            c = self.nav.nearest_walk(x, y)
            if c is not None:
                tx, ty = self.nav.center(*c)
        dx, dy = tx - x, ty - y
        L = math.hypot(dx, dy)
        if L < 0.8:
            self.straight_s = 0.0
            return (0.0, 0.0), False, False
        h = (dx / L, dy / L)
        if gunners and not committed:
            self.commit_dir = h
            self.commit_until = self.t + self.rng.uniform(0.6, 1.1)
        if self.heading is not None and h[0] * self.heading[0] + h[1] * self.heading[1] > 0.95:
            self.straight_s += 0.1
        else:
            self.straight_s = 0.0
        self.heading = h
        mag = 1.0 if L >= 3.0 else max(0.3, L / 3.0)
        threatened = bool(locks or incoming or shooter_near)
        boost = me["energy"] > P["energy_reserve"] and (
            threatened or urg >= 1.0 or not me.get("in_zone", True))
        if duel_th is not None and P["duel_boost"] and dmeE[duel_th.id] > P["duel_r"] + 15 \
                and me["energy"] > 20 and h[0] * (duel_th.pos[0] - x) + h[1] * (duel_th.pos[1] - y) > 0:
            boost, mag = True, 1.0
        if gunners and min(dmeE[th.id] for th in gunners) < P["dodge_boost_r"] and me["energy"] > 15:
            boost, mag = True, 1.0
        # acoustic stealth: no combat in sight and no zone pressure -> walk at <= 5 m/s, no boost
        recent = any(not th.est and self.t - th.seen < 3.0 for th in enemies if th.id < 100)
        if not threatened and not recent and urg < 0.5 and me.get("in_zone", True) \
                and self.t - self.last_hit_t > 5.0 and best_loot < 0.5:
            mag = min(mag, P["silent_move"])
            boost = False
        return (h[0] * mag, h[1] * mag), boost, False

    # ================================================================== loot
    def weapon_pref(self, kinds) -> Dict[str, float]:
        """Wanted pickups: any weapon for an empty slot; with swap_mortar a carried mortar
        (laser cook-off bait, slow shells) is traded for a laser or lightning."""
        pref = {"lightning": 1.0, "laser": 0.95}
        if self.P["mortar_ok"]:
            pref["mortar"] = 0.6
        if None in kinds:
            return pref
        if self.P["no_gun"] and "gun" in kinds:
            other = [k for k in kinds if k in ("laser", "lightning")]
            if other:
                return {"lightning" if other[0] == "laser" else "laser": 1.0}
        if self.P["swap_mortar"] and "mortar" in kinds:
            return {"lightning": 0.9, "laser": 0.9}
        return {}

    def loot_objects(self, obs: dict, kinds) -> List[Tuple[float, float, float]]:
        """O_t with need(o, inventory, cubes, h) in [0, 1]: (a) a weapon for the empty slot,
        (b) cubes up to stat_cap 10 (past it only when 1-2 short of the next upgrade threshold,
        since every cube also grows the hitbox), (c) supply drops about to land inside the zone."""
        x, y, z = self.pos
        out = []
        me = self.me
        cubes = int(me.get("cubes", 0))
        hp = me["hp"] / max(1.0, me["max_hp"])
        nxt = next((s for s in UPGRADE_STEPS if s > cubes), 99)
        need_cube = 0.9 if cubes < 10 else (0.4 if nxt - cubes <= 2 else 0.05)
        for c in obs.get("cubes", []):
            p = c["pos"]
            if math.dist(p[:2], (x, y)) < 120 and abs(p[2] - z) < 3.5:
                out.append((p[0], p[1], need_cube))
        P = self.P
        for p in obs.get("pickups", []):
            self.loot_mem[p["id"]] = (p["kind"], tuple(p["pos"]), self.t)
        seen = {p["id"] for p in obs.get("pickups", [])}
        for pid in [k for k, v in self.loot_mem.items()
                    if pid_gone(k, v, seen, x, y, self.t)]:
            del self.loot_mem[pid]
        pref = self.weapon_pref(kinds)
        if pref:
            src = ([(k, v[0], v[1]) for k, v in self.loot_mem.items()] if P["loot_mem"]
                   else [(p["id"], p["kind"], p["pos"]) for p in obs.get("pickups", [])])
            for _, kind, q in src:
                if kind in kinds or kind not in pref:
                    continue
                if math.dist(q[:2], (x, y)) < P["loot_r"] and abs(q[2] - z) < 3.5:
                    out.append((q[0], q[1], pref[kind]))
        if self.offense and cubes < 10:
            for c in obs.get("crates", []):
                q = c["pos"]
                if math.dist(q[:2], (x, y)) < 120:
                    out.append((q[0], q[1], 0.6))
        for d in obs.get("drops", []):
            h = d.get("holder")
            lands = float(d.get("lands_in", 99.0))
            if (h is not None and h != me["id"]) or lands > 12.0:
                continue
            q = d["pos"]
            if math.dist(q[:2], (x, y)) < 150 and self.time_to_edge(q[0], q[1]) > 14.0 + lands:
                out.append((q[0], q[1], 1.0 if (cubes < 10 or hp < 0.7) else 0.5))
        return out

    def plan_take(self, obs: dict, target) -> dict:
        """Fill the empty slot with the best complement to the gun, never in an open duel (the
        equip lock leaves the slot dead for max(1, cooldown) s)."""
        x, y, z = self.pos
        weapons = self.me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        want = self.weapon_pref(kinds)
        if not want:
            return {}
        if not self.P["take_any"] and target is not None and self.aims_at_me(target[0]) \
                and math.dist(target[0].pos[:2], (x, y)) < 120:
            return {}
        reach = (self.cfg.get("loot") or {}).get("pickup_radius", 4.0) - 0.3
        pref = {k: -v for k, v in want.items()}
        best = None
        for p in obs.get("pickups", []):
            if p["kind"] in kinds or p["kind"] not in pref:
                continue
            d = math.dist(p["pos"][:2], (x, y))
            if d <= reach and abs(p["pos"][2] - z) < 3.8:
                if best is None or pref[p["kind"]] < best[0]:
                    best = (pref[p["kind"]], p["id"])
        if not best:
            return {}
        if None in kinds:
            return {"take": best[1]}
        slot = kinds.index("mortar") if "mortar" in kinds else kinds.index("gun")
        w = weapons[slot]
        if w.get("cooldown", 0) > 0.5:
            return {}
        return {"take": best[1], "drop": slot}

    # ================================================================== aim / fire
    def plan_aim_fire(self, locks, target):
        me = self.me
        x, y, z = self.pos
        t = self.t
        eye = (x, y, z + me["stats"].get("muzzle_height", self.mech.get("muzzle_height", 7.0)))
        weapons = me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        rear = [r for r in self.rear_hits if t - r[1] < 1.2]
        if len(rear) >= 2 and (target is None or (rear[-1][2] != target[0].id
                                                  and target[0].hp / max(1.0, target[0].max_hp) > 0.3)):
            # taking x1.5 hits from behind: turn the front arc to that attacker and shoot back
            src = self.threats.get(rear[-1][2]) if rear[-1][2] is not None else None
            if src is not None and self.t - src.seen < 5.0 and not src.est:
                return self.weapon_choice(src, weapons, kinds, eye, math.dist(src.pos[:2], (x, y)), False,
                                          self._shot_clear(np.array(eye), src))
            return [rear[-1][0], 0.0], [False, False]
        if target is not None:
            th, s, clear = target
            d = math.dist(th.pos[:2], (x, y))
            short = th.charging and "lightning" in kinds and d <= 44
            lob = "mortar" in kinds and 25.0 <= d <= 235.0
            # regen discipline: hurt, no finish in sight and not under fire -> no speculative long shots
            hp = me["hp"] / max(1.0, me["max_hp"])
            idle_shot = not self.offense and hp < self.P["regen_hp"] and d > 120 \
                and not self.aims_at_me(th, memory=8.0) and th.hp / max(1.0, th.max_hp) > 0.3
            fmin = self.P["fire_min_off"] if self.offense else self.P["fire_min"]
            if self.P["surv"] and d > self.P["surv_fire_r"] and not self.aims_at_me(th, memory=8.0):
                idle_shot = True
            if getattr(self, "hiding", False) and d > self.P["amb_fire_r"]:
                idle_shot = True
            if ((s >= fmin and (clear or lob)) or short) and not idle_shot:
                return self.weapon_choice(th, weapons, kinds, eye, d, short, clear)
            return list(yaw_pitch(eye, (th.pos[0], th.pos[1], th.pos[2] + 4.5))), [False, False]
        # no firing target: keep the front arc toward the most dangerous known enemy
        enemies = self.enemies()
        if self.offense and not locks and "gun" in kinds:
            crate = self.crate_shot(eye, enemies)
            if crate is not None:
                return crate
        if locks:
            th = min(locks, key=lambda l: math.dist(l.pos[:2], self.pos[:2]))
            return list(yaw_pitch(eye, (th.pos[0], th.pos[1], th.pos[2] + 4.5))), [False, False]
        if enemies:
            th = max(enemies, key=lambda th: th.conf * (weapon_danger(th.weapons, math.dist(th.pos[:2], (x, y))) + 0.05)
                     * math.exp(-(t - th.seen) / self.P["decay_s"]))
            if t - th.seen < 4.0:
                return list(yaw_pitch(eye, (th.pos[0], th.pos[1], th.pos[2] + 4.5))), [False, False]
        if self.last_dmg_bearing and t - self.last_dmg_bearing[1] < 3.0:
            return [self.last_dmg_bearing[0], 0.0], [False, False]
        if self.blobs:
            b = max(self.blobs, key=lambda b: b[4] / (1.0 + t - b[2]))
            if t - b[2] < 4.0:
                return [math.atan2(b[1] - y, b[0] - x), 0.0], [False, False]
        return self.scan(me), [False, False]

    def crate_shot(self, eye, enemies):
        """Offense: break the nearest visible crate (1 cube, 50% weapon) when no mech is close."""
        if self.me.get("cubes", 0) >= 10 or not self.crates:
            return None
        if any(0 <= th.id < 100 and self.t - th.seen < 3.0 and math.dist(th.pos[:2], self.pos[:2]) < 150
               for th in enemies):
            return None
        x, y, _ = self.pos
        best, bd = None, self.P["crate_r"]
        for c in self.crates:
            d = math.dist(c["pos"][:2], (x, y))
            if d < bd:
                best, bd = c, d
        if best is None:
            return None
        tgt = (best["pos"][0], best["pos"][1], best["pos"][2] + 1.2)
        if not bool(self.world.los_batch([eye], [tgt], step=1.0)[0]):
            return None
        kinds = [w["kind"] if w else None for w in self.me["weapons"]]
        s = kinds.index("gun")
        w = self.me["weapons"][s]
        aim = list(yaw_pitch(eye, tgt))
        err = abs(wrap(aim[0] - self.me["aim"][0])) + abs(aim[1] - self.me["aim"][1])
        fire = [False, False]
        fire[s] = err < max(0.02, math.atan2(1.0, max(bd, 1.0))) and w.get("ammo", 1) > 0
        return aim, fire

    def _mortar_aim(self, th: Threat, w: dict) -> Optional[List[float]]:
        """Ballistic elevation from the tube (pos.z + 9.5) onto the target's feet; prefer the
        low arc and fall back to the high one when a prism sits on the low path."""
        x, y, z = self.pos
        st = w.get("stats", {})
        v = float(st.get("speed", 72.0))
        g = float(st.get("gravity", (self.cfg.get("weapons") or {}).get("mortar", {}).get("gravity", 22.0)))
        tube = z + float(self.me["stats"].get("height", 9.0)) + 0.5
        tof = math.dist(th.pos[:2], (x, y)) / max(1.0, v * 0.7)
        tx, ty = th.pos[0] + th.vel[0] * tof, th.pos[1] + th.vel[1] * tof
        d = math.hypot(tx - x, ty - y)
        dz = th.pos[2] - tube
        yaw = math.atan2(ty - y, tx - x)
        for high in (False, True):
            pitch = mortar_pitch(d, dz, v, g, high)
            if pitch is None:
                continue
            vx, vz = v * math.cos(pitch), v * math.sin(pitch)
            T = d / max(vx, 1e-3)
            ts = np.linspace(0.0, T, 9)
            P = np.stack([x + math.cos(yaw) * vx * ts, y + math.sin(yaw) * vx * ts,
                          tube + vz * ts - 0.5 * g * ts * ts], axis=1)
            if bool(self.world.los_batch(P[:-2], P[1:-1], step=3.0).all()):
                return [yaw, pitch]
        return None

    def weapon_choice(self, th: Threat, weapons, kinds, eye, d: float, short: bool, clear: bool = True):
        fire = [False, False]
        centre = (th.pos[0], th.pos[1], th.pos[2] + (1.75 if th.id >= 100 else 4.5))
        me = self.me
        aim = list(yaw_pitch(eye, centre))
        if "lightning" in kinds and d <= 44 and me["energy"] >= 3 and clear:
            s = kinds.index("lightning")
            cone = math.radians(weapons[s]["stats"].get("cone_deg", 30)) / 2
            fire[s] = abs(wrap(aim[0] - me["aim"][0])) < cone * 0.8
            if self.P["dual"] and "gun" in kinds:
                aim, gs, gf = self._gun_shot(th, weapons, kinds, eye, centre, d)
                fire[gs] = gf
            return aim, fire
        if short:
            return aim, fire
        if "laser" in kinds and d <= 280 and clear:
            s = kinds.index("laser")
            w = weapons[s]
            err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
            on = err < math.atan2(2.0, max(d, 1.0))
            ch = float(w.get("charge", 0))
            full_s = float(w["stats"].get("charge", 0.75))
            arc_near = any("lightning" in e.weapons and math.dist(e.pos[:2], self.pos[:2]) < 50
                           for e in self.enemies() if e.id < 100)
            # hold to >= 0.5 s (or full) on an exposed target; release early past min_charge when a
            # lightning carrier closes in; let go on target so the 2 s auto-fire never wastes a shot
            ready = ch >= min(self.P["laser_hold"], full_s * 0.97) or (arc_near and ch >= 0.15)
            release = on and ready
            if w.get("cooldown", 0) <= 0.05 or ch > 0:
                fire[s] = not release
                if self.P["dual"] and not ready and "gun" in kinds:
                    aim, gs, gf = self._gun_shot(th, weapons, kinds, eye, centre, d)
                    fire[gs] = gf
                return aim, fire
        if "mortar" in kinds and 25.0 <= d <= 235.0 and (not clear or "gun" not in kinds or d > 150):
            s = kinds.index("mortar")
            w = weapons[s]
            ma = self._mortar_aim(th, w)
            if ma is not None:
                err_y = abs(wrap(ma[0] - me["aim"][0]))
                fire[s] = err_y < math.atan2(3.0, max(d, 1.0)) and abs(ma[1] - me["aim"][1]) < 0.02 \
                    and w.get("cooldown", 0) <= 0.0
                return ma, fire
        if "gun" in kinds and clear:
            aim, s, fire[s] = self._gun_shot(th, weapons, kinds, eye, centre, d)
        return aim, fire

    def _gun_shot(self, th: Threat, weapons, kinds, eye, centre, d: float):
        me = self.me
        s = kinds.index("gun")
        w = weapons[s]
        p = lead(eye, centre, th.vel, w["stats"].get("speed", 252.0), 1.0)
        aim = list(yaw_pitch(eye, p))
        err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
        ok = err < max(0.01, math.atan2(self.P["aim_tol"], max(d, 1.0))) and w.get("ammo", 1) > 0 \
            and d <= w["stats"].get("range", 230.0)
        return aim, s, ok

    def scan(self, me: dict) -> List[float]:
        t = self.t
        if self.scan_yaw is None or t > self.scan_t:
            J, I = self.nav.cell(self.pos[0], self.pos[1])
            free = self.nav.ray_free[:, J, I]
            stale = np.minimum(t - self.look_age, 8.0)
            k = int(np.argmax(free * (0.3 + stale)))
            self.scan_yaw = wrap(2 * math.pi * k / NDIR)
            self.scan_t = t + 1.1
        yaw = me["aim"][0]
        for k in range(NDIR):
            if abs(wrap(2 * math.pi * k / NDIR - yaw)) < math.radians(55):
                self.look_age[k] = t
        return [self.scan_yaw, 0.0]

    # ================================================================== upgrades
    def pick_upgrade(self, offer: dict) -> int:
        ids = [c["id"] for c in offer["choices"]]
        order = UPGRADE_ORDER
        if self.P["upg_weapon"]:
            first = ["laser_damage", "laser_charge", "lightning_efficiency", "lightning_range", "laser_pierce",
                     "lightning_chain"]
            order = first + [u for u in UPGRADE_ORDER if u not in first]
        rank = {u: n for n, u in enumerate(order)}
        return min(range(len(ids)), key=lambda i: rank.get(ids[i], 99))
