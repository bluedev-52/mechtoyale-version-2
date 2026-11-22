"""Fallback survival bot: no map, no heavy preprocessing, always answers in ~1 ms.

Used if the full Agent fails to initialise (missing map files, unexpected config, OOM).
Rotates early into the next circle, keeps distance from visible mechs, faces the most
dangerous known bearing so rear hits don't multiply, and fires back only when very close.
"""
from __future__ import annotations

import math
from typing import List, Optional


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SimpleBot:
    name = "survivor-lite"

    UPGRADES = ["plating", "armor", "nanites", "servos", "sensors", "jets", "capacitor",
                "reactor", "gyros", "lightning_efficiency", "gun_mag", "gun_rate", "gun_damage"]

    def __init__(self, hello: dict):
        self.cfg = hello.get("config", {})
        self.last_bearing: Optional[tuple] = None  # (world yaw toward attacker, t)
        self.last_hit_t = -1e9

    def _nearest(self, mechs: List[dict], x: float, y: float, kind: str = "mech"):
        best, bd = None, 1e9
        for m in mechs:
            if m.get("kind") != kind:
                continue
            d = math.hypot(m["pos"][0] - x, m["pos"][1] - y)
            if d < bd:
                best, bd = m, d
        return best, bd

    def act(self, obs: dict) -> dict:
        me = obs.get("self", {})
        x, y, z = me.get("pos", (0, 0, 0))
        t = float(obs.get("t", 0))
        a: dict = {}
        offer = obs.get("offer")
        if offer:
            rank = {u: i for i, u in enumerate(self.UPGRADES)}
            ids = [c.get("id", "") for c in offer.get("choices", [])]
            a["pick"] = min(range(len(ids)), key=lambda i: rank.get(ids[i], 99)) if ids else 0

        for e in obs.get("events", []):
            if e.get("type") == "damaged" and e.get("weapon") != "zone":
                self.last_hit_t = t
                if e.get("bearing") is not None:
                    self.last_bearing = (e["bearing"], t)

        zone = obs.get("zone", {})
        nc = zone.get("next_center", [0, 0])
        nr = float(zone.get("next_radius", 50) or 50)
        zc = zone.get("center", nc)
        zr = float(zone.get("radius", 1e9) or 1e9)
        mechs = obs.get("mechs", [])
        foe, fd = self._nearest(mechs, x, y)

        # movement: weighted blend of zone pull and threat repulsion
        mvx = mvy = 0.0
        dz_next = math.hypot(nc[0] - x, nc[1] - y)
        outside = math.hypot(x - zc[0], y - zc[1]) > zr
        need_zone = outside or dz_next > nr * 0.7
        if need_zone and dz_next > 1:
            w = 3.0 if outside else 1.5
            mvx += w * (nc[0] - x) / dz_next
            mvy += w * (nc[1] - y) / dz_next
        elif dz_next > 1:
            ang = math.atan2(y - nc[1], x - nc[0]) + 0.6  # drift around the circle, off-centre
            rr = min(nr * 0.6, dz_next)
            gx, gy = nc[0] + rr * math.cos(ang), nc[1] + rr * math.sin(ang)
            L = math.hypot(gx - x, gy - y) or 1.0
            mvx += 0.5 * (gx - x) / L
            mvy += 0.5 * (gy - y) / L
        if foe is not None:
            away = math.atan2(y - foe["pos"][1], x - foe["pos"][0])
            w = 2.5 if fd < 60 else 1.2 if fd < 140 else 0.3
            mvx += w * math.cos(away)
            mvy += w * math.sin(away)
            # strafe so we are a harder target
            mvx += 0.4 * math.cos(away + math.pi / 2)
            mvy += 0.4 * math.sin(away + math.pi / 2)
        L = math.hypot(mvx, mvy)
        move = [mvx / L, mvy / L] if L > 0.05 else [0.0, 0.0]
        if foe is not None and fd < 30 and obs.get("alive", 16) <= 6:
            move = [move[0] * 0.4, move[1] * 0.4]  # quiet footsteps late

        # aim: at the visible foe, else at the last damage bearing, else scan
        aim = [math.atan2(math.sin(t), math.cos(t)), 0.0]
        fire = [False, False]
        if foe is not None:
            fyaw = math.atan2(foe["pos"][1] - y, foe["pos"][0] - x)
            fp = math.atan2(foe["pos"][2] + 4.5 - (z + 7), max(fd, 1.0))
            aim = [fyaw, fp]
            endgame = float(zone.get("radius", 1e9)) <= 30 or obs.get("alive", 16) <= 2
            if fd < 35 or endgame or t - self.last_hit_t < 1.5:
                # fire only when roughly aligned (shots come from the actual torso aim)
                cur = me.get("aim") or [0.0, 0.0]
                if abs(_wrap(cur[0] - fyaw)) < 0.12:
                    fire = [True, True]
        elif self.last_bearing and t - self.last_bearing[1] < 3.0:
            aim = [self.last_bearing[0], 0.0]
        elif foe is None and mechs:
            npc, _ = self._nearest(mechs, x, y, "npc")
            if npc is not None:
                aim = [math.atan2(npc["pos"][1] - y, npc["pos"][0] - x), 0.0]

        boost = bool(outside or (foe is not None and fd < 80 and me.get("energy", 0) > 40))
        a.update(move=move, aim=aim, fire=fire, boost=boost, jump=False)
        return a
