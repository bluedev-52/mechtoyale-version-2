"""Briefly suspend one weapon objective when the remaining approach has become risky."""
from collections import deque
import math


class WeaponRouteGuard:
    def __init__(self):
        self.hits = deque()
        self.previous = None
        self.candidate = None
        self.since = 0.0
        self.start_distance = 0.0
        self.pause_until = 0.0
        self.paused_id = None
        self.last_pause = {}

    def update(self, agent, obs, wanted):
        now, own = obs["t"], obs["self"]
        if now >= 45 or not wanted or not own["grounded"]:
            self.candidate = None
            return None
        pos = own["pos"]
        pool = [(math.dist(item["pos"][:2], pos[:2]), item) for item in obs["pickups"]
                if item["kind"] in wanted and abs(item["pos"][2] - pos[2]) <= 3.5]
        distance, item = min(pool, key=lambda pair: pair[0], default=(math.inf, None))
        if item is None or distance >= agent.P["grab_r"]:
            self.candidate = None
            return None
        pickup = item["id"]
        if now < self.pause_until and self.paused_id == pickup:
            return pickup
        if pickup != self.candidate:
            self.candidate, self.since, self.start_distance = pickup, now, distance
            self.hits.clear()
        fresh = False
        for event in obs["events"]:
            by = event.get("by")
            if event["type"] == "damaged" and by is not None and 0 <= by < agent.cfg["match"]["mechs"]:
                self.hits.append((now, event["amount"], by))
                fresh = True
        while self.hits and self.hits[0][0] < now - 1:
            self.hits.popleft()
        if (not fresh or now - self.since < 0.39 or self.previous is None
                or self.start_distance - distance < 1 or now < self.last_pause.get(pickup, -math.inf) + 4
                or not own["in_zone"]):
            return None
        px, py = item["pos"][:2]
        mx, my = self.previous
        if (mx * (px - pos[0]) + my * (py - pos[1]) < 0.7 * distance
                or not agent.nav.clear_line(*pos[:2], px, py)):
            return None
        walk_speed = own["stats"].get("walk_speed", agent.mech["walk_speed"])
        remaining = max(0.0, distance - agent.cfg["loot"]["pickup_radius"])
        eta = 0.2 + remaining / walk_speed
        sources = {by for _, _, by in self.hits}
        if eta <= 0.85 and len(sources) == 1:
            return None
        health, energy = own["hp"], own["energy"] / max(1, own["max_energy"])
        value = 1.0 if item["kind"] == "lightning" else 0.9
        risk = sum(amount for _, amount, _ in self.hits) * eta * (1 + 0.3 * (len(sources) - 1))
        if risk <= health * (0.06 + 0.045 * energy) * value:
            return None
        self.paused_id = pickup
        self.last_pause[pickup] = now
        self.pause_until = now + min(2.2, eta)
        return pickup
