"""One-shot interventions for controlled pickup-routing replays."""
from collections import deque
import math
import os


class RouteProbe:
    def __init__(self, mode, tick, pickup):
        self.mode = mode
        self.tick = tick
        self.pickup = pickup
        self.damage = deque()
        self.item = None
        self.finished = False
        self.applied = None

    @classmethod
    def from_env(cls):
        value = os.environ.get("MECH_W_PROBE")
        if not value:
            return None
        mode, tick, pickup = value.split(":")
        if mode not in ("continue", "abandon", "reconsider"):
            raise ValueError(f"Unknown pickup probe: {mode}")
        return cls(mode, int(tick), int(pickup))

    def observe(self, obs, wanted):
        self.item = None
        self.applied = None
        tick, now = obs["tick"], obs["t"]
        for event in obs["events"]:
            if event["type"] == "damaged":
                self.damage.append((now, event["amount"], event.get("by")))
        while self.damage and self.damage[0][0] < now - 1:
            self.damage.popleft()
        if self.finished or tick < self.tick:
            return
        if tick > self.tick + 44:
            self.finished = True
            return
        self.item = next((item for item in obs["pickups"]
                          if item["id"] == self.pickup and item["kind"] in wanted), None)
        if self.item is None:
            self.finished = True

    def consider(self, agent, obs, dist, cross_delta, area_danger):
        """Allow a short bias toward a reachable pickup while projected loss stays affordable."""
        own, zone = obs["self"], obs["zone"]
        point = self.item["pos"]
        eta = 0.2 + max(0, dist - agent.cfg["loot"]["pickup_radius"]) / agent.mech["walk_speed"]
        danger = sum(amount for _, amount, _ in self.damage)
        sources = len({by for _, _, by in self.damage if by is not None})
        reserve = own["energy"] / max(1, own["max_energy"])
        health = own["hp"]
        value = 1.0 if self.item["kind"] == "lightning" else 0.9
        loss = danger * eta * (1 + 0.35 * max(0, sources - 1) + max(0, cross_delta))
        affordable = health * (0.09 + 0.06 * reserve) * value
        if (eta > 2.2 or loss > affordable or area_danger > 0.3 or cross_delta > 0.2
                or math.dist(point[:2], zone["center"]) > zone["radius"]):
            self.finished = True
            return False
        return True
