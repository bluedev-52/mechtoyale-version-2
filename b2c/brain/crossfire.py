"""Evaluate firing lanes at the current position and along reachable moves."""
from collections import deque
import math
import time

import numpy as np


DIRECT_WEAPONS = ("gun", "laser", "lightning")
CROSS_ANGLE = math.radians(100.0)
HIT_WINDOW = 2.0
PATH_TIMES = (0.35, 0.75, 1.2)


class Crossfire:
    def __init__(self, config):
        self.config = config
        self.hits = deque()
        self.sample = None

    def observe(self, obs):
        self.sample = None
        now = obs["t"]
        while self.hits and now - self.hits[0][0] > HIT_WINDOW:
            self.hits.popleft()
        for event in obs["events"]:
            attacker = event.get("by")
            if (event["type"] == "damaged" and attacker is not None
                    and 0 <= attacker < self.config["match"]["mechs"]):
                self.hits.append((now, attacker, event["amount"]))

    def sources(self, agent, obs):
        weapons = self.config["weapons"]
        size_per_cube = self.config["cubes"]["size_per_cube"]
        muzzle = agent.mech["muzzle_height"]
        rows = []
        for entity in obs["mechs"]:
            if entity["kind"] != "mech":
                continue
            reach = max((weapons[k]["range"] for k in entity.get("weapons", ())
                         if k in DIRECT_WEAPONS), default=0.0)
            if reach:
                x, y, z = entity["pos"]
                height = muzzle * (1.0 + size_per_cube * entity.get("cubes", 0))
                rows.append((entity["id"], (x, y, z + height), reach, 1.0))
        return rows

    def exposure(self, agent, targets, origins, reach, weights):
        delta = origins - targets[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        source, point = np.nonzero(distance <= reach[:, None])
        clear = np.zeros(distance.shape, dtype=bool)
        clear[source, point] = agent.world.los_batch(origins[source, point], targets[point], step=2.0)
        angles = np.arctan2(delta[..., 1], delta[..., 0])
        pressure = (clear * weights[:, None] / (1.0 + (distance / 100.0) ** 2)).sum(0)
        result = []
        for k in range(len(targets)):
            directions = np.sort(angles[clear[:, k], k])
            if len(directions) >= 2:
                gaps = np.diff(directions, append=directions[0] + math.tau)
                spread = math.tau - float(gaps.max())
            else:
                spread = 0.0
            count = int(len(directions))
            result.append((count, spread, float(pressure[k])))
        return result, clear, angles

    @staticmethod
    def risk(exposure):
        count, spread, pressure = exposure
        crossfire = count >= 1.5 and spread > CROSS_ANGLE
        # Group nearby values so an equivalent route keeps the base planner's choice.
        return (count >= 2.5 or crossfire, int(count + 0.5), crossfire,
                int(math.degrees(spread) / 30.0), int(pressure * 4))

    def paths(self, agent, pts, utility, veto, best, aoe):
        x, y, z = agent.pos
        me = agent.me
        stats = me["stats"]
        at = np.array([x, y])
        speed = stats.get("walk_speed", agent.mech["walk_speed"])
        accel = stats.get("ground_accel", agent.mech["ground_accel"])
        initial = np.asarray(me["vel"][:2])
        choices = [best]
        for j in range(12):
            # Use the furthest viable point in each direction to look beyond the nearest corner.
            for i in (25 + j, 13 + j, 1 + j):
                if not veto[i] and aoe[i] <= aoe[best] + 0.15:
                    choices.append(i)
                    break
        choices = list(dict.fromkeys(choices))
        steps = np.asarray(PATH_TIMES)[None, :, None]
        delta = np.asarray([pts[i] for i in choices]) - at
        dist = np.linalg.norm(delta, axis=1)
        desired = delta * (speed * np.minimum(1.0, dist / 3.0) / np.maximum(dist, 1e-6))[:, None]
        dv = desired - initial
        settle = np.linalg.norm(dv, axis=1)[:, None, None] / accel
        ramp = np.minimum(steps, settle)
        drift = steps * desired[:, None, :] - dv[:, None, :] * (
            ramp - ramp * ramp / (2 * np.maximum(settle, 1e-6)))
        xy = at + drift
        # The directive will replan once it reaches its local destination.
        lengths = np.linalg.norm(xy - at, axis=2)
        xy = at + (xy - at) * np.minimum(1.0, dist[:, None] / np.maximum(lengths, 1e-6))[..., None]
        torso = stats.get("height", agent.mech["height"]) * 0.55
        elevated = z > agent.world.ground_at(x, y) + 2.5
        zc = agent.zone["center"]
        zone_radius = agent.zone["radius"]
        start_zone_distance = math.hypot(x - zc[0], y - zc[1])
        allowed = []
        routes = []
        for i, path in zip(choices, xy):
            points = []
            valid = True
            for px, py in path:
                zz = agent.world.surface(px, py, z, stats.get("step_up", 1.2)) if elevated \
                    else agent.world.ground_at(px, py)
                if (not agent.world.in_bounds(px, py)
                        or math.hypot(px - zc[0], py - zc[1]) > max(zone_radius, start_zone_distance)
                        or (not elevated and not agent.nav.clear_line(x, y, px, py))
                        or (elevated and (z - zz > 5 or agent.world.solid_point(px, py, zz + torso)))):
                    valid = False
                points.append((px, py, zz + torso))
            allowed.append(valid)
            routes.extend(points)
        return choices, np.asarray([(x, y, z + torso), *routes]), allowed

    def select(self, agent, obs, pts, utility, veto, best, aoe):
        if not agent.me.get("grounded", True):
            return best
        rows = self.sources(agent, obs)
        if len(rows) < 2 or time.perf_counter() + 0.003 >= agent.deadline:
            return best
        choices, targets, allowed = self.paths(agent, pts, utility, veto, best, aoe)
        if len(choices) < 2 or not any(allowed[1:]):
            return best
        ids, origins, reach, _ = zip(*rows)
        damage = dict.fromkeys(ids, 0.0)
        for _, attacker, amount in self.hits:
            if attacker in damage:
                damage[attacker] += amount
        weights = 1.0 + np.array([damage[i] for i in ids]) / agent.me["max_hp"]
        velocities = {e["id"]: e["vel"] for e in obs["mechs"]}
        times = np.tile(PATH_TIMES, len(choices))
        current_origins = np.asarray(origins, dtype=float)
        velocities = np.asarray([velocities[i] for i in ids])
        predicted = current_origins[:, None, :] + velocities[:, None, :] * np.r_[0.0, times][None, :, None]
        # Ground movement can stop at a wall. In that case, keep the last known muzzle.
        latest = predicted[:, -1, :]
        reachable = agent.world.los_batch(current_origins, latest, step=2.0)
        predicted[~reachable] = current_origins[~reachable, None, :]
        nominal, lanes, bearings = self.exposure(agent, targets, predicted, np.asarray(reach), weights)
        risks = [self.risk(e) for e in nominal]
        current = risks[0]
        self.sample = {"lanes": nominal[0][0], "spread_deg": round(math.degrees(nominal[0][1]), 1),
                       "danger": current[0], "changed": False, "reduction": 0, "cover_run": False}
        if not current[0] and not any(r[0] for r in risks[1:1 + len(PATH_TIMES)]):
            return best
        if time.perf_counter() >= agent.deadline:
            return best

        # Stress-test the last position with a plausible raised muzzle from each reported
        # rival. Clear vertical travel is required. Do not take jet access as certainty.
        ends = np.arange(len(PATH_TIMES), len(targets), len(PATH_TIMES))
        takeoff = predicted[:, ends[0], :].copy()
        lifted = takeoff.copy()
        lifted[:, 2] += min(10.0, agent.mech["max_rise_speed"] * PATH_TIMES[-1])
        can_rise = agent.world.los_batch(takeoff, lifted, step=1.0)
        lifted[~can_rise] = takeoff[~can_rise]
        raised = np.broadcast_to(lifted[:, None, :], (len(ids), len(ends), 3))
        jet, _, _ = self.exposure(agent, targets[ends], raised, np.asarray(reach), weights)
        costs = []
        existing = bearings[lanes[:, 0], 0]
        for j, end in enumerate(ends):
            start = end - len(PATH_TIMES) + 1
            opened = np.nonzero(lanes[:, start:end + 1].any(axis=1) & ~lanes[:, 0])[0]
            new_sectors = 0
            for idx in opened:
                directions = bearings[idx, start:end + 1][lanes[idx, start:end + 1]]
                if len(existing) and any(
                    np.all(np.abs((yaw - existing + math.pi) % math.tau - math.pi) > math.pi / 3)
                    for yaw in directions
                ):
                    new_sectors += 1
            peak = max(r[1] for r in risks[start:end + 1])
            costs.append((peak > current[1] + 1, risks[end][0], new_sectors, risks[end][1],
                          self.risk(jet[j])[:3], peak, sum(r[1] for r in risks[start:end + 1]),
                          risks[end][3:]))
        baseline = choices.index(best)
        legal = [j for j, okay in enumerate(allowed) if okay]
        selected = min(legal, key=lambda j: (costs[j], j != baseline, -utility[choices[j]])) if legal else baseline
        end = ends[selected]
        reduction = nominal[0][0] - nominal[end][0]
        self.sample.update(changed=selected != baseline, reduction=reduction,
                           cover_run=current[0] and reduction > 0 and jet[selected][0] < nominal[0][0]
                           and nominal[end - 1][0] >= nominal[0][0])
        return choices[selected]
