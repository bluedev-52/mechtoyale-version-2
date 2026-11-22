#!/usr/bin/env python3
"""Measure JSONL round trips and exercise high-load observations in the match sandbox."""

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check.check import Bot, Schema


def run_command(image, debug):
    flags = [
        "docker", "run", "--rm", "-i",
        "--network", "none", "--cpus", "1",
        "--memory", "512m", "--memory-swap", "512m",
        "--pids-limit", "128", "--cap-drop", "ALL",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m",
        "--user", "65534:65534", "-v", f"{ROOT / 'arena'}:/arena:ro",
    ]
    if debug:
        flags += ["-e", "DEBUG_DRAW=1"]
    return [*flags, image]


def dense_observation(source, hello, i):
    """Stress the parser, tracks, projectile predictor and weapon solvers."""
    config = hello["config"]
    tick_step = config["match"]["decide_every"]
    dt = tick_step / config["match"]["tick_hz"]
    own = dict(source["self"], weapons=list(source["self"]["weapons"]))
    x, y, z = own["pos"]
    slot = hello["slot"]
    tick = source["tick"] + (i + 1) * tick_step
    now = source["t"] + (i + 1) * dt
    kind = ("mortar", "laser", "lightning")[i % 3]
    own["weapons"][1] = {
        "kind": kind, "cooldown": 0, "stats": config["weapons"][kind],
        "charge": 0.55 if kind == "laser" else None,
        "ammo": None, "reload": None,
    }
    if i % 11 == 0:
        own["weapons"][1] = None
    mechs = []
    for j in range(15):
        angle = j * math.tau / 15 + now * 0.06
        radius = 25 + 6 * j
        mechs.append({
            "id": j if j < slot else j + 1, "kind": "mech",
            "pos": [x + radius * math.cos(angle), y + radius * math.sin(angle), z],
            "vel": [-radius * 0.06 * math.sin(angle), radius * 0.06 * math.cos(angle), 0],
            "aim": [angle - math.pi, 0], "hp": None if j % 4 == 0 else 450 + 65 * j,
            "max_hp": 1400, "cubes": j % 6, "weapons": ["gun", "laser", "mortar"][0:1 + j % 3],
            "charging": j % 4 == 1, "future_field": True,
        })
    projectiles = []
    for j in range(40):
        angle = j * math.tau / 40
        mortar = j % 5 == 0
        distance = 17 + j * 2
        vx, vy = math.cos(angle), math.sin(angle)
        projectiles.append({
            "id": 700 + j, "kind": "mortar" if mortar else "gun",
            "pos": [x + vx * distance, y + vy * distance, z + (32 if mortar else 5)],
            "vel": [-vx * (50 if mortar else 250), -vy * (50 if mortar else 250), -8 if mortar else 0],
        })
    sounds = [
        {"kind": "gun", "bearing": j * math.tau / 30, "dist": ("near", "mid", "far")[j % 3],
         "age": j * 0.002} for j in range(30)
    ]
    return dict(source, tick=tick, t=now, alive=16, self=own,
                mechs=mechs, projectiles=projectiles, sounds=sounds,
                beams=[], events=[{"type": "damaged", "amount": 20, "weapon": "zone",
                                   "by": None, "bearing": None, "rear": None}],
                drops=[], offer=None, future_field={"new": 1})


def benchmark(args):
    with gzip.open(ROOT / "check/stream.jsonl.gz", "rt") as fixture:
        stream = [json.loads(line) for line in fixture]
    hello, source, end = stream[0], stream[1], stream[-1]
    hello["map"]["dir"] = "/arena"
    hello["future_field"] = {"enabled": True}
    schema = Schema(ROOT / "mechfight.schema.json")
    command = run_command(args.image, args.debug)
    if args.binary:
        command = [str(args.binary.resolve())]
        hello["map"]["dir"] = str(ROOT / "arena")
    bot = Bot(command)
    tick_budget = hello["config"]["bot"]["tick_budget_ms"]
    max_line = hello["config"]["bot"]["max_line_bytes"]

    def receive(deadline, message_type):
        result = bot.next(deadline)
        if result is None or result[1] is None:
            raise RuntimeError(f"missing {message_type} before deadline")
        arrival, raw = result
        if len(raw.rstrip(b"\n")) > max_line:
            raise RuntimeError(f"{message_type} exceeds line budget")
        message = json.loads(raw)
        errors = schema.errors(message, message_type)
        if errors:
            raise RuntimeError("; ".join(errors))
        return arrival, message

    try:
        start = time.perf_counter()
        bot.send(hello)
        arrival, _ = receive(start + hello["config"]["bot"]["hello_budget_ms"] / 1000, "Ready")
        ready_ms = (arrival - start) * 1000
        samples = []
        last_tick = -1
        for i in range(args.samples):
            obs = dense_observation(source, hello, i)
            start = time.perf_counter()
            bot.send(obs)
            arrival, action = receive(start + tick_budget / 1000, "Action")
            if action["tick"] != obs["tick"]:
                raise RuntimeError(f"expected tick {obs['tick']}, got {action['tick']}")
            last_tick = action["tick"]
            samples.append((arrival - start) * 1000)

        batch = [dense_observation(source, hello, i) for i in range(args.samples, args.samples + 10)]
        bot.send_many(batch)
        pending = {obs["tick"] for obs in batch}
        deadline = time.perf_counter() + tick_budget / 1000
        answered = 0
        while last_tick != batch[-1]["tick"]:
            _, action = receive(deadline, "Action")
            if action["tick"] not in pending or action["tick"] <= last_tick:
                raise RuntimeError(f"invalid queued tick {action['tick']} after {last_tick}")
            last_tick = action["tick"]
            answered += 1
        if answered == len(batch):
            raise RuntimeError("bot responded to every queued observation")

        retreat = dense_observation(source, hello, args.samples + 10)
        retreat["self"].update(hp=40, in_zone=True)
        retreat["self"]["weapons"][1] = {"kind": "laser", "cooldown": 0, "charge": 0.55}
        bot.send(retreat)
        _, action = receive(time.perf_counter() + tick_budget / 1000, "Action")
        if action["tick"] != retreat["tick"] or any(action["fire"]):
            raise RuntimeError("expected both triggers released during regeneration")
        if args.debug and not action.get("debug"):
            raise RuntimeError("DEBUG_DRAW is enabled but the action contains no drawing")

        bot.send(end)
        bot.proc.stdin.close()
        if bot.proc.wait(timeout=2) != 0:
            raise RuntimeError("bot did not exit successfully")
    finally:
        if bot.proc.poll() is None:
            bot.proc.kill()
            bot.proc.wait()

    samples.sort()

    def percentile(p):
        return samples[max(0, math.ceil(p * len(samples)) - 1)]

    print(f"ready {ready_ms:.1f} ms | {len(samples)} dense observations | "
          f"p50 {percentile(.50):.2f} ms | p95 {percentile(.95):.2f} ms | "
          f"p99 {percentile(.99):.2f} ms | max {samples[-1]:.2f} ms")
    print("Newest of 10 queued observations, optional fields, regeneration and clean exit: PASS")
    return percentile(.99) < 15 and samples[-1] < 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="mechtoyale:v2")
    parser.add_argument("--binary", type=Path, help="run a native binary outside Docker")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--debug", action="store_true", help="also validate optional debug drawings")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.binary and args.debug:
        parser.error("--debug requires Docker")
    if not benchmark(args):
        raise SystemExit("latency targets exceeded (p99 < 15 ms, max < 30 ms)")


if __name__ == "__main__":
    main()
