"""Static map model (protocol.md §5 and §11), shared by the bot and the local simulator.

Everything is rasterized once onto the terrain's own grid: cell (j, i) is the point
x = minX + i*step, y = minY + j*step. Prisms become two solid layers (ground-standing
buildings/stadium, and bridges) stored as [lo, hi) height ranges; tree crowns become a
soft canopy layer that only matters for vision.
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

EMPTY_LO = np.float32(1e6)
EMPTY_HI = np.float32(-1e6)


def _flat_xy(flat: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(flat, dtype=np.float64)
    return a[0::2], a[1::2]


def _even_odd(cx: np.ndarray, cy: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Point-in-polygon (even-odd) for every (cy[r], cx[c]) grid point."""
    inside = np.zeros((len(cy), len(cx)), dtype=bool)
    n = len(xs)
    for k in range(n):
        x0, y0 = xs[k], ys[k]
        x1, y1 = xs[(k + 1) % n], ys[(k + 1) % n]
        if y0 == y1:
            continue
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        r0 = int(np.searchsorted(cy, lo, "left"))
        r1 = int(np.searchsorted(cy, hi, "left"))
        if r1 <= r0:
            continue
        yy = cy[r0:r1]
        xint = x0 + (yy - y0) * (x1 - x0) / (y1 - y0)
        inside[r0:r1] ^= cx[None, :] < xint[:, None]
    return inside


class World:
    NEAR_R = 8  # cells; footprint queries with a smaller radius skip cells far from any prism

    def __init__(self, map_dir: str, files: Optional[List[str]] = None):
        files = files or ["v2/world.json", "v2/terrain.i16"]
        wfile = next(f for f in files if f.endswith(".json"))
        with open(os.path.join(map_dir, wfile)) as fh:
            w = json.load(fh)
        t = w["terrain"]
        self.minx, self.miny = float(t["minX"]), float(t["minY"])
        self.step = float(t["step"])
        self.nx, self.ny = int(t["nx"]), int(t["ny"])
        tfile = os.path.join(map_dir, os.path.dirname(wfile), t["file"])
        raw = np.fromfile(tfile, dtype="<i2", count=self.nx * self.ny)
        self.ground = raw.reshape(self.ny, self.nx).astype(np.float32) * np.float32(t.get("scale", 0.01))
        b = w["bounds"]
        self.bounds = (float(b["minX"]), float(b["minY"]), float(b["maxX"]), float(b["maxY"]))
        rules = w.get("rules", {})
        dens = float(rules.get("canopyDensity", 0.35))
        blk = float(rules.get("canopyBlockT", 0.25))
        # canopy transmittance exp(-density * metres of crown) below blockT hides the target
        self.canopy_block_len = math.log(1.0 / blk) / dens if dens > 0 and 0 < blk < 1 else 1e9

        shape = (self.ny, self.nx)
        self.s_lo = np.full(shape, EMPTY_LO, np.float32)
        self.s_hi = np.full(shape, EMPTY_HI, np.float32)
        self.b_lo = np.full(shape, EMPTY_LO, np.float32)
        self.b_hi = np.full(shape, EMPTY_HI, np.float32)
        self.c_lo = np.full(shape, EMPTY_LO, np.float32)
        self.c_hi = np.full(shape, EMPTY_HI, np.float32)
        self.prisms = w["prisms"]
        for p in self.prisms:
            mask, j0, i0 = self._poly_mask(p["ring"], p.get("holes") or [])
            if mask is None or not mask.any():
                continue
            lo, hi = (self.b_lo, self.b_hi) if p["kind"] == "bridge" else (self.s_lo, self.s_hi)
            sl = (slice(j0, j0 + mask.shape[0]), slice(i0, i0 + mask.shape[1]))
            lo[sl] = np.where(mask, np.minimum(lo[sl], np.float32(p["base"])), lo[sl])
            hi[sl] = np.where(mask, np.maximum(hi[sl], np.float32(p["top"])), hi[sl])
        self._rasterize_trees(w.get("trees") or {})
        solid = (self.s_hi > EMPTY_HI) | (self.b_hi > EMPTY_HI)
        rows = solid.copy()
        for d in range(1, self.NEAR_R + 1):
            rows[d:, :] |= solid[:-d, :]
            rows[:-d, :] |= solid[d:, :]
        near = rows.copy()
        for d in range(1, self.NEAR_R + 1):
            near[:, d:] |= rows[:, :-d]
            near[:, :-d] |= rows[:, d:]
        self._near = near.ravel().astype(np.uint8).tobytes()
        self._g = self.ground.ravel()
        self._slo, self._shi = self.s_lo.ravel(), self.s_hi.ravel()
        self._blo, self._bhi = self.b_lo.ravel(), self.b_hi.ravel()
        self._clo, self._chi = self.c_lo.ravel(), self.c_hi.ravel()

    # ------------------------------------------------------------------ rasterizing
    def _poly_mask(self, ring, holes):
        xs, ys = _flat_xy(ring)
        if len(xs) < 3:
            return None, 0, 0
        i0 = max(0, int(math.floor((xs.min() - self.minx) / self.step)))
        i1 = min(self.nx - 1, int(math.ceil((xs.max() - self.minx) / self.step)))
        j0 = max(0, int(math.floor((ys.min() - self.miny) / self.step)))
        j1 = min(self.ny - 1, int(math.ceil((ys.max() - self.miny) / self.step)))
        if i1 < i0 or j1 < j0:
            return None, 0, 0
        cx = self.minx + np.arange(i0, i1 + 1) * self.step
        cy = self.miny + np.arange(j0, j1 + 1) * self.step
        mask = _even_odd(cx, cy, xs, ys)
        for h in holes:
            hx, hy = _flat_xy(h)
            if len(hx) >= 3:
                mask &= ~_even_odd(cx, cy, hx, hy)
        return mask, j0, i0

    def _rasterize_trees(self, trees) -> None:
        fields = trees.get("fields") or []
        data = trees.get("data") or []
        if not fields or not data:
            return
        f = {k: n for n, k in enumerate(fields)}
        nf = len(fields)
        arr = np.asarray(data, dtype=np.float64).reshape(-1, nf)
        for row in arr:
            z = row[f["z"]]
            cx = row[f["crownX"]] if "crownX" in f else row[f["x"]]
            cy = row[f["crownY"]] if "crownY" in f else row[f["y"]]
            r = row[f["crownR"]]
            lo = z + row[f["crownBase"]]
            hi = z + row[f["height"]]
            if r <= 0 or hi <= lo:
                continue
            i0 = max(0, int(math.floor((cx - r - self.minx) / self.step)))
            i1 = min(self.nx - 1, int(math.ceil((cx + r - self.minx) / self.step)))
            j0 = max(0, int(math.floor((cy - r - self.miny) / self.step)))
            j1 = min(self.ny - 1, int(math.ceil((cy + r - self.miny) / self.step)))
            if i1 < i0 or j1 < j0:
                continue
            gx = self.minx + np.arange(i0, i1 + 1) * self.step - cx
            gy = self.miny + np.arange(j0, j1 + 1) * self.step - cy
            m = gx[None, :] ** 2 + gy[:, None] ** 2 <= r * r
            sl = (slice(j0, j1 + 1), slice(i0, i1 + 1))
            self.c_lo[sl] = np.where(m, np.minimum(self.c_lo[sl], lo), self.c_lo[sl])
            self.c_hi[sl] = np.where(m, np.maximum(self.c_hi[sl], hi), self.c_hi[sl])

    # ------------------------------------------------------------------ indexing
    def ij(self, x: float, y: float) -> Tuple[int, int]:
        i = int(round((x - self.minx) / self.step))
        j = int(round((y - self.miny) / self.step))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def flat_idx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        i = np.clip(np.rint((x - self.minx) / self.step), 0, self.nx - 1).astype(np.intp)
        j = np.clip(np.rint((y - self.miny) / self.step), 0, self.ny - 1).astype(np.intp)
        return j * self.nx + i

    def ground_at(self, x: float, y: float) -> float:
        fx = (x - self.minx) / self.step
        fy = (y - self.miny) / self.step
        i = min(max(int(math.floor(fx)), 0), self.nx - 2)
        j = min(max(int(math.floor(fy)), 0), self.ny - 2)
        tx = min(max(fx - i, 0.0), 1.0)
        ty = min(max(fy - j, 0.0), 1.0)
        g = self.ground
        a = float(g[j, i]) * (1 - tx) + float(g[j, i + 1]) * tx
        b = float(g[j + 1, i]) * (1 - tx) + float(g[j + 1, i + 1]) * tx
        return a * (1 - ty) + b * ty

    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.bounds
        return x0 + margin <= x <= x1 - margin and y0 + margin <= y <= y1 - margin

    # ------------------------------------------------------------------ queries
    def surface(self, x: float, y: float, z: float, step_up: float) -> float:
        """Highest walkable surface under (x, y) that a mech whose feet are at z can stand on."""
        i, j = self.ij(x, y)
        s = self.ground_at(x, y)
        hi = float(self.s_hi[j, i])
        if s < hi <= z + step_up:
            s = hi
        bh = float(self.b_hi[j, i])
        if s < bh <= z + step_up:
            s = bh
        return s

    def roof_top(self, x: float, y: float) -> float:
        i, j = self.ij(x, y)
        return max(float(self.ground[j, i]), float(self.s_hi[j, i]), float(self.b_hi[j, i]))

    def solid_point(self, x: float, y: float, z: float) -> bool:
        i, j = self.ij(x, y)
        if z < self.ground[j, i]:
            return True
        if self.s_lo[j, i] <= z < self.s_hi[j, i]:
            return True
        return bool(self.b_lo[j, i] <= z < self.b_hi[j, i])

    def _samples(self, A: np.ndarray, B: np.ndarray, step: float):
        d = B - A
        L = np.sqrt((d * d).sum(1))
        n = int(min(max(2, math.ceil(float(L.max()) / step)), 600))
        t = np.arange(1, n) / n
        P = A[:, None, :] + t[None, :, None] * d[:, None, :]
        k = self.flat_idx(P[..., 0], P[..., 1])
        z = P[..., 2]
        blk = z < self._g[k]
        blk |= (z >= self._slo[k]) & (z < self._shi[k])
        blk |= (z >= self._blo[k]) & (z < self._bhi[k])
        return blk, k, z, L, n, t

    def los_batch(self, A, B, vision: bool = False, step: float = 1.0) -> np.ndarray:
        """True where the straight segment A[n]->B[n] crosses no building, bridge or terrain.
        With vision=True dense canopy also blocks (shots ignore trees)."""
        A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
        B = np.asarray(B, dtype=np.float64).reshape(-1, 3)
        if len(A) == 0:
            return np.zeros(0, dtype=bool)
        blk, k, z, L, n, _ = self._samples(A, B, step)
        ok = ~blk.any(1)
        if vision:
            inside = (z >= self._clo[k]) & (z < self._chi[k])
            ok &= inside.sum(1) * (L / n) < self.canopy_block_len
        return ok

    def los(self, a, b, vision: bool = False) -> bool:
        return bool(self.los_batch([a], [b], vision)[0])

    def first_block(self, A, B, step: float = 0.5) -> np.ndarray:
        """Fraction along A->B of the first solid sample (1.0 when the segment is clear)."""
        A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
        B = np.asarray(B, dtype=np.float64).reshape(-1, 3)
        if len(A) == 0:
            return np.zeros(0)
        blk, _, _, _, n, t = self._samples(A, B, step)
        anyb = blk.any(1)
        first = np.argmax(blk, axis=1)
        out = np.ones(len(A))
        out[anyb] = t[first[anyb]]
        return out

    def disc_window(self, x: float, y: float, r: float):
        i0 = max(0, int(math.floor((x - r - self.minx) / self.step)))
        i1 = min(self.nx - 1, int(math.ceil((x + r - self.minx) / self.step)))
        j0 = max(0, int(math.floor((y - r - self.miny) / self.step)))
        j1 = min(self.ny - 1, int(math.ceil((y + r - self.miny) / self.step)))
        dx = self.minx + np.arange(i0, i1 + 1) * self.step - x
        dy = self.miny + np.arange(j0, j1 + 1) * self.step - y
        m = dx[None, :] ** 2 + dy[:, None] ** 2 <= r * r
        return (slice(j0, j1 + 1), slice(i0, i1 + 1)), m, dx, dy

    def far_from_solid(self, x: float, y: float, r: float) -> bool:
        if r >= (self.NEAR_R - 1) * self.step:
            return False
        i, j = self.ij(x, y)
        return not self._near[j * self.nx + i]

    def wall_contact(self, x: float, y: float, z: float, r: float, h: float, step_up: float):
        """Solid cells inside the mech's footprint that it cannot step onto.
        Returns (blocked, nx, ny) with (nx, ny) the push-out direction."""
        if self.far_from_solid(x, y, r):
            return False, 0.0, 0.0
        sl, m, dx, dy = self.disc_window(x, y, r)
        zlo, zhi = z + step_up, z + h
        wall = ((self.s_hi[sl] > zlo) & (self.s_lo[sl] < zhi)) | ((self.b_hi[sl] > zlo) & (self.b_lo[sl] < zhi))
        wall &= m
        if not wall.any():
            return False, 0.0, 0.0
        jj, ii = np.nonzero(wall)
        nx = -float(dx[ii].sum())
        ny = -float(dy[jj].sum())
        L = math.hypot(nx, ny)
        if L < 1e-9:
            return True, 0.0, 0.0
        return True, nx / L, ny / L

    def ceiling_hit(self, x: float, y: float, z0: float, z1: float, r: float, h: float) -> bool:
        """Would rising from feet z0 to z1 push the head (z+h) into an underside?"""
        if self.far_from_solid(x, y, r):
            return False
        sl, m, _, _ = self.disc_window(x, y, r)
        a, b = z0 + h - 1e-3, z1 + h
        hit = ((self.s_lo[sl] >= a) & (self.s_lo[sl] < b)) | ((self.b_lo[sl] >= a) & (self.b_lo[sl] < b))
        return bool((hit & m).any())

    def edge_support(self, x: float, y: float, z0: float, z1: float, r: float):
        """While falling from z0 to z1: the highest top inside the footprint crossed on the way
        down (a roof edge we overhang), and the direction to slide off it."""
        if self.far_from_solid(x, y, r):
            return None, 0.0, 0.0
        sl, m, dx, dy = self.disc_window(x, y, r)
        tops = np.maximum(self.s_hi[sl], self.b_hi[sl])
        cross = m & (tops <= z0 + 1e-3) & (tops > z1)
        if not cross.any():
            return None, 0.0, 0.0
        top = float(tops[cross].max())
        jj, ii = np.nonzero(cross)
        nx, ny = -float(dx[ii].sum()), -float(dy[jj].sum())
        L = math.hypot(nx, ny) or 1.0
        return top, nx / L, ny / L
