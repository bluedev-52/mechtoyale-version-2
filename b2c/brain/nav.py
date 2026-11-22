"""Ground navigation for a mech: walkable grid, wavefront distance fields and a cover map.

The walkable grid is coarse (default 4 m) and sampled from a 1 m "blocked" mask that is the
ground-level obstacles grown by the mech's radius, so a cell is walkable when a mech can stand
with its centre there. Distance fields are computed with a vectorized wavefront that can be
spread over several ticks (NavField.run with a time budget).
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np

SQ2 = math.sqrt(2.0)
DIRS8 = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
         (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


def _pair(d: int, n: int) -> Tuple[slice, slice]:
    """(dst, src) slices along one axis such that dst index + d == src index."""
    if d > 0:
        return slice(0, n - d), slice(d, n)
    if d < 0:
        return slice(-d, n), slice(0, n + d)
    return slice(0, n), slice(0, n)


def shift(mask: np.ndarray, dj: int, di: int, fill=False) -> np.ndarray:
    """out[j, i] = mask[j + dj, i + di], `fill` outside."""
    out = np.full_like(mask, fill)
    (dj_s, sj_s), (di_s, si_s) = _pair(dj, mask.shape[0]), _pair(di, mask.shape[1])
    out[dj_s, di_s] = mask[sj_s, si_s]
    return out


def dilate_disc(mask: np.ndarray, r_cells: float) -> np.ndarray:
    R = int(math.ceil(r_cells))
    out = mask.copy()
    for dj in range(-R, R + 1):
        for di in range(-R, R + 1):
            if (dj or di) and dj * dj + di * di <= r_cells * r_cells:
                out |= shift(mask, dj, di)
    return out


class NavGrid:
    def __init__(self, world, mech: dict, res: float = 4.0, clearance: Optional[float] = None,
                 cover_range: float = 160.0, cover_dirs: int = 16):
        self.w = world
        step_up = float(mech.get("step_up", 1.2))
        height = float(mech.get("height", 9.0))
        radius = float(mech.get("radius", 2.5))
        clearance = radius + 0.5 if clearance is None else clearance
        g = world.ground
        obs = ((world.s_hi > g + step_up) & (world.s_lo < g + height)) | \
              ((world.b_hi > g + step_up) & (world.b_lo < g + height))
        gy, gx = np.gradient(g, world.step)
        obs |= np.hypot(gx, gy) > math.tan(math.radians(float(mech.get("max_slope_deg", 50))))
        x0, y0, x1, y1 = world.bounds
        xs = world.minx + np.arange(world.nx) * world.step
        ys = world.miny + np.arange(world.ny) * world.step
        obs |= (xs[None, :] < x0 + 1) | (xs[None, :] > x1 - 1) | (ys[:, None] < y0 + 1) | (ys[:, None] > y1 - 1)
        self.obstacle1 = obs
        self.blocked1 = dilate_disc(obs, clearance / world.step)
        self._blocked_bytes = self.blocked1.ravel().astype(np.uint8).tobytes()
        k = max(1, int(round(res / world.step)))
        self.k = k
        self.res = k * world.step
        self.NY, self.NX = world.ny // k, world.nx // k
        self.jj = np.arange(self.NY) * k + k // 2
        self.ii = np.arange(self.NX) * k + k // 2
        self.walk = ~self.blocked1[np.ix_(self.jj, self.ii)]
        # cost_full[d, J, I]: metres to step from (J, I) to (J + dj, I + di), inf if not walkable
        self.cost_full = np.full((8, self.NY, self.NX), np.inf, np.float32)
        for d, (dj, di, L) in enumerate(DIRS8):
            (dj_s, sj_s), (di_s, si_s) = _pair(dj, self.NY), _pair(di, self.NX)
            ok = self.walk[dj_s, di_s] & self.walk[sj_s, si_s]
            for s in range(1, k):
                mj = self.jj[dj_s] + dj * s
                mi = self.ii[di_s] + di * s
                ok &= ~self.blocked1[np.ix_(mj, mi)]
            self.cost_full[d][dj_s, di_s] = np.where(ok, np.float32(L * self.res), np.float32(np.inf))
        self.cx = (world.minx + self.ii * world.step).astype(np.float32)
        self.cy = (world.miny + self.jj * world.step).astype(np.float32)
        self._cover(cover_range, cover_dirs)

    # ------------------------------------------------------------------ cover
    def _cover(self, maxd: float, ndirs: int) -> None:
        """ray_free[d, J, I]: metres from the cell to the first eye-height building in direction d."""
        w = self.w
        g = w.ground
        tall = (w.s_hi > g + 6.0) & (w.s_lo < g + 6.0)
        k = self.k
        t = tall[: self.NY * k, : self.NX * k].reshape(self.NY, k, self.NX, k).any(axis=(1, 3))
        self.tall = t
        K = max(1, int(maxd / self.res))
        free = np.empty((ndirs, self.NY, self.NX), np.float32)
        for d in range(ndirs):
            a = 2 * math.pi * d / ndirs
            c, s = math.cos(a), math.sin(a)
            hit = np.full((self.NY, self.NX), K, np.float32)
            for st in range(1, K + 1):
                sh = shift(t, int(round(st * s)), int(round(st * c)))
                newly = sh & (hit == K)
                if newly.any():
                    hit[newly] = st
            free[d] = hit
        self.cover_dirs = ndirs
        self.cover_max = K * self.res
        self.ray_free = free * self.res
        self.openness = free.mean(0) / K
        canopy = (w.c_lo < g + 8.0) & (w.c_hi > g + 6.0)
        self.canopy = canopy[: self.NY * k, : self.NX * k].reshape(self.NY, k, self.NX, k).mean(axis=(1, 3))

    def covered_from(self, x: float, y: float, tx: float, ty: float) -> bool:
        """Cheap test: is there an eye-height building between (x, y) and (tx, ty)?"""
        J, I = self.cell(x, y)
        d = math.hypot(tx - x, ty - y)
        a = math.atan2(ty - y, tx - x) % (2 * math.pi)
        di = int(round(a / (2 * math.pi) * self.cover_dirs)) % self.cover_dirs
        return float(self.ray_free[di, J, I]) < d - self.res

    # ------------------------------------------------------------------ cells
    def cell(self, x: float, y: float) -> Tuple[int, int]:
        w = self.w
        I = int(round(((x - w.minx) / w.step - self.k // 2) / self.k))
        J = int(round(((y - w.miny) / w.step - self.k // 2) / self.k))
        return min(max(J, 0), self.NY - 1), min(max(I, 0), self.NX - 1)

    def center(self, J: int, I: int) -> Tuple[float, float]:
        w = self.w
        return (w.minx + (I * self.k + self.k // 2) * w.step, w.miny + (J * self.k + self.k // 2) * w.step)

    def center_batch(self, J: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = self.w
        return (w.minx + (I * self.k + self.k // 2) * w.step,
                w.miny + (J * self.k + self.k // 2) * w.step)

    def walkable_at(self, x: float, y: float) -> bool:
        i, j = self.w.ij(x, y)
        return not self._blocked_bytes[j * self.w.nx + i]

    def nearest_walk(self, x: float, y: float, rmax: int = 8) -> Optional[Tuple[int, int]]:
        J, I = self.cell(x, y)
        if self.walk[J, I]:
            return J, I
        j0, j1 = max(0, J - rmax), min(self.NY, J + rmax + 1)
        i0, i1 = max(0, I - rmax), min(self.NX, I + rmax + 1)
        sub = self.walk[j0:j1, i0:i1]
        if not sub.any():
            return None
        jj, ii = np.nonzero(sub)
        cx = self.w.minx + ((ii + i0) * self.k + self.k // 2) * self.w.step
        cy = self.w.miny + ((jj + j0) * self.k + self.k // 2) * self.w.step
        n = int(np.argmin((cx - x) ** 2 + (cy - y) ** 2))
        return int(jj[n] + j0), int(ii[n] + i0)

    def clear_line(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Can a mech walk the straight line? A blocked prefix is tolerated (we may already be
        hugging a wall); after the first free sample every sample must stay free."""
        w = self.w
        b = self._blocked_bytes
        nx, ny = w.nx, w.ny
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(d / w.step) + 1)
        fx0, fy0 = (x0 - w.minx) / w.step, (y0 - w.miny) / w.step
        dfx, dfy = (x1 - x0) / w.step / (n - 1), (y1 - y0) / w.step / (n - 1)
        seen_free = False
        for s in range(n):
            i = int(fx0 + dfx * s + 0.5)
            j = int(fy0 + dfy * s + 0.5)
            if i < 0 or j < 0 or i >= nx or j >= ny:
                return False
            if b[j * nx + i]:
                if seen_free:
                    return False
            else:
                seen_free = True
        return seen_free

    def circle_mask(self, cx: float, cy: float, r: float) -> np.ndarray:
        xs = self.w.minx + self.ii * self.w.step
        ys = self.w.miny + self.jj * self.w.step
        return (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2 <= r * r


class NavField:
    """Shortest walking distance (metres) from every cell of a window to the nearest goal cell.

    The window defaults to the whole grid; a small window around the mech gives a cheap local
    field (e.g. "nearest cover")."""

    def __init__(self, nav: NavGrid, goal: np.ndarray, window: Optional[Tuple[int, int, int, int]] = None):
        self.nav = nav
        self.j0, self.j1, self.i0, self.i1 = window or (0, nav.NY, 0, nav.NX)
        sl = (slice(self.j0, self.j1), slice(self.i0, self.i1))
        g = goal[sl] if goal.shape == nav.walk.shape else goal
        self.D = np.where(g & nav.walk[sl], 0.0, np.inf).astype(np.float32)
        h, w = self.D.shape
        self.steps = []
        for d, (dj, di, _) in enumerate(DIRS8):
            (dj_s, sj_s), (di_s, si_s) = _pair(dj, h), _pair(di, w)
            cost = nav.cost_full[d][sl][dj_s, di_s]
            self.steps.append((dj_s, di_s, sj_s, si_s, cost))
        self.done = not bool(np.isfinite(self.D).any())
        self.iters = 0

    def run(self, budget_s: Optional[float] = None, max_iters: int = 100000) -> bool:
        t0 = time.perf_counter()
        D = self.D
        while not self.done and self.iters < max_iters:
            before = D.copy()
            for _ in range(4):
                for dj_s, di_s, sj_s, si_s, cost in self.steps:
                    dst = D[dj_s, di_s]
                    np.minimum(dst, D[sj_s, si_s] + cost, out=dst)
                self.iters += 1
            if np.array_equal(before, D):
                self.done = True
            if budget_s is not None and time.perf_counter() - t0 > budget_s:
                break
        return self.done

    def at(self, J: int, I: int) -> float:
        if self.j0 <= J < self.j1 and self.i0 <= I < self.i1:
            return float(self.D[J - self.j0, I - self.i0])
        return math.inf

    def dist(self, x: float, y: float) -> float:
        c = self.nav.nearest_walk(x, y)
        if c is None:
            return float("inf")
        cx, cy = self.nav.center(*c)
        return self.at(*c) + math.hypot(cx - x, cy - y)

    def steer(self, x: float, y: float, look: int = 4) -> Optional[Tuple[float, float]]:
        """Unit direction to walk to go down the field, or None at the goal / unreachable."""
        nav = self.nav
        c = nav.nearest_walk(x, y)
        if c is None:
            return None
        J, I = c
        if self.at(J, I) == 0.0:
            return None
        j0, j1 = max(self.j0, J - look), min(self.j1, J + look + 1)
        i0, i1 = max(self.i0, I - look), min(self.i1, I + look + 1)
        if j1 <= j0 or i1 <= i0:
            return None
        sub = self.D[j0 - self.j0:j1 - self.j0, i0 - self.i0:i1 - self.i0]
        jj, ii = np.nonzero(np.isfinite(sub))
        if len(jj) == 0:
            return None
        cx = nav.w.minx + ((ii + i0) * nav.k + nav.k // 2) * nav.w.step
        cy = nav.w.miny + ((jj + j0) * nav.k + nav.k // 2) * nav.w.step
        score = sub[jj, ii] + np.hypot(cx - x, cy - y)
        order = np.argsort(score)
        for n in order[:6]:
            tx, ty = float(cx[n]), float(cy[n])
            if math.hypot(tx - x, ty - y) < 0.5:
                continue
            if nav.clear_line(x, y, tx, ty):
                L = math.hypot(tx - x, ty - y)
                return (tx - x) / L, (ty - y) / L
        cx0, cy0 = nav.center(J, I)
        L = math.hypot(cx0 - x, cy0 - y)
        if L > 0.5:
            return (cx0 - x) / L, (cy0 - y) / L
        return None
