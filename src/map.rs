use crate::math::{V2, V3};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::path::Path;

const HASH_STEP: f64 = 16.0;
pub const NAV_STEP: f64 = 3.0;
pub type Cell = (i32, i32);

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TerrainInfo {
    min_x: f64,
    min_y: f64,
    step: f64,
    nx: usize,
    ny: usize,
    scale: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}
impl Bounds {
    pub fn inset(&self, p: V2, r: f64) -> V2 {
        V2(
            p.0.clamp(self.min_x + r, self.max_x - r),
            p.1.clamp(self.min_y + r, self.max_y - r),
        )
    }
}

#[derive(Deserialize)]
struct WorldFile {
    bounds: Bounds,
    terrain: TerrainInfo,
    prisms: Vec<PrismFile>,
}

#[derive(Deserialize)]
struct PrismFile {
    base: f64,
    top: f64,
    ring: Vec<f64>,
    holes: Vec<Vec<f64>>,
}

struct Prism {
    base: f64,
    top: f64,
    rings: Vec<Vec<V2>>,
    min: V2,
    max: V2,
}

fn edges(ring: &[V2]) -> impl Iterator<Item = (V2, V2)> + '_ {
    ring.iter()
        .copied()
        .zip(ring.iter().copied().cycle().skip(1))
}

fn inside(p: V2, ring: &[V2]) -> bool {
    let mut odd = false;
    for (a, b) in edges(ring) {
        if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
            odd = !odd;
        }
    }
    odd
}

impl Prism {
    fn contains(&self, p: V2) -> bool {
        p.0 >= self.min.0
            && p.0 <= self.max.0
            && p.1 >= self.min.1
            && p.1 <= self.max.1
            && inside(p, &self.rings[0])
            && !self.rings[1..].iter().any(|ring| inside(p, ring))
    }

    fn intersects(&self, start: V3, end: V3) -> bool {
        let delta = end - start;
        let (lo, hi) = if delta.2.abs() < 1e-9 {
            if start.2 < self.base || start.2 > self.top {
                return false;
            }
            (0.0, 1.0)
        } else {
            let a = (self.base - start.2) / delta.2;
            let b = (self.top - start.2) / delta.2;
            (a.min(b).max(0.0), a.max(b).min(1.0))
        };
        if hi < lo {
            return false;
        }
        if self.contains((start + delta * lo).xy()) || self.contains((start + delta * hi).xy()) {
            return true;
        }
        self.rings.iter().any(|ring| {
            edges(ring).any(|(a, b)| {
                let edge = b - a;
                let divisor = delta.xy().cross(edge);
                if divisor.abs() < 1e-9 {
                    return false;
                }
                let offset = a - start.xy();
                let t = offset.cross(edge) / divisor;
                let u = offset.cross(delta.xy()) / divisor;
                (lo..=hi).contains(&t) && (0.0..=1.0).contains(&u)
            })
        })
    }
}

fn cell(p: V2, step: f64) -> Cell {
    ((p.0 / step).floor() as i32, (p.1 / step).floor() as i32)
}

pub fn nav_cell(p: V2) -> Cell {
    cell(p, NAV_STEP)
}
pub fn nav_point((x, y): Cell) -> V2 {
    V2(
        (f64::from(x) + 0.5) * NAV_STEP,
        (f64::from(y) + 0.5) * NAV_STEP,
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub z: f64,
    pub clearance: f64,
}

pub struct Arena {
    pub bounds: Bounds,
    terrain: TerrainInfo,
    samples: Vec<i16>,
    solids: Vec<Prism>,
    hash: HashMap<Cell, Vec<usize>>,
    grid: Vec<Vec<Surface>>,
    grid_min: Cell,
    grid_width: usize,
}

impl Arena {
    pub fn load(dir: &Path) -> io::Result<Self> {
        let world: WorldFile = serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(dir.join("v2/world.json"))?,
        ))?;
        let data = std::fs::read(dir.join("v2/terrain.i16"))?;
        let terrain = world.terrain;
        if terrain.nx < 2
            || terrain.ny < 2
            || terrain.step <= 0.0
            || terrain
                .nx
                .checked_mul(terrain.ny)
                .and_then(|n| n.checked_mul(2))
                != Some(data.len())
        {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "invalid terrain dimensions",
            ));
        }
        let samples = data
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        let mut hash: HashMap<Cell, Vec<usize>> = HashMap::new();
        let mut solids = Vec::new();
        for p in world.prisms {
            let rings: Vec<Vec<V2>> = std::iter::once(p.ring)
                .chain(p.holes)
                .map(|ring| ring.chunks_exact(2).map(|xy| V2(xy[0], xy[1])).collect())
                .collect();
            if rings[0].len() < 3 {
                continue;
            }
            let min = rings[0]
                .iter()
                .fold(V2(f64::INFINITY, f64::INFINITY), |a, b| {
                    V2(a.0.min(b.0), a.1.min(b.1))
                });
            let max = rings[0]
                .iter()
                .fold(V2(f64::NEG_INFINITY, f64::NEG_INFINITY), |a, b| {
                    V2(a.0.max(b.0), a.1.max(b.1))
                });
            let start = cell(min, HASH_STEP);
            let end = cell(max, HASH_STEP);
            for y in start.1..=end.1 {
                for x in start.0..=end.0 {
                    hash.entry((x, y)).or_default().push(solids.len());
                }
            }
            solids.push(Prism {
                base: p.base,
                top: p.top,
                rings,
                min,
                max,
            });
        }
        let bounds = world.bounds;
        let grid_min = nav_cell(V2(bounds.min_x, bounds.min_y));
        let grid_max = nav_cell(V2(bounds.max_x, bounds.max_y));
        let grid_width = (grid_max.0 - grid_min.0 + 1) as usize;
        let mut map = Self {
            bounds,
            terrain,
            samples,
            solids,
            hash,
            grid: Vec::new(),
            grid_min,
            grid_width,
        };
        map.grid = (grid_min.1..=grid_max.1)
            .flat_map(|y| (grid_min.0..=grid_max.0).map(move |x| (x, y)))
            .map(|key| map.surfaces(nav_point(key)))
            .collect();
        Ok(map)
    }

    pub fn ground(&self, p: V2) -> f64 {
        let t = &self.terrain;
        let x = ((p.0 - t.min_x) / t.step).clamp(0.0, (t.nx - 1) as f64);
        let y = ((p.1 - t.min_y) / t.step).clamp(0.0, (t.ny - 1) as f64);
        let i = (x as usize).min(t.nx - 2);
        let j = (y as usize).min(t.ny - 2);
        let u = x - i as f64;
        let v = y - j as f64;
        let sample = |i, j: usize| f64::from(self.samples[j * t.nx + i]) * t.scale;
        (sample(i, j) * (1.0 - u) + sample(i + 1, j) * u) * (1.0 - v)
            + (sample(i, j + 1) * (1.0 - u) + sample(i + 1, j + 1) * u) * v
    }

    pub fn ground_slope_walkable(&self, a: V2, b: V2, max_slope: f64) -> bool {
        let delta = b - a;
        let length = delta.norm();
        if length < 1e-8 {
            return true;
        }
        let t = &self.terrain;
        let crossings = |p: f64, d: f64, origin: f64| {
            if d == 0.0 {
                return (f64::INFINITY, f64::INFINITY);
            }
            let cell = (p - origin) / t.step;
            let edge = if d > 0.0 {
                cell.floor() + 1.0
            } else {
                cell.ceil() - 1.0
            };
            ((edge - cell) * t.step / d, t.step / d.abs())
        };
        let (mut tx, dx) = crossings(a.0, delta.0, t.min_x);
        let (mut ty, dy) = crossings(a.1, delta.1, t.min_y);
        let mut start = 0.0;
        let mut z0 = self.ground(a);
        while start < 1.0 {
            let end = tx.min(ty).min(1.0);
            let middle_rise = self.ground(a + delta * ((start + end) * 0.5)) - z0;
            let z1 = self.ground(a + delta * end);
            let rise = z1 - z0;
            // Within a bilinear cell the height along this segment is quadratic.
            // Its steepest uphill grade occurs at one of the two endpoints.
            let max_rise = (4.0 * middle_rise - rise).max(3.0 * rise - 4.0 * middle_rise);
            if max_rise > max_slope * length * (end - start) + 1e-8 {
                return false;
            }
            if tx <= end {
                tx += dx;
            }
            if ty <= end {
                ty += dy;
            }
            start = end;
            z0 = z1;
        }
        true
    }

    fn candidates(&self, key: Cell) -> impl Iterator<Item = &Prism> {
        self.hash
            .get(&key)
            .into_iter()
            .flatten()
            .map(|i| &self.solids[*i])
    }

    pub fn surfaces(&self, p: V2) -> Vec<Surface> {
        let ground = self.ground(p);
        let local: Vec<_> = self
            .candidates(cell(p, HASH_STEP))
            .filter(|s| s.contains(p))
            .collect();
        let mut layers = vec![ground];
        layers.extend(local.iter().map(|s| s.top).filter(|&top| top > ground));
        layers.sort_by(f64::total_cmp);
        layers.dedup_by(|a, b| (*a - *b).abs() < 0.02);
        layers
            .into_iter()
            .map(|z| {
                let clearance = local
                    .iter()
                    .filter(|s| s.top > z + 0.02)
                    .map(|s| (s.base - z).max(0.0))
                    .fold(f64::INFINITY, f64::min);
                Surface { z, clearance }
            })
            .collect()
    }

    pub fn layers(&self, key: Cell) -> &[Surface] {
        let x = key.0 - self.grid_min.0;
        let y = key.1 - self.grid_min.1;
        if x < 0 || y < 0 || x as usize >= self.grid_width {
            return &[];
        }
        self.grid
            .get(y as usize * self.grid_width + x as usize)
            .map_or(&[], Vec::as_slice)
    }

    pub fn floor_below(&self, p: V3) -> f64 {
        self.surfaces(p.xy())
            .iter()
            .filter(|s| s.z <= p.2 + 0.1)
            .map(|s| s.z)
            .fold(self.ground(p.xy()), f64::max)
    }

    pub fn clear(&self, a: V3, b: V3) -> bool {
        let delta = b - a;
        let count = (delta.xy().norm() / 4.0).ceil().max(1.0) as usize;
        if (0..=count).any(|i| {
            let p = a + delta * (i as f64 / count as f64);
            p.2 < self.ground(p.xy()) - 0.03
        }) {
            return false;
        }

        let mut at = cell(a.xy(), HASH_STEP);
        let end = cell(b.xy(), HASH_STEP);
        let axis = |coord: f64, d: f64, index: i32| {
            if d == 0.0 {
                return (0, f64::INFINITY, f64::INFINITY);
            }
            let direction = if d > 0.0 { 1 } else { -1 };
            let boundary = f64::from(index + i32::from(d > 0.0)) * HASH_STEP;
            (direction, (boundary - coord) / d, HASH_STEP / d.abs())
        };
        let (sx, mut tx, dx) = axis(a.0, delta.0, at.0);
        let (sy, mut ty, dy) = axis(a.1, delta.1, at.1);
        loop {
            if self.candidates(at).any(|p| p.intersects(a, b)) {
                return false;
            }
            if at == end {
                return true;
            }
            if at.1 == end.1 || (at.0 != end.0 && tx <= ty) {
                at.0 += sx;
                tx += dx;
            } else {
                at.1 += sy;
                ty += dy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn loads_official_arena_and_can_query_1000_rays() {
        let t = Instant::now();
        let map = Arena::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("arena")).unwrap();
        assert!(t.elapsed() < Duration::from_millis(2500));
        assert_eq!(map.solids.len(), 259);
        assert!(!map.layers(nav_cell(V2(0.0, 0.0))).is_empty());
        let t = Instant::now();
        let mut clear = 0;
        for i in 0..1000 {
            let y = (f64::from(i) * 0.41).sin() * 160.0;
            clear += usize::from(map.clear(V3(-125.0, y, 9.0), V3(55.0, y + 14.0, 9.0)));
        }
        assert!((1..1000).contains(&clear));
        eprintln!("1000 arena rays: {:?}", t.elapsed());
    }

    #[test]
    fn bridge_height_and_hole_block_only_solid_segments() {
        let square = |r: f64| vec![V2(-r, -r), V2(r, -r), V2(r, r), V2(-r, r)];
        let p = Prism {
            base: 10.0,
            top: 12.0,
            min: V2(-10.0, -10.0),
            max: V2(10.0, 10.0),
            rings: vec![square(10.0), square(2.0)],
        };
        assert!(!p.intersects(V3(-20.0, 0.0, 7.0), V3(20.0, 0.0, 7.0)));
        assert!(!p.intersects(V3(0.0, 0.0, 7.0), V3(0.0, 0.0, 20.0)));
        assert!(p.intersects(V3(-20.0, 0.0, 11.0), V3(20.0, 0.0, 11.0)));
        assert!(p.intersects(V3(5.0, 0.0, 7.0), V3(5.0, 0.0, 20.0)));
    }
}
