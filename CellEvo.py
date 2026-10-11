import pygame
import random
import math
import sys
import csv
import pickle
import os
from collections import deque
import numpy as np

# ===================== NUMBA JIT (with graceful fallback) =====================
try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("[perf] Numba JIT compilation enabled")
except ImportError:
    HAS_NUMBA = False
    print("[perf] Numba not available — using pure NumPy fallback")


# ===================== PERFORMANCE: JIT Distance Kernels =====================

if HAS_NUMBA:
    @njit(fastmath=True)
    def _pairwise_distances_numba(pos):
        """Compute NxN pairwise Euclidean distance matrix (float32)."""
        n = pos.shape[0]
        out = np.empty((n, n), dtype=np.float32)
        for i in range(n):
            xi = pos[i, 0]
            yi = pos[i, 1]
            for j in range(i + 1, n):
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                d = math.sqrt(dx * dx + dy * dy)
                out[i, j] = d
                out[j, i] = d
            out[i, i] = 0.0
        return out

    @njit(fastmath=True)
    def _cross_distances_numba(cell_pos, other_pos):
        """Compute N x M distance matrix between two position sets."""
        n = cell_pos.shape[0]
        m = other_pos.shape[0]
        out = np.empty((n, m), dtype=np.float32)
        for i in range(n):
            xi = cell_pos[i, 0]
            yi = cell_pos[i, 1]
            for j in range(m):
                dx = xi - other_pos[j, 0]
                dy = yi - other_pos[j, 1]
                out[i, j] = math.sqrt(dx * dx + dy * dy)
        return out

    @njit(fastmath=True)
    def _genetic_distance_matrix_numba(genomes, morph_size):
        """Compute NxN mean absolute genetic distance on morph genes."""
        n = genomes.shape[0]
        out = np.empty((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                total = 0.0
                for k in range(morph_size):
                    total += abs(genomes[i, k] - genomes[j, k])
                d = total / morph_size
                out[i, j] = d
                out[j, i] = d
            out[i, i] = 0.0
        return out

    @njit(fastmath=True)
    def _batch_hypot_numba(cx, cy, fx, fy):
        """Compute scalar distance from (cx,cy) to each (fx,fy) in arrays."""
        n = fx.shape[0]
        out = np.empty(n, dtype=np.float32)
        for j in range(n):
            dx = cx - fx[j]
            dy = cy - fy[j]
            out[j] = math.sqrt(dx * dx + dy * dy)
        return out

else:
    # Pure NumPy fallbacks (still vectorized, no Python loops)
    def _pairwise_distances_numba(pos):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)

    def _cross_distances_numba(cell_pos, other_pos):
        diff = cell_pos[:, np.newaxis, :] - other_pos[np.newaxis, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)

    def _genetic_distance_matrix_numba(genomes, morph_size):
        diff = np.abs(genomes[:, np.newaxis, :morph_size] - genomes[np.newaxis, :, :morph_size])
        return np.mean(diff, axis=2).astype(np.float32)

    def _batch_hypot_numba(cx, cy, fx, fy):
        dx = cx - fx
        dy = cy - fy
        return np.sqrt(dx * dx + dy * dy).astype(np.float32)


# ===================== SPATIAL HASH GRID =====================
class SpatialHashGrid:
    """Fixed-cell-size spatial hash for O(1) approximate neighbor queries.

    Programming Pattern: Spatial Partitioning (optimization pattern).
    Reduces neighbor search from O(N) to O(k) where k = avg neighbors per cell.
    """

    __slots__ = ['cell_size', 'cols', 'rows', 'buckets']

    def __init__(self, cell_size=120, world_w=1024, world_h=768):
        self.cell_size = cell_size
        self.cols = max(1, world_w // cell_size + 1)
        self.rows = max(1, world_h // cell_size + 1)
        self.buckets = {}

    def clear(self):
        self.buckets.clear()

    def _key(self, x, y):
        return (int(x) // self.cell_size, int(y) // self.cell_size)

    def insert(self, idx, x, y):
        key = self._key(x, y)
        bucket = self.buckets.get(key)
        if bucket is None:
            self.buckets[key] = [idx]
        else:
            bucket.append(idx)

    def query_radius(self, x, y, radius):
        """Return list of indices potentially within radius of (x,y)."""
        r_cells = int(radius // self.cell_size) + 1
        cx, cy = int(x) // self.cell_size, int(y) // self.cell_size
        result = []
        buckets = self.buckets
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                bucket = buckets.get((cx + dx, cy + dy))
                if bucket is not None:
                    result.extend(bucket)
        return result

    def query_range(self, x, y, half_w=120, half_h=120):
        """Return list of indices in a rectangular region."""
        cx, cy = int(x) // self.cell_size, int(y) // self.cell_size
        hw = max(1, int(half_w // self.cell_size) + 1)
        hh = max(1, int(half_h // self.cell_size) + 1)
        result = []
        buckets = self.buckets
        for dx in range(-hw, hw + 1):
            for dy in range(-hh, hh + 1):
                bucket = buckets.get((cx + dx, cy + dy))
                if bucket is not None:
                    result.extend(bucket)
        return result


# ===================== SIMULATION CACHE =====================
class SimulationCache:
    """Pre-computed per-frame data to eliminate redundant O(N²) loops.

    Programming Pattern: Cache (structural pattern).
    Computes distance matrices and genetic distances once per frame using
    vectorized NumPy/Numba, then provides O(1) lookup for all cell methods.
    """

    __slots__ = [
        'cell_pos', 'food_pos', 'food_energy',
        'hazard_pos', 'hazard_radii',
        'obstacle_pos', 'obstacle_radii',
        'cell_dist', 'food_dist', 'hazard_dist', 'obstacle_dist',
        'genetic_dist', 'cell_dx', 'cell_dy', 'food_dx', 'food_dy',
        'is_predator', 'n_cells', 'n_food',
        'spatial', 'food_spatial',
        'valid'
    ]

    def __init__(self):
        self.valid = False
        self.cell_pos = np.empty((0, 2), dtype=np.float32)
        self.food_pos = np.empty((0, 2), dtype=np.float32)
        self.food_energy = np.empty(0, dtype=np.float32)
        self.hazard_pos = np.empty((0, 2), dtype=np.float32)
        self.hazard_radii = np.empty(0, dtype=np.float32)
        self.obstacle_pos = np.empty((0, 2), dtype=np.float32)
        self.obstacle_radii = np.empty(0, dtype=np.float32)
        self.cell_dist = np.empty((0, 0), dtype=np.float32)
        self.food_dist = np.empty((0, 0), dtype=np.float32)
        self.hazard_dist = np.empty((0, 0), dtype=np.float32)
        self.obstacle_dist = np.empty((0, 0), dtype=np.float32)
        self.genetic_dist = np.empty((0, 0), dtype=np.float32)
        self.cell_dx = np.empty((0, 0), dtype=np.float32)
        self.cell_dy = np.empty((0, 0), dtype=np.float32)
        self.food_dx = np.empty((0, 0), dtype=np.float32)
        self.food_dy = np.empty((0, 0), dtype=np.float32)
        self.is_predator = np.empty(0, dtype=bool)
        self.n_cells = 0
        self.n_food = 0
        self.spatial = SpatialHashGrid(cell_size=120, world_w=1024, world_h=768)
        self.food_spatial = SpatialHashGrid(cell_size=80, world_w=1024, world_h=768)

    def build(self, cells, food_list, hazards, obstacles):
        """Build all caches from current simulation state. Call once per frame."""
        n = len(cells)
        m = len(food_list)

        # Cell positions
        self.cell_pos = np.empty((n, 2), dtype=np.float32)
        self.is_predator = np.empty(n, dtype=bool)
        genomes = np.empty((n, MORPH_SIZE), dtype=np.float32)
        for i, c in enumerate(cells):
            self.cell_pos[i, 0] = c.x
            self.cell_pos[i, 1] = c.y
            self.is_predator[i] = c.is_predator
            genomes[i] = c.genome[:MORPH_SIZE]
        self.n_cells = n

        # Food positions
        self.food_pos = np.empty((m, 2), dtype=np.float32)
        self.food_energy = np.empty(m, dtype=np.float32)
        for i, f in enumerate(food_list):
            self.food_pos[i, 0] = f[0]
            self.food_pos[i, 1] = f[1]
            self.food_energy[i] = f[2]
        self.n_food = m

        # Hazard / obstacle positions
        nh = len(hazards)
        no = len(obstacles)
        self.hazard_pos = np.empty((nh, 2), dtype=np.float32)
        self.hazard_radii = np.empty(nh, dtype=np.float32)
        for i, h in enumerate(hazards):
            self.hazard_pos[i, 0] = h.x
            self.hazard_pos[i, 1] = h.y
            self.hazard_radii[i] = h.radius
        self.obstacle_pos = np.empty((no, 2), dtype=np.float32)
        self.obstacle_radii = np.empty(no, dtype=np.float32)
        for i, o in enumerate(obstacles):
            self.obstacle_pos[i, 0] = o.x
            self.obstacle_pos[i, 1] = o.y
            self.obstacle_radii[i] = o.radius

        # --- Vectorized distance matrices (Numba JIT or NumPy) ---
        if n > 0:
            self.cell_dist = _pairwise_distances_numba(self.cell_pos)
            self.genetic_dist = _genetic_distance_matrix_numba(genomes, MORPH_SIZE)
            # Direction vectors for angle computation
            self.cell_dx = self.cell_pos[:, np.newaxis, 0] - self.cell_pos[np.newaxis, :, 0]
            self.cell_dy = self.cell_pos[:, np.newaxis, 1] - self.cell_pos[np.newaxis, :, 1]
        else:
            self.cell_dist = np.empty((0, 0), dtype=np.float32)
            self.genetic_dist = np.empty((0, 0), dtype=np.float32)
            self.cell_dx = np.empty((0, 0), dtype=np.float32)
            self.cell_dy = np.empty((0, 0), dtype=np.float32)

        if n > 0 and m > 0:
            self.food_dist = _cross_distances_numba(self.cell_pos, self.food_pos)
            self.food_dx = self.cell_pos[:, np.newaxis, 0] - self.food_pos[np.newaxis, :, 0]
            self.food_dy = self.cell_pos[:, np.newaxis, 1] - self.food_pos[np.newaxis, :, 1]
        else:
            self.food_dist = np.empty((0, 0), dtype=np.float32)
            self.food_dx = np.empty((0, 0), dtype=np.float32)
            self.food_dy = np.empty((0, 0), dtype=np.float32)

        if n > 0 and nh > 0:
            self.hazard_dist = _cross_distances_numba(self.cell_pos, self.hazard_pos)
        else:
            self.hazard_dist = np.empty((0, 0), dtype=np.float32)

        if n > 0 and no > 0:
            self.obstacle_dist = _cross_distances_numba(self.cell_pos, self.obstacle_pos)
        else:
            self.obstacle_dist = np.empty((0, 0), dtype=np.float32)

        # --- Spatial hash grids ---
        self.spatial.clear()
        for i in range(n):
            self.spatial.insert(i, cells[i].x, cells[i].y)
        self.food_spatial.clear()
        for i in range(m):
            self.food_spatial.insert(i, food_list[i][0], food_list[i][1])

        self.valid = True

    def neighbors_of(self, idx, radius=120.0):
        """Get spatial-hash filtered neighbors within radius."""
        candidates = self.spatial.query_radius(
            self.cell_pos[idx, 0], self.cell_pos[idx, 1], radius)
        dists = self.cell_dist[idx]
        return [(j, dists[j]) for j in candidates if j != idx and dists[j] < radius]

    def food_near(self, idx, radius):
        """Get food items within radius of cell idx."""
        if self.n_food == 0:
            return []
        candidates = self.food_spatial.query_radius(
            self.cell_pos[idx, 0], self.cell_pos[idx, 1], radius)
        dists = self.food_dist[idx]
        return [(j, dists[j]) for j in candidates if dists[j] < radius]


# ===================== SURFACE CACHE (Flyweight Pattern) =====================
class SurfaceCache:
    """Caches frequently-used pygame surfaces to avoid per-frame allocation.

    Programming Pattern: Flyweight + Cache.
    Glow circles, gradient overlays, and UI panels are cached by (color, size)
    and reused across frames, eliminating thousands of Surface allocations.
    """

    __slots__ = ['_glow_cache', '_circle_cache', '_max_size']

    def __init__(self, max_cache=200):
        self._glow_cache = {}
        self._circle_cache = {}
        self._max_size = max_cache

    def get_glow(self, color, radius):
        """Get or create a radial gradient glow surface."""
        radius = max(2, int(radius))
        key = (color[:3], radius)
        surf = self._glow_cache.get(key)
        if surf is not None:
            return surf
        size = radius * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        steps = min(radius, 12)
        for i in range(steps, 0, -1):
            frac = i / steps
            r = int(radius * frac)
            alpha = int(55 * (1 - frac))
            pygame.draw.circle(surf, (*color[:3], alpha), (radius, radius), r)
        if len(self._glow_cache) < self._max_size:
            self._glow_cache[key] = surf
        return surf

    def get_circle_alpha(self, color, radius, alpha):
        """Get or create a solid alpha circle surface."""
        radius = max(1, int(radius))
        alpha = max(0, min(255, int(alpha)))
        key = (color[:3], radius, alpha)
        surf = self._circle_cache.get(key)
        if surf is not None:
            return surf
        size = radius * 2 + 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color[:3], alpha), (radius + 1, radius + 1), radius)
        if len(self._circle_cache) < self._max_size:
            self._circle_cache[key] = surf
        return surf

    def clear(self):
        """Clear caches (call on window resize)."""
        self._glow_cache.clear()
        self._circle_cache.clear()


# ===================== CONSTANTS =====================
WIDTH, HEIGHT = 1024, 768
FPS = 60
FPS_CAP = 60

INITIAL_ENERGY = 100
ENERGY_DRAIN_BASE = 0.08
REPRODUCE_THRESHOLD = 200
MATE_DIST = 50
MATE_ENERGY_THRESHOLD = 150
MUTATION_RATE_BASE = 0.15
MUTATION_AMOUNT = 0.3

FOOD_COUNT = 80
FOOD_ENERGY = 20
FOOD_RADIUS = 4
CLUSTER_COUNT = 6
CLUSTER_RADIUS = 80

HIDDEN_SIZE = 8
INPUT_SIZE = 18
OUTPUT_SIZE = 11

PERCEPTION_BASE = 150

HAZARD_COUNT = 3
HAZARD_RADIUS = 60
HAZARD_DRAIN = 0.5

OBSTACLE_COUNT = 5
OBSTACLE_RADIUS = 30

ELITE_FRACTION = 0.1
ELITE_INTERVAL = 50

RITUAL_DURATION = 20
GROUP_SIZE_THRESHOLD = 3
GROUP_DEFENSE_BONUS = 0.5

DAY_LENGTH = 3600
NIGHT_DRAIN_MULT = 1.5
NIGHT_FOOD_MULT = 0.3
NIGHT_VISION_MULT = 0.6

MAX_LIFESPAN_BASE = 3600
SENESCENCE_START = 0.7

PHEROMONE_EMIT_RATE = 0.5
PHEROMONE_DECAY = 0.98
PHEROMONE_GRID_SIZE = 20
PHEROMONE_MAX = 10.0

BIOME_COUNT = 4

SOFT_CAP = 200
HARD_CAP = 300

ZOOM_MIN = 0.5
ZOOM_MAX = 3.0
ZOOM_STEP = 0.1

TRAIL_LENGTH = 30

# ===== COLOR PALETTE =====
BACKGROUND = (12, 12, 24)
FOOD_COLOR = (46, 204, 113)
FOOD_COLOR_HIGH = (255, 215, 0)
OBSTACLE_COLOR = (86, 90, 100)

BIOME_COLORS = {
    0: (34, 85, 51),    # Plains
    1: (16, 52, 36),    # Forest
    2: (120, 90, 51),   # Desert
    3: (22, 42, 82),    # Water
}
BIOME_NAMES = {0: "Plains", 1: "Forest", 2: "Desert", 3: "Water"}
BIOME_ENERGY_MULT = {0: 1.0, 1: 1.1, 2: 1.3, 3: 0.7}
BIOME_SPEED_MULT = {0: 1.0, 1: 0.85, 2: 0.9, 3: 0.7}
BIOME_FOOD_MULT = {0: 1.0, 1: 1.3, 2: 0.5, 3: 0.2}

SPECIES_PALETTE = [
    (0, 191, 255),   (255, 99, 72),   (46, 213, 115),  (255, 71, 177),
    (255, 234, 0),   (0, 230, 230),   (155, 89, 255),   (255, 159, 67),
    (88, 214, 141),  (116, 185, 255), (241, 196, 15),  (214, 48, 49),
    (9, 132, 227),   (253, 167, 223), (38, 166, 154),   (175, 122, 197),
]

LOG_INTERVAL = 60
LOG_FILENAME = "evolution_kin_groups_log.csv"
SAVE_FILENAME = "evolution_kin_groups.pkl"

global_time = 0


# ===== DRAWING UTILITIES =====
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def lerp_color(c1, c2, t):
    t = clamp(t, 0, 1)
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))

def brighten(color, amount=40):
    return (min(255, color[0] + amount),
            min(255, color[1] + amount),
            min(255, color[2] + amount))

def darken(color, amount=40):
    return (max(0, color[0] - amount),
            max(0, color[1] - amount),
            max(0, color[2] - amount))

def draw_gradient_circle(surface, color, center, radius, surf_cache=None):
    """Draw a soft radial gradient circle (glow effect), optionally using cache."""
    size = radius * 2
    if size < 2:
        return
    if surf_cache is not None:
        glow = surf_cache.get_glow(color, radius)
    else:
        glow = pygame.Surface((size, size), pygame.SRCALPHA)
        steps = min(radius, 12)
        for i in range(steps, 0, -1):
            frac = i / steps
            r = int(radius * frac)
            alpha = int(55 * (1 - frac))
            pygame.draw.circle(glow, (*color[:3], alpha), (radius, radius), r)
    surface.blit(glow, (center[0] - radius, center[1] - radius))

def draw_rounded_rect_alpha(surface, rect, color, alpha, radius=10, border_color=None, border_alpha=180):
    x, y, w, h = rect
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color[:3], alpha), (0, 0, w, h), border_radius=radius)
    highlight = pygame.Surface((w, h // 3), pygame.SRCALPHA)
    pygame.draw.rect(highlight, (255, 255, 255, 18), (0, 0, w, h // 3), border_radius=radius)
    s.blit(highlight, (0, 0))
    if border_color:
        pygame.draw.rect(s, (*border_color[:3], border_alpha), (0, 0, w, h), 1, border_radius=radius)
    surface.blit(s, (x, y))


# ===================== PARTICLE SYSTEM (Object Pool Pattern) =====================
class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'color', 'life', 'max_life', 'radius', 'active']

    def __init__(self):
        self.active = False
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.color = (255, 255, 255)
        self.life = 0
        self.max_life = 1
        self.radius = 2

    def reset(self, x, y, vx, vy, color, life, radius=2):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life
        self.radius = radius
        self.active = True


class ParticleSystem:
    """Object Pool pattern for particles — avoids per-frame allocation/GC.

    Programming Pattern: Object Pool.
    Pre-allocates a fixed pool of Particle objects; emit() activates from pool,
    update() recycles dead particles. Eliminates list append/append churn.
    """

    __slots__ = ['pool', '_free_indices', '_next_free']

    def __init__(self, max_particles=500):
        self.pool = [Particle() for _ in range(max_particles)]
        self._free_indices = list(range(max_particles))
        self._next_free = 0

    def _acquire(self):
        if self._next_free >= len(self._free_indices):
            # Pool exhausted — find a dead particle to recycle
            for p in self.pool:
                if not p.active:
                    return p
            return None  # truly exhausted
        idx = self._free_indices[self._next_free]
        self._next_free += 1
        p = self.pool[idx]
        if p.active:
            # This slot is still alive; find another
            for i in range(self._next_free, len(self._free_indices)):
                idx = self._free_indices[i]
                if not self.pool[idx].active:
                    self._next_free = i + 1
                    return self.pool[idx]
            return None
        return p

    def emit_burst(self, x, y, color, count=8, speed=2.0, life=25, radius=2):
        for _ in range(count):
            p = self._acquire()
            if p is None:
                break
            angle = random.uniform(0, 2 * math.pi)
            spd = random.uniform(0.5, speed)
            c = brighten(color, random.randint(-20, 20))
            c = tuple(clamp(v, 0, 255) for v in c)
            p.reset(x, y, math.cos(angle) * spd, math.sin(angle) * spd,
                    c, random.randint(life // 2, life), radius)

    def emit_food_sparkle(self, x, y, color=(255, 255, 150), count=4):
        for _ in range(count):
            p = self._acquire()
            if p is None:
                break
            angle = random.uniform(0, 2 * math.pi)
            spd = random.uniform(0.3, 1.2)
            p.reset(x, y, math.cos(angle) * spd, math.sin(angle) * spd,
                    color, random.randint(10, 20), 1)

    def emit_death(self, x, y, color, count=12, speed=2.5, life=35):
        self.emit_burst(x, y, color, count, speed, life, 2)

    def update(self):
        decay = 0.96
        for p in self.pool:
            if not p.active:
                continue
            p.x += p.vx
            p.y += p.vy
            p.vx *= decay
            p.vy *= decay
            p.life -= 1
            if p.life <= 0:
                p.active = False

    def draw(self, screen, surf_cache=None):
        for p in self.pool:
            if not p.active:
                continue
            frac = p.life / p.max_life
            alpha = int(255 * frac)
            r = max(1, int(p.radius * frac))
            if surf_cache is not None:
                s = surf_cache.get_circle_alpha(p.color, r + 1, alpha)
                screen.blit(s, (int(p.x - r - 1), int(p.y - r - 1)))
            else:
                s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
                c = (*p.color[:3], clamp(alpha, 0, 255))
                pygame.draw.circle(s, c, (r + 1, r + 1), r)
                if r > 1:
                    pygame.draw.circle(s, (*brighten(p.color, 60), clamp(alpha // 2, 0, 255)),
                                       (r + 1, r + 1), max(1, r // 2))
                screen.blit(s, (int(p.x - r - 1), int(p.y - r - 1)))

    @property
    def active_count(self):
        return sum(1 for p in self.pool if p.active)


# ---------- Biome Map ----------
class BiomeMap:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.grid_w = width // 10
        self.grid_h = height // 10
        self.grid = np.zeros((self.grid_w, self.grid_h), dtype=np.int8)
        self._generate()

    def _generate(self):
        gw, gh = self.grid.shape
        seeds_x = np.random.randint(0, gw, size=BIOME_COUNT)
        seeds_y = np.random.randint(0, gh, size=BIOME_COUNT)
        # Vectorized Voronoi using broadcasting
        xs = np.arange(gw).reshape(-1, 1)
        ys = np.arange(gh).reshape(1, -1)
        best_dist = np.full((gw, gh), np.inf, dtype=np.float32)
        best_id = np.zeros((gw, gh), dtype=np.int8)
        for i in range(BIOME_COUNT):
            dx = xs - seeds_x[i]
            dy = ys - seeds_y[i]
            d = dx * dx + dy * dy
            mask = d < best_dist
            best_dist = np.where(mask, d, best_dist)
            best_id = np.where(mask, i, best_id)
        self.grid = best_id

    def biome_at(self, x, y):
        gx = int(x) // 10
        gy = int(y) // 10
        gx = max(0, min(self.grid.shape[0] - 1, gx))
        gy = max(0, min(self.grid.shape[1] - 1, gy))
        return self.grid[gx, gy]

    def pre_render(self):
        """Pre-render biome with soft boundary blending."""
        surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                b = self.grid[x, y]
                color = BIOME_COLORS.get(b, (30, 30, 30))
                pygame.draw.rect(surf, (*color, 50), (x * 10, y * 10, 10, 10))
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                b = self.grid[x, y]
                neighbors = set()
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        if self.grid[nx, ny] != b:
                            neighbors.add(self.grid[nx, ny])
                for nb in neighbors:
                    nc = BIOME_COLORS.get(nb, (30, 30, 30))
                    blend = lerp_color(BIOME_COLORS.get(b, (30,30,30)), nc, 0.4)
                    pygame.draw.rect(surf, (*blend, 25), (x * 10, y * 10, 10, 10))
        return surf


# ---------- Day/Night ----------
def time_of_day(t):
    return (math.sin(t / DAY_LENGTH * 2 * math.pi - math.pi / 2) + 1) / 2

def daylight_factor(t):
    return time_of_day(t)

def night_overlay_alpha(t):
    return int(160 * (1 - daylight_factor(t)))


# ---------- Temperature & background ----------
def temperature_at(x, y, t=None):
    if t is None:
        t = global_time
    tx = math.sin(x / 120.0 + t * 0.001) * 0.5 + 0.5
    ty = math.sin(y / 120.0 + 1.2 + t * 0.0015) * 0.5 + 0.5
    day_mod = 0.7 + 0.3 * daylight_factor(t)
    return (tx + ty) / 2.0 * day_mod

def background_color_at(x, y):
    r = int(30 + 80 * (0.5 + 0.5 * math.sin(x / 200.0)))
    g = int(50 + 100 * (0.5 + 0.5 * math.sin(y / 150.0 + 1.0)))
    b = int(20 + 60 * (0.5 + 0.5 * math.cos((x + y) / 180.0)))
    return (r, g, b)


# ---------- Pheromone Grid ----------
class PheromoneGrid:
    def __init__(self, width, height):
        self.w = width // PHEROMONE_GRID_SIZE
        self.h = height // PHEROMONE_GRID_SIZE
        self.layers = np.zeros((3, self.w, self.h), dtype=np.float32)

    def emit(self, x, y, layer=0, amount=1.0):
        gx = int(x) // PHEROMONE_GRID_SIZE
        gy = int(y) // PHEROMONE_GRID_SIZE
        if 0 <= gx < self.w and 0 <= gy < self.h:
            self.layers[layer, gx, gy] = min(PHEROMONE_MAX, self.layers[layer, gx, gy] + amount)

    def sense_at(self, x, y, layer=0, radius=2):
        gx = int(x) // PHEROMONE_GRID_SIZE
        gy = int(y) // PHEROMONE_GRID_SIZE
        x_lo = max(0, gx - radius)
        x_hi = min(self.w, gx + radius + 1)
        y_lo = max(0, gy - radius)
        y_hi = min(self.h, gy + radius + 1)
        if x_lo >= x_hi or y_lo >= y_hi:
            return 0.0
        total = np.sum(self.layers[layer, x_lo:x_hi, y_lo:y_hi])
        count = (x_hi - x_lo) * (y_hi - y_lo)
        return total / count if count > 0 else 0.0

    def sense_gradient(self, x, y, layer=0):
        gx = int(x) // PHEROMONE_GRID_SIZE
        gy = int(y) // PHEROMONE_GRID_SIZE
        val = self.sense_at(x, y, layer, 1)
        if val < 0.01:
            return None
        best_angle = 0
        best_val = 0
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            sx = x + math.cos(rad) * 30
            sy = y + math.sin(rad) * 30
            sv = self.sense_at(sx, sy, layer, 1)
            if sv > best_val:
                best_val = sv
                best_angle = rad
        return best_angle if best_val > val else None

    def decay(self):
        self.layers *= PHEROMONE_DECAY

    def draw(self, screen, camera_x, camera_y, zoom):
        colors = [(0, 200, 255), (255, 60, 60), (46, 204, 113)]
        gs = PHEROMONE_GRID_SIZE
        for layer in range(3):
            threshold = self.layers[layer] > 0.1
            indices = np.argwhere(threshold)
            for gx, gy in indices:
                val = self.layers[layer, gx, gy]
                alpha = min(80, int(val * 12))
                c = colors[layer]
                rx = int(gx * gs * zoom - camera_x)
                ry = int(gy * gs * zoom - camera_y)
                rs = int(gs * zoom)
                if -rs < rx < WIDTH and -rs < ry < HEIGHT:
                    s = pygame.Surface((rs, rs), pygame.SRCALPHA)
                    s.fill((*c, alpha))
                    screen.blit(s, (rx, ry))


# ---------- RNN ----------
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

class RNN:
    def __init__(self, weights=None, hidden_state=None):
        if weights is None:
            self.W_in = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
            self.W_hid = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.5
            self.b_hid = np.random.randn(HIDDEN_SIZE) * 0.5
            self.W_out = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
            self.b_out = np.random.randn(OUTPUT_SIZE) * 0.5
        else:
            self.W_in, self.W_hid, self.b_hid, self.W_out, self.b_out = weights
        self.hidden = np.zeros(HIDDEN_SIZE) if hidden_state is None else hidden_state

    def forward(self, inputs):
        x = np.array(inputs)
        self.hidden = np.tanh(np.dot(x, self.W_in) + np.dot(self.hidden, self.W_hid) + self.b_hid)
        output = np.tanh(np.dot(self.hidden, self.W_out) + self.b_out)
        return output

    def mutate(self, rate, amount):
        """Vectorized mutation — replaces nested Python loop with numpy ops.

        Programming Pattern: Vectorization (performance pattern).
        Uses numpy random mask + gaussian noise instead of Python for-loop
        over every weight element.
        """
        for arr in [self.W_in, self.W_hid, self.b_hid, self.W_out, self.b_out]:
            mask = np.random.random(arr.shape) < rate
            noise = np.random.normal(0, amount, arr.shape)
            arr += mask * noise
            np.clip(arr, -5, 5, out=arr)

    def get_weights(self):
        return (self.W_in.copy(), self.W_hid.copy(), self.b_hid.copy(),
                self.W_out.copy(), self.b_out.copy())

    def reset_hidden(self):
        self.hidden = np.zeros(HIDDEN_SIZE)


# ---------- Genome ----------
MORPH_SIZE = 18

def random_genome():
    morph = [random.random() for _ in range(MORPH_SIZE)]
    rnn = RNN()
    flat_weights = []
    for arr in [rnn.W_in, rnn.W_hid, rnn.b_hid, rnn.W_out, rnn.b_out]:
        flat_weights.extend(arr.flatten().tolist())
    return morph + flat_weights

def genome_to_phenotype(genome):
    radius = 5 + genome[0] * 10
    speed = 1 + genome[1] * 3
    r_base, g_base, b_base = genome[2], genome[3], genome[4]
    is_predator = genome[5] >= 0.5
    temp_gene = genome[6]
    perception = 50 + genome[7] * 200
    mut_rate = 0.01 + genome[8] * 0.49
    planning_depth = int(genome[9] * 5)
    camouflage = genome[10] if not is_predator else 0.0
    predator_vision = genome[11] if is_predator else 0.0
    parasitic_tendency = genome[12]
    kin_sacrifice_tendency = genome[13]
    group_cohesion = genome[14]
    ritual_enthusiasm = genome[15]
    lifespan_gene = int(MAX_LIFESPAN_BASE * (0.5 + genome[16] * 1.5))
    biome_preference = genome[17]
    flat_weights = genome[MORPH_SIZE:]
    rnn = RNN()
    idx = 0
    for arr in [rnn.W_in, rnn.W_hid, rnn.b_hid, rnn.W_out, rnn.b_out]:
        size = arr.size
        arr_flat = np.array(flat_weights[idx:idx + size]).reshape(arr.shape)
        arr[:] = arr_flat
        idx += size
    return (radius, speed, (r_base, g_base, b_base), is_predator, temp_gene, perception,
            mut_rate, planning_depth, camouflage, predator_vision, parasitic_tendency,
            kin_sacrifice_tendency, group_cohesion, ritual_enthusiasm,
            lifespan_gene, biome_preference, rnn)

def mutate_genome(genome):
    new_genome = genome[:]
    for i in range(MORPH_SIZE):
        if random.random() < MUTATION_RATE_BASE:
            delta = (random.random() * 2 - 1) * MUTATION_AMOUNT
            new_genome[i] += delta
            new_genome[i] = max(0, min(1, new_genome[i]))
    return new_genome

def crossover_genome(g1, g2):
    child = []
    for a, b in zip(g1, g2):
        child.append(random.choice([a, b]))
    return child

def genetic_distance(g1, g2):
    dist = 0.0
    for i in range(MORPH_SIZE):
        dist += abs(g1[i] - g2[i])
    return dist / MORPH_SIZE


# ---------- Species color clustering ----------
def compute_species_colors(cells, threshold=0.18):
    if not cells:
        return
    colors_assigned = {}
    color_idx = 0
    for cell in cells:
        assigned = False
        for rep_genome, rep_color in colors_assigned.items():
            if genetic_distance(cell.genome, rep_genome) < threshold:
                cell.species_color = rep_color
                assigned = True
                break
        if not assigned:
            c = SPECIES_PALETTE[color_idx % len(SPECIES_PALETTE)]
            colors_assigned[cell.genome] = c
            cell.species_color = c
            color_idx += 1


# ---------- Hazard, Obstacle, FoodCluster ----------
class Hazard:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, screen, surf_cache=None):
        cx, cy = int(self.x), int(self.y)
        pulse = 0.7 + 0.3 * math.sin(global_time * 0.05)
        r = int(self.radius * pulse)
        draw_gradient_circle(screen, (220, 40, 40), (cx, cy), r, surf_cache)
        inner_r = int(r * (0.5 + 0.5 * math.sin(global_time * 0.08)))
        s = pygame.Surface((inner_r * 2 + 4, inner_r * 2 + 4), pygame.SRCALPHA)
        alpha = int(80 + 40 * math.sin(global_time * 0.1))
        pygame.draw.circle(s, (255, 80, 60, alpha), (inner_r + 2, inner_r + 2), inner_r, 2)
        screen.blit(s, (cx - inner_r - 2, cy - inner_r - 2))
        s2 = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s2, (180, 20, 20, 45), (r, r), r)
        screen.blit(s2, (cx - r, cy - r))
        s3 = pygame.Surface((20, 20), pygame.SRCALPHA)
        f = pygame.font.SysFont("Arial", 16, bold=True)
        txt = f.render("!", True, (255, 100, 80))
        s3.blit(txt, (2, 0))
        screen.blit(s3, (cx - 10, cy - 10))

    def shrink(self, amount):
        self.radius = max(5, self.radius - amount)


class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.bumps = []
        for _ in range(random.randint(5, 9)):
            a = random.uniform(0, 2 * math.pi)
            d = random.uniform(0.4, 0.9)
            r = random.uniform(0.3, 0.6)
            self.bumps.append((a, d, r))

    def draw(self, screen, surf_cache=None):
        cx, cy = int(self.x), int(self.y)
        r = self.radius
        shadow_s = pygame.Surface((r * 2 + 10, r * 2 + 10), pygame.SRCALPHA)
        pygame.draw.circle(shadow_s, (0, 0, 0, 35), (r + 5, r + 7), r + 2)
        screen.blit(shadow_s, (cx - r - 5, cy - r - 7))
        pygame.draw.circle(screen, OBSTACLE_COLOR, (cx, cy), r)
        for a, d, br in self.bumps:
            bx = cx + int(math.cos(a) * r * d)
            by = cy + int(math.sin(a) * r * d)
            br_int = int(r * br)
            c = brighten(OBSTACLE_COLOR, random.randint(-5, 15))
            pygame.draw.circle(screen, c, (bx, by), br_int)
        hl_s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(hl_s, (255, 255, 255, 25), (int(r * 0.7), int(r * 0.7)), int(r * 0.6))
        screen.blit(hl_s, (cx - r, cy - r))
        pygame.draw.circle(screen, darken(OBSTACLE_COLOR, 30), (cx, cy), r, 2)


class FoodCluster:
    def __init__(self):
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.dx = random.uniform(-0.3, 0.3)
        self.dy = random.uniform(-0.3, 0.3)
        self.food_type = random.choice(['normal', 'high'])

    def update(self):
        self.x += self.dx
        self.y += self.dy
        if self.x < 0 or self.x > WIDTH:
            self.dx *= -1
        if self.y < 0 or self.y > HEIGHT:
            self.dy *= -1
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

    def generate_food(self, food_list, count_per_cluster):
        for _ in range(count_per_cluster):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.gauss(0, CLUSTER_RADIUS / 2)
            x = self.x + math.cos(angle) * dist
            y = self.y + math.sin(angle) * dist
            x = max(0, min(WIDTH, x))
            y = max(0, min(HEIGHT, y))
            energy = FOOD_ENERGY if self.food_type == 'normal' else FOOD_ENERGY * 2
            food_list.append((x, y, energy))


# ---------- Cell Class ----------
class Cell:

    def __init__(self, x, y, genome=None, energy=None, generation=0):
        self.x = x
        self.y = y
        if genome is None:
            genome = random_genome()
        self.genome = genome
        (self.radius, self.speed, self.base_color, self.is_predator,
         self.temp_gene, self.perception, self.mut_rate, self.planning_depth,
         self.camouflage, self.predator_vision, self.parasitic_tendency,
         self.kin_sacrifice_tendency, self.group_cohesion, self.ritual_enthusiasm,
         self.lifespan, self.biome_preference, self.brain) = genome_to_phenotype(genome)
        self.energy = energy if energy is not None else INITIAL_ENERGY
        self.angle = random.uniform(0, 2 * math.pi)
        self.mate_cooldown = 0
        self.speed_mod = 1.0
        self.turn = 0.0
        self.reproduce_signal = 0.0
        self.share_signal = 0.0
        self.invest_size = 0.0
        self.invest_speed = 0.0
        self.parasite_attach = 0.0
        self.env_mod_food = 0.0
        self.env_mod_clear = 0.0
        self.kin_sacrifice = 0.0
        self.group_attraction = 0.0
        self.mating_readiness = 0.0

        self.total_energy_gained = 0.0
        self.offspring_count = 0
        self.age = 0
        self.is_elite = False
        self.fitness = 0
        self.generation = generation

        self.host = None
        self.parasite_attached = False
        self.ritual_partner = None
        self.ritual_countdown = 0

        self.trail = deque(maxlen=TRAIL_LENGTH)
        self.species_color = None
        self.death_reason = None

    @property
    def age_fraction(self):
        return min(1.0, self.age / max(1, self.lifespan))

    @property
    def senescence_factor(self):
        if self.age_fraction < SENESCENCE_START:
            return 1.0
        return max(0.3, 1.0 - (self.age_fraction - SENESCENCE_START) / (1.0 - SENESCENCE_START))

    @property
    def preferred_biome(self):
        return int(self.biome_preference * (BIOME_COUNT - 1))

    def get_display_color(self, species_mode=False):
        if species_mode and self.species_color:
            base = self.species_color
            if self.is_predator:
                return (min(255, base[0] + 80), base[1] // 2, base[2] // 2)
            return base
        if not self.is_predator:
            bg = background_color_at(self.x, self.y)
            cam = self.camouflage
            r = int((1 - cam) * self.base_color[0] * 255 + cam * bg[0])
            g = int((1 - cam) * self.base_color[1] * 255 + cam * bg[1])
            b = int((1 - cam) * self.base_color[2] * 255 + cam * bg[2])
            return (clamp(r,0,255), clamp(g,0,255), clamp(b,0,255))
        else:
            vision = self.predator_vision
            r = int(self.base_color[0] * 255 + 50 * vision)
            g = int(self.base_color[1] * 255)
            b = int(self.base_color[2] * 255)
            return (min(255, r), min(255, g), min(255, b))

    # ===================== OPTIMIZED SENSE (uses cache) =====================
    def sense(self, food_list, cells, hazards, pheromones, biome_map, cache=None):
        """Gather sensory inputs, using pre-computed SimulationCache if available.

        Without cache: O(N*M + N*N) per call (original behavior).
        With cache:    O(K) where K = number of neighbors (typically < 20).
        """
        inputs = []
        inputs.append(self.energy / 300.0)
        inputs.append(temperature_at(self.x, self.y))

        if cache is not None and cache.valid and self._idx is not None:
            return self._sense_cached(food_list, cells, hazards, pheromones, biome_map, cache)

        # Fallback: original loop-based sensing (used when cache is unavailable)
        min_food_dist = self.perception
        for fx, fy, _ in food_list:
            d = math.hypot(self.x - fx, self.y - fy)
            if d < min_food_dist:
                min_food_dist = d
        inputs.append(1.0 - min_food_dist / self.perception)
        target_type = not self.is_predator
        min_target_dist = self.perception
        for other in cells:
            if other is self:
                continue
            if other.is_predator == target_type:
                d = math.hypot(self.x - other.x, self.y - other.y)
                if d < min_target_dist:
                    min_target_dist = d
        inputs.append(1.0 - min_target_dist / self.perception)
        nearest_food = None
        min_fd = self.perception
        for fx, fy, fe in food_list:
            d = math.hypot(self.x - fx, self.y - fy)
            if d < min_fd:
                min_fd = d
                nearest_food = (fx, fy)
        if nearest_food:
            dx = nearest_food[0] - self.x
            dy = nearest_food[1] - self.y
            angle = math.atan2(dy, dx)
            inputs.append(math.sin(angle))
            inputs.append(math.cos(angle))
        else:
            inputs.append(0.0)
            inputs.append(0.0)
        nearest_target = None
        min_td = self.perception
        for other in cells:
            if other is self:
                continue
            if other.is_predator == target_type:
                d = math.hypot(self.x - other.x, self.y - other.y)
                if d < min_td:
                    min_td = d
                    nearest_target = other
        if nearest_target:
            dx = nearest_target.x - self.x
            dy = nearest_target.y - self.y
            angle = math.atan2(dy, dx)
            inputs.append(math.sin(angle))
            inputs.append(math.cos(angle))
        else:
            inputs.append(0.0)
            inputs.append(0.0)
        min_hazard_dist = self.perception
        for h in hazards:
            d = math.hypot(self.x - h.x, self.y - h.y) - h.radius
            if d < min_hazard_dist:
                min_hazard_dist = d
        inputs.append(1.0 - min_hazard_dist / self.perception)
        kin_count = 0
        for other in cells:
            if other is self:
                continue
            if math.hypot(self.x - other.x, self.y - other.y) < 100:
                if genetic_distance(self.genome, other.genome) < 0.15:
                    kin_count += 1
        inputs.append(min(1.0, kin_count / 5.0))
        if not self.is_predator:
            bg = background_color_at(self.x, self.y)
            match = 1.0 - (abs(bg[0] - self.base_color[0] * 255) / 255 +
                           abs(bg[1] - self.base_color[1] * 255) / 255 +
                           abs(bg[2] - self.base_color[2] * 255) / 255) / 3.0
            inputs.append(match)
        else:
            inputs.append(self.predator_vision)
        nearby_energy_avg = 0
        nearby_count = 0
        for other in cells:
            if other is self:
                continue
            if math.hypot(self.x - other.x, self.y - other.y) < 60:
                nearby_energy_avg += other.energy
                nearby_count += 1
        if nearby_count > 0:
            nearby_energy_avg /= nearby_count
            inputs.append(1.0 - nearby_energy_avg / 200.0)
        else:
            inputs.append(0.0)
        inputs.append(1.0)
        group_size = 0
        for other in cells:
            if other is self:
                continue
            if other.is_predator == self.is_predator:
                if math.hypot(self.x - other.x, self.y - other.y) < 100:
                    group_size += 1
        inputs.append(min(1.0, group_size / 10.0))
        kin_low_energy = 0
        for other in cells:
            if other is self:
                continue
            if math.hypot(self.x - other.x, self.y - other.y) < 60:
                if genetic_distance(self.genome, other.genome) < 0.15 and other.energy < 30:
                    kin_low_energy = 1
                    break
        inputs.append(kin_low_energy)
        ritual_opportunity = 0
        for other in cells:
            if other is self:
                continue
            if math.hypot(self.x - other.x, self.y - other.y) < MATE_DIST:
                if other.ritual_enthusiasm > 0.5 and other.mating_readiness > 0.5:
                    ritual_opportunity = 1
                    break
        inputs.append(ritual_opportunity)
        inputs.append(min(1.0, group_size / 5.0))
        inputs.append(daylight_factor(global_time))
        current_biome = biome_map.biome_at(self.x, self.y)
        inputs.append(1.0 if current_biome == self.preferred_biome else 0.0)
        phero_food = pheromones.sense_at(self.x, self.y, layer=2, radius=2) / PHEROMONE_MAX
        inputs.append(phero_food)
        if len(inputs) < INPUT_SIZE:
            inputs.extend([0.0] * (INPUT_SIZE - len(inputs)))
        elif len(inputs) > INPUT_SIZE:
            inputs = inputs[:INPUT_SIZE]
        return inputs

    def _sense_cached(self, food_list, cells, hazards, pheromones, biome_map, cache):
        """Cache-optimized sensing — O(K) instead of O(N)."""
        idx = self._idx
        n = cache.n_cells
        inputs = []
        inputs.append(self.energy / 300.0)
        inputs.append(temperature_at(self.x, self.y))

        # Food distance (vectorized lookup)
        if cache.n_food > 0:
            food_dists = cache.food_dist[idx]
            min_fd = np.min(food_dists)
            inputs.append(1.0 - min(min_fd, self.perception) / self.perception)
            nearest_fi = np.argmin(food_dists)
            dx = cache.food_dx[idx, nearest_fi]
            dy = cache.food_dy[idx, nearest_fi]
            angle = math.atan2(-dy, -dx)
            inputs.append(math.sin(angle))
            inputs.append(math.cos(angle))
        else:
            inputs.append(0.0)
            inputs.append(0.0)
            inputs.append(0.0)

        # Target distance and angle (prey finds predator, predator finds prey)
        target_type = not self.is_predator
        target_mask = cache.is_predator == target_type
        if np.any(target_mask):
            target_dists = cache.cell_dist[idx].copy()
            target_dists[~target_mask] = np.inf
            target_dists[idx] = np.inf
            min_td = np.min(target_dists)
            nearest_ti = np.argmin(target_dists)
            inputs.append(1.0 - min(min_td, self.perception) / self.perception)
            dx = cache.cell_dx[idx, nearest_ti]
            dy = cache.cell_dy[idx, nearest_ti]
            angle = math.atan2(-dy, -dx)
            inputs.append(math.sin(angle))
            inputs.append(math.cos(angle))
        else:
            inputs.append(0.0)
            inputs.append(0.0)
            inputs.append(0.0)

        # Hazard distance
        if cache.hazard_dist.shape[1] > 0:
            h_dists = cache.hazard_dist[idx] - cache.hazard_radii
            min_hd = np.min(h_dists) if len(h_dists) > 0 else self.perception
            inputs.append(1.0 - min(min_hd, self.perception) / self.perception)
        else:
            inputs.append(1.0)

        # Kin count (using spatial hash + genetic distance matrix)
        neighbors = cache.neighbors_of(idx, 100.0)
        gen_dists = cache.genetic_dist[idx]
        kin_count = sum(1 for j, _ in neighbors if gen_dists[j] < 0.15)
        inputs.append(min(1.0, kin_count / 5.0))

        # Camouflage match / predator vision
        if not self.is_predator:
            bg = background_color_at(self.x, self.y)
            match = 1.0 - (abs(bg[0] - self.base_color[0] * 255) / 255 +
                           abs(bg[1] - self.base_color[1] * 255) / 255 +
                           abs(bg[2] - self.base_color[2] * 255) / 255) / 3.0
            inputs.append(match)
        else:
            inputs.append(self.predator_vision)

        # Nearby energy average (spatial hash filtered)
        nearby = cache.neighbors_of(idx, 60.0)
        if nearby:
            nearby_energy_avg = sum(cells[j].energy for j, _ in nearby) / len(nearby)
            inputs.append(1.0 - nearby_energy_avg / 200.0)
        else:
            inputs.append(0.0)

        inputs.append(1.0)  # bias

        # Group size
        group_members = [(j, d) for j, d in neighbors
                        if cache.is_predator[j] == self.is_predator and d < 100]
        group_size = len(group_members)
        inputs.append(min(1.0, group_size / 10.0))

        # Kin low energy nearby
        kin_low = 0
        for j, _ in nearby:
            if gen_dists[j] < 0.15 and cells[j].energy < 30:
                kin_low = 1
                break
        inputs.append(kin_low)

        # Ritual opportunity (spatial hash filtered)
        ritual_opp = 0
        ritual_candidates = cache.spatial.query_radius(self.x, self.y, MATE_DIST)
        for j in ritual_candidates:
            if j == idx:
                continue
            if d := cache.cell_dist[idx, j] < MATE_DIST:
                if cells[j].ritual_enthusiasm > 0.5 and cells[j].mating_readiness > 0.5:
                    ritual_opp = 1
                    break
        inputs.append(ritual_opp)
        inputs.append(min(1.0, group_size / 5.0))
        inputs.append(daylight_factor(global_time))
        current_biome = biome_map.biome_at(self.x, self.y)
        inputs.append(1.0 if current_biome == self.preferred_biome else 0.0)
        phero_food = pheromones.sense_at(self.x, self.y, layer=2, radius=2) / PHEROMONE_MAX
        inputs.append(phero_food)

        if len(inputs) < INPUT_SIZE:
            inputs.extend([0.0] * (INPUT_SIZE - len(inputs)))
        elif len(inputs) > INPUT_SIZE:
            inputs = inputs[:INPUT_SIZE]
        return inputs

    def _parse_outputs(self, output):
        self.turn = output[0]
        self.speed_mod = (output[1] + 1.0) / 2.0
        self.reproduce_signal = (output[2] + 1.0) / 2.0
        self.share_signal = (output[3] + 1.0) / 2.0
        self.invest_size = (output[4] + 1.0) / 2.0
        self.invest_speed = (output[5] + 1.0) / 2.0
        self.parasite_attach = (output[6] + 1.0) / 2.0 if len(output) > 6 else 0
        self.env_mod_food = (output[7] + 1.0) / 2.0 if len(output) > 7 else 0
        self.env_mod_clear = (output[8] + 1.0) / 2.0 if len(output) > 8 else 0
        self.kin_sacrifice = (output[9] + 1.0) / 2.0 if len(output) > 9 else 0
        self.group_attraction = (output[10] + 1.0) / 2.0 if len(output) > 10 else 0
        self.mating_readiness = (output[11] + 1.0) / 2.0 if len(output) > 11 else 0

    def _parse_outputs_speed_only(self, output):
        self.speed_mod = (output[1] + 1.0) / 2.0
        self.reproduce_signal = (output[2] + 1.0) / 2.0
        self.share_signal = (output[3] + 1.0) / 2.0
        self.invest_size = (output[4] + 1.0) / 2.0
        self.invest_speed = (output[5] + 1.0) / 2.0
        self.parasite_attach = (output[6] + 1.0) / 2.0 if len(output) > 6 else 0
        self.env_mod_food = (output[7] + 1.0) / 2.0 if len(output) > 7 else 0
        self.env_mod_clear = (output[8] + 1.0) / 2.0 if len(output) > 8 else 0
        self.kin_sacrifice = (output[9] + 1.0) / 2.0 if len(output) > 9 else 0
        self.group_attraction = (output[10] + 1.0) / 2.0 if len(output) > 10 else 0
        self.mating_readiness = (output[11] + 1.0) / 2.0 if len(output) > 11 else 0

    def plan_action(self, food_list, cells, hazards, obstacles, pheromones, biome_map, cache=None):
        if self.planning_depth == 0:
            inputs = self.sense(food_list, cells, hazards, pheromones, biome_map, cache)
            output = self.brain.forward(inputs)
            self._parse_outputs(output)
            return
        best_utility = -float('inf')
        best_angle = self.angle
        for angle_offset in [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4,
                             0, math.pi/4, math.pi/2, 3*math.pi/4]:
            test_angle = self.angle + angle_offset
            sim_x, sim_y, sim_energy, utility = self.x, self.y, self.energy, 0.0
            step = 0
            while step < self.planning_depth:
                dx = math.cos(test_angle) * self.speed * 0.8
                dy = math.sin(test_angle) * self.speed * 0.8
                sim_x += dx; sim_y += dy
                sim_x %= WIDTH; sim_y %= HEIGHT
                temp = temperature_at(sim_x, sim_y)
                penalty = 1.0 + 0.5 * abs(self.temp_gene - temp)
                b = biome_map.biome_at(sim_x, sim_y)
                penalty *= BIOME_ENERGY_MULT[b]
                drain = ENERGY_DRAIN_BASE * penalty
                if daylight_factor(global_time) < 0.3:
                    drain *= NIGHT_DRAIN_MULT
                sim_energy -= drain
                for h in hazards:
                    if math.hypot(sim_x - h.x, sim_y - h.y) < h.radius:
                        sim_energy -= HAZARD_DRAIN; utility -= 10
                for fx, fy, fe in food_list:
                    if math.hypot(sim_x - fx, sim_y - fy) < self.radius + FOOD_RADIUS:
                        utility += 20; sim_energy += fe
                if self.is_predator:
                    for other in cells:
                        if other is self or other.is_predator: continue
                        gs = sum(1 for c in cells if not c.is_predator and math.hypot(c.x-other.x,c.y-other.y)<100)
                        gain = 15 if gs < GROUP_SIZE_THRESHOLD else 15*(1-GROUP_DEFENSE_BONUS)
                        if math.hypot(sim_x-other.x,sim_y-other.y) < self.radius+other.radius:
                            utility += gain; sim_energy += 40 if gs < GROUP_SIZE_THRESHOLD else 20
                else:
                    gs2 = sum(1 for c in cells if not c.is_predator and math.hypot(c.x-self.x,c.y-self.y)<100)
                    pp = 30 if gs2 < GROUP_SIZE_THRESHOLD else 10
                    for other in cells:
                        if other is self or not other.is_predator: continue
                        if math.hypot(sim_x-other.x,sim_y-other.y) < self.radius+other.radius+20:
                            utility -= pp
                for other in cells:
                    if other is self: continue
                    if math.hypot(sim_x-other.x,sim_y-other.y)<60:
                        if genetic_distance(self.genome,other.genome)<0.15: utility += 5
                stc = sum(1 for o in cells if o is not self and o.is_predator==self.is_predator and math.hypot(sim_x-o.x,sim_y-o.y)<100)
                utility += self.group_cohesion * min(5, stc * 0.5)
                for other in cells:
                    if other is self: continue
                    if math.hypot(sim_x-other.x,sim_y-other.y)<MATE_DIST:
                        if other.ritual_enthusiasm>0.5 and other.mating_readiness>0.5: utility += 10
                step += 1
            if utility > best_utility:
                best_utility = utility; best_angle = test_angle
        self.turn = max(-1, min(1, (best_angle - self.angle) / math.pi))
        inputs = self.sense(food_list, cells, hazards, pheromones, biome_map, cache)
        output = self.brain.forward(inputs)
        self._parse_outputs_speed_only(output)

    def act(self, food_list, cells, hazards, obstacles, pheromones, biome_map, cache=None):
        if self.planning_depth > 0:
            self.plan_action(food_list, cells, hazards, obstacles, pheromones, biome_map, cache)
        else:
            inputs = self.sense(food_list, cells, hazards, pheromones, biome_map, cache)
            output = self.brain.forward(inputs)
            self._parse_outputs(output)

    def move(self, food_list, obstacles, hazards, cells, biome_map, pheromones, cache=None):
        global global_time
        temp = temperature_at(self.x, self.y)
        temp_penalty = 1.0 + 0.5 * abs(self.temp_gene - temp)
        current_biome = biome_map.biome_at(self.x, self.y)
        biome_penalty = BIOME_ENERGY_MULT[current_biome]
        day_factor = daylight_factor(global_time)
        night_mult = NIGHT_DRAIN_MULT if day_factor < 0.3 else 1.0
        senescence_drain = 1.0 / max(0.3, self.senescence_factor)
        drain = ENERGY_DRAIN_BASE * temp_penalty * biome_penalty * night_mult * senescence_drain

        # Hazard drain using cache or loops
        if cache is not None and cache.valid and self._idx is not None and cache.hazard_dist.shape[1] > 0:
            h_dists = cache.hazard_dist[self._idx]
            for hi in range(len(hazards)):
                if h_dists[hi] < hazards[hi].radius:
                    drain += HAZARD_DRAIN
        else:
            for h in hazards:
                if math.hypot(self.x - h.x, self.y - h.y) < h.radius:
                    drain += HAZARD_DRAIN
        self.energy -= drain
        self.age += 1
        if self.age >= self.lifespan:
            self.energy = -999; self.death_reason = "old age"; return
        if self.parasite_attached and self.host:
            drain_amount = 0.5
            if self.host.energy > drain_amount:
                self.host.energy -= drain_amount; self.energy += drain_amount; self.total_energy_gained += drain_amount
            else:
                self.host.energy = -999; self.host.death_reason = "parasitized"
                self.parasite_attached = False; self.host = None
        if self.energy < 80 and not self.is_predator:
            trail_angle = pheromones.sense_gradient(self.x, self.y, layer=2)
            if trail_angle is not None:
                ad = trail_angle - self.angle
                while ad > math.pi: ad -= 2*math.pi
                while ad < -math.pi: ad += 2*math.pi
                self.angle += ad * 0.03
        dv = pheromones.sense_at(self.x, self.y, layer=1, radius=2)
        if dv > 0.5 and not self.is_predator:
            fa = pheromones.sense_gradient(self.x, self.y, layer=1)
            if fa is not None:
                ad = fa - self.angle
                while ad > math.pi: ad -= 2*math.pi
                while ad < -math.pi: ad += 2*math.pi
                self.angle -= ad * 0.05
        if self.group_attraction > 0.5 and self.group_cohesion > 0.5:
            cx2, cy2, cnt = 0, 0, 0
            # Use spatial hash if available
            if cache is not None and cache.valid and self._idx is not None:
                for j, d in cache.neighbors_of(self._idx, 100.0):
                    if cache.is_predator[j] == self.is_predator and d < 100:
                        cx2 += cells[j].x; cy2 += cells[j].y; cnt += 1
            else:
                for other in cells:
                    if other is self: continue
                    if other.is_predator == self.is_predator and math.hypot(self.x-other.x,self.y-other.y)<100:
                        cx2 += other.x; cy2 += other.y; cnt += 1
            if cnt > 0:
                cx2 /= cnt; cy2 /= cnt
                if math.hypot(cx2-self.x, cy2-self.y) > 5:
                    ta = math.atan2(cy2-self.y, cx2-self.x)
                    ad = ta - self.angle
                    while ad > math.pi: ad -= 2*math.pi
                    while ad < -math.pi: ad += 2*math.pi
                    self.angle += ad * 0.05 * self.group_attraction
        if self.energy > 50:
            if self.invest_size > 0.5:
                inv = (self.invest_size-0.5)*2.0*0.5
                if self.energy > inv:
                    self.energy -= inv; self.radius = min(25, self.radius + inv*0.01)
            if self.invest_speed > 0.5:
                inv = (self.invest_speed-0.5)*2.0*0.5
                if self.energy > inv:
                    self.energy -= inv; self.speed = min(6, self.speed + inv*0.005)
        if self.env_mod_food > 0.7 and self.energy > 20:
            for _ in range(3):
                a = random.uniform(0, 2*math.pi); d = random.uniform(10, 30)
                fx = max(0, min(WIDTH, self.x+math.cos(a)*d))
                fy = max(0, min(HEIGHT, self.y+math.sin(a)*d))
                food_list.append((fx, fy, FOOD_ENERGY))
                pheromones.emit(fx, fy, layer=2, amount=0.5)
            self.energy -= 20
        if self.env_mod_clear > 0.7 and self.energy > 30:
            for h in hazards:
                if math.hypot(self.x-h.x, self.y-h.y) < h.radius+30:
                    h.shrink(5); self.energy -= 30; break
        if not self.is_predator and self.energy > 60:
            pheromones.emit(self.x, self.y, layer=0, amount=PHEROMONE_EMIT_RATE*0.3)
        if self.is_predator:
            pheromones.emit(self.x, self.y, layer=1, amount=PHEROMONE_EMIT_RATE)
        self.angle += self.turn * 0.1 * self.senescence_factor
        speed = self.speed * self.speed_mod * BIOME_SPEED_MULT[current_biome] * self.senescence_factor
        dx = math.cos(self.angle) * speed; dy = math.sin(self.angle) * speed
        nx, ny = self.x + dx, self.y + dy
        # Obstacle collision using cache or loops
        if cache is not None and cache.valid and self._idx is not None and cache.obstacle_dist.shape[1] > 0:
            o_dists = cache.obstacle_dist[self._idx]
            for oi in range(len(obstacles)):
                dist = o_dists[oi]
                if dist < self.radius + obstacles[oi].radius:
                    self.angle += math.pi + random.uniform(-0.3, 0.3)
                    overlap = (self.radius + obstacles[oi].radius) - dist
                    if dist > 0:
                        ao = math.atan2(ny-obstacles[oi].y, nx-obstacles[oi].x)
                        nx += math.cos(ao)*overlap; ny += math.sin(ao)*overlap
                    break
        else:
            for obs in obstacles:
                dist = math.hypot(nx-obs.x, ny-obs.y)
                if dist < self.radius + obs.radius:
                    self.angle += math.pi + random.uniform(-0.3, 0.3)
                    overlap = (self.radius+obs.radius) - dist
                    if dist > 0:
                        ao = math.atan2(ny-obs.y, nx-obs.x)
                        nx += math.cos(ao)*overlap; ny += math.sin(ao)*overlap
                    break
        self.trail.append((self.x, self.y))
        self.x = nx % WIDTH; self.y = ny % HEIGHT
        if self.mate_cooldown > 0: self.mate_cooldown -= 1
        if self.energy > 1000: self.energy = 1000

    def eat_or_hunt_or_parasite(self, food_list, cells, pheromones, cache=None):
        # Kin sacrifice using cache
        if self.kin_sacrifice > 0.5 and self.kin_sacrifice_tendency > 0.5 and self.energy > 30:
            bk, bd = None, float('inf')
            if cache is not None and cache.valid and self._idx is not None:
                gen_dists = cache.genetic_dist[self._idx]
                for j, d in cache.neighbors_of(self._idx, 60.0):
                    if gen_dists[j] < 0.15 and cells[j].energy < 30 and d < bd:
                        bk, bd = cells[j], d
            else:
                for other in cells:
                    if other is self: continue
                    if genetic_distance(self.genome,other.genome)<0.15 and other.energy<30:
                        d = math.hypot(self.x-other.x, self.y-other.y)
                        if d < 60 and d < bd: bk, bd = other, d
            if bk:
                t = min(self.energy*0.4, 30); self.energy -= t; bk.energy += t
        if not self.parasite_attached and self.parasitic_tendency > 0.6 and self.parasite_attach > 0.5:
            if cache is not None and cache.valid and self._idx is not None:
                for j, d in cache.neighbors_of(self._idx, self.radius + 30):
                    if d < self.radius + cells[j].radius + 10:
                        self.parasite_attached = True; self.host = cells[j]
                        self.energy += 5; break
            else:
                for other in cells:
                    if other is self: continue
                    if math.hypot(self.x-other.x, self.y-other.y) < self.radius+other.radius+10:
                        self.parasite_attached = True; self.host = other; self.energy += 5; break
        if not self.is_predator:
            for food in food_list[:]:
                if math.hypot(self.x-food[0], self.y-food[1]) < self.radius+FOOD_RADIUS:
                    fe = food[2]
                    # Group sharing using spatial hash
                    gs = 0
                    if cache is not None and cache.valid and self._idx is not None:
                        for j, d in cache.neighbors_of(self._idx, 100.0):
                            if not cache.is_predator[j]:
                                gs += 1
                    else:
                        gs = sum(1 for c in cells if not c.is_predator and math.hypot(c.x-self.x,c.y-self.y)<100)
                    if gs >= GROUP_SIZE_THRESHOLD:
                        sh = fe/gs
                        if cache is not None and cache.valid and self._idx is not None:
                            for j, d in cache.neighbors_of(self._idx, 100.0):
                                if not cache.is_predator[j]:
                                    cells[j].energy += sh; cells[j].total_energy_gained += sh
                        else:
                            for o in cells:
                                if not o.is_predator and math.hypot(o.x-self.x,o.y-self.y)<100:
                                    o.energy += sh; o.total_energy_gained += sh
                        self.energy += sh; self.total_energy_gained += sh
                    else:
                        self.energy += fe; self.total_energy_gained += fe
                    pheromones.emit(self.x, self.y, layer=2, amount=1.0)
                    food_list.remove(food)
        else:
            nf = daylight_factor(global_time)
            vm = NIGHT_VISION_MULT if nf < 0.3 else 1.0
            vm += self.predator_vision * 0.4 * (1-nf)
            for other in cells[:]:
                if other is self or other.is_predator: continue
                d = math.hypot(self.x-other.x, self.y-other.y)
                if d < self.radius+other.radius and d < self.perception*vm:
                    gs = sum(1 for c in cells if not c.is_predator and math.hypot(c.x-other.x,c.y-other.y)<100)
                    gain = 40 if gs < GROUP_SIZE_THRESHOLD else 20
                    self.energy += gain; self.total_energy_gained += gain
                    other.energy = -999; other.death_reason = "hunted"
                    pheromones.emit(other.x, other.y, layer=1, amount=3.0)

    def share_energy(self, cells, cache=None):
        if self.share_signal > 0.7 and self.energy > 50:
            bk, bd = None, float('inf')
            if cache is not None and cache.valid and self._idx is not None:
                gen_dists = cache.genetic_dist[self._idx]
                for j, d in cache.neighbors_of(self._idx, 60.0):
                    if cache.is_predator[j] == self.is_predator and d < 60 and gen_dists[j] < 0.2 and d < bd:
                        bk, bd = cells[j], d
            else:
                for other in cells:
                    if other is self or other.is_predator != self.is_predator: continue
                    d = math.hypot(self.x-other.x, self.y-other.y)
                    if d < 60 and d < bd and genetic_distance(self.genome,other.genome)<0.2:
                        bk, bd = other, d
            if bk and bk.energy < self.energy:
                t = min(10, self.energy*0.1); self.energy -= t; bk.energy += t

    def update_ritual(self, cells, cache=None):
        if self.ritual_partner:
            partner = self.ritual_partner
            if partner.is_dead() or math.hypot(self.x-partner.x,self.y-partner.y) > MATE_DIST:
                self.ritual_partner = None; self.ritual_countdown = 0
                if partner.ritual_partner == self: partner.ritual_partner = None; partner.ritual_countdown = 0
                return
            if (self.mating_readiness>0.7 and partner.mating_readiness>0.7
                and self.energy>MATE_ENERGY_THRESHOLD and partner.energy>MATE_ENERGY_THRESHOLD):
                self.ritual_countdown += 1; partner.ritual_countdown += 1
                if self.ritual_countdown >= RITUAL_DURATION:
                    self.perform_mating(partner)
                    self.ritual_partner = None; self.ritual_countdown = 0
                    partner.ritual_partner = None; partner.ritual_countdown = 0
            else:
                self.ritual_partner = None; self.ritual_countdown = 0
                if partner.ritual_partner == self: partner.ritual_partner = None; partner.ritual_countdown = 0
        else:
            if self.mating_readiness>0.7 and self.energy>MATE_ENERGY_THRESHOLD and self.ritual_enthusiasm>0.5:
                if cache is not None and cache.valid and self._idx is not None:
                    for j, d in cache.spatial.query_radius(self.x, self.y, MATE_DIST):
                        if j == self._idx: continue
                        other = cells[j]
                        if other.is_predator != self.is_predator: continue
                        if (other.ritual_partner is None and other.mating_readiness>0.7 and other.ritual_enthusiasm>0.5):
                            if d < MATE_DIST:
                                self.ritual_partner = other; self.ritual_countdown = 0
                                other.ritual_partner = self; other.ritual_countdown = 0; break
                else:
                    for other in cells:
                        if other is self or other.is_predator!=self.is_predator: continue
                        if (other.ritual_partner is None and other.mating_readiness>0.7 and other.ritual_enthusiasm>0.5):
                            if math.hypot(self.x-other.x,self.y-other.y) < MATE_DIST:
                                self.ritual_partner = other; self.ritual_countdown = 0
                                other.ritual_partner = self; other.ritual_countdown = 0; break

    def perform_mating(self, other):
        cg = crossover_genome(self.genome, other.genome)
        cg = mutate_genome(cg)
        ce = (self.energy+other.energy)/4.0
        self.energy *= 0.7; other.energy *= 0.7
        self.mate_cooldown = 30; other.mate_cooldown = 30
        gen = max(self.generation, other.generation) + 1
        child = Cell(self.x+random.uniform(-10,10), self.y+random.uniform(-10,10), cg, ce, generation=gen)
        child.mutate_network()
        self.offspring_count += 1; other.offspring_count += 1
        return child

    def asexual_reproduce(self):
        if self.energy >= REPRODUCE_THRESHOLD and self.reproduce_signal > 0.5:
            ce = self.energy // 2; self.energy = ce
            cg = mutate_genome(self.genome)
            child = Cell(self.x+random.uniform(-10,10), self.y+random.uniform(-10,10), cg, ce, generation=self.generation+1)
            child.mutate_network(); self.offspring_count += 1; return child
        return None

    def mutate_network(self):
        self.brain.mutate(self.mut_rate, MUTATION_AMOUNT)

    def compute_fitness(self):
        return self.total_energy_gained + self.offspring_count * 30 + self.age * 0.1

    def is_dead(self):
        return self.energy <= 0

    def draw(self, screen, species_mode=False, show_trails=False, particles=None, surf_cache=None):
        color = self.get_display_color(species_mode)
        cx, cy = int(self.x), int(self.y)
        r_draw = max(3, int(self.radius))

        if show_trails and len(self.trail) > 1:
            points = list(self.trail)
            for i in range(1, len(points)):
                frac = i / len(points)
                w = max(1, int(2 * frac))
                pygame.draw.line(screen, color[:3],
                    (int(points[i-1][0]), int(points[i-1][1])),
                    (int(points[i][0]), int(points[i][1])), w)

        if self.is_elite:
            pulse = 0.7 + 0.3 * math.sin(global_time * 0.1)
            glow_r = int(r_draw * 2.5 * pulse)
            draw_gradient_circle(screen, (255, 215, 0), (cx, cy), glow_r, surf_cache)

        glow_size = r_draw + 6
        if surf_cache is not None:
            glow_s = surf_cache.get_circle_alpha(color, glow_size, 35)
        else:
            glow_s = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (*color, 35), (glow_size, glow_size), glow_size)
        screen.blit(glow_s, (cx - glow_size, cy - glow_size))

        pygame.draw.circle(screen, color, (cx, cy), r_draw)

        if r_draw > 4:
            hl_s = pygame.Surface((r_draw*2, r_draw*2), pygame.SRCALPHA)
            hl_r = max(2, int(r_draw * 0.6))
            pygame.draw.circle(hl_s, (*brighten(color, 60), 50), (int(r_draw*0.65), int(r_draw*0.65)), hl_r)
            screen.blit(hl_s, (cx - r_draw, cy - r_draw))

        rim_color = darken(color, 50)
        pygame.draw.circle(screen, rim_color, (cx, cy), r_draw, 1)

        activity = abs(self.turn) + self.speed_mod
        nuc_r = max(1, int(r_draw * 0.25))
        nuc_brightness = int(140 + 115 * min(1, activity))
        nuc_color = (nuc_brightness, nuc_brightness, clamp(nuc_brightness + 30, 0, 255))
        pygame.draw.circle(screen, nuc_color, (cx, cy), nuc_r)

        if r_draw > 5:
            dx = math.cos(self.angle) * (r_draw - 2)
            dy = math.sin(self.angle) * (r_draw - 2)
            pygame.draw.circle(screen, brighten(color, 40), (int(cx+dx), int(cy+dy)), max(1, r_draw // 5))

        if self.is_predator:
            num_spikes = 6
            for i in range(num_spikes):
                a = self.angle + (2 * math.pi * i / num_spikes) + math.sin(global_time * 0.05 + i) * 0.2
                sx = cx + math.cos(a) * (r_draw + 4)
                sy = cy + math.sin(a) * (r_draw + 4)
                spk_color = brighten(color, 30)
                pygame.draw.circle(screen, spk_color, (int(sx), int(sy)), max(1, r_draw // 4))

        if self.parasite_attached:
            dash_offset = global_time * 0.15
            for i in range(8):
                a_start = dash_offset + (math.pi * 2 * i / 8)
                a_end = a_start + math.pi / 12
                steps = 4
                for s in range(steps):
                    frac = s / steps
                    a = a_start + (a_end - a_start) * frac
                    px = cx + math.cos(a) * (r_draw + 3)
                    py = cy + math.sin(a) * (r_draw + 3)
                    pygame.draw.circle(screen, (200, 50, 220), (int(px), int(py)), 1)

        if self.ritual_partner:
            pulse = 0.5 + 0.5 * math.sin(global_time * 0.15)
            ring_r = r_draw + 5 + int(3 * pulse)
            alpha = int(120 + 80 * pulse)
            ring_s = pygame.Surface((ring_r*2+4, ring_r*2+4), pygame.SRCALPHA)
            pygame.draw.circle(ring_s, (255, 200, 50, clamp(alpha, 0, 255)), (ring_r+2, ring_r+2), ring_r, 2)
            screen.blit(ring_s, (cx-ring_r-2, cy-ring_r-2))
            ring_r2 = r_draw + 8 + int(5 * pulse)
            ring_s2 = pygame.Surface((ring_r2*2+4, ring_r2*2+4), pygame.SRCALPHA)
            pygame.draw.circle(ring_s2, (255, 230, 100, clamp(alpha//2, 0, 255)), (ring_r2+2, ring_r2+2), ring_r2, 1)
            screen.blit(ring_s2, (cx-ring_r2-2, cy-ring_r2-2))

        bar_w = r_draw * 2
        bar_h = max(2, int(r_draw * 0.2))
        bar_x = cx - r_draw
        bar_y = cy - r_draw - bar_h - 6
        pygame.draw.rect(screen, (30, 30, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=1)
        fill = clamp(self.energy / REPRODUCE_THRESHOLD, 0, 1)
        if fill > 0.01:
            fw = int(bar_w * fill)
            if fill < 0.33:
                bar_color = lerp_color((220, 50, 50), (255, 180, 0), fill / 0.33)
            elif fill < 0.66:
                bar_color = lerp_color((255, 180, 0), (50, 220, 100), (fill - 0.33) / 0.33)
            else:
                bar_color = lerp_color((50, 220, 100), (0, 230, 230), (fill - 0.66) / 0.34)
            pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fw, bar_h), border_radius=1)

        age_h = 2
        age_y = bar_y - age_h - 1
        af = self.age_fraction
        age_color = (100, 160, 255) if af < SENESCENCE_START else lerp_color((100,160,255), (255,80,80), (af-SENESCENCE_START)/(1-SENESCENCE_START))
        pygame.draw.rect(screen, (25, 25, 35), (bar_x, age_y, bar_w, age_h), border_radius=1)
        if af > 0.01:
            pygame.draw.rect(screen, age_color, (bar_x, age_y, max(1, int(bar_w * af)), age_h), border_radius=1)


# ---------- Population Graph ----------
class PopulationGraph:
    def __init__(self, x, y, w, h, max_points=200):
        self.x = x; self.y = y; self.w = w; self.h = h
        self.max_points = max_points
        self.total_history = deque(maxlen=max_points)
        self.pred_history = deque(maxlen=max_points)
        self.prey_history = deque(maxlen=max_points)
        self.food_history = deque(maxlen=max_points)
        self.surface = pygame.Surface((w, h), pygame.SRCALPHA)

    def update(self, cells, food_list):
        total = len(cells)
        preds = sum(1 for c in cells if c.is_predator)
        self.total_history.append(total)
        self.pred_history.append(preds)
        self.prey_history.append(total - preds)
        self.food_history.append(len(food_list))

    def draw(self, screen):
        draw_rounded_rect_alpha(screen, (self.x, self.y, self.w, self.h),
                              (15, 15, 30), 190, radius=8,
                              border_color=(80, 120, 180), border_alpha=120)
        if len(self.total_history) < 2:
            return
        max_val = max(max(self.total_history, default=1), 1)
        margin_top = 14
        margin_bottom = 4
        margin_lr = 6
        graph_h = self.h - margin_top - margin_bottom
        graph_w = self.w - margin_lr * 2

        def draw_filled_line(data, color, alpha_fill=30):
            if len(data) < 2: return
            points = []
            for i, v in enumerate(data):
                px = self.x + margin_lr + int(i * graph_w / self.max_points)
                py = self.y + margin_top + graph_h - int(v / max_val * graph_h)
                points.append((px, py))
            fill_points = [(points[0][0], self.y + margin_top + graph_h)] + points + [(points[-1][0], self.y + margin_top + graph_h)]
            fill_s = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            adjusted = [(p[0] - self.x, p[1] - self.y) for p in fill_points]
            if len(adjusted) >= 3:
                pygame.draw.polygon(fill_s, (*color, alpha_fill), adjusted)
            screen.blit(fill_s, (self.x, self.y))
            if len(points) >= 2:
                pygame.draw.lines(screen, color, False, points, 2)

        draw_filled_line(self.total_history, (200, 210, 230), 20)
        draw_filled_line(self.pred_history, (255, 99, 72), 35)
        draw_filled_line(self.prey_history, (46, 204, 113), 35)
        draw_filled_line(self.food_history, (116, 185, 255), 25)

        font = pygame.font.SysFont("Arial", 10)
        labels = [("Total", (200, 210, 230)), ("Pred", (255, 99, 72)),
                  ("Prey", (46, 204, 113)), ("Food", (116, 185, 255))]
        lx = self.x + margin_lr
        for text, color in labels:
            surf = font.render(text, True, color)
            screen.blit(surf, (lx, self.y + 2))
            lx += surf.get_width() + 10


# ---------- Cell Inspector ----------
class CellInspector:
    def __init__(self):
        self.selected = None

    def select(self, cell, mx, my):
        self.selected = cell

    def draw(self, screen, font_small):
        cell = self.selected
        if cell is None or cell.is_dead():
            self.selected = None
            return
        panel_w, panel_h = 210, 290
        px, py = WIDTH - panel_w - 8, 8
        draw_rounded_rect_alpha(screen, (px, py, panel_w, panel_h),
                              (12, 12, 28), 210, radius=10,
                              border_color=(100, 130, 200), border_alpha=140)
        color = cell.get_display_color(False)
        pygame.draw.circle(screen, color, (px + panel_w - 18, py + 16), 8)
        pygame.draw.circle(screen, brighten(color, 40), (px + panel_w - 18, py + 16), 8, 1)

        lines = [
            ("CELL INSPECTOR", (170, 190, 255)),
            (f"{'PREDATOR' if cell.is_predator else 'PREY'}",
             (255, 99, 72) if cell.is_predator else (46, 213, 115)),
            (f"Energy  {cell.energy:.1f} / {REPRODUCE_THRESHOLD}", (255, 234, 100)),
            (f"Age     {cell.age} / {cell.lifespan}  ({cell.age_fraction:.0%})", (190, 190, 210)),
            (f"Senesc  {cell.senescence_factor:.2f}", (210, 160, 160)),
            (f"Gen     {cell.generation}    Fit {cell.compute_fitness():.0f}", (190, 190, 210)),
            (f"Offspring {cell.offspring_count}", (190, 190, 210)),
            ("", (0,0,0)),
            ("TRAITS", (130, 140, 200)),
            (f"Spd {cell.speed:.1f}  Rad {cell.radius:.0f}  Per {cell.perception:.0f}", (180, 185, 200)),
            (f"Plan {cell.planning_depth}  Mut {cell.mut_rate:.2f}", (180, 185, 200)),
            (f"Temp {cell.temp_gene:.2f}  Bio {BIOME_NAMES[cell.preferred_biome]}", (180, 185, 200)),
            (f"Camo {cell.camouflage:.2f}  Vis {cell.predator_vision:.2f}", (180, 185, 200)),
            (f"Para {cell.parasitic_tendency:.2f}  Kin {cell.kin_sacrifice_tendency:.2f}", (180, 185, 200)),
            (f"Grp {cell.group_cohesion:.2f}  Rit {cell.ritual_enthusiasm:.2f}", (180, 185, 200)),
        ]
        if cell.is_elite:
            lines.append(("ELITE", (255, 215, 0)))
        y = py + 6
        for text, c in lines:
            if text:
                surf = font_small.render(text, True, c)
                screen.blit(surf, (px + 10, y))
            y += 16


# ---------- Save/Load ----------
def save_simulation(state, filename=SAVE_FILENAME):
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print("Simulation saved.")

def load_simulation(filename=SAVE_FILENAME):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        print("Simulation loaded.")
        return state
    else:
        print("No save file found.")
        return None


# ---------- Main ----------
def main():
    global global_time
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("CellEvo - Cellular Evolution Simulator [Numba Optimized]")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14)
    font_small = pygame.font.SysFont("Arial", 11)
    font_title = pygame.font.SysFont("Arial", 16, bold=True)

    # ===== Performance infrastructure =====
    surf_cache = SurfaceCache(max_cache=300)
    sim_cache = SimulationCache()

    biome_map = BiomeMap(WIDTH, HEIGHT)
    biome_surface = biome_map.pre_render()
    pheromones = PheromoneGrid(WIDTH, HEIGHT)
    particles = ParticleSystem(max_particles=600)

    obstacles = []
    for _ in range(OBSTACLE_COUNT):
        x = random.randint(OBSTACLE_RADIUS, WIDTH - OBSTACLE_RADIUS)
        y = random.randint(OBSTACLE_RADIUS, HEIGHT - OBSTACLE_RADIUS)
        obstacles.append(Obstacle(x, y, OBSTACLE_RADIUS))

    hazards = []
    for _ in range(HAZARD_COUNT):
        x = random.randint(HAZARD_RADIUS, WIDTH - HAZARD_RADIUS)
        y = random.randint(HAZARD_RADIUS, HEIGHT - HAZARD_RADIUS)
        hazards.append(Hazard(x, y, HAZARD_RADIUS))

    clusters = [FoodCluster() for _ in range(CLUSTER_COUNT)]
    food = []
    for cluster in clusters:
        cluster.generate_food(food, FOOD_COUNT // CLUSTER_COUNT)

    cells = []
    for _ in range(30):
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        cells.append(Cell(x, y))

    log_file = open(LOG_FILENAME, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "frame", "total_cells", "predators", "prey",
        "avg_energy", "avg_speed", "avg_radius", "avg_temp_match",
        "avg_mut_rate", "avg_planning_depth",
        "avg_camouflage", "avg_predator_vision", "avg_parasitic_tendency",
        "avg_kin_sacrifice", "avg_group_cohesion", "avg_ritual_enthusiasm",
        "avg_age", "avg_lifespan", "avg_biome_match"
    ])

    running = True
    paused = False
    frame = 0
    elites = []
    sim_speed = 1
    zoom = 1.0
    camera_x, camera_y = 0.0, 0.0
    panning = False
    pan_start = (0, 0)
    cam_start = (0, 0)
    show_species_colors = False
    show_trails = False
    show_pheromones = False
    show_biomes = True
    species_update_counter = 0

    pop_graph = PopulationGraph(WIDTH - 220, HEIGHT - 118, 212, 110)
    inspector = CellInspector()
    night_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # ===== Performance tracking =====
    perf_update_ms = 0.0
    perf_draw_ms = 0.0
    perf_cache_ms = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_f:
                    for _ in range(10):
                        fx = random.randint(0, WIDTH); fy = random.randint(0, HEIGHT)
                        food.append((fx, fy, FOOD_ENERGY))
                        pheromones.emit(fx, fy, layer=2, amount=0.3)
                if event.key == pygame.K_c:
                    cells = [Cell(random.randint(50,WIDTH-50), random.randint(50,HEIGHT-50)) for _ in range(30)]
                    food = []
                    for cluster in clusters: cluster.generate_food(food, FOOD_COUNT // CLUSTER_COUNT)
                    pheromones = PheromoneGrid(WIDTH, HEIGHT); frame = 0; global_time = 0
                if event.key == pygame.K_o:
                    obstacles.append(Obstacle(random.randint(OBSTACLE_RADIUS,WIDTH-OBSTACLE_RADIUS),
                                               random.randint(OBSTACLE_RADIUS,HEIGHT-OBSTACLE_RADIUS), OBSTACLE_RADIUS))
                if event.key == pygame.K_h:
                    hazards.append(Hazard(random.randint(HAZARD_RADIUS,WIDTH-HAZARD_RADIUS),
                                           random.randint(HAZARD_RADIUS,HEIGHT-HAZARD_RADIUS), HAZARD_RADIUS))
                if event.key == pygame.K_s:
                    save_simulation({'frame':frame,'cells':cells,'food':food,'obstacles':obstacles,
                        'hazards':hazards,'clusters':clusters,'global_time':global_time,
                        'elites':elites,'biome_map':biome_map,'pheromones':pheromones})
                if event.key == pygame.K_l:
                    state = load_simulation()
                    if state:
                        frame=state['frame'];cells=state['cells'];food=state['food']
                        obstacles=state['obstacles'];hazards=state['hazards'];clusters=state['clusters']
                        global_time=state['global_time'];elites=state.get('elites',[])
                        biome_map=state.get('biome_map',biome_map)
                        pheromones=state.get('pheromones',PheromoneGrid(WIDTH,HEIGHT))
                        biome_surface=biome_map.pre_render()
                if event.key == pygame.K_1: sim_speed = 1
                elif event.key == pygame.K_2: sim_speed = 2
                elif event.key == pygame.K_3: sim_speed = 5
                elif event.key == pygame.K_4: sim_speed = 10
                if event.key == pygame.K_g: show_species_colors = not show_species_colors
                if event.key == pygame.K_t: show_trails = not show_trails
                if event.key == pygame.K_p: show_pheromones = not show_pheromones
                if event.key == pygame.K_b: show_biomes = not show_biomes
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    wx, wy = (mx+camera_x)/zoom, (my+camera_y)/zoom
                    cc = None; bd = float('inf')
                    for cell in cells:
                        d = math.hypot(wx-cell.x, wy-cell.y)
                        if d < cell.radius+5 and d < bd: cc = cell; bd = d
                    if cc: inspector.select(cc, mx, my)
                    else:
                        food.append((wx, wy, FOOD_ENERGY))
                        pheromones.emit(wx, wy, layer=2, amount=0.3)
                        particles.emit_food_sparkle(wx, wy)
                elif event.button == 3:
                    mx, my = event.pos
                    wx, wy = (mx+camera_x)/zoom, (my+camera_y)/zoom
                    new_c = Cell(wx, wy)
                    cells.append(new_c)
                    particles.emit_burst(wx, wy, new_c.get_display_color(), count=6, speed=1.5, life=20)
                elif event.button == 2:
                    panning = True; pan_start = event.pos; cam_start = (camera_x, camera_y)
                elif event.button == 4: zoom = min(ZOOM_MAX, zoom + ZOOM_STEP)
                elif event.button == 5: zoom = max(ZOOM_MIN, zoom - ZOOM_STEP)
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2: panning = False
            if event.type == pygame.MOUSEMOTION:
                if panning:
                    dx = event.pos[0]-pan_start[0]; dy = event.pos[1]-pan_start[1]
                    camera_x = max(0, cam_start[0] - dx/zoom)
                    camera_y = max(0, cam_start[1] - dy/zoom)

        if not paused:
            import time as _time
            t0 = _time.perf_counter()

            # Track dying cells for particle effects
            dying_positions = []
            for c in cells:
                if c.energy <= -900 and c.death_reason is not None:
                    dying_positions.append((c.x, c.y, c.get_display_color(), c.death_reason))

            steps = sim_speed
            for _ in range(steps):
                frame += 1; global_time += 1
                for cluster in clusters: cluster.update()
                df = daylight_factor(global_time)
                fgm = NIGHT_FOOD_MULT if df < 0.3 else 1.0
                while len(food) < int(FOOD_COUNT * fgm):
                    chosen = random.choice(clusters)
                    b = biome_map.biome_at(chosen.x, chosen.y)
                    if random.random() < BIOME_FOOD_MULT[b]:
                        chosen.generate_food(food, 1)
                if frame % 10 == 0: pheromones.decay()

                # ===== Build simulation cache (once per substep) =====
                t_cache = _time.perf_counter()
                sim_cache.build(cells, food, hazards, obstacles)
                # Assign cell indices for cache lookups
                for i, cell in enumerate(cells):
                    cell._idx = i
                perf_cache_ms += (_time.perf_counter() - t_cache) * 1000

                for cell in cells:
                    cell.act(food, cells, hazards, obstacles, pheromones, biome_map, sim_cache)
                for cell in cells:
                    cell.move(food, obstacles, hazards, cells, biome_map, pheromones, sim_cache)
                for cell in cells:
                    cell.eat_or_hunt_or_parasite(food, cells, pheromones, sim_cache)
                for cell in cells:
                    cell.share_energy(cells, sim_cache)
                for cell in cells:
                    cell.update_ritual(cells, sim_cache)

                # Capture death positions before removal
                for c in cells:
                    if c.is_dead() and c.death_reason and c.death_reason != "old age":
                        dying_positions.append((c.x, c.y, c.get_display_color(), c.death_reason))

                cells = [c for c in cells if not c.is_dead()]
                if len(cells) > SOFT_CAP:
                    er = (len(cells)-SOFT_CAP)/(HARD_CAP-SOFT_CAP)
                    for cell in cells:
                        if random.random() < er*0.1: cell.energy -= 2
                if len(cells) > HARD_CAP:
                    cells.sort(key=lambda c: c.energy, reverse=True)
                    for c in cells[HARD_CAP:]: c.energy = -999; c.death_reason = "overpopulation"
                    cells = cells[:HARD_CAP]
                if frame % ELITE_INTERVAL == 0 and len(cells) > 10:
                    for cell in cells: cell.fitness = cell.compute_fitness()
                    cells.sort(key=lambda c: c.fitness, reverse=True)
                    ec = max(1, int(len(cells)*ELITE_FRACTION))
                    elites = cells[:ec]
                    for e in elites: e.is_elite = True
                else:
                    for cell in cells: cell.is_elite = False
                new_cells = []
                for cell in cells:
                    if cell.is_elite: continue
                    child = cell.asexual_reproduce()
                    if child:
                        new_cells.append(child)
                        particles.emit_burst(child.x, child.y, (150, 255, 200), count=5, speed=1.0, life=15, radius=1)
                cells.extend(new_cells)
                species_update_counter += 1
                if show_species_colors and species_update_counter >= 30:
                    compute_species_colors(cells); species_update_counter = 0
                if frame % LOG_INTERVAL == 0:
                    total = len(cells)
                    if total > 0:
                        preds = sum(1 for c in cells if c.is_predator)
                        log_writer.writerow([frame, total, preds, total-preds,
                            sum(c.energy for c in cells)/total,
                            sum(c.speed for c in cells)/total,
                            sum(c.radius for c in cells)/total,
                            sum(1-abs(c.temp_gene-temperature_at(c.x,c.y)) for c in cells)/total,
                            sum(c.mut_rate for c in cells)/total,
                            sum(c.planning_depth for c in cells)/total,
                            sum(c.camouflage for c in cells if not c.is_predator)/(total-preds) if preds<total else 0,
                            sum(c.predator_vision for c in cells if c.is_predator)/preds if preds else 0,
                            sum(c.parasitic_tendency for c in cells)/total,
                            sum(c.kin_sacrifice_tendency for c in cells)/total,
                            sum(c.group_cohesion for c in cells)/total,
                            sum(c.ritual_enthusiasm for c in cells)/total,
                            sum(c.age for c in cells)/total,
                            sum(c.lifespan for c in cells)/total,
                            sum(1.0 if biome_map.biome_at(c.x,c.y)==c.preferred_biome else 0.0 for c in cells)/total])
                        log_file.flush()

            # Emit death particles for tracked dying cells
            for dx, dy, dc, dr in dying_positions:
                if dr == "hunted":
                    particles.emit_death(dx, dy, (255, 80, 60), count=10)
                elif dr == "parasitized":
                    particles.emit_death(dx, dy, (180, 50, 200), count=8)
                elif dr == "old age":
                    particles.emit_death(dx, dy, (100, 100, 140), count=5, speed=1.0, life=20)
                else:
                    particles.emit_death(dx, dy, dc, count=6)

            perf_update_ms = (_time.perf_counter() - t0) * 1000

        particles.update()
        if frame % 6 == 0: pop_graph.update(cells, food)

        # ===== DRAW =====
        t_draw = _time.perf_counter()
        screen.fill(BACKGROUND)

        if show_biomes:
            screen.blit(biome_surface, (0, 0))

        if daylight_factor(global_time) < 0.5:
            star_alpha = int(60 * (1 - daylight_factor(global_time) * 2))
            for i in range(30):
                sx = (i * 137 + 50) % WIDTH
                sy = (i * 211 + 30) % HEIGHT
                twinkle = 0.5 + 0.5 * math.sin(global_time * 0.03 + i * 1.7)
                a = int(star_alpha * twinkle)
                if a > 5:
                    pygame.draw.circle(screen, (200, 210, 255), (sx, sy), 1)

        for obs in obstacles: obs.draw(screen, surf_cache)
        for h in hazards: h.draw(screen, surf_cache)

        if show_pheromones: pheromones.draw(screen, camera_x, camera_y, zoom)

        for fx, fy, fe in food:
            ix, iy = int(fx), int(fy)
            is_high = fe > FOOD_ENERGY
            base_c = FOOD_COLOR_HIGH if is_high else FOOD_COLOR
            pulse = 0.8 + 0.2 * math.sin(global_time * 0.08 + fx * 0.1 + fy * 0.1)
            r = int((FOOD_RADIUS + (1 if is_high else 0)) * pulse)
            glow_r = r + 4
            if surf_cache is not None:
                glow_s = surf_cache.get_circle_alpha(base_c, glow_r, 30 if not is_high else 50)
            else:
                glow_s = pygame.Surface((glow_r*2, glow_r*2), pygame.SRCALPHA)
                gc = (*base_c, 30 if not is_high else 50)
                pygame.draw.circle(glow_s, gc, (glow_r, glow_r), glow_r)
            screen.blit(glow_s, (ix - glow_r, iy - glow_r))
            pygame.draw.circle(screen, base_c, (ix, iy), r)
            pygame.draw.circle(screen, brighten(base_c, 80), (ix, iy), max(1, r // 2))
            if is_high:
                sparkle_a = int(100 + 100 * math.sin(global_time * 0.12 + fx))
                for sa in range(4):
                    a = math.pi / 4 * sa + global_time * 0.02
                    sx2 = ix + int(math.cos(a) * (r + 3))
                    sy2 = iy + int(math.sin(a) * (r + 3))
                    pygame.draw.circle(screen, (255, 255, 200), (sx2, sy2), 1)

        for cell in cells:
            cell.draw(screen, show_species_colors, show_trails, particles, surf_cache)

        particles.draw(screen, surf_cache)

        na = night_overlay_alpha(global_time)
        if na > 0:
            night_surface.fill((5, 5, 35, na))
            screen.blit(night_surface, (0, 0))
            if na > 40:
                vig = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                for cx_v, cy_v in [(0,0),(WIDTH,0),(0,HEIGHT),(WIDTH,HEIGHT)]:
                    for vr in range(3):
                        rr = 200 + vr * 100
                        aa = clamp(int(na * 0.15 * (3 - vr) / 3), 0, 255)
                        pygame.draw.circle(vig, (5, 5, 25, aa), (cx_v, cy_v), rr)
                screen.blit(vig, (0, 0))

        # Day/Night Sky Bar
        day_f = daylight_factor(global_time)
        bar_x, bar_y, bar_w, bar_h = WIDTH//2-60, 4, 120, 12
        sky_color = lerp_color((15, 15, 45), (70, 130, 200), day_f)
        draw_rounded_rect_alpha(screen, (bar_x, bar_y, bar_w, bar_h), sky_color, 220, radius=6,
                              border_color=(100, 140, 200), border_alpha=80)
        sun_x = bar_x + int(8 + (bar_w - 16) * day_f)
        if day_f > 0.3:
            glow_s = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (255, 230, 80, 60), (10, 10), 8)
            pygame.draw.circle(glow_s, (255, 245, 150, 180), (10, 10), 5)
            screen.blit(glow_s, (sun_x - 10, bar_y + 1))
        else:
            glow_s = pygame.Surface((18, 18), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (200, 210, 240, 120), (9, 9), 6)
            pygame.draw.circle(glow_s, (230, 240, 255, 200), (9, 9), 4)
            screen.blit(glow_s, (sun_x - 9, bar_y + 2))

        if day_f < 0.5:
            for i in range(8):
                sx = bar_x + 10 + i * 13
                if (i + int(global_time * 0.02)) % 3 != 0:
                    tw = 0.5 + 0.5 * math.sin(global_time * 0.04 + i * 2)
                    a = int(180 * (0.5 - day_f) * tw)
                    if a > 10:
                        pygame.draw.circle(screen, (220, 230, 255), (sx, bar_y + 6), 1)

        # UI Panel
        info_x = 10; info_y = 20
        panel_w = 265; panel_h = 360
        draw_rounded_rect_alpha(screen, (info_x, info_y, panel_w, panel_h),
                              (12, 14, 28), 200, radius=10,
                              border_color=(70, 100, 180), border_alpha=100)

        title = font_title.render("CellEvo", True, (180, 210, 255))
        screen.blit(title, (info_x + 10, info_y + 6))
        pygame.draw.circle(screen, (46, 204, 113), (info_x + 68, info_y + 14), 3)

        info_y += 28
        total_cells = len(cells)
        preds = sum(1 for c in cells if c.is_predator)
        prey_n = total_cells - preds
        day_name = "Day" if day_f > 0.5 else "Night"

        actual_fps = clock.get_fps()
        info_lines = [
            (f"Cells {total_cells}   Pred {preds}   Prey {prey_n}", (200, 210, 230)),
            (f"Food {len(food)}   Frame {frame}", (180, 190, 210)),
            (f"{day_name} {day_f:.0%}   Season {math.sin(global_time*0.001):+.2f}", (160, 180, 220)),
            (f"Speed {sim_speed}x{'   PAUSED' if paused else ''}", (255, 200, 100) if paused else (160, 170, 200)),
            (f"Zoom {zoom:.1f}x   Elites {len(elites)}", (160, 170, 200)),
            (f"Rituals {sum(1 for c in cells if c.ritual_partner is not None)//2}", (160, 170, 200)),
            (f"Particles {particles.active_count}", (130, 140, 170)),
            (f"FPS {actual_fps:.0f}  Upd {perf_update_ms:.1f}ms", (130, 170, 130)),
            (f"Numba {'ON' if HAS_NUMBA else 'OFF'}", (100, 200, 100) if HAS_NUMBA else (200, 100, 100)),
            (None, None),
            ("CONTROLS", (100, 120, 190)),
            ("SPACE pause   1-4 speed", (130, 140, 170)),
            ("LClick select/food   RClick spawn", (130, 140, 170)),
            ("Scroll zoom   MidDrag pan", (130, 140, 170)),
            ("F food  O obstacle  H hazard", (130, 140, 170)),
            ("S save   L load   C reset", (130, 140, 170)),
            (None, None),
            ("TOGGLES", (100, 120, 190)),
            (f"G species [{'ON' if show_species_colors else 'off'}]  "
             f"T trails [{'ON' if show_trails else 'off'}]",
             (100, 255, 100) if (show_species_colors or show_trails) else (130, 140, 170)),
            (f"P pheromones [{'ON' if show_pheromones else 'off'}]  "
             f"B biomes [{'ON' if show_biomes else 'off'}]",
             (100, 255, 100) if (show_pheromones or show_biomes) else (130, 140, 170)),
        ]
        for text, color in info_lines:
            if text is None:
                info_y += 4; continue
            surf = font_small.render(text, True, color)
            screen.blit(surf, (info_x + 12, info_y))
            info_y += 15

        pop_graph.draw(screen)
        inspector.draw(screen, font_small)
        pygame.display.flip()
        perf_draw_ms = (_time.perf_counter() - t_draw) * 1000
        clock.tick(FPS)

    log_file.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()