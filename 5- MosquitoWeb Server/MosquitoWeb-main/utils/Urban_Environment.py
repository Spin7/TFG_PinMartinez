# Urban_Environment.py

import json
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
from Agent import State


# ============================================================
# CONTAINER
# ============================================================

@dataclass
class Container:
    id: str
    pos: Tuple[int, int]
    capacity: float = 100.0


# ============================================================
# ENVIRONMENT
# ============================================================

class Environment:

    def __init__(self,
                 urban_grid_path: str,
                 delta_time: float,
                 config_path: Optional[str] = None,
                 trap_path: Optional[str] = None,
                 grid_to_meters: Optional[float] = None,
                 num_containers: int = 20,
                 num_random_breeding_sites: int = 50,
                 seed: Optional[int] = 0,
                 config: Optional[dict] = None,
                 center_lat: Optional[float] = None,
                 center_lon: Optional[float] = None,
                 cell_size_m: Optional[float] = None,
                 map_size_m: Optional[float] = None,
                 compute_potential: bool = False):

        self.rng = np.random.RandomState(seed)
        self.delta_time = delta_time

        trap_records = None

        # spatial index
        self.spatial_grid = {}
        self.spatial_cell_size = 10   # cells for neighborhood lookup

        # buffer for newborn juveniles
        self.newborn_buffer = []

        # =====================================================
        # LOAD CONFIG FILE
        # =====================================================

        if config_path is not None:

            with open(config_path, "r", encoding="utf-8") as f:
                cfg_json = json.load(f)

            center_lat = cfg_json.get("center_lat", center_lat)
            center_lon = cfg_json.get("center_lon", center_lon)
            cell_size_m = cfg_json.get("cell_size_m", cell_size_m)
            map_size_m = cfg_json.get("map_size_m", map_size_m)

            trap_records = cfg_json.get("traps", None)

            print(f"[Environment] Loaded config: {config_path}")

        # Geographic parameters
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.cell_size_m = cell_size_m if cell_size_m is not None else 1.0
        self.map_size_m = map_size_m
        self.grid_to_meters = grid_to_meters if grid_to_meters is not None else self.cell_size_m

        # =====================================================
        # LOAD GRID
        # =====================================================

        self.urban_grid = np.load(urban_grid_path).astype(np.uint8)
        self.grid_height, self.grid_width = self.urban_grid.shape

        if self.map_size_m is None:
            self.map_size_m = float(self.grid_height * self.cell_size_m)

        print(f"[Environment] Grid: {self.grid_height} x {self.grid_width}")
        print(f"[Environment] Cell size: {self.cell_size_m} m")
        print(f"[Environment] Map size: {self.map_size_m:.2f} m")

        # =====================================================
        # CONTAINERS
        # =====================================================

        self.containers: List[Container] = []

        if trap_records is not None:
            self._load_containers_from_records(trap_records)

        elif trap_path is not None:
            self._load_containers_from_json(trap_path)

        else:
            self._place_containers(num_containers)

        # -----------------------------------------------------
        # ADD RANDOM BREEDING SITES (ON VEGETATION)
        # -----------------------------------------------------

        self._generate_random_breeding_sites(num_random_breeding_sites)

        # Buffers
        self.newborn_buffer: List[Tuple[int, int]] = []
        self.spatial_index = {}

        # =====================================================
        # BIOLOGICAL PARAMETERS
        # =====================================================

        cfg = config or {}

        self.mu_J = cfg.get("mu_J", 0.05)
        self.mu_M = cfg.get("mu_M", 0.1)
        self.mu_F = cfg.get("mu_F", 0.08)

        self.gamma = cfg.get("gamma", 0.07)
        self.alpha = cfg.get("alpha", 0.001)
        self.beta = cfg.get("beta", 0.02)
        self.f = cfg.get("f", 5.0)

        self.D_A = cfg.get("D_A", 0.5)
        self.Kc = cfg.get("Kc", 100.0)

        self.turning_sigma = cfg.get("turning_sigma", math.pi / 4)

        self.mating_radius = int(cfg.get("mating_radius", 5))
        self.density_radius = int(cfg.get("density_radius", 3))
        self.oviposition_radius = float(cfg.get("oviposition_radius", 10.0))

        # Discrete lattice movement (paper model)
        # R: max jump radius in grid cells
        # beta_0: exploration bias (additive constant in weights)
        # forbidden_types: cell types adults cannot enter
        self.movement_radius = int(cfg.get("movement_radius", 3))
        self.beta_0 = float(cfg.get("beta_0", 10.0))
        self.forbidden_types = set(cfg.get("forbidden_types", [1, 3]))

        self.time = 0.0

        # Potential field is only needed for visualize_potential_field().
        # Skipped by default — agents do not use it during simulation.
        if compute_potential:
            self.compute_potential_field()
        else:
            self.potential_field = None
            self.grad_y = None
            self.grad_x = None

    # =========================================================
    # RANDOM BREEDING SITE GENERATION
    # =========================================================

    def _generate_random_breeding_sites(self, n):

        vegetation_cells = np.argwhere(self.urban_grid == 2)

        if len(vegetation_cells) == 0:
            print("[Environment] No vegetation cells available")
            return

        existing = {tuple(c.pos) for c in self.containers}

        candidates = [tuple(v) for v in vegetation_cells if tuple(v) not in existing]

        if len(candidates) == 0:
            print("[Environment] No free vegetation cells")
            return

        idxs = self.rng.choice(len(candidates),
                               size=min(n, len(candidates)),
                               replace=False)

        for i in idxs:

            y, x = candidates[i]

            cid = f"b_{y}_{x}"

            self.containers.append(Container(cid, (y, x)))

        print(f"[Environment] Added {len(idxs)} random breeding sites")

    # =========================================================
    # CONTAINER PLACEMENT (GENERIC)
    # =========================================================

    def _place_containers(self, n):

        suitable = np.argwhere(self.urban_grid != 3)

        if len(suitable) == 0:
            return

        idxs = self.rng.choice(len(suitable),
                               size=min(n, len(suitable)),
                               replace=False)

        for i in idxs:

            y, x = suitable[i]
            cid = f"c_{y}_{x}"

            self.containers.append(Container(cid, (y, x)))

    # =========================================================
    # LOAD TRAPS JSON
    # =========================================================

    def _load_containers_from_json(self, path: str):

        with open(path, "r", encoding="utf-8") as fh:
            records = json.load(fh)

        self._load_containers_from_records(records)

    # =========================================================
    # LOAD TRAPS FROM CONFIG
    # =========================================================

    def _load_containers_from_records(self, records):

        placed = 0

        for rec in records:

            r = int(rec.get("row", 0))
            c = int(rec.get("col", 0))

            if not (0 <= r < self.grid_height and 0 <= c < self.grid_width):
                continue

            if self.urban_grid[r, c] == 3:
                continue

            cid = rec.get("code", f"trap_{r}_{c}")

            self.containers.append(Container(cid, (r, c)))

            placed += 1

        print(f"[Environment] Loaded {placed} containers")

    # =========================================================
    # SPATIAL INDEX
    # =========================================================
    def rebuild_spatial_index(self, agents):

        self.spatial_grid = {}

        for a in agents:

            y, x = a.pos

            cy = int(y // self.spatial_cell_size)
            cx = int(x // self.spatial_cell_size)

            key = (cy, cx)

            if key not in self.spatial_grid:
                self.spatial_grid[key] = []

            self.spatial_grid[key].append(a)

    # =========================================================
    # NEIGHBORHOOD QUERY
    # =========================================================

    def neighborhood_counts(self, pos, radius):

        y, x = pos

        cy = int(y // self.spatial_cell_size)
        cx = int(x // self.spatial_cell_size)

        r_cells = int(radius // self.spatial_cell_size) + 1

        counts = {
            "J": 0,   # Juvenile
            "M": 0,   # Male
            "FU": 0,  # Female Unmated
            "FG": 0   # Female Gravid
        }

        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):

                key = (cy + dy, cx + dx)

                if key not in self.spatial_grid:
                    continue

                for a in self.spatial_grid[key]:

                    ay, ax = a.pos
                    d = np.sqrt((ay - y) ** 2 + (ax - x) ** 2)

                    if d <= radius:

                        if a.state.name == "JUVENILE":
                            counts["J"] += 1

                        elif a.state.name == "ADULT_MALE":
                            counts["M"] += 1

                        elif a.state.name == "ADULT_FEMALE_U":
                            counts["FU"] += 1

                        elif a.state.name == "ADULT_FEMALE_G":
                            counts["FG"] += 1

        return counts

    # =========================================================
    # REPRODUCTION BUFFER
    # =========================================================

    def register_new_juvenile(self, pos):
        """
        Called by gravid females when they oviposit.
        The birth only succeeds if the female is within `oviposition_radius`
        grid cells of at least one breeding site. The juvenile is placed at
        the nearest breeding site, not at the female's position.
        """
        if not self.containers:
            return  # no breeding sites at all → birth fails

        y, x = pos
        min_dist = float("inf")
        nearest_pos = None

        for c in self.containers:
            cy, cx = c.pos
            dist = math.sqrt((cy - y) ** 2 + (cx - x) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_pos = c.pos

        if min_dist <= self.oviposition_radius:
            self.newborn_buffer.append(nearest_pos)

    # =========================================================
    # BOUNDARY CONDITIONS
    # =========================================================

    def apply_boundary(self, pos):
        """
        Keep agents inside the simulation grid.
        Reflective boundary.
        """

        x, y = pos

        x = max(0, min(x, self.grid_width - 1))
        y = max(0, min(y, self.grid_height - 1))

        return (y, x)

    # =========================================================
    # DISCRETE LATTICE MOVEMENT  (paper model)
    # =========================================================

    def sample_movement(self, pos, rng):
        """
        Sample the next position for an adult agent according to:

            N_R(x) = { y in G | ||y-x||_2 <= R, c(y) not in forbidden_types }

            w̃(y|x) = (Π(y) + β₀) / (||y-x||_2 + 1)

            P(y|x)  = w̃(y|x) / Σ_{z in N_R(x)} w̃(z|x)

        Returns the sampled next position (row, col).
        If no valid neighbour exists, the agent stays in place.
        """
        y0, x0 = int(pos[0]), int(pos[1])
        R = self.movement_radius

        # ── Bounding box clipped to grid ──────────────────────────────────────
        y_lo = max(0, y0 - R)
        y_hi = min(self.grid_height - 1, y0 + R)
        x_lo = max(0, x0 - R)
        x_hi = min(self.grid_width - 1, x0 + R)

        # ── Build index arrays for the bounding box ───────────────────────────
        yy, xx = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]

        # ── Euclidean distances ───────────────────────────────────────────────
        dist = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)

        # ── Cell-type mask: within radius AND not forbidden ───────────────────
        cell_types = self.urban_grid[yy, xx]
        forbidden_arr = np.array(list(self.forbidden_types), dtype=np.uint8)
        is_forbidden = np.isin(cell_types, forbidden_arr)
        mask = (dist <= R) & ~is_forbidden

        if not mask.any():
            return pos   # no valid cell → stay

        valid_y   = yy[mask]
        valid_x   = xx[mask]
        valid_dist = dist[mask]

        # ── Potential field Π(y) (zeros if not computed) ──────────────────────
        if self.potential_field is not None:
            pi_vals = self.potential_field[valid_y, valid_x]
        else:
            pi_vals = np.zeros(len(valid_y), dtype=float)

        # ── Weights and normalisation ─────────────────────────────────────────
        weights = (pi_vals + self.beta_0) / (valid_dist + 1.0)
        weights /= weights.sum()

        # ── Sample one destination ────────────────────────────────────────────
        idx = rng.choice(len(valid_y), p=weights)
        return (int(valid_y[idx]), int(valid_x[idx]))

    # =========================================================
    # POTENTIAL FIELD
    # =========================================================

    def compute_potential_field(self):

        H, W = self.grid_height, self.grid_width

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        potential = np.zeros((H, W))

        lam = 10.0

        for c in self.containers:

            cy, cx = c.pos

            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

            potential += np.exp(-dist / lam)

        if potential.max() > 0:
            potential /= potential.max()

        self.potential_field = potential

        self.grad_y, self.grad_x = np.gradient(self.potential_field)

    # =========================================================
    # VISUALIZATION
    # =========================================================

    def visualize_map(self, show=True):

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap([
            [0.9, 0.9, 0.9],
            [0.6, 0.6, 0.6],
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 0.8],
            [0.5, 0.3, 0.1],
        ])

        plt.figure(figsize=(6, 6))

        plt.imshow(self.urban_grid, cmap=cmap, origin="upper")

        if self.containers:

            ys = [c.pos[0] for c in self.containers]
            xs = [c.pos[1] for c in self.containers]

            plt.scatter(xs, ys, c="purple", s=40)

        plt.axis("off")

        if show:
            plt.show()

    # =========================================================
    # VISUALIZE POTENTIAL FIELD
    # =========================================================

    def visualize_potential_field(self, show=True):

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # ── 5-class terrain colormap (same as visualize_map) ──────────────────
        terrain_cmap = ListedColormap([
            [0.9, 0.9, 0.9],   # 0 Empty
            [0.6, 0.6, 0.6],   # 1 Road
            [0.0, 0.6, 0.0],   # 2 Vegetation
            [0.0, 0.0, 0.8],   # 3 Water
            [0.5, 0.3, 0.1],   # 4 Building
        ])

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        # ── Layer 1: terrain grid (base) ───────────────────────────────────────
        ax.imshow(
            self.urban_grid,
            cmap=terrain_cmap,
            vmin=0,
            vmax=4,
            origin="upper",
            alpha=0.85
        )

        # ── Layer 2: potential field overlay ──────────────────────────────────
        im = ax.imshow(
            self.potential_field,
            cmap="inferno",
            vmin=0,
            vmax=1,
            origin="upper",
            alpha=0.55
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attraction potential", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # ── Breeding sites ────────────────────────────────────────────────────
        if self.containers:
            ys = [c.pos[0] for c in self.containers]
            xs = [c.pos[1] for c in self.containers]
            ax.scatter(xs, ys, c="cyan", s=20, marker="^",
                       label="Breeding sites", zorder=5)
            ax.legend(loc="upper right", fontsize=8,
                      facecolor="#222222", labelcolor="white")

        ax.set_title("Potential field overlay", color="white", fontsize=10)
        ax.axis("off")
        plt.tight_layout()

        if show:
            plt.show()


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    env = Environment(
        urban_grid_path="grid_data.npy",
        delta_time=1.0,
        config_path="environment_params.json",
        num_random_breeding_sites=20,
        seed=42
    )

    print(f"Grid size: {env.grid_height} x {env.grid_width}")
    print(f"Containers: {len(env.containers)}")

    env.visualize_map()
    env.visualize_potential_field()