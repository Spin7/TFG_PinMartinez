from Agent import Agent, State
from Urban_Environment import Environment
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Optional, Tuple
import csv

class Simulation:

    def __init__(self,
                 grid_path: str,
                 delta_t: float = 0.1,
                 containers: int = 20,
                 num_random_breeding_sites: int = 50,
                 seed: int = 0,
                 grid_to_meters: float = 1.0,
                 init_counts: Optional[Dict[str, int]] = None,
                 env_config: Optional[Dict[str, float]] = None,
                 trap_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 center_lat: Optional[float] = None,
                 center_lon: Optional[float] = None,
                 cell_size_m: float = 1.0,
                 map_size_m: Optional[float] = None,
                 compute_potential: bool = False):

        # -----------------------------------------------------
        # LOAD JSON CONFIG IF PROVIDED
        # -----------------------------------------------------

        if config_path is not None:

            with open(config_path, "r", encoding="utf-8") as f:
                cfg_json = json.load(f)

            center_lat = cfg_json.get("center_lat", center_lat)
            center_lon = cfg_json.get("center_lon", center_lon)
            cell_size_m = cfg_json.get("cell_size_m", cell_size_m)
            map_size_m = cfg_json.get("map_size_m", map_size_m)

            print(f"[Simulation] Loaded config from {config_path}")

        # -----------------------------------------------------
        # ENVIRONMENT
        # -----------------------------------------------------

        self.env = Environment(
            urban_grid_path=grid_path,
            delta_time=delta_t,
            grid_to_meters=grid_to_meters,
            num_containers=containers,
            num_random_breeding_sites=num_random_breeding_sites,
            seed=seed,
            config=env_config,
            trap_path=trap_path,
            config_path=config_path,
            center_lat=center_lat,
            center_lon=center_lon,
            cell_size_m=cell_size_m,
            map_size_m=map_size_m,
            compute_potential=compute_potential
        )

        self.delta_t = delta_t
        self.time = 0.0
        self.rng = self.env.rng

        self.agents: List[Agent] = []

        self.history: Dict[str, List[float]] = {
            "time": [],
            "J": [],
            "M": [],
            "FU": [],
            "FG": [],
            "total": []
        }

        if init_counts:
            self._spawn_from_counts(init_counts)

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def _sample_random_cell(self) -> Tuple[int, int]:

        mask = self.env.urban_grid != 3
        sites = np.argwhere(mask)

        if len(sites) == 0:
            return (0, 0)

        idx = self.rng.randint(0, len(sites))
        return (int(sites[idx][0]), int(sites[idx][1]))

    def _spawn_from_counts(self, counts: Dict[str, int]):

        for _ in range(counts.get("JUVENILE", 0)):

            cont = self.rng.choice(self.env.containers)

            self.agents.append(
                Agent(State.JUVENILE, cont.pos, self.rng)
            )

        for _ in range(counts.get("ADULT_MALE", 0)):

            self.agents.append(
                Agent(State.ADULT_MALE,
                      self._sample_random_cell(),
                      self.rng)
            )

        for _ in range(counts.get("ADULT_FEMALE_U", 0)):

            self.agents.append(
                Agent(State.ADULT_FEMALE_U,
                      self._sample_random_cell(),
                      self.rng)
            )

        for _ in range(counts.get("ADULT_FEMALE_G", 0)):

            self.agents.append(
                Agent(State.ADULT_FEMALE_G,
                      self._sample_random_cell(),
                      self.rng)
            )

    # =========================================================
    # STEP
    # =========================================================

    def step(self):

        self.rng.shuffle(self.agents)

        self.env.rebuild_spatial_index(self.agents)

        for a in self.agents:

            if a.state != State.DEAD:
                a.step(self.delta_t, self.env)

        self.agents = [a for a in self.agents if a.state != State.DEAD]

        for pos in self.env.newborn_buffer:

            self.agents.append(
                Agent(State.JUVENILE, pos, self.rng)
            )

        self.env.newborn_buffer.clear()

        for a in self.agents:
            a.move(self.delta_t, self.env)

        self.time += self.delta_t

        self._record_history()

    # =========================================================

    def _record_history(self):

        nJ = sum(1 for a in self.agents if a.state == State.JUVENILE)
        nM = sum(1 for a in self.agents if a.state == State.ADULT_MALE)
        nFU = sum(1 for a in self.agents if a.state == State.ADULT_FEMALE_U)
        nFG = sum(1 for a in self.agents if a.state == State.ADULT_FEMALE_G)

        self.history["time"].append(self.time)
        self.history["J"].append(nJ)
        self.history["M"].append(nM)
        self.history["FU"].append(nFU)
        self.history["FG"].append(nFG)
        self.history["total"].append(nJ + nM + nFU + nFG)

    # =========================================================
    # MOSQUITO HEATMAP
    # =========================================================

    def compute_density_heatmap(self):

        heatmap = np.zeros_like(self.env.urban_grid, dtype=float)

        for a in self.agents:

            y, x = int(a.pos[0]), int(a.pos[1])

            if 0 <= y < self.env.grid_height and 0 <= x < self.env.grid_width:
                heatmap[y, x] += 1

        return heatmap

    # =========================================================

    def plot_heatmap(self, sigma: float = 5.0):
        """
        Overlay a smoothed mosquito-density heatmap on the urban grid.

        Parameters
        ----------
        sigma : float
            Standard deviation (grid cells) for Gaussian smoothing.
            Larger values = wider blobs, more visible at low density.
        """
        heatmap = self.compute_density_heatmap()

        # --- smooth so sparse counts become visible blobs ---
        heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)

        # --- normalise to [0,1] so vmin/vmax work consistently ---
        h_max = heatmap_smooth.max()
        if h_max > 0:
            heatmap_norm = heatmap_smooth / h_max
        else:
            heatmap_norm = heatmap_smooth  # all zeros edge-case

        from matplotlib.colors import ListedColormap

        terrain_cmap = ListedColormap([
            [0.9, 0.9, 0.9],   # 0 Empty
            [0.6, 0.6, 0.6],   # 1 Road
            [0.0, 0.6, 0.0],   # 2 Vegetation
            [0.0, 0.0, 0.8],   # 3 Water
            [0.5, 0.3, 0.1],   # 4 Building
        ])

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        # urban grid with proper terrain colors as base
        ax.imshow(
            self.env.urban_grid,
            cmap=terrain_cmap,
            vmin=0,
            vmax=4,
            alpha=0.9,
            origin="upper"
        )

        # smoothed density overlay
        im = ax.imshow(
            heatmap_norm,
            cmap="inferno",
            alpha=0.85,
            vmin=0,
            vmax=1,
            origin="upper"
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative mosquito density", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # breeding sites
        if self.env.containers:
            ys = [c.pos[0] for c in self.env.containers]
            xs = [c.pos[1] for c in self.env.containers]
            ax.scatter(
                xs, ys,
                c="cyan", s=20, marker="^",
                label="Breeding sites",
                zorder=5
            )

        # population counts as text
        nJ  = sum(1 for a in self.agents if a.state == State.JUVENILE)
        nM  = sum(1 for a in self.agents if a.state == State.ADULT_MALE)
        nFU = sum(1 for a in self.agents if a.state == State.ADULT_FEMALE_U)
        nFG = sum(1 for a in self.agents if a.state == State.ADULT_FEMALE_G)
        txt = f"t={self.time:.1f}d  J={nJ}  M={nM}  FU={nFU}  FG={nFG}  Total={nJ+nM+nFU+nFG}"
        ax.set_title(txt, color="white", fontsize=10)

        ax.legend(
            loc="upper right",
            fontsize=8,
            facecolor="#222222",
            labelcolor="white"
        )
        ax.axis("off")
        plt.tight_layout()
        
        filename = f"mosquito_heatmap_t{self.time:.1f}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"[Saved] {filename}")

    # =========================================================

    def run(self, t_max: float, verbose: bool = True):

        steps = int(t_max / self.delta_t)

        for i in range(steps):

            self.step()

            if verbose and i % 50 == 0:
                print(f"t={self.time:.2f} | agents={len(self.agents)}")

        self.plot_results()
        self.plot_heatmap()
        # --- Save datasets ---
        self.save_population_history()
        self.save_final_agents()

    # =========================================================

    def plot_results(self):

        t = self.history["time"]

        plt.figure(figsize=(10,5))

        plt.plot(t, self.history["J"], label="Juveniles")
        plt.plot(t, self.history["M"], label="Males")
        plt.plot(t, self.history["FU"], label="Females U")
        plt.plot(t, self.history["FG"], label="Females G")
        plt.plot(t, self.history["total"], label="Total", linewidth=2)

        plt.legend()

        plt.xlabel("Time (Days)")
        plt.ylabel("Population")

        plt.title("Population Dynamics")

        plt.tight_layout()
        
        filename = "population_dynamics.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"[Saved] {filename}")

    def save_population_history(self, filename="population_timeseries.csv"):

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
    
            writer.writerow(["time", "J", "M", "FU", "FG", "total"])
    
            for i in range(len(self.history["time"])):
                writer.writerow([
                    self.history["time"][i],
                    self.history["J"][i],
                    self.history["M"][i],
                    self.history["FU"][i],
                    self.history["FG"][i],
                    self.history["total"][i],
                ])
    
        print(f"[Saved] {filename}")

    def save_final_agents(self, filename="agents_final_state.csv"):
    
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
    
            writer.writerow(["id", "state", "y", "x"])
    
            for i, a in enumerate(self.agents):
    
                writer.writerow([
                    i,
                    a.state.name,
                    float(a.pos[0]),
                    float(a.pos[1]),
                ])
    
        print(f"[Saved] {filename}")


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    init_counts = {
        "JUVENILE": 300,
        "ADULT_MALE": 1000,
        "ADULT_FEMALE_U": 500,
        "ADULT_FEMALE_G": 500
    }

    env_config = {
        # ── All rates in day⁻¹  (delta_t = 0.1 d ≈ 2.4 h per step) ────────────
        # Sources: Brady et al. 2013, Otero et al. 2006, Styer et al. 2007

        # Larval/pupal mortality.  Lab: 0.01–0.05 d⁻¹; field closer to 0.05
        "mu_J":  0.05,

        # Adult male lifespan ≈ 7–10 d  →  mu_M ≈ 0.12 d⁻¹
        "mu_M":  0.12,

        # Adult female lifespan ≈ 14–21 d in field  →  mu_F ≈ 0.06 d⁻¹
        "mu_F":  0.06,

        # Aquatic stage duration (egg→adult) ≈ 8–12 d at 25 °C  →  gamma ≈ 0.10
        "gamma": 0.10,

        # Density-dependence strength (dimensionless crowding coefficient)
        "alpha": 1.0,

        # Larval carrying capacity per density-radius neighbourhood
        # (roughly: max larvae per ~10×10 cell patch around a breeding site)
        "Kc":    60.0,

        # Per-male mating rate (encounters day⁻¹ per male in radius).
        # beta=0.5 → P(mating within 10 d | 1 male nearby) ≈ 99 %
        "beta":  0.5,

        # Daily egg clutch rate.  Aedes aegypti lays ~100 eggs every 3–4 d;
        # with per-egg survival ~5 % → effective juvenile input ≈ 1.5 d⁻¹.
        # Keep at 1.0 and rely on density-dependence to regulate growth.
        "f":     1.0,

        # Diffusion coefficient (cells² d⁻¹).  Aedes flight range ≈ 50–200 m;
        # with cell_size=3 m → effective range 17–67 cells.  D_A=0.5 gives
        # step_length ≈ 0.32 cells per step, realistic short-range foraging.
        "D_A":          0.5,
        "turning_sigma": 0.5,   # rad, correlated random walk persistence

        # Neighbourhood radii (grid cells, cell_size=3 m)
        #   mating_radius 15 cells ≈ 45 m  (realistic encounter range)
        #   density_radius 5 cells ≈ 15 m  (larval competition neighbourhood)
        "mating_radius":  5,
        "density_radius":  5,

        # Oviposition: female must be within this many cells of a breeding site.
        # 10 cells × 3 m = 30 m  (Aedes typically oviposit within 50–100 m
        # of a water source, 30 m is a conservative realistic value).
        "oviposition_radius": 5,
    }

    sim = Simulation(

        grid_path="grid_data.npy",

        delta_t=0.1,

        seed=42,

        containers=0,

        num_random_breeding_sites=20,

        init_counts=init_counts,

        env_config=env_config,

        config_path="environment_params.json"
    )

    print("Simulation started...")
    print(f"Grid: {sim.env.grid_height} x {sim.env.grid_width}")
    print(f"Containers: {len(sim.env.containers)}")

    sim.run(t_max=100, verbose=True)

    print("Simulation finished.")