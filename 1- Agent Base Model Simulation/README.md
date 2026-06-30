# Agent Base Model (ABM) Simulation

This folder contains all versions of the **agent-based simulation** of *Aedes aegypti* mosquito population dynamics over a realistic urban grid.

## Overview

The simulation models individual mosquitoes as agents with biologically-grounded states, movement, and life-cycle rules. An urban grid (derived from OpenStreetMap data) encodes terrain types — roads, buildings, vegetation, water, and empty cells — which influence mosquito behavior.

### Mosquito Life Cycle States

```
Egg/Larva → JUVENILE → ADULT_MALE
                    ↘
                      ADULT_FEMALE_U (unfed/unmated)
                           ↓ (mating)
                      ADULT_FEMALE_G (gravid/mated)
                           ↓ (oviposition near water)
                      [New JUVENILEs spawned]
```

---

## Folder Structure

```
1- Agent Base Model Simulation/
│
├── Agent_simulationV0_preliminary_model/   # Initial NetLogo-style HTML prototype
├── Agent_simulationV1/                     # First Python version (grid + matplotlib)
└── Agent_simulationV2/                     # Final, full-featured Python simulation
```

---

## Version History

### V0 — Preliminary Model (`Agent_simulationV0_preliminary_model/`)
- Two HTML files using the NetLogo-inspired **AgentScript** framework.
- Proof-of-concept: mosquito agents on a Moore-neighborhood space-time grid.
- Files: `agentBaseMosquitos.html`, `agentesEspacioTemporalesMoore.html`

### V1 — First Python Version (`Agent_simulationV1/`)
- Transition to Python with a proper grid generator using OSMnx.
- Scripts: `Generacion_del_gird.py`, `Simulacion_Agentes.py`
- Produces population time series and saves results to `simulations_results/`.

### V2 — Final Version (`Agent_simulationV2/`) 
The production simulation. Fully modular, configurable via JSON.

| File | Role |
|------|------|
| `Agent.py` | Agent class: state machine, movement, mating, oviposition, mortality |
| `Urban_Environment.py` | Environment: urban grid, breeding sites, spatial index, trap placement |
| `Grid_Generator.py` | Downloads OSM data and rasterizes terrain into a `.npy` grid |
| `Simulation_Mosquitoes.py` | Orchestrates the simulation run, records history, saves results |
| `sensitivity.py` | Sensitivity analysis (Morris / Sobol) of model parameters |
| `plot_agent_distribution.py` | Plots final agent spatial distribution over the urban map |
| `plot_population_timeseries.py` | Plots population dynamics over time |
| `environment_params.json` | Configuration: center coordinates, cell size, map size |
| `grid_data.npy` | Pre-generated urban grid (NumPy array) |
| `traps_coordinates_clean.csv` | GPS coordinates of deployed traps |
| `population_timeseries.csv` | Simulation output: population counts per time step |
| `agents_final_state.csv` | Final positions and states of all agents |

---

## Running the Simulation

```bash
cd Agent_simulationV2
python Simulation_Mosquitoes.py
```

**Key parameters** (in `Simulation_Mosquitoes.py` `__main__` block):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta_t` | `0.1` | Time step in days (~2.4 hours) |
| `t_max` | `100` | Total simulation time in days |
| `seed` | `42` | Random seed for reproducibility |
| `num_random_breeding_sites` | `20` | Number of random water breeding sites |
| `init_counts` | `J:300, M:1000, FU:500, FG:500` | Initial population |

**Biological parameters** (in `env_config` dict):

| Parameter | Value | Source |
|-----------|-------|--------|
| `mu_J` | 0.05 d⁻¹ | Juvenile mortality (Brady et al. 2013) |
| `mu_M` | 0.12 d⁻¹ | Male mortality (~7–10 day lifespan) |
| `mu_F` | 0.06 d⁻¹ | Female mortality (~14–21 day lifespan) |
| `gamma` | 0.10 d⁻¹ | Aquatic stage maturation rate |
| `beta` | 0.5 | Per-male mating rate |
| `f` | 1.0 | Daily egg clutch rate |
| `D_A` | 0.5 | Diffusion coefficient (cells² d⁻¹) |

---

## Outputs

- `population_dynamics.png` — Time series plot of all life stages
- `mosquito_heatmap_t*.png` — Spatial density heatmap at time `t`
- `population_timeseries.csv` — Raw population data per step
- `agents_final_state.csv` — Final agent positions (used by the web dashboard)
