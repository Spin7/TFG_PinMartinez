"""
generate_preview.py
Generates the grid preview image from the real grid_data.npy produced by
utils/Grid_Generator.py (CENTER_LAT=-25.2637, CENTER_LON=-57.5759,
MAP_SIZE_M=1000, CELL_SIZE_M=3 → 333×333 cells).
Saves to static/grid_preview.png and copies
figures/satellite_reference.png → static/satellite_preview.png.
"""
import numpy as np
import os, shutil

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    raise

# ── Color palette (must match Grid_Generator.py) ────────────────────────────
CELL_TYPES = {
    0: [242, 239, 233],   # empty    (cream)
    1: [100, 100, 100],   # road     (grey)
    2: [  0, 150,   0],   # vegetation (green)
    3: [  0,   0, 255],   # water    (blue)
    4: [139,  69,  19],   # building (brown)
    5: [255,   0, 255],   # trap     (magenta)
}

BASE = os.path.dirname(__file__)

# ── 1. Load real grid from utils/grid_data.npy ───────────────────────────────
grid_path = os.path.join(BASE, "utils", "grid_data.npy")
if not os.path.exists(grid_path):
    raise FileNotFoundError(
        f"grid_data.npy not found at {grid_path}.\n"
        "Run utils/Grid_Generator.py first to generate it."
    )

grid = np.load(grid_path)
grid_h, grid_w = grid.shape
cell_size_m = 3
map_size_m  = grid_h * cell_size_m   # 333 × 3 = 999 m  (≈1000 m)

print("Loaded grid: {}x{} cells  |  cell={} m  |  area~{}x{} m".format(grid_h, grid_w, cell_size_m, map_size_m, map_size_m))

# ── 2. Render and save ───────────────────────────────────────────────────────
colors = [np.array(CELL_TYPES[k]) / 255.0 for k in sorted(CELL_TYPES)]
cmap   = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(CELL_TYPES) - 1, interpolation="nearest")
ax.set_title(
    f"Grid {grid_h}\u00d7{grid_w} \u2014 celda {cell_size_m}\u202fm  |  "
    f"\u00c1rea \u2248{map_size_m}\u00d7{map_size_m}\u202fm  |  Asunci\u00f3n, PY",
    fontsize=10, pad=8
)
ax.axis("off")

os.makedirs(os.path.join(BASE, "static"), exist_ok=True)
out_grid = os.path.join(BASE, "static", "grid_preview.png")
fig.savefig(out_grid, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Grid preview saved  -> {out_grid}")

# ── 3. Copy satellite reference image ────────────────────────────────────────
sat_src = os.path.join(BASE, "figures", "satellite_reference.png")
sat_dst = os.path.join(BASE, "static", "satellite_preview.png")
if os.path.exists(sat_src):
    shutil.copy2(sat_src, sat_dst)
    print(f"Satellite image copied -> {sat_dst}")
else:
    print(f"WARNING: satellite_reference.png not found at {sat_src}")

print("Done!")
