import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_agents(grid_path, agents_csv):

    # --- load data ---
    grid = np.load(grid_path)
    agents = pd.read_csv(agents_csv)

    # --- terrain colormap (same as simulation) ---
    terrain_cmap = ListedColormap([
        [0.9, 0.9, 0.9],   # 0 Empty
        [0.6, 0.6, 0.6],   # 1 Road
        [0.0, 0.6, 0.0],   # 2 Vegetation
        [0.0, 0.0, 0.8],   # 3 Water
        [0.5, 0.3, 0.1],   # 4 Building
    ])

    fig, ax = plt.subplots(figsize=(5,5))

    # --- draw base urban map ---
    ax.imshow(
        grid,
        cmap=terrain_cmap,
        vmin=0,
        vmax=4,
        origin="upper",
        alpha=0.9
    )

    # --- overlay all agents as black points ---
    ax.scatter(
        agents["x"],
        agents["y"],
        s=4,
        c="black",
        marker="o",
        alpha=0.8
    )

    ax.set_title("Final Mosquito Distribution")
    ax.axis("off")

    plt.tight_layout()

    plt.savefig("agents_overlay_map.png", dpi=300)
    plt.close()

    print("[Saved] agents_overlay_map.png")


if __name__ == "__main__":

    plot_agents(
        "grid_data.npy",
        "agents_final_state.csv"
    )