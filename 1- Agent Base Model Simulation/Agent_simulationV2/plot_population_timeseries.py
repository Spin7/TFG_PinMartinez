import pandas as pd
import matplotlib.pyplot as plt


def plot_population(csv_path):

    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract columns
    t = df["time"]
    J = df["J"]
    M = df["M"]
    FU = df["FU"]
    FG = df["FG"]
    total = df["total"]

    # Plot
    plt.figure(figsize=(10,5.5))

    plt.plot(t, J, label="Juveniles")
    plt.plot(t, M, label="Males")
    plt.plot(t, FU, label="Females U")
    plt.plot(t, FG, label="Females G")
    plt.plot(t, total, label="Total", linewidth=2)

    plt.xlabel("Time (Days)")
    plt.ylabel("Population")

    plt.title("Mosquito Population Dynamics")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save figure
    plt.savefig("population_timeseries_plot.png", dpi=300)
    plt.close()

    print("[Saved] population_timeseries_plot.png")


if __name__ == "__main__":

    plot_population("population_timeseries.csv")