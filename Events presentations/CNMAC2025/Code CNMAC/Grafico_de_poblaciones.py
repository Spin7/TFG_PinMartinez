import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def analizar_resultados(pattern="simulacion_*.xlsx"):
    # Buscar todos los archivos de simulación
    files = sorted(glob.glob(pattern))
    if not files:
        print("⚠️ No se encontraron archivos de simulación.")
        return

    print(f"Archivos encontrados: {files}")

    # Leer todos los DataFrames
    dfs = [pd.read_excel(f) for f in files]

    # Usamos la columna 'time' como referencia
    time = dfs[0]["time"].values

    # Apilamos resultados en matrices
    total_matrix = np.array([df["total"].values for df in dfs])
    adults_matrix = np.array([df["adults"].values for df in dfs])
    juveniles_matrix = np.array([df["juveniles"].values for df in dfs])

    # === Calcular estadísticos ===
    stats = {
        "total_mean": total_matrix.mean(axis=0),
        "total_var": total_matrix.var(axis=0),
        "total_std": total_matrix.std(axis=0),

        "adults_mean": adults_matrix.mean(axis=0),
        "adults_var": adults_matrix.var(axis=0),
        "adults_std": adults_matrix.std(axis=0),

        "juveniles_mean": juveniles_matrix.mean(axis=0),
        "juveniles_var": juveniles_matrix.var(axis=0),
        "juveniles_std": juveniles_matrix.std(axis=0),
    }

    # === Graficar ===
    plt.figure(figsize=(7, 6))

    # Total
    plt.plot(time, stats["total_mean"], "k-", label="Total (mean)")
    plt.fill_between(time,
                     stats["total_mean"] - stats["total_std"],
                     stats["total_mean"] + stats["total_std"],
                     color="gray", alpha=0.3, label="Total ± std")

    # Adults
    plt.plot(time, stats["adults_mean"], "r-", label="Adults (mean)")
    plt.fill_between(time,
                     stats["adults_mean"] - stats["adults_std"],
                     stats["adults_mean"] + stats["adults_std"],
                     color="red", alpha=0.2, label="Adults ± std")

    # Juveniles
    plt.plot(time, stats["juveniles_mean"], "b-", label="Juveniles (mean)")
    plt.fill_between(time,
                     stats["juveniles_mean"] - stats["juveniles_std"],
                     stats["juveniles_mean"] + stats["juveniles_std"],
                     color="blue", alpha=0.2, label="Juveniles ± std")

    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.title("Population dynamics (mean ± std across simulations)")
    plt.legend()
    plt.savefig("population_stats.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("✅ Gráfico generado: population_stats.png")

    # === Graficar juveniles por sitio ===
    # Identificar columnas de sitios (site_x_y)
    site_columns = [c for c in dfs[0].columns if c.startswith("site_")]

    if site_columns:
        plt.figure(figsize=(7, 6))
        for site in site_columns:
            # Construimos matriz para este sitio en todas las simulaciones
            site_matrix = np.array([df[site].values for df in dfs])
            site_mean = site_matrix.mean(axis=0)
            plt.plot(time, site_mean, label=site)

        plt.xlabel("Days")
        plt.ylabel("Juveniles")
        plt.title("Mean juveniles per breeding site across simulations")
        plt.legend(ncol=2, fontsize=8)
        plt.savefig("juveniles_per_site_mean.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("✅ Gráfico generado: juveniles_per_site_mean.png")

# Ejecutar análisis
if __name__ == "__main__":
    analizar_resultados("simulacion_*.xlsx")
