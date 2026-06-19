"""
sensitivity_fast.py
===================
Análisis de sensibilidad global mediante Latin Hypercube Sampling (LHS) + PRCC.
Versión optimizada para ejecución rápida (~10-20 minutos según hardware).

OPTIMIZACIONES vs sensitivity_lhs.py / sobol_sensitivity.py
------------------------------------------------------------
1. Paralelismo real  : multiprocessing.Pool evalúa las muestras en todos
   los núcleos disponibles (detectable automáticamente).
2. Configuración reducida: N_SAMPLES=40, N_RUNS=1, T_MAX=60 días.
   Con 13 parámetros y 40 muestras el PRCC tiene buena estabilidad estadística
   (regla empírica: N > 4k → 40 > 52 es límite; usar N=60 si se quiere margen).
3. Early stopping: si la población se extingue antes de T_MAX, el loop
   de steps se interrumpe → simulaciones de extinción son muy rápidas.
4. Supresión de stdout: los prints de Simulation/Environment se silencian
   en los workers para no saturar la consola.
5. Caché de parámetros de movimiento: se pasa env_config directamente;
   la simulación ya tiene compute_potential=False por defecto.

MÉTODO
------
- LHS estratificado (pyDOE2 o implementación propia).
- PRCC (Partial Rank Correlation Coefficient) como métrica de sensibilidad.

SALIDAS
-------
    figures/sensitivity_fast/prcc_heatmap.png
    figures/sensitivity_fast/tornado_<qoi>.png
    results/fast_lhs_samples.csv
    results/fast_lhs_qoi.csv
    results/fast_lhs_prcc.csv
"""

import os
import sys
import time
import contextlib
import io
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

# ── Asegurar que el simulador es importable ───────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

# =============================================================================
# CONFIGURACIÓN  ← ajusta aquí
# =============================================================================

N_SAMPLES   = 100        # muestras LHS  (regla: N > 3·k; k=13 → 40 es cómodo)
N_RUNS      = 1         # réplicas por muestra (1 es suficiente para screening)
T_MAX       = 80.0      # días simulados (60 d captura dinámica transiente)
DELTA_T     = 1.0       # paso de tiempo (días)
SEED_LHS    = 2024
GRID_PATH   = os.path.join(_DIR, "grid_data.npy")
CONFIG_PATH = os.path.join(_DIR, "environment_params.json")
N_BREEDING  = 20        # sitios de cría aleatorios

# Número de workers paralelos (None → detectar automáticamente)
# Poner PARALLEL = False si hay problemas con multiprocessing en Windows
PARALLEL    = True
N_WORKERS   = None      # p.ej. 4 para forzar 4 núcleos

INIT_COUNTS = {
    "JUVENILE":        200,   # reducido vs original para acelerar cada step
    "ADULT_MALE":       80,
    "ADULT_FEMALE_U":   40,
    "ADULT_FEMALE_G":   40,
}

OUT_DIR = os.path.join(_DIR, "figures", "sensitivity_fast")
RES_DIR = os.path.join(_DIR, "results")

# =============================================================================
# PARÁMETROS Y RANGOS  (13 parámetros, ±50 % alrededor del valor base)
# =============================================================================

PARAMS = {
    "mu_J":              (0.05,   0.025,  0.10),
    "mu_M":              (0.10,   0.05,   0.20),
    "mu_F":              (0.08,   0.04,   0.16),
    "gamma":             (0.07,   0.035,  0.14),
    "alpha":             (0.001,  0.0005, 0.002),
    "Kc":                (100.0,  50.0,   200.0),
    "beta":              (0.02,   0.01,   0.04),
    "f":                 (5.0,    2.5,    10.0),
    "mating_radius":     (5,      2,      10),
    "density_radius":    (3,      1,      6),
    "oviposition_radius":(10,     5,      20),
    "movement_radius":   (3,      1,      6),
    "beta_0":            (10.0,   5.0,    20.0),
}

PARAM_LABELS = {
    "mu_J":              r"$\mu_J$ (mort. juvenil)",
    "mu_M":              r"$\mu_M$ (mort. macho)",
    "mu_F":              r"$\mu_F$ (mort. hembra)",
    "gamma":             r"$\gamma$ (maduración)",
    "alpha":             r"$\alpha$ (compet. larval)",
    "Kc":                r"$K_c$ (cap. de carga)",
    "beta":              r"$\beta$ (apaream.)",
    "f":                 r"$f$ (oviposición)",
    "mating_radius":     r"$r_m$ (radio apaream.)",
    "density_radius":    r"$r_d$ (radio densidad)",
    "oviposition_radius":r"$r_o$ (radio ovipos.)",
    "movement_radius":   r"$R$ (radio movim.)",
    "beta_0":            r"$\beta_0$ (sesgo explor.)",
}

QOI_LABELS = {
    "N_total_final": "Población total final",
    "N_FG_mean":     "Hembras grávidas (media)",
    "peak_total":    "Pico población total",
    "N_J_mean":      "Juveniles (media)",
    "extinction":    "Extinción (0/1)",
}

# =============================================================================
# GENERADOR LHS
# =============================================================================

def latin_hypercube_sample(n_samples, param_dict, seed):
    """
    Genera una matriz LHS de forma (n_samples, n_params).
    Sin dependencias externas (no requiere pyDOE2).
    """
    rng = np.random.RandomState(seed)
    keys = list(param_dict.keys())
    k = len(keys)

    cut = np.linspace(0, 1, n_samples + 1)
    lhs_unit = np.zeros((n_samples, k))

    for j in range(k):
        u = rng.uniform(low=cut[:-1], high=cut[1:])
        lhs_unit[:, j] = rng.permutation(u)

    samples = np.zeros_like(lhs_unit)
    for j, key in enumerate(keys):
        _, lo, hi = param_dict[key]
        samples[:, j] = lo + lhs_unit[:, j] * (hi - lo)

    return pd.DataFrame(samples, columns=keys)


# =============================================================================
# WRAPPER DE SIMULACIÓN  (diseñado para ejecutarse en un worker)
# =============================================================================

def _silence():
    """Context manager que suprime stdout durante la construcción del entorno."""
    return contextlib.redirect_stdout(io.StringIO())


def _run_single(args):
    """
    Función de top-level (necesaria para multiprocessing en Windows).
    args = (params_dict, sim_seed, grid_path, config_path, init_counts,
            n_breeding, t_max, delta_t)
    Devuelve dict con QoIs.
    """
    (params_dict, sim_seed,
     grid_path, config_path, init_counts,
     n_breeding, t_max, delta_t) = args

    # Importar aquí para que cada worker tenga su propio módulo
    from Simulation_Mosquitoes import Simulation

    env_config = {k: float(v) for k, v in params_dict.items()}
    for rk in ("mating_radius", "density_radius", "movement_radius"):
        env_config[rk] = max(1, int(round(env_config[rk])))
    env_config["oviposition_radius"] = float(env_config["oviposition_radius"])

    with _silence():
        sim = Simulation(
            grid_path=grid_path,
            delta_t=delta_t,
            containers=0,
            num_random_breeding_sites=n_breeding,
            seed=sim_seed,
            init_counts=init_counts,
            env_config=env_config,
            config_path=config_path,
            compute_potential=False,
        )

    steps = int(t_max / delta_t)
    for _ in range(steps):
        sim.step()
        # Early stopping: extinción → salir antes de T_MAX
        if len(sim.agents) == 0:
            break

    history = sim.history
    total   = np.array(history["total"])
    fg      = np.array(history["FG"])
    juv     = np.array(history["J"])

    return {
        "N_total_final": float(total[-1]) if len(total) else 0.0,
        "N_FG_mean":     float(fg.mean())  if len(fg)    else 0.0,
        "peak_total":    float(total.max()) if len(total) else 0.0,
        "N_J_mean":      float(juv.mean()) if len(juv)   else 0.0,
        "extinction":    float(total[-1] == 0) if len(total) else 1.0,
    }


def evaluate_all(samples_df, n_runs, base_seed,
                 grid_path, config_path, init_counts, n_breeding,
                 t_max, delta_t, n_workers, parallel=True):
    """
    Evalúa todas las filas de samples_df.
    Si parallel=True usa multiprocessing.Pool; si falla, cae en modo secuencial.
    Cada muestra se corre n_runs veces y se promedia.
    """
    # Construir lista de tareas
    tasks = []
    for i, (_, row) in enumerate(samples_df.iterrows()):
        for r in range(n_runs):
            seed = base_seed + i * 100 + r
            tasks.append((
                row.to_dict(), seed,
                grid_path, config_path, init_counts,
                n_breeding, t_max, delta_t
            ))

    n_total = len(tasks)
    n_sims  = len(samples_df)
    print(f"  Total simulaciones: {n_total}  ({n_sims} muestras × {n_runs} réplica/s)")

    results_flat = []
    t0 = time.time()

    if parallel:
        workers = n_workers or max(1, mp.cpu_count() - 1)
        print(f"  Modo paralelo: {workers} workers  (CPUs: {mp.cpu_count()})")
        try:
            with mp.Pool(processes=workers) as pool:
                for k, res in enumerate(pool.imap(_run_single, tasks), 1):
                    results_flat.append(res)
                    elapsed = time.time() - t0
                    rate    = elapsed / k
                    remaining = rate * (n_total - k)
                    print(f"  [{k:4d}/{n_total}]  "
                          f"N_tot={res['N_total_final']:.0f}  "
                          f"ext={res['extinction']:.0f}  "
                          f"ETA {remaining/60:.1f} min", end="\r", flush=True)
            print(f"\n  Completado en {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"\n  AVISO: paralelismo falló ({e}). Cambiando a modo secuencial...")
            results_flat = []
            parallel = False

    if not parallel:
        print(f"  Modo secuencial (sin paralelismo)")
        for k, task in enumerate(tasks, 1):
            res = _run_single(task)
            results_flat.append(res)
            elapsed = time.time() - t0
            rate    = elapsed / k
            remaining = rate * (n_total - k)
            print(f"  [{k:4d}/{n_total}]  "
                  f"N_tot={res['N_total_final']:.0f}  "
                  f"ext={res['extinction']:.0f}  "
                  f"ETA {remaining/60:.1f} min", end="\r", flush=True)
        print(f"\n  Completado en {(time.time()-t0)/60:.1f} min")

    # Promediar réplicas por muestra
    qoi_rows = []
    for i in range(n_sims):
        chunk = results_flat[i * n_runs: (i + 1) * n_runs]
        mean_qoi = {}
        for key in chunk[0]:
            mean_qoi[key] = float(np.mean([c[key] for c in chunk]))
        qoi_rows.append(mean_qoi)

    return pd.DataFrame(qoi_rows)


# =============================================================================
# PRCC
# =============================================================================

def _ols_residuals(y, X_mat):
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)
    if X_mat.shape[1] == 0:
        return y - y.mean()
    X_aug = np.column_stack([np.ones(len(y)), X_mat])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return y - X_aug @ beta


def compute_prcc(X: pd.DataFrame, Y: pd.DataFrame):
    X_rank = X.rank()
    Y_rank = Y.rank()
    params = list(X.columns)
    qois   = list(Y.columns)
    prcc   = pd.DataFrame(index=params, columns=qois, dtype=float)

    for px in params:
        other_X = [p for p in params if p != px]
        res_x = _ols_residuals(X_rank[px].values, X_rank[other_X].values)
        for qoi in qois:
            res_y = _ols_residuals(Y_rank[qoi].values, X_rank[other_X].values)
            r, _ = stats.pearsonr(res_x, res_y)
            prcc.loc[px, qoi] = r

    return prcc.astype(float)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_COL  = "white"
GRID_COL  = "#2d3542"


def plot_prcc_heatmap(prcc: pd.DataFrame, out_dir: str):
    labels_x = [PARAM_LABELS.get(p, p) for p in prcc.index]
    labels_y = [QOI_LABELS.get(q, q) for q in prcc.columns]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    data = prcc.values
    vmax = max(abs(data.min()), abs(data.max()), 0.01)
    im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PRCC", color=TEXT_COL, fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    ax.set_xticks(range(len(labels_y)))
    ax.set_xticklabels(labels_y, color=TEXT_COL, fontsize=9, rotation=25, ha="right")
    ax.set_yticks(range(len(labels_x)))
    ax.set_yticklabels(labels_x, color=TEXT_COL, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(data[i, j]) > 0.4 else TEXT_COL)

    ax.set_title("PRCC – Sensibilidad Global (LHS rápido)", color=TEXT_COL, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT_COL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    plt.tight_layout()
    path = os.path.join(out_dir, "prcc_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Saved] {path}")


def plot_tornado(prcc: pd.DataFrame, qoi: str, out_dir: str):
    col    = prcc[qoi].sort_values(key=abs)
    labels = [PARAM_LABELS.get(p, p) for p in col.index]
    values = col.values
    colors = ["#e05252" if v < 0 else "#52aee0" for v in values]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    bars = ax.barh(range(len(values)), values, color=colors, edgecolor=GRID_COL, height=0.6)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, color=TEXT_COL, fontsize=10)
    ax.set_xlabel("PRCC", color=TEXT_COL, fontsize=11)
    ax.set_title(f"Tornado PRCC – {QOI_LABELS.get(qoi, qoi)}", color=TEXT_COL, fontsize=12)
    ax.axvline(0,    color=TEXT_COL, linewidth=0.8)
    ax.axvline( 0.5, color=TEXT_COL, linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(-0.5, color=TEXT_COL, linewidth=0.5, linestyle="--", alpha=0.4)
    ax.tick_params(colors=TEXT_COL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    for i, (v, _bar) in enumerate(zip(values, bars)):
        ax.text(v + 0.01 * np.sign(v) if v != 0 else 0.01,
                i, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right",
                color=TEXT_COL, fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"tornado_{qoi}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Saved] {path}")


def plot_scatter_top(X: pd.DataFrame, Y: pd.DataFrame,
                     prcc: pd.DataFrame, qoi: str,
                     out_dir: str, top_n: int = 4):
    top_params = prcc[qoi].abs().nlargest(top_n).index.tolist()
    fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 4))
    fig.patch.set_facecolor(DARK_BG)
    if top_n == 1:
        axes = [axes]

    for ax, px in zip(axes, top_params):
        ax.set_facecolor(PANEL_BG)
        ax.scatter(X[px], Y[qoi], c=Y[qoi], cmap="plasma",
                   s=30, alpha=0.8, edgecolors="none")
        m, b, *_ = stats.linregress(X[px], Y[qoi])
        xline = np.linspace(X[px].min(), X[px].max(), 100)
        ax.plot(xline, m * xline + b, color="#ffd700", linewidth=1.5, linestyle="--")
        ax.set_xlabel(PARAM_LABELS.get(px, px), color=TEXT_COL, fontsize=9)
        ax.set_ylabel(QOI_LABELS.get(qoi, qoi) if ax == axes[0] else "",
                      color=TEXT_COL, fontsize=9)
        ax.set_title(f"PRCC={prcc.loc[px, qoi]:.2f}", color=TEXT_COL, fontsize=9)
        ax.tick_params(colors=TEXT_COL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)

    fig.suptitle(f"Top {top_n} parámetros – {QOI_LABELS.get(qoi, qoi)}",
                 color=TEXT_COL, fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"scatter_{qoi}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Saved] {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    t_total = time.time()
    print("=" * 65)
    print("  Análisis de Sensibilidad LHS+PRCC (versión rápida)")
    print(f"  N_SAMPLES={N_SAMPLES}  N_RUNS={N_RUNS}  T_MAX={T_MAX}d  dt={DELTA_T}d")
    print(f"  Parámetros: {len(PARAMS)}  |  Grid: {GRID_PATH}")
    print("=" * 65)

    # ── 1. Generar diseño LHS ─────────────────────────────────────────────────
    print("\n[1/4] Generando diseño LHS...")
    samples_df = latin_hypercube_sample(N_SAMPLES, PARAMS, SEED_LHS)
    samples_df.to_csv(os.path.join(RES_DIR, "fast_lhs_samples.csv"), index=False)
    print(f"  {N_SAMPLES} muestras generadas → results/fast_lhs_samples.csv")

    # ── 2. Evaluar en paralelo ────────────────────────────────────────────────
    print(f"\n[2/4] Evaluando muestras en paralelo...")
    qoi_df = evaluate_all(
        samples_df, N_RUNS, SEED_LHS,
        GRID_PATH, CONFIG_PATH, INIT_COUNTS, N_BREEDING,
        T_MAX, DELTA_T, N_WORKERS,
        parallel=PARALLEL
    )
    qoi_df.to_csv(os.path.join(RES_DIR, "fast_lhs_qoi.csv"), index=False)
    print(f"  QoI guardadas → results/fast_lhs_qoi.csv")

    # Verificar varianza
    for col in qoi_df.columns:
        if qoi_df[col].std() < 1e-10:
            print(f"  AVISO: varianza nula en '{col}' — añadiendo ruido mínimo.")
            qoi_df[col] += np.random.default_rng(42).normal(0, 1e-6, len(qoi_df))

    # ── 3. Calcular PRCC ──────────────────────────────────────────────────────
    print("\n[3/4] Calculando PRCC...")
    prcc = compute_prcc(samples_df, qoi_df)

    prcc_csv = prcc.copy()
    prcc_csv.index = [PARAM_LABELS.get(p, p) for p in prcc.index]
    prcc_csv.to_csv(os.path.join(RES_DIR, "fast_lhs_prcc.csv"))
    print("  PRCC guardada → results/fast_lhs_prcc.csv")

    print("\n  ── Resumen PRCC (N_total_final, ordenado por |PRCC|) ──")
    sorted_prcc = prcc["N_total_final"].abs().sort_values(ascending=False)
    for p, v in sorted_prcc.items():
        sign  = "+" if prcc.loc[p, "N_total_final"] >= 0 else "-"
        stars = "***" if v > 0.5 else ("**" if v > 0.3 else ("*" if v > 0.1 else ""))
        print(f"    {sign}{v:.3f}  {PARAM_LABELS.get(p, p):<35} {stars}")

    # ── 4. Figuras ────────────────────────────────────────────────────────────
    print("\n[4/4] Generando figuras...")
    plot_prcc_heatmap(prcc, OUT_DIR)
    for qoi in qoi_df.columns:
        plot_tornado(prcc, qoi, OUT_DIR)
        plot_scatter_top(samples_df, qoi_df, prcc, qoi, OUT_DIR)

    elapsed = time.time() - t_total
    print(f"\n✓ Análisis completado en {elapsed/60:.1f} minutos.")
    print(f"  Figuras → {OUT_DIR}")
    print(f"  Datos   → {RES_DIR}")


if __name__ == "__main__":
    # Necesario en Windows para multiprocessing
    mp.freeze_support()
    main()
