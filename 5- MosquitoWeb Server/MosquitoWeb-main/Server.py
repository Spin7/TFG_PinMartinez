import uvicorn
import json
import ast
import csv
import base64
import io
import math
import asyncio
import threading
import numpy as np
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from pydantic import BaseModel
import requests
import os

# Load .env file when running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required in production (env vars set by platform)

# Cascade inference
from utils.cascade_inference import CascadeInference

# ==============================
# MODEL DIR
# ==============================
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_cascade: CascadeInference | None = None


def _load_models_background():
    """Download ONNX models and load them. Runs in a background thread so
    uvicorn can start accepting requests (and pass the healthcheck) immediately."""
    global _cascade

    SUPABASE_URL_LOCAL = os.getenv("SUPABASE_URL")
    os.makedirs(_MODEL_DIR, exist_ok=True)
    model_files = ["yolo_model.onnx", "Mobilnet_mode.onnx"]

    if SUPABASE_URL_LOCAL:
        for model_file in model_files:
            local_path = os.path.join(_MODEL_DIR, model_file)
            if os.path.exists(local_path):
                print(f"[models] ✓ {model_file} cached ({os.path.getsize(local_path) // 1024} KB).")
                continue
            url = f"{SUPABASE_URL_LOCAL}/storage/v1/object/public/models/{model_file}"
            print(f"[models] Downloading {model_file} ...")
            try:
                import requests as _req
                r = _req.get(url, stream=True, timeout=180)
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"[models] ✓ {model_file} downloaded ({os.path.getsize(local_path) // 1024} KB).")
            except Exception as e:
                print(f"[models] ✗ Failed to download {model_file}: {e}")
    else:
        print("[models] SUPABASE_URL not set — skipping download.")

    try:
        _cascade = CascadeInference(
            det_path=os.path.join(_MODEL_DIR, "yolo_model.onnx"),
            cls_path=os.path.join(_MODEL_DIR, "Mobilnet_mode.onnx"),
        )
        print("[models] ✓ ONNX models loaded successfully.")
    except Exception as e:
        print(f"[models] WARNING — Could not load ONNX models: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start model loading in a daemon thread so the server is immediately healthy."""
    thread = threading.Thread(target=_load_models_background, daemon=True, name="model-loader")
    thread.start()
    print("[startup] Model loading started in background thread.")
    yield
    # Nothing special needed on shutdown


# ==============================
# APP
# ==============================
app = FastAPI(
    title="MosquitoAI API",
    version="1.0.0",
    description="REST API for mosquito detection, ABM simulation and dashboard data.",
    lifespan=lifespan,
)

# ==============================
# CONFIGURATION FROM ENV VARS
# ==============================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
NGROK_TOKEN  = os.getenv("NGROK_TOKEN", "")
BUCKET_NAME  = "Mosquitoes"

if not DATABASE_URL:
    print("[Server] WARNING — DATABASE_URL not set. DB endpoints will fail.")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("[Server] WARNING — SUPABASE_URL / SUPABASE_KEY not set. Storage will fail.")

engine   = create_engine(DATABASE_URL) if DATABASE_URL else None
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# ==============================
# CORS — allow all origins (set specific Vercel URL in production)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ==============================
# HELPERS
# ==============================
def smart_parse(value):
    if isinstance(value, (list, dict)):
        return value
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return []


# ==============================
# HEALTH CHECK
# ==============================
@app.get("/api/health")
def health():
    return JSONResponse({
        "status": "ok",
        "models_loaded": _cascade is not None,
        "db_configured": engine is not None,
        "storage_configured": supabase is not None,
    })


# ==============================
# DASHBOARD DATA (replaces Jinja2 table_fragment)
# ==============================
@app.get("/api/dashboard-data")
def dashboard_data():
    """Return the last 50 detection records as a JSON array."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured.")

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM yolo_data ORDER BY timestamp DESC LIMIT 50")
        )
        rows = [dict(r._mapping) for r in result.fetchall()]

    # Serialise timestamps and parse detections
    for r in rows:
        if hasattr(r.get("timestamp"), "isoformat"):
            r["timestamp"] = r["timestamp"].isoformat()
        r["detections_parsed"] = smart_parse(r.get("detections", ""))

    return JSONResponse(content=rows)


# ==============================
# SIMULATION RESULTS API
# ==============================
@app.get("/api/simulation-results")
async def simulation_results():
    """Generate population timeseries + agent distribution plots from stored CSVs.
    Returns both charts as base64 PNG strings plus summary statistics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import ListedColormap

    utils_dir = os.path.join(BASE_DIR, "utils")
    pop_csv  = os.path.join(utils_dir, "population_timeseries.csv")
    ag_csv   = os.path.join(utils_dir, "agents_final_state.csv")
    grid_npy = os.path.join(utils_dir, "grid_data.npy")

    results = {}

    # ── Plot 1: Population Timeseries ──────────────────────────────────────────
    try:
        df = pd.read_csv(pop_csv)
        t, J, M, FU, FG, total = df["time"], df["J"], df["M"], df["FU"], df["FG"], df["total"]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.fill_between(t, total, alpha=0.06, color="#1f2937")
        ax.plot(t, J,     label="Juveniles (J)",  color="#f59e0b", linewidth=1.8)
        ax.plot(t, M,     label="Machos (M)",     color="#3b82f6", linewidth=1.8)
        ax.plot(t, FU,    label="Hembras FU",     color="#ec4899", linewidth=1.8)
        ax.plot(t, FG,    label="Hembras FG",     color="#10b981", linewidth=1.8)
        ax.plot(t, total, label="Total",          color="#1f2937", linewidth=2.8, linestyle="--")
        ax.set_xlabel("Tiempo (dias)", fontsize=11)
        ax.set_ylabel("Poblacion", fontsize=11)
        ax.set_title("Dinamica Poblacional de Mosquitos — Simulacion ABM",
                     fontsize=13, fontweight="bold", pad=14)
        ax.legend(framealpha=0.92, fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        results["population_b64"] = base64.b64encode(buf.read()).decode("utf-8")
        results["population_ok"] = True
    except Exception as exc:
        results["population_b64"] = None
        results["population_ok"] = False
        results["population_error"] = str(exc)

    # ── Plot 2: Agent Distribution Overlay ────────────────────────────────────
    try:
        grid   = np.load(grid_npy)
        agents = pd.read_csv(ag_csv)

        terrain_cmap = ListedColormap([
            [0.95, 0.94, 0.91],   # 0 Empty
            [0.39, 0.39, 0.39],   # 1 Road
            [0.0,  0.59, 0.0 ],   # 2 Vegetation
            [0.0,  0.0,  1.0 ],   # 3 Water
            [0.55, 0.27, 0.07],   # 4 Building
        ])

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(grid, cmap=terrain_cmap, vmin=0, vmax=4, origin="upper", alpha=0.82)

        alive = agents[agents["state"] != "DEAD"] if "state" in agents.columns else agents
        if not alive.empty:
            ax.scatter(alive["x"], alive["y"], s=4,
                       c="black", marker="o", alpha=0.6, label="Agentes")

        ax.set_title("Distribucion Final de Agentes sobre el Mapa Urbano",
                     fontsize=11, fontweight="bold", pad=12)
        ax.legend(loc="upper right", fontsize=9, markerscale=3,
                  framealpha=0.92, edgecolor="#e2e8f0")
        ax.axis("off")
        fig.patch.set_facecolor("#f8fafc")
        fig.tight_layout()

        buf2 = io.BytesIO()
        fig.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf2.seek(0)
        results["distribution_b64"] = base64.b64encode(buf2.read()).decode("utf-8")
        results["distribution_ok"] = True
    except Exception as exc:
        results["distribution_b64"] = None
        results["distribution_ok"] = False
        results["distribution_error"] = str(exc)

    # ── Summary Statistics ─────────────────────────────────────────────────────
    try:
        df_p = pd.read_csv(pop_csv)
        results["total_days"]    = round(float(df_p["time"].max()), 1)
        results["initial_total"] = int(df_p["total"].iloc[0])
        results["peak_total"]    = int(df_p["total"].max())
        results["final_total"]   = int(df_p["total"].iloc[-1])
        results["final_J"]       = int(df_p["J"].iloc[-1])
        results["final_M"]       = int(df_p["M"].iloc[-1])
        results["final_FU"]      = int(df_p["FU"].iloc[-1])
        results["final_FG"]      = int(df_p["FG"].iloc[-1])
    except Exception:
        pass

    try:
        ag = pd.read_csv(ag_csv)
        results["n_agents"] = int(len(ag))
        results["state_counts"] = {k: int(v) for k, v in ag["state"].value_counts().items()}
    except Exception:
        pass

    return JSONResponse(content=results)


# ==============================
# GRID GENERATION API
# ==============================
class GridRequest(BaseModel):
    lat: float
    lon: float
    map_size_m: int = 500
    cell_size_m: int = 5


@app.post("/api/generate-grid")
async def generate_grid_api(req: GridRequest):
    """Generate a grid from OSM data and return satellite + grid images as base64."""
    try:
        import osmnx as ox
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.colors import ListedColormap
        from shapely.geometry import box as shapely_box, Point
        import contextily as ctx
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Dependencia faltante: {e}")

    center_lat  = req.lat
    center_lon  = req.lon
    map_size_m  = req.map_size_m
    cell_size_m = req.cell_size_m

    CELL_TYPES = {
        0: {"color": [242, 239, 233]},
        1: {"color": [100, 100, 100]},
        2: {"color": [0,   150,   0]},
        3: {"color": [0,     0, 255]},
        4: {"color": [139,  69,  19]},
        5: {"color": [255,   0, 255]},
    }

    def _blocking_work():
        import matplotlib.pyplot as plt

        dist = map_size_m // 2
        R    = 6378137

        def safe_features(tags):
            try:
                return ox.features_from_point((center_lat, center_lon), tags, dist=dist)
            except Exception:
                import geopandas as _gpd
                return _gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        buildings  = safe_features({"building": True})
        water      = safe_features({"natural": ["water", "wetland"]})
        vegetation = safe_features({
            "landuse": ["forest","grass","meadow","orchard","vineyard"],
            "leisure": ["park","garden"],
            "natural": ["wood","scrub"],
        })
        graph = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type="all")
        _, roads = ox.graph_to_gdfs(graph)
        roads_proj = roads.to_crs(epsg=3857)

        dlat  = (dist / R) * (180 / math.pi)
        dlon  = (dist / (R * math.cos(math.radians(center_lat)))) * (180 / math.pi)
        north, south = center_lat + dlat, center_lat - dlat
        east,  west  = center_lon + dlon, center_lon - dlon
        bbox_poly = shapely_box(west, south, east, north)

        import geopandas as gpd2
        gdf_bbox = gpd2.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326").to_crs(epsg=3857)

        fig_sat, ax_sat = plt.subplots(figsize=(8, 8))
        gdf_bbox.boundary.plot(ax=ax_sat, linewidth=0)
        ctx.add_basemap(ax_sat, source=ctx.providers.Esri.WorldImagery)
        ax_sat.set_axis_off()
        buf_sat = io.BytesIO()
        fig_sat.savefig(buf_sat, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig_sat)
        buf_sat.seek(0)
        sat_b64 = base64.b64encode(buf_sat.read()).decode("utf-8")

        grid_n = int(map_size_m / cell_size_m)
        half   = map_size_m / 2
        grid   = np.zeros((grid_n, grid_n), dtype=np.uint8)

        def cell_polygon(row, col):
            x   = col * cell_size_m - half
            y   = half - row * cell_size_m
            lat = center_lat + (y / R) * (180 / math.pi)
            lon = center_lon + (x / R) * (180 / math.pi) / math.cos(math.radians(center_lat))
            from shapely.geometry import Point as Pt
            return Pt(lon, lat).buffer(0.00001)

        def rasterize_polygons(gdf, class_id, min_cover):
            if gdf is None or gdf.empty:
                return
            gdf_ll = gdf.to_crs("EPSG:4326")
            for r in range(grid_n):
                for c in range(grid_n):
                    poly = cell_polygon(r, c)
                    hits = gdf_ll[gdf_ll.intersects(poly)]
                    if hits.empty:
                        continue
                    overlap = sum(g.intersection(poly).area for g in hits.geometry)
                    if (overlap / poly.area) >= min_cover and class_id > grid[r, c]:
                        grid[r, c] = class_id

        def rasterize_roads(roads_projected):
            if roads_projected is None or roads_projected.empty:
                return
            rb = roads_projected.copy()
            rb["geometry"] = rb.buffer(3)
            rb = rb.to_crs("EPSG:4326")
            for r in range(grid_n):
                for c in range(grid_n):
                    poly = cell_polygon(r, c)
                    if rb.intersects(poly).any() and grid[r, c] == 0:
                        grid[r, c] = 1

        rasterize_polygons(buildings,  4, 0.30)
        rasterize_polygons(water,      3, 0.20)
        rasterize_polygons(vegetation, 2, 0.45)
        rasterize_roads(roads_proj)

        colors = [np.array(CELL_TYPES[k]["color"]) / 255 for k in sorted(CELL_TYPES)]
        from matplotlib.colors import ListedColormap as LCM
        cmap = LCM(colors)

        fig_grid, ax_grid = plt.subplots(figsize=(8, 8))
        ax_grid.imshow(grid, cmap=cmap, vmin=0, vmax=len(CELL_TYPES) - 1)
        ax_grid.set_title(f"Grid {grid_n}\u00d7{grid_n} — celda {cell_size_m} m", fontsize=11)
        ax_grid.axis("off")
        buf_grid = io.BytesIO()
        fig_grid.savefig(buf_grid, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig_grid)
        buf_grid.seek(0)
        grid_b64 = base64.b64encode(buf_grid.read()).decode("utf-8")

        return {
            "satellite_b64": sat_b64,
            "grid_b64":      grid_b64,
            "grid_size":     grid_n,
            "map_size_m":    map_size_m,
            "cell_size_m":   cell_size_m,
        }

    try:
        result = await asyncio.to_thread(_blocking_work)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# INFERENCE DEMO
# ==============================
@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    """Receive an image, run cascade pipeline, return annotated image (base64 PNG) + detections."""
    if _cascade is None:
        raise HTTPException(
            status_code=503,
            detail="ONNX models are not available on this server."
        )

    file_bytes = await image.read()
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        boxes, preds, det_scores, cls_scores = _cascade.predict(frame)
        annotated_bytes = _cascade.predict_and_draw(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    CLASS_NAMES = ["Female", "Male"]
    detections = []
    for box, pred, d_sc, c_sc in zip(boxes, preds, det_scores, cls_scores):
        x1, y1, x2, y2 = map(int, box)
        detections.append({
            "label":    CLASS_NAMES[pred],
            "cls_id":   pred,
            "det_conf": round(float(d_sc), 4),
            "cls_conf": round(float(c_sc), 4),
            "box":      {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    img_b64 = base64.b64encode(annotated_bytes).decode("utf-8")

    return JSONResponse({
        "total":      len(detections),
        "female":     sum(1 for d in detections if d["label"] == "Female"),
        "male":       sum(1 for d in detections if d["label"] == "Male"),
        "detections": detections,
        "image_b64":  img_b64,
    })


# ==============================
# YOLO TRAINING DATA
# ==============================
@app.get("/api/yolo-training-data")
def yolo_training_data():
    """Read utils/yolo_results.csv and return training metrics as JSON."""
    csv_path = os.path.join(BASE_DIR, "utils", "yolo_results.csv")
    result = {
        "epochs": [], "train_box_loss": [], "train_cls_loss": [], "train_dfl_loss": [],
        "val_box_loss": [], "val_cls_loss": [], "val_dfl_loss": [],
        "precision": [], "recall": [], "map50": [], "map50_95": [],
    }
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    result["epochs"].append(int(float(row["epoch"])))
                    result["train_box_loss"].append(float(row["train/box_loss"]))
                    result["train_cls_loss"].append(float(row["train/cls_loss"]))
                    result["train_dfl_loss"].append(float(row["train/dfl_loss"]))
                    result["val_box_loss"].append(float(row["val/box_loss"]))
                    result["val_cls_loss"].append(float(row["val/cls_loss"]))
                    result["val_dfl_loss"].append(float(row["val/dfl_loss"]))
                    result["precision"].append(float(row["metrics/precision(B)"]))
                    result["recall"].append(float(row["metrics/recall(B)"]))
                    result["map50"].append(float(row["metrics/mAP50(B)"]))
                    result["map50_95"].append(float(row["metrics/mAP50-95(B)"]))
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "CSV not found"})
    return JSONResponse(content=result)


# ==============================
# UPLOAD (IoT device data ingest)
# ==============================
@app.post("/upload")
async def upload_data(
    client_id: str = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    detections: str = Form(...),
    image: UploadFile = File(...),
):
    if supabase is None or engine is None:
        raise HTTPException(status_code=503, detail="Storage or DB not configured.")

    file_bytes = await image.read()

    timestamp = datetime.now().isoformat().replace(":", "-")
    filename  = f"{timestamp}_{client_id}.jpg"

    upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{filename}"
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "image/jpeg",
    }

    r = requests.put(upload_url, headers=headers, data=file_bytes)

    print("\n===== SUPABASE DEBUG =====")
    print("STATUS:", r.status_code)
    print("RESPONSE:", r.text)
    print("==========================\n")

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=r.text)

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"

    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO yolo_data
            (client_id, timestamp, detections, temperature, humidity, image_url)
            VALUES (:client_id, :timestamp, :detections, :temperature, :humidity, :image_url)
            """),
            {
                "client_id":   client_id,
                "timestamp":   datetime.now(timezone.utc),
                "detections":  detections,
                "temperature": temperature,
                "humidity":    humidity,
                "image_url":   public_url,
            },
        )
        conn.commit()

    return {"status": "ok", "image_url": public_url}


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))

    if NGROK_TOKEN:
        try:
            from pyngrok import ngrok
            ngrok_tunnel = ngrok.connect(port)
            print("Public URL:", ngrok_tunnel)
        except Exception as e:
            print(f"[ngrok] Could not start tunnel: {e}")
    else:
        print(f"[ngrok] NGROK_TOKEN not set — running on localhost:{port}")

    uvicorn.run(app, host="0.0.0.0", port=port)
