# 🌐 MosquitoWeb Server

This folder contains the **FastAPI backend server** that acts as the central hub of the system — receiving data from IoT traps, running AI inference, serving the web dashboard, and exposing the ABM simulation results.

## Folder Structure

```
5- MosquitoWeb Server/
└── MosquitoWeb-main/
    ├── Server.py               # Main FastAPI application (all API routes)
    ├── requirements.txt        # Python dependencies
    ├── Dockerfile              # Docker container definition
    ├── Procfile                # Railway deployment process definition
    ├── railway.json            # Railway platform configuration
    ├── .env.example            # Template for environment variables
    ├── generate_preview.py     # Utility to pre-render dashboard preview images
    ├── frontend/               # Frontend web application source
    ├── static/                 # Compiled static assets (CSS, JS, images)
    ├── templates/              # HTML Jinja2 templates
    └── utils/                  # Server-side utilities and data files
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server health check (models loaded, DB configured) |
| `GET` | `/api/dashboard-data` | Last 50 detection records from the database |
| `GET` | `/api/simulation-results` | ABM simulation plots + summary statistics (base64 PNG) |
| `POST` | `/api/generate-grid` | Generate urban grid from OSM data for given coordinates |
| `POST` | `/api/predict` | Run cascade inference on an uploaded image |
| `GET` | `/api/yolo-training-data` | YOLO training metrics from CSV |
| `POST` | `/upload` | IoT device data ingestion (image + detections + environment data) |

---

## Setup & Running Locally

### 1. Clone and install dependencies

```bash
cd MosquitoWeb-main
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your credentials:
```

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (e.g., Neon or Supabase Postgres) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service-role secret key |
| `NGROK_TOKEN` | (Optional) ngrok auth token for local tunnel during development |

### 3. Run the server

```bash
python Server.py
```

The server starts on `http://0.0.0.0:8000`. ONNX models are downloaded from Supabase Storage automatically in a background thread on startup.

### 4. Run with Docker

```bash
docker build -t mosquitoweb .
docker run -p 8000:8000 --env-file .env mosquitoweb
```

---

## Deployment

The server is deployed on **Railway** (`railway.json` / `Procfile`). The frontend is hosted on **Vercel**.

**Live server**: `https://web-production-90e52.up.railway.app`

---

## Database Schema

The server uses a **PostgreSQL** table `yolo_data`:

| Column | Type | Description |
|--------|------|-------------|
| `client_id` | text | Trap identifier (e.g., `rpi_cam1`) |
| `timestamp` | timestamptz | UTC timestamp of the detection |
| `detections` | text (JSON) | Array of detection objects (label, confidence, box) |
| `temperature` | float | Temperature from Arduino BMP280 (°C) |
| `humidity` | float | Humidity / pressure reading |
| `image_url` | text | Public Supabase Storage URL of the captured image |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI + Uvicorn |
| Database ORM | SQLAlchemy |
| Storage | Supabase (PostgreSQL + Object Storage) |
| AI inference | ONNX Runtime, OpenCV, NumPy |
| GIS grid generation | OSMnx, GeoPandas, Shapely, Contextily |
| Data viz | Matplotlib, Pandas |
| Containerization | Docker |
| Deployment | Railway (server), Vercel (frontend) |
