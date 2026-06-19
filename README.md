# TFG — Sistema Inteligente de Monitoreo de Mosquitos
### *Trabajo Final de Grado — Pin Martínez | FIUNA*

> **Automatic detection, sex-classification, and population modeling of *Aedes aegypti* mosquitoes using an IoT smart trap, computer vision, and agent-based simulation.**

---

## Overview

This repository contains all the artifacts of the thesis project: from the preliminary proposal to the final memory document, source code, hardware designs, AI models, and academic event presentations.

The system is composed of four integrated components:

| # | Component | Description |
|---|-----------|-------------|
| 1 | **Agent-Based Simulation** | ABM model of mosquito population dynamics over an urban grid |
| 2 | **Vision AI Model** | Cascade (YOLO + MobileNet) pipeline for detection and sex classification |
| 3 | **Smart Trap Hardware** | Physical trap with Raspberry Pi + Arduino + servo + camera |
| 4 | **MosquitoWeb Server** | FastAPI backend + web dashboard for real-time monitoring |

---

## Repository Structure

```
TFG PinMartinez/
│
├── 0- Ante Proyecto/             # Thesis proposal and preliminary models
├── 1- Agent Base Model Simulation/  # ABM mosquito population simulation (Python)
├── 2- Vision AI Model/           # AI model training: YOLO, MobileNet, Cascade
├── 3- Trap 3D Model/             # SolidWorks 3D design of the physical trap
├── 4- Hardware of the Trap/      # Electrical schematics, flow diagrams, cost sheet
├── 5- MosquitoWeb Server/        # FastAPI server + frontend dashboard (deployed)
├── 6- Client Code/               # Embedded client: Raspberry Pi + Arduino firmware
├── 7- TFG Memory/                # Final thesis document (PDF) + LaTeX source
├── Events presentations/         # Slides and papers for academic conferences
├── Multimedia/                   # Photos and videos of the trap
└── References/                   # Bibliographic references and literature
```

---

## System Architecture

```
[Smart Trap]
  ├── Camera (Pi Camera)  →  [Raspberry Pi]
  │                               │
  │                    Cascade Inference (YOLO + MobileNet ONNX)
  │                               │
  │                    Detection → sort command → [Arduino]
  │                               │                   │
  │                               │               Servo motor sorts
  │                               │               mosquito into A/B bin
  │                               │
  │                    Upload (image + detections + env data)
  │                               │
  │                               ▼
  │                      [MosquitoWeb Server]  ←  FastAPI on Railway
  │                               │
  │                    PostgreSQL (Supabase) + Image Storage
  │                               │
  │                               ▼
  │                      [Web Dashboard]  ←  Vercel frontend
  │
  └── [Agent-Based Simulation]  →  population dynamics visualized on dashboard
```

---

## Quick Start

### Server (MosquitoWeb)
See [`5- MosquitoWeb Server/README.md`](./5-%20MosquitoWeb%20Server/README.md)

### Raspberry Pi Client
See [`6- Client Code/README.md`](./6-%20Client%20Code/README.md)

### Agent Simulation
See [`1- Agent Base Model Simulation/README.md`](./1-%20Agent%20Base%20Model%20Simulation/README.md)

### Vision AI Models
See [`2- Vision AI Model/README.md`](./2-%20Vision%20AI%20Model/README.md)

---

## Thesis Documents

All final documents are in [`7- TFG Memory/`](./7-%20TFG%20Memory/):

- **`TFG_PinMartinez.pdf`** — Full thesis memory
- **`TFG_PinMartinez_ResumenEjecutivo.pdf`** — Executive summary
- **`TFG_PinMartinez_Presentacion_large_version.pdf`** — Defense presentation

---

## Academic Events

Presentations and papers submitted to conferences are in [`Events presentations/`](./Events%20presentations/):
- CNMAC 2025
- CNMAC 2026
- Coloquio de la Sociedad Matemática de Paraguay
- X Encuentro de Investigadores

---

## Key Technologies

| Domain | Technologies |
|--------|-------------|
| AI / Vision | YOLOv8, MobileNet V3, ONNX Runtime, OpenCV |
| Simulation | Python, NumPy, SciPy, Matplotlib, OSMnx |
| Backend | FastAPI, SQLAlchemy, Supabase (PostgreSQL + Storage) |
| Frontend | HTML/CSS/JS, deployed on Vercel |
| Embedded | Raspberry Pi (Python), Arduino (C++) |
| Hardware | Servo motor, Hall sensor, BMP280 (temp/pressure), relay |
| Deployment | Railway (server), Vercel (frontend) |

---

*Universidad Nacional de Asunción — Facultad de Ingeniería (FIUNA)*
