# Vision AI Model — Mosquito Detection & Sex Classification

This folder contains all the work related to the computer vision pipeline used to detect mosquitoes and classify them by sex (Male / Female).

## Overview

A two-stage **Cascade inference pipeline** was developed:
1. **Stage 1 — Detection**: YOLOv8 detects mosquito bounding boxes in the camera frame.
2. **Stage 2 — Classification**: MobileNet V3 classifies each detected crop as **Male** or **Female**.

Both models are exported to **ONNX** format for efficient deployment on the Raspberry Pi and the server.

---

## Folder Structure

```
2- Vision AI Model/
└── WORKSPACE_CASCADE/
    ├── 0_Datasets/          # Image datasets for training
    ├── 1_YOLO/              # YOLOv8 detection model: training, experiments, results
    ├── 2_MOBILENET/         # MobileNet V3 classification model
    ├── 3_CASCADE/           # End-to-end cascade pipeline testing
    ├── Models/              # Trained model files (.onnx, etc.)
    ├── Trainers/            # Training scripts
    └── Utils/               # Shared utility scripts
```

---

## Datasets (`0_Datasets/`)

| Folder | Contents |
|--------|----------|
| `Male_and_female/` | Detection dataset: images + YOLO-format bounding box annotations |
| `Male_and_female_cls/` | Classification dataset: cropped mosquito images organized by sex (Male / Female) |
| `Mosquitoes_dataset/` | Raw image dataset before preprocessing |

The datasets consist of macro photographs of *Aedes aegypti* mosquitoes, with male/female labels.

---

## Stage 1 — YOLO Detection (`1_YOLO/`)

- **Model**: YOLOv8 (Ultralytics), fine-tuned for mosquito detection.
- **Hyperparameter search**: Optuna (`optuna.db`, `trials.csv`) was used to find the best training hyperparameters.
- **Best config**: stored in `best_hyp.yaml`.
- **Results**: `metrics_detection.png`, `comparison_results/`, `runs/`

| File | Description |
|------|-------------|
| `1_Documentation_YOLO.ipynb` | Full training notebook with results and analysis |
| `Yolo_onnx_inference_test.py` | Script to test the exported ONNX model |
| `best_hyp.yaml` | Best hyperparameters found by Optuna |
| `runs/` | Ultralytics training run outputs |

---

## Stage 2 — MobileNet Classification (`2_MOBILENET/`)

- **Model**: MobileNet V3 (Small), trained for binary sex classification (Male / Female).
- Input: cropped mosquito images from YOLO detections.
- **Results**: `metrics_results.png`

| File | Description |
|------|-------------|
| `2_Documentation_MobileNet.ipynb` | Full training notebook |
| `MobileNet_onnx_inference_test.py` | Script to test the exported ONNX model |

---

## Stage 3 — Cascade Pipeline (`3_CASCADE/`)

End-to-end test of the combined YOLO + MobileNet cascade.

| File | Description |
|------|-------------|
| `3_Documentation_Cascade.ipynb` | Full pipeline testing notebook |
| `cascade_model_inference_test.py` | Script to run inference on test images |
| `Exam_Images/` | Test images for evaluating the full pipeline |
| `Output_Images/` | Annotated output images |
| `Crops/` | Intermediate YOLO crops passed to MobileNet |

---

## Cascade Inference (`utils/` server copy)

The server's `utils/Cascade_model_inference_script.py` and the Raspberry Pi's `cascade_inference.py` implement the same two-stage pipeline:

```python
# Pseudocode
boxes = yolo_detect(frame)           # Stage 1: detect bounding boxes
for box in boxes:
    crop = frame[box]
    sex  = mobilenet_classify(crop)  # Stage 2: classify sex
```

---

## ONNX Models (Deployed)

| Model | Architecture | Task |
|-------|-------------|------|
| `yolo_model.onnx` | YOLOv8 | Mosquito detection |
| `Mobilnet_mode.onnx` | MobileNet V3 Small | Sex classification |

The ONNX models are downloaded from Supabase Storage at server startup and loaded locally on the Raspberry Pi.
