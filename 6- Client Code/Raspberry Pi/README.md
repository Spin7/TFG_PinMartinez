# Raspberry Pi Client

This folder contains the **main Python application** that runs on the Raspberry Pi inside the smart trap, along with camera utilities.

## Folder Structure

```
Raspberry Pi/
│
├── Client_rasp1/                  # Main application
│   ├── main_rasp_client.py        # Entry point — main inference and control loop
│   ├── cascade_inference.py       # YOLO + MobileNet two-stage inference engine
│   ├── visualization.py           # Bounding box drawing utilities
│   ├── best.onnx                  # YOLO detection model (ONNX)
│   ├── best_CLmodel.onnx          # MobileNet classification model (ONNX)
│   ├── camera_calibration.json    # Camera intrinsic parameters
│   │
│   ├── camera/                    # Camera abstraction layer
│   │   ├── threaded_camera.py     # Pi Camera v2 (picamera2) threaded reader
│   │   └── threaded_camera2.py    # PC webcam (cv2.VideoCapture) threaded reader
│   │
│   ├── pipeline/                  # Processing pipeline workers
│   │   ├── inference_worker.py    # Async inference thread (frame queue + results)
│   │   ├── uploader.py            # HTTP upload thread (posts to MosquitoWeb server)
│   │   └── serial_manager.py      # Serial communication thread (Arduino I/O)
│   │
│   └── utils/                     # Utility functions
│       └── draw.py                # Draws detection boxes and labels on frames
│
└── Utils/                         # Setup and calibration tools
    ├── camera_calibration.py      # Camera calibration script (checkerboard)
    ├── camera_test.py             # Quick camera test script
    ├── camera_get_information.py  # Print camera properties and capabilities
    └── Pi_camera_firts_use_guide.txt  # Guide for setting up Pi Camera on Raspberry Pi OS
```

---

## Running the Client

### Prerequisites

```bash
# Install Python dependencies (on the Raspberry Pi)
pip install opencv-python-headless onnxruntime numpy requests pyserial picamera2
```

### Configuration

Edit the constants at the top of `main_rasp_client.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `USE_PI_CAMERA` | `True` | `True` = Pi Camera, `False` = USB webcam |
| `SERVER_URL` | Railway URL | MosquitoWeb server base URL |
| `CLIENT_ID` | `"rpi_cam1"` | Unique identifier for this trap |
| `DET_CONF_TH` | `0.6` | YOLO detection confidence threshold |
| `CLS_CONF_TH` | `0.7` | MobileNet classification confidence threshold |
| `SERIAL_PORT` | `"/dev/ttyACM0"` | Arduino serial port |
| `SERIAL_DEBUG` | `False` | `True` = simulate Arduino (no hardware needed) |
| `ENABLE_UPLOAD` | `True` | Enable/disable server uploads |
| `SORT_DURATION` | `5.0` s | Time to wait for Arduino to complete a sort cycle |

### Start the application

```bash
cd Client_rasp1
python main_rasp_client.py
```

Press **ESC** (or `Ctrl+C`) to stop.

---

## Main Loop Logic

```
┌──────────────────────────────────────────────┐
│  1. Read frame from camera (threaded)        │
│  2. Submit frame to InferenceWorker (async)  │
│  3. Get latest inference result              │
│  4. TRAP STATE MACHINE:                      │
│     ┌─ IDLE ──────────────────────┐          │
│     │  if detection found:        │          │
│     │    → send sort cmd to Arduino│         │
│     │    → upload to server        │         │
│     │    → transition to SORTING  │          │
│     └─ SORTING ─────────────────-┘          │
│        wait SORT_DURATION seconds            │
│        → back to IDLE                        │
│  5. Draw HUD overlay on frame                │
│  6. Display (cv2.imshow)                     │
└──────────────────────────────────────────────┘
```

---

## Architecture: Multi-threaded Pipeline

The application uses a producer-consumer pattern to decouple the camera, inference, upload, and serial I/O:

| Thread | Class | Role |
|--------|-------|------|
| Camera thread | `ThreadedCamera` | Continuously captures frames from Pi Camera |
| Inference thread | `InferenceWorker` | Runs ONNX inference on submitted frames |
| Upload thread | `Uploader` | POSTs images + data to the server asynchronously |
| Serial thread | `SerialManager` | Reads sensor data from and writes commands to Arduino |
| Main thread | `main_rasp_client.py` | Orchestrates everything, drives the state machine |

---

## ONNX Models

| File | Model | Task |
|------|-------|------|
| `best.onnx` | YOLOv8 | Mosquito detection (bounding boxes) |
| `best_CLmodel.onnx` | MobileNet V3 Small | Sex classification (Female / Male) |

The cascade pipeline (`cascade_inference.py`) runs both models sequentially on each frame.

---

## Camera Utils (`Utils/`)

| Script | Purpose |
|--------|---------|
| `camera_calibration.py` | Compute camera intrinsic matrix using a checkerboard pattern |
| `camera_test.py` | Quickly verify the camera is working |
| `camera_get_information.py` | Print camera resolution, FPS, format info |
| `Pi_camera_firts_use_guide.txt` | Step-by-step guide to enable the Pi Camera on Raspberry Pi OS |
