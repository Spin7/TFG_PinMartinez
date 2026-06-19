import gc
import os
import torch
import optuna
from ultralytics import YOLO


def objective(trial):

    # -------------------------------
    # Hyperparameter space
    # -------------------------------
    params = {
        'lr0': trial.suggest_float('lr0', 1e-4, 5e-2, log=True),
        'lrf': trial.suggest_float('lrf', 0.05, 0.5),
        'momentum': trial.suggest_float('momentum', 0.7, 0.98),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),

        'warmup_epochs': trial.suggest_float('warmup_epochs', 0.0, 3.0),

        'box': trial.suggest_float('box', 0.05, 0.15),
        'cls': trial.suggest_float('cls', 0.2, 0.6),
        'dfl': trial.suggest_float('dfl', 0.8, 1.5),

        'hsv_h': trial.suggest_float('hsv_h', 0.0, 0.05),
        'hsv_s': trial.suggest_float('hsv_s', 0.2, 0.8),
        'hsv_v': trial.suggest_float('hsv_v', 0.2, 0.8),

        'flipud': trial.suggest_float('flipud', 0.0, 0.3),
        'fliplr': trial.suggest_float('fliplr', 0.3, 0.7),

        'mosaic': trial.suggest_float('mosaic', 0.5, 1.0),
        'mixup': trial.suggest_float('mixup', 0.0, 0.3),
    }

    # -------------------------------
    # GPU assignment (critical)
    # -------------------------------
    device = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))

    model = YOLO("yolov8s.pt")

    try:
        results = model.train(
            data="dataset/data.yaml",   # must contain train + val ONLY
            epochs=15,                 # slightly longer for stability
            imgsz=256,
            batch=16,
            device=device,
            verbose=False,
            **params
        )

        # Validation metric (used by Optuna)
        metrics = model.val(data="dataset/data.yaml", verbose=False)

        map50 = metrics.box.map50

    except Exception as e:
        print(f"Trial failed: {e}")
        map50 = 0.0

    # -------------------------------
    # Cleanup (VERY important)
    # -------------------------------
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return map50