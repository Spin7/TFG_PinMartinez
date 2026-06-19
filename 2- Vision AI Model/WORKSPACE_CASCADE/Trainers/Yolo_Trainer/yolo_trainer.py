from ultralytics import YOLO
import yaml

MODEL = "yolov8s.pt"
DATA = "dataset/data.yaml"

def main():
    with open("best_hyp.yaml") as f:
        hyp = yaml.safe_load(f)

    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=200,
        imgsz=256,
        batch=16,
        device=0,
        workers=4,
        patience=20,
        deterministic=True,
        amp=True,
        close_mosaic=10,
        **hyp
    )

if __name__ == "__main__":
    main()