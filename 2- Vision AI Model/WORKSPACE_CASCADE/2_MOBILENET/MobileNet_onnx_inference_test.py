import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# =========================
# PATHS
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# IMAGE TEST PATH
IMAGE_PATH = ROOT / "2_MOBILENET" / "Test_Images" / "test.jpg" # <-- PUT YOUR IMAGE HERE

# IMAGE OUTPUT

# MODELS
CLASSIFIER_ONNX  = ROOT / "Models" / "best_CLmodel.onnx"
# =========================



# =========================
# CONFIG
# =========================  
CLS_SIZE = 160

CLASS_NAMES = ["Female", "Male"]

# =========================
# NORMALIZATION (MATCH TRAINING)
# =========================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

# =========================
# PREPROCESS
# =========================
def preprocess_cls(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CLS_SIZE, CLS_SIZE))

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW

    img = (img - MEAN) / STD

    return np.expand_dims(img, axis=0).astype(np.float32)

# =========================
# LOAD MODEL
# =========================
session = ort.InferenceSession(CLASSIFIER_ONNX, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# =========================
# INFERENCE
# =========================
def run(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return

    inp = preprocess_cls(img)

    logits = session.run([output_name], {input_name: inp})[0]

    # Remove batch dim
    logits = logits[0]

    # Stable softmax
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / np.sum(exp)

    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    label = f"{CLASS_NAMES[pred]} ({conf:.2f})"

    print("Prediction:", label)

    # Optional visualization
    out = img.copy()
    cv2.putText(out, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run(IMAGE_PATH)