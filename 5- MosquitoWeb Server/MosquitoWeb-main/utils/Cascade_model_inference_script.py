import cv2
import numpy as np
import onnxruntime as ort
import os
import yaml
from glob import glob
import random
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DETECTION_ONNX = "best.onnx"
CLASSIFIER_ONNX = "best_CLmodel_fixed.onnx"
DATA_YAML = "dataset/data.yaml"

OUTPUT_DIR = "results_visual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMGSZ = 256
CLS_SIZE = 160

CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.45

CLASS_NAMES = ["Female", "Male"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

MAX_CROPS = 6  # avoid overcrowding

# =========================
# DATASET PATHS
# =========================
def resolve_test_paths():
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(DATA_YAML))

    base_path = data.get("path", "")
    if base_path:
        if not os.path.isabs(base_path):
            base_path = os.path.normpath(os.path.join(yaml_dir, base_path))
    else:
        base_path = yaml_dir

    test_path = data.get("test", "")

    # =========================
    # Build ALL candidates
    # =========================
    candidates = []

    # 1. Absolute
    if os.path.isabs(test_path):
        candidates.append(test_path)

    # 2. Relative to base_path (YOLO standard)
    candidates.append(os.path.normpath(os.path.join(base_path, test_path)))

    # 3. Relative to yaml dir
    candidates.append(os.path.normpath(os.path.join(yaml_dir, test_path)))

    # 4. Common structures
    candidates.append(os.path.join(base_path, "images", "test"))
    candidates.append(os.path.join(base_path, "test", "images"))

    # =========================
    # DEBUG PRINT
    # =========================
    print("\n=== PATH DEBUG ===")
    print("YAML dir:", yaml_dir)
    print("Base path:", base_path)
    print("Test field:", test_path)
    print("Candidates:")
    for c in candidates:
        print(" -", c)

    # =========================
    # Select valid path
    # =========================
    test_images = None
    for c in candidates:
        if c and os.path.exists(c):
            test_images = c
            break

    if test_images is None:
        raise RuntimeError("Could not resolve dataset paths")

    # =========================
    # Labels path
    # =========================
    if "images" in test_images:
        test_labels = test_images.replace("images", "labels")
    else:
        test_labels = os.path.join(os.path.dirname(test_images), "labels")

    # fallback
    if not os.path.exists(test_labels):
        alt = os.path.join(base_path, "labels", "test")
        if os.path.exists(alt):
            test_labels = alt

    print("\n=== FINAL PATHS ===")
    print("Images:", test_images)
    print("Labels:", test_labels)

    if not os.path.exists(test_images):
        raise RuntimeError(f"Images not found: {test_images}")

    if not os.path.exists(test_labels):
        raise RuntimeError(f"Labels not found: {test_labels}")

    return test_images, test_labels

def get_all_images(path):
    files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        files.extend(glob(os.path.join(path, ext)))
    return files

# =========================
# PREPROCESS
# =========================
def letterbox(img, size):
    h, w = img.shape[:2]
    scale = min(size/w, size/h)

    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh)//2
    left = (size - nw)//2

    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top

def preprocess_det(img):
    img, s, px, py = letterbox(img, IMGSZ)
    img = img.astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))[None]
    return img, s, px, py

def preprocess_cls(crop):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (CLS_SIZE, CLS_SIZE))
    crop = crop.astype(np.float32)/255.0
    crop = np.transpose(crop, (2,0,1))
    crop = (crop - MEAN)/STD
    return crop[None]

# =========================
# NMS + DECODE
# =========================
def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)

def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious <= thr]
    return keep

def decode(out, h, w, scale, px, py):
    out = out[0]
    if out.ndim == 3:
        out = out.squeeze(0)
    if out.shape[0] < out.shape[1]:
        out = out.T

    x,y,bw,bh,conf = out[:,0],out[:,1],out[:,2],out[:,3],out[:,4]

    if np.max(bw) <= 2:
        x1 = (x - bw/2)*IMGSZ
        y1 = (y - bh/2)*IMGSZ
        x2 = (x + bw/2)*IMGSZ
        y2 = (y + bh/2)*IMGSZ
    else:
        x1,y1 = x-bw/2, y-bh/2
        x2,y2 = x+bw/2, y+bh/2

    x1 = (x1 - px)/scale
    x2 = (x2 - px)/scale
    y1 = (y1 - py)/scale
    y2 = (y2 - py)/scale

    boxes = np.stack([x1,y1,x2,y2],1)
    mask = conf > CONF_THRESH

    boxes, conf = boxes[mask], conf[mask]
    keep = nms(boxes, conf, NMS_IOU_THRESH)

    return boxes[keep], conf[keep]

# =========================
# VISUALIZATION (IMPROVED)
# =========================
def visualize_result(img, boxes, preds, save_path):

    img_draw = img.copy()
    crops = []

    for box, pred in zip(boxes, preds):
        x1, y1, x2, y2 = map(int, box)

        color = (0,200,0) if pred == 0 else (200,0,0)
        label = CLASS_NAMES[pred]

        # bbox
        cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 2)

        # label with background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1-th-5), (x1+tw, y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        crop = img[y1:y2, x1:x2]
        if crop.size != 0:
            crop = cv2.resize(crop, (120,120))
            crops.append((crop, label))

    crops = crops[:MAX_CROPS]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_draw_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

    # Layout
    n = len(crops)
    cols = max(3, n)
    fig = plt.figure(figsize=(4*cols, 6))

    # Title
    fig.suptitle("Detection + Classification Results", fontsize=16)

    # Original
    ax1 = plt.subplot(2, cols, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Original")
    ax1.axis("off")

    # Detection
    ax2 = plt.subplot(2, cols, 2)
    ax2.imshow(img_draw_rgb)
    ax2.set_title("Detections")
    ax2.axis("off")

    # Crops grid
    for i, (crop, label) in enumerate(crops):
        ax = plt.subplot(2, cols, cols + i + 1)
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    # Save
    plt.savefig(save_path, dpi=200)
    plt.show()  # show in Jupyter
    plt.close()

# =========================
# LOAD MODELS
# =========================
det = ort.InferenceSession(DETECTION_ONNX)
det_in = det.get_inputs()[0].name

cls = ort.InferenceSession(CLASSIFIER_ONNX)
cls_in = cls.get_inputs()[0].name
cls_out = cls.get_outputs()[0].name

# =========================
# MAIN
# =========================
img_dir, _ = resolve_test_paths()
images = get_all_images(img_dir)

print("Total images:", len(images))

images = random.sample(images, min(10, len(images)))

for i, img_path in enumerate(images):

    frame = cv2.imread(img_path)
    if frame is None:
        continue

    h, w = frame.shape[:2]

    inp, s, px, py = preprocess_det(frame)
    out = det.run(None, {det_in: inp})

    pred_boxes, _ = decode(out, h, w, s, px, py)

    preds = []

    for pb in pred_boxes:
        x1, y1, x2, y2 = map(int, pb)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        inp_cls = preprocess_cls(crop)
        logits = cls.run([cls_out], {cls_in: inp_cls})[0][0]

        preds.append(int(np.argmax(logits)))

    save_path = os.path.join(OUTPUT_DIR, f"result_{i}.png")
    visualize_result(frame, pred_boxes[:len(preds)], preds, save_path)

    print(f"Saved: {save_path}")