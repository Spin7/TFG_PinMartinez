import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path

# =========================
# PATHS
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent 

# IMAGE TEST PATH
IMAGE_PATH = ROOT / "3_CASCADE" / "Exam_Images" / "exam3.jpg" # <-- PUT YOUR IMAGE HERE

# MODELS
DETECTION_ONNX= ROOT / "Models" / "best.onnx"
CLASSIFIER_ONNX  = ROOT / "Models" / "best_CLmodel.onnx"
# =========================

# =========================
# CONFIG
# =========================

IMGSZ = 256
CLS_SIZE = 160

CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.45
PAD_RATIO = 0.0

CLASS_NAMES = ["Female", "Male"]

# =========================
# NORMALIZATION (MATCH TRAINING)
# =========================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

# =========================
# LETTERBOX
# =========================
def letterbox(image, new_size=256, color=(114,114,114)):
    h, w = image.shape[:2]
    scale = min(new_size / w, new_size / h)

    nw, nh = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (nw, nh))

    new_image = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    new_image[top:top+nh, left:left+nw] = image_resized
    return new_image, scale, left, top

# =========================
# PREPROCESS DETECTOR
# =========================
def preprocess_det(frame):
    img, scale, pad_x, pad_y = letterbox(frame, IMGSZ)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img, scale, pad_x, pad_y

# =========================
# PREPROCESS CLASSIFIER (ONNX)
# =========================
def preprocess_cls(crop):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  

    img = cv2.resize(img, (CLS_SIZE, CLS_SIZE))
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))  # CHW
    img = (img - MEAN) / STD

    return np.expand_dims(img, axis=0).astype(np.float32)

# =========================
# NMS
# =========================
def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# =========================
# DECODE OUTPUT
# =========================
def decode_output(output, frame_h, frame_w, scale, pad_x, pad_y):
    out = output[0]

    if out.ndim == 3:
        out = out.squeeze(0)

    if out.shape[0] < out.shape[1]:
        out = out.T

    x, y, w, h, scores = out[:,0], out[:,1], out[:,2], out[:,3], out[:,4]

    if np.max(w) <= 2.0:
        x1 = (x - w/2) * IMGSZ
        y1 = (y - h/2) * IMGSZ
        x2 = (x + w/2) * IMGSZ
        y2 = (y + h/2) * IMGSZ
    else:
        x1, y1 = x - w/2, y - h/2
        x2, y2 = x + w/2, y + h/2

    x1 = (x1 - pad_x) / scale
    x2 = (x2 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    y2 = (y2 - pad_y) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, frame_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, frame_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, frame_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, frame_h)

    mask = scores > CONF_THRESH
    boxes, scores = boxes[mask], scores[mask]

    if len(boxes) == 0:
        return boxes, scores

    keep = nms(boxes, scores, NMS_IOU_THRESH)
    return boxes[keep], scores[keep]

# =========================
# CROP
# =========================
def crop_with_padding(image, box, pad=0.25):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad, bh * pad

    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w, x2 + pad_x))
    y2 = int(min(h, y2 + pad_y))

    return image[y1:y2, x1:x2]

# =========================
# LOAD MODELS
# =========================
det_session = ort.InferenceSession(DETECTION_ONNX, providers=["CPUExecutionProvider"])
det_input = det_session.get_inputs()[0].name

cls_session = ort.InferenceSession(CLASSIFIER_ONNX, providers=["CPUExecutionProvider"])
cls_input = cls_session.get_inputs()[0].name
cls_output = cls_session.get_outputs()[0].name

# =========================
# MAIN PIPELINE
# =========================
def run(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = os.path.join("results", image_name)
    os.makedirs(result_dir, exist_ok=True)

    frame = cv2.imread(image_path)
    if frame is None:
        print("Image not found")
        return

    orig = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # =========================
    # SAVE 1_input
    # =========================
    cv2.imwrite(os.path.join(result_dir, "1_input.jpg"), orig)

    # =========================
    # DETECTION
    # =========================
    inp, scale, px, py = preprocess_det(rgb)
    output = det_session.run(None, {det_input: inp})

    boxes, scores = decode_output(output, h, w, scale, px, py)

    if len(boxes) == 0:
        print("No detections")
        return

    print(f"{len(boxes)} detections")

    # =========================
    # SAVE 2_detect
    # =========================
    detect_img = orig.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(detect_img, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imwrite(os.path.join(result_dir, "2_detect.jpg"), detect_img)

    # =========================
    # PROCESS EACH DETECTION
    # =========================
    for i, box in enumerate(boxes):

        crop = crop_with_padding(orig, box, PAD_RATIO)

        # ---------- SAVE 3_crop ----------
        crop_path = os.path.join(result_dir, f"3_crop_{i}.jpg")
        cv2.imwrite(crop_path, crop)

        # =========================
        # CLASSIFICATION (ONNX) - FIXED
        # =========================
        inp_cls = preprocess_cls(crop)  # PASS BGR DIRECTLY

        logits = cls_session.run([cls_output], {cls_input: inp_cls})[0]

        # Stable softmax
        logits = logits[0]  # remove batch dim
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        label = f"{CLASS_NAMES[pred]} {conf:.2f}"
        print(f"Crop {i} → {label}")

        # ---------- SAVE 4_output ----------
        output_img = crop.copy()
        cv2.putText(output_img, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        output_path = os.path.join(result_dir, f"4_output_{i}.jpg")
        cv2.imwrite(output_path, output_img)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run(IMAGE_PATH)