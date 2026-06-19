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
IMAGE_PATH = ROOT / "1_YOLO" / "Test_Images" / "test2.jpg" # <-- PUT YOUR IMAGE HERE

# IMAGE OUTPUT

# MODELS
DETECTION_ONNX= ROOT / "Models" / "best.onnx"
# =========================

# =========================
# CONFIG
# =========================

IMGSZ = 256
CONF_THRESH = 0.25   # lower for debugging
NMS_IOU_THRESH = 0.45


# =========================
# LETTERBOX (VERY IMPORTANT)
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
# PREPROCESS
# =========================
def preprocess(frame, size=256):
    img, scale, pad_x, pad_y = letterbox(frame, size)

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]

    return img, scale, pad_x, pad_y


# =========================
# NMS
# =========================
def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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
# SIGMOID (for safety)
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================
# DECODE OUTPUT
# =========================
def decode_output(output, frame_h, frame_w, scale, pad_x, pad_y, conf_thresh=0.25):
    out = output[0]

    if out.shape[0] == 6:
        out = out.T

    out = out.astype(np.float32)

    x = out[:, 0].reshape(-1)
    y = out[:, 1].reshape(-1)
    w = out[:, 2].reshape(-1)
    h = out[:, 3].reshape(-1)
    scores = out[:, 4].reshape(-1)

    print("max score:", np.max(scores))

    # AUTO-DETECT SCALE
    if np.max(w) <= 2.0:  
        print("Using NORMALIZED coordinates")
        # normalized → scale
        x1 = (x - w/2) * IMGSZ
        y1 = (y - h/2) * IMGSZ
        x2 = (x + w/2) * IMGSZ
        y2 = (y + h/2) * IMGSZ
    else:
        print("Using ABSOLUTE coordinates")
        # already in pixels
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2

    # remove padding
    x1 -= pad_x
    x2 -= pad_x
    y1 -= pad_y
    y2 -= pad_y

    # back to original image
    x1 /= scale
    x2 /= scale
    y1 /= scale
    y2 /= scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # CLIP BOXES (IMPORTANT)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, frame_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, frame_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, frame_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, frame_h)

    # filter
    mask = scores > conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return boxes, scores

    keep = nms(boxes, scores, NMS_IOU_THRESH)
    boxes = boxes[keep]
    scores = scores[keep]

    order = scores.argsort()[::-1]
    return boxes[order], scores[order]

# =========================
# DRAW
# =========================
def draw_detections(frame, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame


# =========================
# MAIN
# =========================
def main():
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("Error loading image")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Load model
    session = ort.InferenceSession(DETECTION_ONNX, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Preprocess
    inp, scale, pad_x, pad_y = preprocess(frame, IMGSZ)

    # Inference
    output = session.run(None, {input_name: inp})

    print("Output shape:", output[0].shape)

    # Decode
    boxes, scores = decode_output(output, h, w, scale, pad_x, pad_y)

    print("Detections:", len(boxes))

    # Draw
    frame_out = draw_detections(frame.copy(), boxes, scores)

    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
    cv2.imshow("Detections", frame_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()