"""
cascade_inference.py
--------------------
Clase reutilizable para el pipeline de inferencia en cascada:
  YOLOv8s (detección) → MobileNetV3-Large (clasificación binaria Male/Female)

Uso:
    from utils.cascade_inference import CascadeInference
    ci = CascadeInference("models/yolo_model.onnx", "models/Mobilnet_mode.onnx")
    boxes, preds, det_scores, cls_scores = ci.predict(frame_bgr)
    annotated_bytes = ci.predict_and_draw(frame_bgr)
"""

import cv2
import numpy as np
import onnxruntime as ort
import io


# ─── Constantes del pipeline ────────────────────────────────────────────────
IMGSZ         = 256
CLS_SIZE      = 160
CONF_THRESH   = 0.25
NMS_IOU_THRESH = 0.45
CLS_DET_THRESH = 0.40          # umbral pre-clasificación

CLASS_NAMES   = ["Female", "Male"]
CLASS_COLORS  = {               # BGR para OpenCV
    "Female": (180, 60, 220),   # morado
    "Male":   (60, 160, 240),   # azul
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ─── Preprocesado ────────────────────────────────────────────────────────────
def _letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top  = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def _preprocess_det(img: np.ndarray):
    lb, s, px, py = _letterbox(img, IMGSZ)
    tensor = lb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]   # [1, 3, H, W]
    return tensor, s, px, py


def _preprocess_cls(crop: np.ndarray) -> np.ndarray:
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (CLS_SIZE, CLS_SIZE))
    crop = crop.astype(np.float32) / 255.0
    crop = np.transpose(crop, (2, 0, 1))
    crop = (crop - MEAN) / STD
    return crop[None]                                 # [1, 3, H, W]


# ─── NMS ─────────────────────────────────────────────────────────────────────
def _iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-9)


def _nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes)
    scores = np.array(scores)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]
        keep.append(i)
        ious  = np.array([_iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious <= thr]
    return keep


def _decode(out, scale, px, py):
    raw = out[0]
    if raw.ndim == 3:
        raw = raw.squeeze(0)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T

    x, y, bw, bh, conf = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]

    if np.max(bw) <= 2:                     # coordenadas normalizadas
        x1 = (x - bw / 2) * IMGSZ
        y1 = (y - bh / 2) * IMGSZ
        x2 = (x + bw / 2) * IMGSZ
        y2 = (y + bh / 2) * IMGSZ
    else:                                   # coordenadas en píxeles
        x1, y1 = x - bw / 2, y - bh / 2
        x2, y2 = x + bw / 2, y + bh / 2

    # desnormalizar al espacio original
    x1 = (x1 - px) / scale
    x2 = (x2 - px) / scale
    y1 = (y1 - py) / scale
    y2 = (y2 - py) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    mask  = conf > CONF_THRESH
    boxes, conf = boxes[mask], conf[mask]
    keep = _nms(boxes, conf, NMS_IOU_THRESH)
    return boxes[keep], conf[keep]


# ─── Clase principal ──────────────────────────────────────────────────────────
class CascadeInference:
    """
    Pipeline de inferencia en dos etapas (ONNX Runtime).

    Parameters
    ----------
    det_path : str   Ruta al modelo ONNX del detector (YOLOv8s).
    cls_path : str   Ruta al modelo ONNX del clasificador (MobileNetV3-Large).
    """

    def __init__(self, det_path: str, cls_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4

        self._det = ort.InferenceSession(det_path, sess_options=opts)
        self._cls = ort.InferenceSession(cls_path, sess_options=opts)

        self._det_in  = self._det.get_inputs()[0].name
        self._cls_in  = self._cls.get_inputs()[0].name
        self._cls_out = self._cls.get_outputs()[0].name

        print(f"[CascadeInference] Detector  : {det_path}")
        print(f"[CascadeInference] Classifier: {cls_path}")

    # ─── Predicción ──────────────────────────────────────────────────────────
    def predict(self, frame: np.ndarray):
        """
        Ejecuta el pipeline completo sobre un frame BGR.

        Returns
        -------
        final_boxes  : np.ndarray  [N, 4]  coords píxel (x1,y1,x2,y2)
        preds        : list[int]   clase (0=Female, 1=Male)
        det_scores   : np.ndarray  [N]     confianza del detector
        cls_scores   : list[float] confianza del clasificador
        """
        h, w = frame.shape[:2]

        # ── Etapa 1: Detección ──────────────────────────────────────────────
        inp, scale, px, py = _preprocess_det(frame)
        det_out = self._det.run(None, {self._det_in: inp})
        boxes, det_scores = _decode(det_out, scale, px, py)

        # Clamp boxes a límites del frame
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

        # Filtro pre-clasificación
        mask = det_scores >= CLS_DET_THRESH
        boxes, det_scores = boxes[mask], det_scores[mask]

        preds      = []
        cls_scores = []

        # ── Etapa 2: Clasificación por crop ─────────────────────────────────
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                preds.append(0)
                cls_scores.append(0.5)
                continue

            inp_cls = _preprocess_cls(crop)
            logits  = self._cls.run([self._cls_out], {self._cls_in: inp_cls})[0][0]

            # Softmax estable
            e   = np.exp(logits - logits.max())
            prob = e / e.sum()

            pred = int(np.argmax(prob))
            preds.append(pred)
            cls_scores.append(float(prob[pred]))

        return boxes, preds, det_scores, cls_scores

    # ─── Anotación ───────────────────────────────────────────────────────────
    def predict_and_draw(self, frame: np.ndarray) -> bytes:
        """
        Ejecuta predict() y devuelve la imagen anotada como bytes PNG.
        """
        boxes, preds, det_scores, cls_scores = self.predict(frame)
        annotated = frame.copy()

        for box, pred, d_sc, c_sc in zip(boxes, preds, det_scores, cls_scores):
            x1, y1, x2, y2 = map(int, box)
            name  = CLASS_NAMES[pred]
            color = CLASS_COLORS[name]
            label = f"{name}  {c_sc:.0%}"

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Fondo label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".png", annotated)
        if not ok:
            raise RuntimeError("No se pudo codificar la imagen anotada.")
        return buf.tobytes()
