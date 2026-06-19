import cv2
import numpy as np
import onnxruntime as ort

class CascadeInference:

    def __init__(self,
                 det_model_path,
                 cls_model_path,
                 imgsz=256,
                 cls_size=160,
                 conf_thresh=0.25,
                 iou_thresh=0.45,
                 cls_det_thresh=0.4,
                 det_conf_th=0.6,    # final det confidence threshold
                 cls_conf_th=0.7):   # final cls confidence threshold

        self.IMGSZ = imgsz
        self.CLS_SIZE = cls_size
        self.CONF_THRESH = conf_thresh          # detector threshold (low)
        self.CLS_DET_THRESH = cls_det_thresh    # threshold BEFORE classification
        self.NMS_IOU_THRESH = iou_thresh
        self.DET_CONF_TH = det_conf_th          # final det threshold (from main)
        self.CLS_CONF_TH = cls_conf_th          # final cls threshold (from main)

        self.CLASS_NAMES = ["Female", "Male"]

        self.MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
        self.STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

        self.det = ort.InferenceSession(det_model_path)
        self.det_in = self.det.get_inputs()[0].name

        self.cls = ort.InferenceSession(cls_model_path)
        self.cls_in = self.cls.get_inputs()[0].name
        self.cls_out = self.cls.get_outputs()[0].name

    # =========================
    # PREPROCESS
    # =========================
    def letterbox(self, img):
        h, w = img.shape[:2]
        scale = min(self.IMGSZ/w, self.IMGSZ/h)

        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (nw, nh))

        canvas = np.full((self.IMGSZ, self.IMGSZ, 3), 114, dtype=np.uint8)
        top = (self.IMGSZ - nh)//2
        left = (self.IMGSZ - nw)//2

        canvas[top:top+nh, left:left+nw] = resized
        return canvas, scale, left, top

    def preprocess_det(self, img):
        img, s, px, py = self.letterbox(img)
        img = img.astype(np.float32)/255.0
        img = np.transpose(img, (2,0,1))[None]
        return img, s, px, py

    def preprocess_cls(self, crop):
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.CLS_SIZE, self.CLS_SIZE))
        crop = crop.astype(np.float32)/255.0
        crop = np.transpose(crop, (2,0,1))
        crop = (crop - self.MEAN)/self.STD
        return crop[None]

    # =========================
    # NMS  (vectorized)
    # =========================
    def nms(self, boxes: np.ndarray, scores: np.ndarray) -> list:
        """Vectorized NMS – avoids per-pair Python loops."""
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep   = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            rest = order[1:]
            # vectorized intersection
            ix1  = np.maximum(x1[i], x1[rest])
            iy1  = np.maximum(y1[i], y1[rest])
            ix2  = np.minimum(x2[i], x2[rest])
            iy2  = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            iou   = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= self.NMS_IOU_THRESH]

        return keep

    # =========================
    # DECODE
    # =========================
    def decode(self, out, scale, px, py):

        out = out[0]

        if out.ndim == 3:
            out = out.squeeze(0)
        if out.shape[0] < out.shape[1]:
            out = out.T

        x,y,bw,bh,conf = out[:,0],out[:,1],out[:,2],out[:,3],out[:,4]

        if np.max(bw) <= 2:
            x1 = (x - bw/2)*self.IMGSZ
            y1 = (y - bh/2)*self.IMGSZ
            x2 = (x + bw/2)*self.IMGSZ
            y2 = (y + bh/2)*self.IMGSZ
        else:
            x1,y1 = x-bw/2, y-bh/2
            x2,y2 = x+bw/2, y+bh/2

        x1 = (x1 - px)/scale
        x2 = (x2 - px)/scale
        y1 = (y1 - py)/scale
        y2 = (y2 - py)/scale

        boxes = np.stack([x1,y1,x2,y2],1)

        mask = conf > self.CONF_THRESH
        boxes, conf = boxes[mask], conf[mask]

        keep = self.nms(boxes, conf)

        return boxes[keep], conf[keep]

    # =========================
    # MAIN INFERENCE (IMPROVED)
    # =========================
    def predict(self, frame):

        inp, s, px, py = self.preprocess_det(frame)
        out = self.det.run(None, {self.det_in: inp})

        boxes, det_scores = self.decode(out, s, px, py)

        final_boxes = []
        preds = []
        cls_scores = []
        final_det_scores = []

        for box, det_score in zip(boxes, det_scores):

            # NEW: skip weak detections BEFORE classification
            if det_score < self.CLS_DET_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box)

            # clamp (robustness)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            inp_cls = self.preprocess_cls(crop)
            logits = self.cls.run([self.cls_out], {self.cls_in: inp_cls})[0][0]

            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()

            cls_id = int(np.argmax(probs))
            cls_conf = float(probs[cls_id])

            final_boxes.append(box)
            preds.append(cls_id)
            cls_scores.append(cls_conf)
            final_det_scores.append(float(det_score))

        # =========================
        # FINAL THRESHOLD FILTER
        # =========================
        filtered = [
            (b, p, ds, cs)
            for b, p, ds, cs in zip(final_boxes, preds, final_det_scores, cls_scores)
            if ds >= self.DET_CONF_TH and cs >= self.CLS_CONF_TH
        ]

        if filtered:
            final_boxes, preds, final_det_scores, cls_scores = zip(*filtered)
            return list(final_boxes), list(preds), list(final_det_scores), list(cls_scores)

        return [], [], [], []