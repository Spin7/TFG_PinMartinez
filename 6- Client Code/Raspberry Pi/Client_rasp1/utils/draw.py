import cv2

def draw_results(frame, boxes, preds, det_scores, cls_scores, class_names):

    h, w = frame.shape[:2]

    for box, pred, d_conf, c_conf in zip(boxes, preds, det_scores, cls_scores):
        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        color = (0,200,0) if pred == 0 else (200,0,0)

        label = f"{class_names[pred]} | D:{d_conf:.2f} C:{c_conf:.2f}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 2)

    return frame