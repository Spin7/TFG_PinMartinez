import threading
import queue
import time
import requests
import json
import cv2


class Uploader:

    def __init__(self, server_url, client_id, jpeg_quality=80, max_queue=10):
        # 🔥 FORCE correct endpoint
        if not server_url.endswith("/upload"):
            server_url = server_url.rstrip("/") + "/upload"

        self.server_url = server_url
        self.client_id = client_id
        self.jpeg_quality = jpeg_quality

        self.queue = queue.Queue(maxsize=max_queue)
        self.running = False

        # Optional SerialManager – injected after construction via set_serial_manager()
        self._serial_manager = None

        print(f"[UPLOADER INIT] URL = {self.server_url}")  # DEBUG

    def set_serial_manager(self, serial_manager):
        """Attach a SerialManager so the uploader can read live sensor data."""
        self._serial_manager = serial_manager
        print("[UPLOADER] SerialManager attached")

    def start(self):
        self.running = True
        threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self.running = False

    def submit(self, frame, boxes, preds, det_scores, cls_scores):

        if self.queue.full():
            print("[UPLOAD] queue full → dropping")
            return

        # ---- Sensor readings (live via SerialManager, or simulated fallback) ----
        if self._serial_manager is not None:
            temperature, humidity = self._serial_manager.get_sensors()
        else:
            # No SerialManager set – use static placeholder values
            temperature = 25.0
            humidity = 60.0

        detections = []

        for box, pred, det_s, cls_s in zip(boxes, preds, det_scores, cls_scores):
            x1, y1, x2, y2 = map(int, box)
            combined_conf = round(float(det_s) * float(cls_s), 4)  # DET_CONF * CLS_CONF

            detections.append({
                "class_id": int(pred),
                "class_name": "Female" if pred == 0 else "Male",
                "confidence": combined_conf,
                "bbox": [x1, y1, x2, y2]
            })

        # ---- Encode JPEG ----
        ok, jpeg_buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )

        if not ok:
            print("[UPLOAD] JPEG encode failed")
            return

        data = {
            "client_id": self.client_id,
            "temperature": str(temperature),
            "humidity": str(humidity),
            "detections": json.dumps(detections)
        }

        try:
            self.queue.put_nowait((data, jpeg_buf.tobytes()))
        except queue.Full:
            print("[UPLOAD] queue full")

    def run(self):

        while self.running:
            try:
                data, jpeg_bytes = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            files    = {"image": ("frame.jpg", jpeg_bytes, "image/jpeg")}
            success  = False

            for attempt in range(1, 4):   # up to 3 attempts
                try:
                    resp = requests.post(
                        self.server_url,
                        data=data,
                        files=files,
                        timeout=10,
                        verify=False          # suppress SSL errors on ngrok
                    )
                    if resp.status_code == 200:
                        print(f"[UPLOAD] OK ({resp.status_code})")
                        success = True
                        break
                    else:
                        # HTTP error (4xx / 5xx) – count as a failed attempt
                        print(f"[UPLOAD] HTTP {resp.status_code} (attempt {attempt}/3)")

                except Exception as e:
                    backoff = 1.5 ** attempt
                    print(f"[UPLOAD] Error attempt {attempt}/3 → {e}  (retry in {backoff:.1f}s)")
                    time.sleep(backoff)

            if not success:
                print("[UPLOAD] FAILED after 3 attempts – packet discarded")

            self.queue.task_done()