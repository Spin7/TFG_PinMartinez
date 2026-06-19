import threading
import queue
import time

class InferenceWorker:
    def __init__(self, model):
        self.model = model

        # Single-slot queue → prevents latency buildup
        self.input_q = queue.Queue(maxsize=1)

        # Store only latest result (no queue needed)
        self.output = None

        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.running:
            try:
                # Block instead of busy-wait → saves CPU
                frame = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.time()

            # ---- Inference ----
            boxes, preds = self.model.predict(frame)

            latency = time.time() - t0

            # Store latest result only (overwrite)
            with self.lock:
                self.output = (frame, boxes, preds, latency)

    def submit(self, frame):
        # Drop frame if busy (critical for real-time)
        if self.input_q.full():
            return
        self.input_q.put(frame)

    def get(self):
        with self.lock:
            return self.output  # non-blocking, latest result

    def is_ready(self):
        return not self.input_q.full()

    def stop(self):
        self.running = False