import threading
import time


class InferenceWorker:
    """
    Runs model.predict() in a dedicated background thread.

    Design
    ------
    - submit() stores the latest frame (drops the previous one if not yet
      consumed, keeping only the freshest data).
    - An Event wakes the thread immediately when a frame is available,
      avoiding the old 1 ms busy-poll that burned CPU on the Pi.
    - stop() signals the thread and wakes the event so it exits cleanly.
    """

    def __init__(self, model):
        self.model = model

        self._frame  = None
        self._result = None

        self._lock        = threading.Lock()
        self._frame_ready = threading.Event()
        self._running     = False
        self._thread: threading.Thread | None = None

        # Stats
        self.frames_dropped = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True,
                                         name="InferenceWorker")
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._running = False
        self._frame_ready.set()          # unblock the thread if it is waiting
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(self, frame):
        """Hand a new frame to the worker. Drops the previous pending frame."""
        with self._lock:
            if self._frame is not None:
                self.frames_dropped += 1
            self._frame = frame.copy()
        self._frame_ready.set()          # wake the worker thread

    def is_ready(self) -> bool:
        """True when no frame is waiting to be processed."""
        with self._lock:
            return self._frame is None

    def get(self):
        """Consume and return the latest result, or None if not available."""
        with self._lock:
            res          = self._result
            self._result = None
        return res

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self):
        while self._running:
            # Block until a frame is submitted (or stop() is called)
            self._frame_ready.wait()
            self._frame_ready.clear()

            # Fetch frame
            with self._lock:
                frame        = self._frame
                self._frame  = None

            if frame is None:
                continue

            # Inference
            t0 = time.perf_counter()
            boxes, preds, det_scores, cls_scores = self.model.predict(frame)
            latency = time.perf_counter() - t0

            # Store result (overwrite old if not yet consumed)
            with self._lock:
                self._result = (frame, boxes, preds, det_scores, cls_scores, latency)