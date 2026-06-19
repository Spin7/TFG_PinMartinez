import threading
import cv2


class ThreadedCamera2:
    """
    PC-webcam version of ThreadedCamera.
    Uses cv2.VideoCapture instead of picamera2 so the rest of
    main_rasp_client.py (cam.start / cam.read / cam.stop) works identically.
    """

    def __init__(self, size=(640, 480), device_index=0):
        """
        Parameters
        ----------
        size : tuple
            (width, height) to request from the camera.
        device_index : int
            OpenCV device index (0 = default webcam, 1 = second camera …).
        """
        self.size = size
        self.device_index = device_index

        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[ThreadedCamera2] Could not open camera at index {self.device_index}"
            )

        self.frame   = None
        self.running = False
        self.lock    = threading.Lock()

        print(f"[ThreadedCamera2] Opened camera {self.device_index} "
              f"at {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # =========================
    # THREADING
    # =========================
    def start(self):
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()
        print("[ThreadedCamera2] Camera released.")
