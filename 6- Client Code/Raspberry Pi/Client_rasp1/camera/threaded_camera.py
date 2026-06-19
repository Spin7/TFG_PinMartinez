import threading
import cv2
import json
import os
from picamera2 import Picamera2


class ThreadedCamera:
    def __init__(self, size=(1280, 720), calib_path="camera_calibration.json"):

        self.picam2 = Picamera2()

        # CONFIG
        config = self.picam2.create_video_configuration(
            main={"size": size, "format": "RGB888"},
            buffer_count=2
        )
        self.picam2.configure(config)

        # Load calibration
        self.calib_path = calib_path
        self.calib = self.load_calibration()

        # Apply calibration
        self.apply_calibration()

        self.frame = None
        self.running = False
        self.lock = threading.Lock()

        self.enable_lux_filter = True

    # =========================
    # CALIBRATION
    # =========================
    def load_calibration(self):
        if not os.path.exists(self.calib_path):
            print("Calibration file not found. Using defaults.")
            return {
                "LensPosition": 1.0,
                "Zoom": 1.0,
                "ExposureTime": 13000,
                "AnalogueGain": 1.5,
                "ColourGains": [1.0, 1.0],
                "Sharpness": 1.0,
                "Contrast": 1.0,
                "Saturation": 1.0
            }

        with open(self.calib_path, "r") as f:
            calib = json.load(f)

        print("\n✅ Loaded calibration:")
        for k, v in calib.items():
            print(f"{k}: {v}")
        print()

        return calib

    def apply_calibration(self):
        c = self.calib

        # =========================
        # CORE CAMERA CONTROLS
        # =========================
        controls = {
            "AfMode": 0,
            "LensPosition": c.get("LensPosition", 1.0),

            "AwbEnable": False,
            "AeEnable": False,

            "ExposureTime": c.get("ExposureTime", 13000),
            "AnalogueGain": c.get("AnalogueGain", 1.5),

            "ColourGains": tuple(c.get("ColourGains", [1.0, 1.0])),

            "Sharpness": c.get("Sharpness", 1.0),
            "Contrast": c.get("Contrast", 1.0),
            "Saturation": c.get("Saturation", 1.0)
        }

        self.picam2.set_controls(controls)

        # =========================
        # ZOOM (ScalerCrop)
        # =========================
        zoom = c.get("Zoom", 1.0)

        sensor_w, sensor_h = self.picam2.camera_properties['PixelArraySize']

        new_w = int(sensor_w / zoom)
        new_h = int(sensor_h / zoom)

        x = (sensor_w - new_w) // 2
        y = (sensor_h - new_h) // 2

        self.picam2.set_controls({
            "ScalerCrop": (x, y, new_w, new_h)
        })

        print(f"🔍 Zoom applied: {zoom}")

    def reload_calibration(self):
        """Reload JSON without restarting program"""
        self.calib = self.load_calibration()
        self.apply_calibration()
        print("🔄 Calibration reloaded")

    # =========================
    # CAMERA CONTROL
    # =========================
    def set_focus(self, lens_position):
        self.picam2.set_controls({
            "AfMode": 0,
            "LensPosition": lens_position
        })

    def set_zoom(self, zoom_factor):
        sensor_w, sensor_h = self.picam2.camera_properties['PixelArraySize']

        new_w = int(sensor_w / zoom_factor)
        new_h = int(sensor_h / zoom_factor)

        x = (sensor_w - new_w) // 2
        y = (sensor_h - new_h) // 2

        self.picam2.set_controls({
            "ScalerCrop": (x, y, new_w, new_h)
        })

    # =========================
    # THREADING
    # =========================
    def start(self):
        self.picam2.start()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            frame = self.picam2.capture_array()

            # ---- Convert to BGR ----
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.picam2.stop()