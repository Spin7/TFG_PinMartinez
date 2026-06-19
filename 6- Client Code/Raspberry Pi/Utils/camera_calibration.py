import cv2
import numpy as np
import json
import time
from picamera2 import Picamera2

# =========================
# INIT CAMERA
# =========================
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()

time.sleep(2)

# =========================
# WINDOW
# =========================
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

def nothing(x): pass

# =========================
# TRACKBARS
# =========================

# Focus / Zoom
cv2.createTrackbar("Lens x10", "Calibration", 10, 100, nothing)
cv2.createTrackbar("Zoom x10", "Calibration", 10, 50, nothing)

# Color (REAL ISP)
cv2.createTrackbar("Red Gain x100", "Calibration", 84, 300, nothing)
cv2.createTrackbar("Blue Gain x100", "Calibration", 155, 300, nothing)

# Exposure / Gain
cv2.createTrackbar("Exposure_us", "Calibration", 13000, 50000, nothing)
cv2.createTrackbar("AnalogGain x10", "Calibration", 15, 80, nothing)

# Image tuning
cv2.createTrackbar("Sharpness x10", "Calibration", 10, 30, nothing)
cv2.createTrackbar("Contrast x10", "Calibration", 10, 30, nothing)
cv2.createTrackbar("Saturation x10", "Calibration", 10, 30, nothing)

# Extra
cv2.createTrackbar("DigitalGain x10", "Calibration", 10, 40, nothing)

print("""
Controls:
  q → quit
  s → save calibration
  w → lock AWB
  u → unlock AWB
  e → lock AE
  r → unlock AE
  f → autofocus trigger
""")

saved = {}

# =========================
# MAIN LOOP
# =========================
while True:

    frame = picam2.capture_array()

    # =========================
    # READ TRACKBARS
    # =========================
    lens = cv2.getTrackbarPos("Lens x10", "Calibration") / 10.0
    zoom = cv2.getTrackbarPos("Zoom x10", "Calibration") / 10.0

    gain_r = cv2.getTrackbarPos("Red Gain x100", "Calibration") / 100.0
    gain_b = cv2.getTrackbarPos("Blue Gain x100", "Calibration") / 100.0

    exposure = cv2.getTrackbarPos("Exposure_us", "Calibration")
    analog_gain = cv2.getTrackbarPos("AnalogGain x10", "Calibration") / 10.0
    digital_gain = cv2.getTrackbarPos("DigitalGain x10", "Calibration") / 10.0

    sharpness = cv2.getTrackbarPos("Sharpness x10", "Calibration") / 10.0
    contrast = cv2.getTrackbarPos("Contrast x10", "Calibration") / 10.0
    saturation = cv2.getTrackbarPos("Saturation x10", "Calibration") / 10.0

    # =========================
    # APPLY CAMERA CONTROLS
    # =========================
    picam2.set_controls({
        "AfMode": 0,
        "LensPosition": lens,

        "ColourGains": (gain_r, gain_b),

        "ExposureTime": exposure,
        "AnalogueGain": analog_gain,

        "Sharpness": sharpness,
        "Contrast": contrast,
        "Saturation": saturation,
    })

    # Zoom (ScalerCrop)
    sensor_w, sensor_h = picam2.camera_properties['PixelArraySize']

    new_w = int(sensor_w / zoom)
    new_h = int(sensor_h / zoom)

    x = (sensor_w - new_w) // 2
    y = (sensor_h - new_h) // 2

    picam2.set_controls({
        "ScalerCrop": (x, y, new_w, new_h)
    })

    # =========================
    # METADATA (REAL ISP STATE)
    # =========================
    meta = picam2.capture_metadata()

    # =========================
    # DISPLAY
    # =========================
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    overlay = {
        "Lens": lens,
        "Zoom": zoom,
        "Exp": meta.get("ExposureTime"),
        "Gain": meta.get("AnalogueGain"),
        "Lux": round(meta.get("Lux", 0), 1),
        "CG": tuple(round(x,2) for x in meta.get("ColourGains", (0,0)))
    }

    y0 = 25
    for i, (k, v) in enumerate(overlay.items()):
        cv2.putText(frame, f"{k}: {v}", (10, y0 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # KEY CONTROLS
    # =========================
    if key == ord('q'):
        break

    elif key == ord('w'):
        gains = meta["ColourGains"]
        picam2.set_controls({
            "AwbEnable": False,
            "ColourGains": gains
        })
        print("AWB locked:", gains)

    elif key == ord('u'):
        picam2.set_controls({"AwbEnable": True})
        print("AWB unlocked")

    elif key == ord('e'):
        picam2.set_controls({"AeEnable": False})
        print("AE locked")

    elif key == ord('r'):
        picam2.set_controls({"AeEnable": True})
        print("AE unlocked")

    elif key == ord('f'):
        picam2.set_controls({"AfTrigger": 0})
        print("Autofocus triggered")

    elif key == ord('s'):
        saved = {
            "LensPosition": lens,
            "Zoom": zoom,
            "ExposureTime": exposure,
            "AnalogueGain": analog_gain,
            "ColourGains": (gain_r, gain_b),
            "Sharpness": sharpness,
            "Contrast": contrast,
            "Saturation": saturation
        }

        print("\n=== FINAL CALIBRATION ===")
        for k, v in saved.items():
            print(f"{k}: {v}")
        print("========================")

        # Save JSON
        with open("camera_calibration.json", "w") as f:
            json.dump(saved, f, indent=4)

        print("Saved to camera_calibration.json\n")

# =========================
# CLEANUP
# =========================
picam2.stop()
cv2.destroyAllWindows()