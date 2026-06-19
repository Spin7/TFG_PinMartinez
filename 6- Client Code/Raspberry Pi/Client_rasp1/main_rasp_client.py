import os
import sys
import cv2
import time
import threading

from cascade_inference import CascadeInference
from pipeline.inference_worker import InferenceWorker
from pipeline.uploader import Uploader
from pipeline.serial_manager import SerialManager
from utils.draw import draw_results


# ─────────────────────────────────────────────────────────────────────────────
# LED blinker — toggles the LED every LED_INTERVAL seconds in a daemon thread
# ─────────────────────────────────────────────────────────────────────────────

class LedBlinker:
    """
    Toggles the Arduino LED on/off at a fixed interval.

    The LED starts ON.  After ``interval`` seconds it turns OFF, after another
    ``interval`` seconds it turns ON again, and so on.

    Parameters
    ----------
    serial_manager : SerialManager
        The already-started SerialManager instance to send commands through.
    interval : float
        Seconds between each toggle (default 60 s → 1 minute on, 1 minute off).
    """

    def __init__(self, serial_manager: SerialManager, interval: float = 60.0):
        self._sm       = serial_manager
        self._interval = interval
        self._stop     = threading.Event()
        self._thread   = threading.Thread(
            target=self._run, name="LedBlinker", daemon=True
        )

    def start(self):
        self._thread.start()
        print(f"[LedBlinker] started  interval={self._interval}s")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self._interval + 1)
        print("[LedBlinker] stopped")

    def _run(self):
        led_on = True
        # LED was already turned ON before this thread starts, so first
        # action is to turn it OFF after one interval.
        while not self._stop.wait(timeout=self._interval):
            led_on = not led_on
            state  = 0 if led_on else 1          # 0 = ON, 1 = OFF
            label  = "ON" if led_on else "OFF"
            self._sm.send_led(state)
            print(f"[LedBlinker] LED {label}")

# =============================================================================
# CONFIG
# =============================================================================

# ---- Camera source ----
# True  → Raspberry Pi camera (picamera2 / ThreadedCamera)
# False → PC webcam           (cv2.VideoCapture / ThreadedCamera2)
USE_PI_CAMERA = True
PC_CAM_INDEX  = 0          # OpenCV device index for the PC webcam

# Lazy import: only load the module that matches the chosen camera.
if USE_PI_CAMERA:
    from camera.threaded_camera  import ThreadedCamera
else:
    from camera.threaded_camera2 import ThreadedCamera2

ENABLE_UPLOAD  = True
ENABLE_DRAW    = True
ENABLE_DISPLAY = True

# ---- Detection thresholds ----
DET_CONF_TH = 0.6
CLS_CONF_TH = 0.7

# ---- Behaviour ----
USE_TOP1         = False   # True = keep only the single best detection per frame
TARGET_FPS       = 10      # target inference throughput (controls dynamic SKIP)
COOLDOWN         = 1.0     # min seconds between uploads for the same presence event
PRESENCE_TIMEOUT = 1.0     # seconds of silence before declaring target gone

# ---- Trap state machine ----
# SORT_DURATION: seconds to wait for the Arduino sorting mechanism to finish
# before accepting new detections.  Match this to your hardware timing.
SORT_DURATION = 5.0

# Symbolic state names (plain strings — easy to print in the HUD)
STATE_IDLE    = "IDLE"     # watching for an insect
STATE_SORTING = "SORTING"  # command sent, mechanism running — vision blocked

# ---- Model paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
det_path = os.path.join(BASE_DIR, "best.onnx")
cls_path = os.path.join(BASE_DIR, "best_CLmodel.onnx")

# ---- Server ----
#SERVER_URL = "https://unrupturable-lunatically-jeana.ngrok-free.dev"
SERVER_URL = "https://web-production-90e52.up.railway.app"
CLIENT_ID  = "rpi_cam1"

# ---- LED blinker ----
LED_INTERVAL = 60.0    # seconds between each ON↔OFF toggle (1 minute)

# ---- Serial / Arduino ----
# SERIAL_DEBUG = True  → run on PC without hardware (simulated sensor values,
#                         detection commands are printed instead of written).
# SERIAL_DEBUG = False → real mode (opens SERIAL_PORT, reads/writes Arduino).
SERIAL_DEBUG = False            # <── flip to False on the Raspberry Pi
SERIAL_PORT  = "/dev/ttyACM0" # e.g. "/dev/ttyACM0" on some boards
SERIAL_BAUD  = 9600

# =============================================================================
# INIT
# =============================================================================
model = CascadeInference(
    det_path, cls_path,
    det_conf_th=DET_CONF_TH,
    cls_conf_th=CLS_CONF_TH,
)

if USE_PI_CAMERA:
    cam = ThreadedCamera(size=(640, 480))
    print("[main] Using Raspberry Pi camera (ThreadedCamera)")
else:
    cam = ThreadedCamera2(size=(640, 480), device_index=PC_CAM_INDEX)
    print("[main] Using PC webcam (ThreadedCamera2)")
cam.start()

worker = InferenceWorker(model)
worker.start()

if ENABLE_UPLOAD:
    uploader = Uploader(SERVER_URL, CLIENT_ID)
    uploader.start()
else:
    uploader = None

serial_mgr = SerialManager(port=SERIAL_PORT, baudrate=SERIAL_BAUD, debug=SERIAL_DEBUG)
serial_mgr.start()

if uploader is not None:
    uploader.set_serial_manager(serial_mgr)

# Arduino resets when the serial connection is opened.
# Wait for it to finish booting before sending the first commands.
print("[main] Waiting for Arduino to boot...")
time.sleep(2.0)   # ← add this line

serial_mgr.send_relay(0)  # → "RELE ON\n"
serial_mgr.send_led(0)    # → "LED ON\n"  (blinker takes over from here)

#led_blinker = LedBlinker(serial_mgr, interval=LED_INTERVAL)
#led_blinker.start()

# =============================================================================
# STATE
# =============================================================================
prev_time    = time.perf_counter()
fps_cam      = 0.0
fps_inf      = 0.0
latency      = 0.0

frame_id     = 0
SKIP         = 2

# ---- Trap state machine ----
trap_state      = STATE_IDLE   # current state
sort_start_time = 0.0          # perf_counter timestamp when SORTING began
sort_class      = None         # class label of the detection that triggered the sort

last_seen_time   = 0.0         # last time any detection was visible
last_upload_time = 0.0

last_result = None   # most recent inference output (may be from a previous frame)

# =============================================================================
# MAIN LOOP
# =============================================================================
try:
    while True:
        frame = cam.read()
        if frame is None:
            continue

        frame_id += 1

        # ---- Camera FPS ------------------------------------------------
        now       = time.perf_counter()
        delta     = now - prev_time
        fps_cam   = 1.0 / max(delta, 1e-9)
        prev_time = now

        # ---- Submit frame to inference worker ---------------------------
        # Dynamic SKIP: target TARGET_FPS inference throughput.
        # Only submit when the worker is idle (no frame waiting).
        if frame_id % SKIP == 0 and worker.is_ready():
            worker.submit(frame)

        # ---- Consume latest result --------------------------------------
        result = worker.get()
        if result is not None:
            last_result = result
            latency     = result[5]
            fps_inf     = 1.0 / latency if latency > 0 else 0.0
            # Adaptive SKIP based on measured latency
            SKIP = max(1, int(fps_cam / TARGET_FPS))

        # ---- Unpack detections -----------------------------------------
        # last_result carries the detections from the most recent inference.
        # The model already applied DET_CONF_TH / CLS_CONF_TH internally.
        if last_result is not None:
            _, boxes, preds, det_scores, cls_scores, _ = last_result
        else:
            boxes, preds, det_scores, cls_scores = [], [], [], []

        # ---- Optional TOP-1 filter ------------------------------------
        if USE_TOP1 and len(boxes) > 0:
            # Pick the single detection with the highest combined score
            best = max(
                zip(boxes, preds, det_scores, cls_scores),
                key=lambda t: t[2] * t[3],
            )
            boxes, preds, det_scores, cls_scores = (
                [best[0]], [best[1]], [best[2]], [best[3]]
            )

        has_detection = len(boxes) > 0

        # =================================================================
        # TRAP STATE MACHINE
        # =================================================================
        if trap_state == STATE_SORTING:
            # ── Arduino is busy ──────────────────────────────────────────────
            # Camera and inference keep running so the display stays live,
            # but we do NOT trigger any new events until the sort finishes.
            elapsed = now - sort_start_time
            if elapsed >= SORT_DURATION:
                trap_state = STATE_IDLE
                sort_class  = None
                print(f"[TRAP] Sort complete ({elapsed:.1f}s) → IDLE")

        else:  # STATE_IDLE ─────────────────────────────────────────────────
            if has_detection:
                last_seen_time = now

                if (now - last_upload_time) > COOLDOWN:
                    # ── TRIGGER ──────────────────────────────────────────────
                    top_pred   = int(preds[0])
                    sort_class = model.CLASS_NAMES[top_pred]
                    cmd_char   = 'A' if top_pred == 0 else 'B'

                    # 1. Send sorting command to Arduino
                    serial_mgr.send_detection(top_pred)

                    # 2. Upload frame + detections to server
                    if uploader is not None:
                        uploader.submit(frame, boxes, preds, det_scores, cls_scores)

                    # 3. Transition to SORTING — blocks new triggers
                    sort_start_time  = now
                    last_upload_time = now
                    trap_state       = STATE_SORTING
                    print(f"[TRAP] {sort_class} detected → sending '{cmd_char}' → SORTING")

        # =================================================================
        # DRAW
        # =================================================================
        if ENABLE_DRAW:
            if has_detection:
                # draw_results works on a copy so the original is untouched
                frame_draw = draw_results(
                    frame.copy(),
                    boxes, preds, det_scores, cls_scores,
                    model.CLASS_NAMES,
                )
            else:
                frame_draw = frame         # no copy needed — only used for display

            # ---- HUD overlay ----------------------------------------
            # Build the state line: colour-code by trap state
            if trap_state == STATE_SORTING:
                remaining    = max(0.0, SORT_DURATION - (now - sort_start_time))
                state_text   = f"SORTING {sort_class or ''} [{remaining:.1f}s]"
                state_colour = (0, 100, 255)   # orange-red
            else:
                state_text   = f"IDLE"
                state_colour = (200, 200, 255)  # soft white

            hud = [
                (f"CAM FPS : {fps_cam:.1f}",          (255, 255,   0)),
                (f"INF FPS : {fps_inf:.1f}",           (  0, 255,   0)),
                (f"Latency : {latency*1000:.1f} ms",   (  0, 255,   0)),
                (f"SKIP    : {SKIP}",                  (  0, 200, 255)),
                (f"STATE   : {state_text}",             state_colour),
                (f"DET     : {len(boxes)}",             (  0, 255, 255)),
                (f"DROPPED : {worker.frames_dropped}",  (  0, 100, 255)),
            ]

            # Ensure we have a writable copy for the HUD text
            if not has_detection:
                frame_draw = frame.copy()

            for row, (text, colour) in enumerate(hud):
                cv2.putText(
                    frame_draw, text,
                    (20, 30 + row * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
                    lineType=cv2.LINE_AA,
                )

        # ---- SORTING banner (subtle, top-right) ------------------------
        if trap_state == STATE_SORTING:
            remaining = max(0.0, SORT_DURATION - (now - sort_start_time))
            banner    = f"Sorting... {remaining:.1f}s"

            fh, fw = frame_draw.shape[:2]

            font_scale = 0.7
            thickness  = 2

            (tw, th), _ = cv2.getTextSize(
                banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            margin = 20

            # Top-right corner
            bx = fw - tw - margin
            by = margin + th

            # Subtle transparent background
            overlay = frame_draw.copy()
            cv2.rectangle(
                overlay,
                (bx - 10, by - th - 8),
                (bx + tw + 10, by + 6),
                (0, 0, 0),
                -1
            )
            alpha = 0.35
            cv2.addWeighted(overlay, alpha, frame_draw, 1 - alpha, 0, frame_draw)

            # Text
            cv2.putText(
                frame_draw,
                banner,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (220, 220, 220),
                thickness,
                lineType=cv2.LINE_AA,
            )

        else:
            frame_draw = frame

        # =================================================================
        # DISPLAY
        # =================================================================
        if ENABLE_DISPLAY:
            cv2.imshow("Real-Time Detection", frame_draw)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                break

except KeyboardInterrupt:
    print("\n[main] KeyboardInterrupt — shutting down...")

finally:
    print("[main] Stopping all workers...")
    #led_blinker.stop()
    worker.stop()
    serial_mgr.send_relay(1)   # → "RELE OFF\n"
    serial_mgr.send_led(1)     # → "LED OFF\n"
    #serial_mgr.send_led(0)
    serial_mgr.stop()
    if uploader is not None:
        uploader.stop()
    cam.stop()
    cv2.destroyAllWindows()
    print("[main] Clean exit.")
    sys.exit(0)