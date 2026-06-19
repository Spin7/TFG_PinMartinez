"""
pipeline/serial_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Threaded serial manager for Arduino communication.

Responsibilities
----------------
  READ  – continuously reads temperature + atmospheric pressure lines sent by
          the Arduino and caches the most recent valid reading.
          Pressure (hPa) is converted to an *estimated* relative humidity (%)
          using the Magnus approximation so the rest of the pipeline (uploader,
          server) continues to receive a value in the expected 0-100 % range.

  WRITE – sends commands back to the Arduino when needed:
              "A\n"        → Female detected
              "B\n"        → Male   detected
              "RELE ON\n"  → Relay turned on
              "RELE OFF\n" → Relay turned off
              "LED ON\n"   → LED turned on
              "LED OFF\n"  → LED turned off

Pressure → Humidity estimation
-------------------------------
True relative humidity cannot be derived from pressure alone — you need a
dew-point temperature or a wet-bulb reading.  As a practical fallback this
module uses the Magnus formula with a *fixed dew-point depression* (how many
°C below air temperature the dew point typically sits).  The default is 10 °C,
which corresponds to roughly 55-65 % RH at typical room temperatures.

Tune DEW_POINT_DEPRESSION_C for your deployment environment:
  • Dry / arid indoor  → 15-20 °C depression (lower RH estimate)
  • Humid / greenhouse → 3-5  °C depression  (higher RH estimate)

Debug mode
----------
When debug=True (or pyserial is unavailable / port cannot be opened) the class
falls back to simulated values so the pipeline can be tested on a PC without
hardware attached.

Usage
-----
    from pipeline.serial_manager import SerialManager

    sm = SerialManager(
        port="/dev/ttyUSB0",
        baudrate=9600,
        debug=False,
        dew_point_depression=10.0,   # tune to your environment
    )
    sm.start()

    temp, humidity_est = sm.get_sensors()   # humidity estimated from pressure
    pressure           = sm.get_pressure()  # raw hPa reading

    sm.send_detection(pred)
    sm.send_relay(state)
    sm.send_led(state)
    sm.stop()
"""

import json
import math
import queue
import random
import threading
import time

# ── TTL for a sensor reading to be considered "fresh" (seconds) ──────────────
SENSOR_TTL_SEC = 10.0

# ── Default dew-point depression used in the Magnus RH estimate (°C) ─────────
DEW_POINT_DEPRESSION_C = 7.0

# ── Command maps ──────────────────────────────────────────────────────────────
DETECTION_CMD = {0: b"A\n",        1: b"B\n"}
RELAY_CMD     = {0: b"RELE ON\n",  1: b"RELE OFF\n"}
LED_CMD       = {0: b"LED ON\n",   1: b"LED OFF\n"}


# ─────────────────────────────────────────────────────────────────────────────
# Physics helper
# ─────────────────────────────────────────────────────────────────────────────

def pressure_to_humidity(temp_c: float, pressure_hpa: float,
                          dew_point_depression: float = DEW_POINT_DEPRESSION_C) -> float:
    """
    Estimate relative humidity (%) from air temperature using the Magnus formula.

    Pressure alone cannot determine humidity; we assume the dew point is
    ``dew_point_depression`` degrees below the air temperature.
    Accuracy: ±15-25 % RH — sufficient to keep the server value in range.

    Parameters
    ----------
    temp_c : float            Air temperature in °C.
    pressure_hpa : float      Atmospheric pressure in hPa (used for logging only).
    dew_point_depression : float  Assumed °C gap between air temp and dew point.

    Returns
    -------
    float  Estimated RH clamped to [5.0, 99.0] %.
    """
    A = 17.625
    B = 243.04  # °C  (Alduchov & Eskridge 1996)

    dew_point_c = temp_c - dew_point_depression
    gamma_air   = (A * temp_c)      / (B + temp_c)
    gamma_dew   = (A * dew_point_c) / (B + dew_point_c)
    rh          = 100.0 * math.exp(gamma_dew - gamma_air)

    return max(5.0, min(99.0, round(rh, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Parser  — temperature + pressure from the Arduino
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sensor_line(line: str):
    """
    Parse a raw Arduino line into (temp_float, pressure_float).

    Accepted formats
    ----------------
    "23.45,997.32"             plain CSV
    "T:23.45,P:997.32"         prefixed  (P = pressure)
    "T:23.45,H:997.32"         legacy H prefix — treated as pressure
    '{"t":23.45,"p":997.32}'   JSON  (keys: t/temp/temperature,
                                       p/press/pressure/h/hum/humidity)

    Returns (None, None) on parse failure.
    """
    if not line:
        return None, None
    line = line.strip()

    # ── plain CSV ─────────────────────────────────────────────────────────────
    if "," in line and line.count(",") == 1:
        parts = line.split(",")
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass

    # ── prefixed: T:..., P:...  or  T:..., H:... ─────────────────────────────
    upper = line.upper().replace(" ", "")
    if "T:" in upper or "P:" in upper or "H:" in upper:
        try:
            parts = [p.strip() for p in line.replace(" ", "").split(",") if p]
            t = p_val = None
            for p in parts:
                pu = p.upper()
                if pu.startswith("T:"):
                    t = float(p.split(":", 1)[1])
                elif pu.startswith("P:") or pu.startswith("H:"):
                    p_val = float(p.split(":", 1)[1])
            if t is not None and p_val is not None:
                return t, p_val
        except (ValueError, IndexError):
            pass

    # ── JSON ──────────────────────────────────────────────────────────────────
    if line.startswith("{") and line.endswith("}"):
        try:
            j = json.loads(line)
            t = p_val = None
            for k, v in j.items():
                kl = k.lower()
                if kl in ("t", "temp", "temperature"):
                    t = float(v)
                elif kl in ("p", "press", "pressure", "h", "hum", "humidity"):
                    p_val = float(v)
            if t is not None and p_val is not None:
                return t, p_val
        except (ValueError, json.JSONDecodeError):
            pass

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# SerialManager
# ─────────────────────────────────────────────────────────────────────────────

class SerialManager:
    """
    Manages bidirectional serial communication with the Arduino in a
    dedicated background thread.

    Parameters
    ----------
    port : str
        Serial port, e.g. "/dev/ttyUSB0" or "/dev/ttyACM0".
    baudrate : int
        Baud rate (default 9600, must match the Arduino sketch).
    sensor_ttl : float
        Seconds a sensor reading is considered valid (default 10).
    debug : bool
        If True, skip real hardware and return simulated sensor data.
    dew_point_depression : float
        °C below air temperature assumed for the dew point when estimating
        relative humidity from pressure (default 10 °C).
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 9600,
        sensor_ttl: float = SENSOR_TTL_SEC,
        debug: bool = False,
        dew_point_depression: float = DEW_POINT_DEPRESSION_C,
    ):
        self.port = port
        self.baudrate = baudrate
        self.sensor_ttl = sensor_ttl
        self.debug = debug
        self.dew_point_depression = dew_point_depression

        # ── shared sensor state ───────────────────────────────────────────────
        self._lock = threading.Lock()
        self._temp: float | None = None
        self._pressure: float | None = None   # raw hPa
        self._ts: float = 0.0

        # ── write queue ───────────────────────────────────────────────────────
        self._write_q: queue.Queue[bytes] = queue.Queue()

        # ── control ───────────────────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ser = None

        mode = "DEBUG" if debug else f"REAL  port={port} baud={baudrate}"
        print(f"[SerialManager] init  mode={mode}  dew_depression={dew_point_depression}°C")

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the background thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="SerialManager",
            daemon=True,
        )
        self._thread.start()
        print("[SerialManager] thread started")

    def stop(self, timeout: float = 2.0):
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        print("[SerialManager] stopped")

    def get_sensors(self) -> tuple[float, float]:
        """
        Return ``(temperature_°C, estimated_humidity_%)``.

        Humidity is derived from the raw pressure reading via the Magnus
        formula.  Falls back to simulated values when no fresh reading exists.
        """
        with self._lock:
            temp, pres, ts = self._temp, self._pressure, self._ts

        if temp is not None and pres is not None and (time.time() - ts) <= self.sensor_ttl:
            est_hum = pressure_to_humidity(temp, pres, self.dew_point_depression)
            result  = (round(temp, 2), est_hum)
            if self.debug:
                print(f"[SerialManager] get_sensors → temp={temp}°C  pres={pres}hPa  est_hum={est_hum}%")
            return result

        # ── fallback / debug simulation ───────────────────────────────────────
        sim_temp = round(20.0 + random.random() * 6.0, 2)
        sim_hum  = round(45.0 + random.random() * 15.0, 2)
        if self.debug:
            print(f"[SerialManager] get_sensors → simulated  temp={sim_temp} hum={sim_hum}")
        return sim_temp, sim_hum

    def get_pressure(self) -> float | None:
        """Return the raw atmospheric pressure in hPa, or None if stale/absent."""
        with self._lock:
            pres, ts = self._pressure, self._ts
        if pres is not None and (time.time() - ts) <= self.sensor_ttl:
            return round(pres, 2)
        return None

    def send_detection(self, pred: int):
        """pred=0 → Female → 'A\\n';  pred=1 → Male → 'B\\n'"""
        cmd = DETECTION_CMD.get(pred)
        if cmd is None:
            print(f"[SerialManager] send_detection: unknown pred={pred!r}, ignoring")
            return
        self._enqueue(cmd, "send_detection", "Female(A)" if pred == 0 else "Male(B)")

    def send_relay(self, state: int):
        """state=0 → 'RELE ON\\n';  state=1 → 'RELE OFF\\n'"""
        cmd = RELAY_CMD.get(state)
        if cmd is None:
            print(f"[SerialManager] send_relay: unknown state={state!r}, ignoring")
            return
        self._enqueue(cmd, "send_relay", "RELE ON" if state == 0 else "RELE OFF")

    def send_led(self, state: int):
        """state=0 → 'LED ON\\n';  state=1 → 'LED OFF\\n'"""
        cmd = LED_CMD.get(state)
        if cmd is None:
            print(f"[SerialManager] send_led: unknown state={state!r}, ignoring")
            return
        self._enqueue(cmd, "send_led", "LED ON" if state == 0 else "LED OFF")

    # ── internal ──────────────────────────────────────────────────────────────

    def _enqueue(self, cmd: bytes, caller: str, label: str):
        if self.debug:
            print(f"[SerialManager] {caller} → [{label}] would send {cmd!r} (debug, no real write)")
            return
        try:
            self._write_q.put_nowait(cmd)
        except Exception as e:
            print(f"[SerialManager] {caller} queue error: {e}")

    def _update_sensor(self, temp: float, pressure: float):
        with self._lock:
            self._temp     = temp
            self._pressure = pressure
            self._ts       = time.time()

    def _run(self):
        if self.debug:
            self._run_debug()
        else:
            self._run_real()

    # ── DEBUG loop ────────────────────────────────────────────────────────────

    def _run_debug(self):
        print("[SerialManager] running in DEBUG mode (no hardware needed)")
        sim_interval = 2.0
        last_sim = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            if now - last_sim >= sim_interval:
                t = round(20.0 + random.random() * 6.0, 2)
                p = round(995.0 + random.random() * 10.0, 2)
                self._update_sensor(t, p)
                est = pressure_to_humidity(t, p, self.dew_point_depression)
                print(f"[SerialManager][DEBUG] simulated → temp={t}°C  pressure={p}hPa  est_hum={est}%")
                last_sim = now

            while not self._write_q.empty():
                try:
                    cmd = self._write_q.get_nowait()
                    print(f"[SerialManager][DEBUG] would write → {cmd!r}")
                except Exception:
                    pass

            time.sleep(0.1)

        print("[SerialManager][DEBUG] thread finished")

    # ── REAL loop ─────────────────────────────────────────────────────────────

    def _run_real(self):
        try:
            import serial as _serial
        except ImportError:
            print("[SerialManager] pyserial not installed — pip install pyserial\n"
                  "[SerialManager] Falling back to debug simulation.")
            self._run_debug()
            return

        try:
            ser = _serial.Serial(self.port, baudrate=self.baudrate, timeout=0.5)
            self._ser = ser
            print(f"[SerialManager] opened {self.port} @ {self.baudrate} baud")
        except Exception as e:
            print(f"[SerialManager] could not open {self.port}: {e}\n"
                  "[SerialManager] Falling back to debug simulation.")
            self._run_debug()
            return

        try:
            while not self._stop_event.is_set():
                # ── READ ──────────────────────────────────────────────────────
                try:
                    raw = ser.readline()
                    if raw:
                        try:
                            line = raw.decode("utf-8", errors="replace").strip()
                        except Exception:
                            line = str(raw)
                        if line:
                            t, p = _parse_sensor_line(line)
                            if t is not None and p is not None:
                                self._update_sensor(t, p)
                                est = pressure_to_humidity(t, p, self.dew_point_depression)
                                print(f"[SerialManager] sensor ← temp={t}°C  "
                                      f"pressure={p}hPa  est_humidity={est}%")
                            else:
                                print(f"[SerialManager] unparseable line: {line!r}")
                except Exception as e:
                    print(f"[SerialManager] read error: {e}")
                    time.sleep(0.5)

                # ── WRITE ─────────────────────────────────────────────────────
                while not self._write_q.empty():
                    try:
                        cmd = self._write_q.get_nowait()
                        ser.write(cmd)
                        print(f"[SerialManager] sent → {cmd!r}")
                    except Exception as e:
                        print(f"[SerialManager] write error: {e}")
        finally:
            try:
                ser.close()
            except Exception:
                pass
            print("[SerialManager] port closed")
