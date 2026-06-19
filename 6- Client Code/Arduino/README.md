# Arduino Firmware

This folder contains the **Arduino sketch** that runs on the Arduino Uno inside the smart trap.

## File

| File | Description |
|------|-------------|
| `Client_Arduino.ino` | Arduino sketch — motor control, sensor reading, serial protocol |

---

## Functionality

The Arduino acts as the **real-time hardware controller**, handling:

1. **Serial command reception** — receives commands from the Raspberry Pi over USB serial (9600 baud).
2. **Servo motor control** — sorts mosquitoes into the correct bin (A = Female, B = Male) using a state machine.
3. **BMP280 sensor reading** — reads temperature and atmospheric pressure every 2 seconds and sends them to the Raspberry Pi.
4. **Relay control** — activates/deactivates an attractant device (UV light or CO₂ emitter).
5. **LED indicator** — status LED for visual feedback.

---

## Pin Map

| Pin | Component | Mode |
|-----|-----------|------|
| 9 | Servo motor (signal) | PWM output |
| 11 | Hall effect sensor | Digital input |
| 8 | LED indicator | Digital output |
| 12 | Relay module | Digital output (active-LOW) |
| SDA/SCL | BMP280 (I²C addr: 0x76) | I²C |

---

## State Machine

The servo sorting cycle follows a 4-state machine:

```
IDLE
  │  (receives "A" or "B" command)
  ▼
BUSCANDO  ← servo rotating toward target
  │  (Hall sensor detects magnet → target reached)
  ▼
ESPERANDO ← waits 2 seconds at target position
  │  (timeout)
  ▼
REGRESANDO ← servo rotating back to home
  │  (Hall sensor detects magnet → home position)
  ▼
IDLE
```

---

## Serial Protocol

**Commands received** (from Raspberry Pi):

| Command | Action |
|---------|--------|
| `A` | Sort Female: servo → 100° (forward), return → 80° |
| `B` | Sort Male: servo → 80° (forward), return → 100° |
| `LED ON` | Turn on LED (pin 8 HIGH) |
| `LED OFF` | Turn off LED (pin 8 LOW) |
| `RELE ON` | Activate relay (pin 12 LOW) |
| `RELE OFF` | Deactivate relay (pin 12 HIGH) |

**Data sent** (to Raspberry Pi, every 2 seconds):
```
<temperature_celsius>,<pressure_hPa>
```
Example: `28.50,1013.25`

---

## Required Libraries

Install via Arduino IDE Library Manager:
- `Servo` (built-in)
- `Wire` (built-in, for I²C)
- `Adafruit BMP280 Library`
- `Adafruit Unified Sensor`

---

## Uploading

1. Open `Client_Arduino.ino` in the **Arduino IDE** (2.x recommended).
2. Select board: **Arduino Uno**.
3. Select the correct COM port.
4. Click **Upload**.

> **Note**: The Arduino resets when a serial connection is opened. The Raspberry Pi waits 2 seconds after connecting before sending the first commands to allow for this boot delay.
