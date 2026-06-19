# Hardware of the Trap

This folder contains all electrical and logical design documents for the smart trap hardware.

## Contents

| File | Description |
|------|-------------|
| `Diagrama_Electrico_TFG.fzz` | Full electrical schematic (Fritzing format) |
| `Diagrama_de_Bloques.drawio` | Block diagram of the system architecture |
| `Diagrama_flujo.drawio` | Control flow diagram of the trap's state machine |
| `Planilla de Costos Tesis.xlsx` | Bill of materials (BOM) and cost breakdown |

---

## Hardware Components

### Main Controllers

| Component | Role |
|-----------|------|
| **Raspberry Pi** (3B+ or 4) | Main computer: AI inference, camera capture, server communication |
| **Arduino Uno** | Real-time hardware controller: servo, relay, Hall sensor, BMP280 |

### Sensors & Actuators

| Component | Pin / Interface | Function |
|-----------|----------------|----------|
| **Pi Camera** | CSI port | Captures mosquito images for AI inference |
| **Servo motor** | Pin 9 (PWM) | Drives the sex-sorting diverter mechanism |
| **Hall effect sensor** | Pin 11 (digital) | Detects home position of the servo disc |
| **BMP280** | I²C (0x76) | Reads temperature (°C) and pressure (hPa) |
| **Relay module** | Pin 12 (digital) | Controls attractant (UV light / CO₂ emitter) |
| **LED indicator** | Pin 8 (digital) | Status indicator light |

---

## Electrical Schematic

Open `Diagrama_Electrico_TFG.fzz` with **Fritzing** (free download at [fritzing.org](https://fritzing.org)).

The schematic shows:
- Arduino ↔ Servo wiring (5V power + PWM signal)
- Arduino ↔ BMP280 I²C bus (SDA/SCL)
- Arduino ↔ Hall sensor (digital input + pull-up)
- Arduino ↔ Relay (digital output, active-LOW)
- Raspberry Pi ↔ Arduino (USB serial, `/dev/ttyACM0`)
- Pi Camera ribbon connection

---

## Serial Communication Protocol (Raspberry Pi ↔ Arduino)

Commands sent from the Raspberry Pi to the Arduino over UART at **9600 baud**:

| Command string | Action |
|----------------|--------|
| `A\n` | Sort Female: servo goes to 100°, returns to 80° |
| `B\n` | Sort Male: servo goes to 80°, returns to 100° |
| `LED ON\n` | Turn on status LED |
| `LED OFF\n` | Turn off status LED |
| `RELE ON\n` | Activate relay (attractant ON) |
| `RELE OFF\n` | Deactivate relay (attractant OFF) |

Data sent from Arduino to Raspberry Pi (every 2 seconds):
```
<temperature>,<pressure_hPa>\n
```

---

## Diagrams

Open `.drawio` files with **draw.io / diagrams.net** (free at [app.diagrams.net](https://app.diagrams.net)).

- **Block diagram**: shows the high-level component interconnections.
- **Flow diagram**: shows the state machine logic (IDLE → BUSCANDO → ESPERANDO → REGRESANDO → IDLE).

---

## Cost Breakdown

See `Planilla de Costos Tesis.xlsx` for the full bill of materials with unit prices in PYG/USD.
