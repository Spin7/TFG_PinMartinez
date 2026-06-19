# Trap 3D Model

This folder contains the **SolidWorks 3D assembly model** of the physical smart mosquito trap.

## Contents

| File | Description |
|------|-------------|
| `Ensamblaje_3D_TFG.SLDASM` | Full 3D assembly of the trap (SolidWorks 2020+) |

## Trap Design

The trap is designed to house:
- A **Raspberry Pi** (main controller + AI inference)
- An **Arduino Uno** (motor control + sensor reading)
- A **Pi Camera** module (for image capture)
- A **servo motor** (drives the sorting mechanism)
- A **Hall effect sensor** (home position detection for the servo)
- A **BMP280 sensor** (temperature and atmospheric pressure)
- A **relay module** (controls an attractant device, e.g., UV light or CO₂)
- Two **collection bins** — one for female, one for male mosquitoes

## Sorting Mechanism

When a mosquito is detected by the AI:
1. The Raspberry Pi sends a serial command (`A` for Female, `B` for Male) to the Arduino.
2. The Arduino activates the **servo motor**, which rotates a diverter/flap.
3. The Hall effect sensor detects when the servo reaches the target position.
4. After a 2-second dwell, the servo returns to the home position.
5. The mosquito is deposited into the corresponding bin.

## Opening the Model

Requires **SolidWorks 2020** or later. To view without a license, use the free **eDrawings Viewer**.
