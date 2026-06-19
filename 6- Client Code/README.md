# Client Code

This folder contains the embedded software that runs on the IoT trap hardware: the **Raspberry Pi** (main AI client) and the **Arduino** (real-time hardware controller).

## Folder Structure

```
6- Client Code/
│
├── Arduino/
│   └── Client_Arduino/
│       └── Client_Arduino.ino      # Arduino firmware
│
└── Raspberry Pi/
    ├── Client_rasp1/               # Main Raspberry Pi client application
    └── Utils/                      # Camera setup utilities and calibration tools
```

---

## Arduino Firmware (`Arduino/Client_Arduino/`)

See [`Arduino/README.md`](./Arduino/README.md) for details.

**Summary**: The Arduino controls the servo sorting mechanism, reads the BMP280 temperature/pressure sensor, and manages the LED indicator and relay. It communicates with the Raspberry Pi via serial (USB UART at 9600 baud).

---

## Raspberry Pi Client (`Raspberry Pi/Client_rasp1/`)

See [`Raspberry Pi/README.md`](./Raspberry%20Pi/README.md) for details.

**Summary**: The Raspberry Pi runs the AI inference pipeline in real time, manages the camera, communicates detections to the Arduino, and uploads results to the MosquitoWeb server.

---

## Communication Overview

```
[Pi Camera]
     │  (frames @ 10 FPS)
     ▼
[Raspberry Pi]
  ├── YOLO detection
  ├── MobileNet classification  →  Sex: Male / Female
  ├── Upload to server (HTTP POST /upload)
  │       image + detections + temperature + humidity
  └── Serial command to Arduino
          │  "A\n" (Female) or "B\n" (Male)
          ▼
      [Arduino]
        ├── Drives servo to sort mosquito
        ├── Reads BMP280 → sends "temp,pressure\n" to Pi
        └── Controls relay (attractant) and LED
```
