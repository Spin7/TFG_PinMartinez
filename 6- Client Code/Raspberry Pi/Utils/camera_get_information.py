from picamera2 import Picamera2, Preview
import time
import pprint

def main():

    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (640, 480)}  # FIX
    )
    picam2.configure(config)

    picam2.start_preview(Preview.QTGL)
    picam2.start()

    time.sleep(2)  # let AE/AWB settle

    print("\nReading live ISP parameters (Ctrl+C to stop)\n")

    try:
        while True:
            metadata = picam2.capture_metadata()

            # Extract key parameters
            data = {
                "ExposureTime": metadata.get("ExposureTime"),
                "AnalogueGain": metadata.get("AnalogueGain"),
                "ColourGains": metadata.get("ColourGains"),
                "ColourTemperature": metadata.get("ColourTemperature"),
                "Lux": metadata.get("Lux"),
                "FocusFoM": metadata.get("FocusFoM"),
                "LensPosition": metadata.get("LensPosition"),
            }

            pprint.pprint(data)
            print("-" * 40)

            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping...")

    picam2.stop()
    picam2.stop_preview()


if __name__ == "__main__":
    main()