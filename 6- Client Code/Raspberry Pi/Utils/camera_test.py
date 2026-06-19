from picamera2 import Picamera2, Preview
import time

def main():
    picam2 = Picamera2()

    # Configuración de video (640x480 es liviano)
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)

    # Mostrar ventana de video (X11 Wayland o tu escritorio)
    picam2.start_preview(Preview.QTGL)

    # Iniciar cámara
    picam2.start()

    print("Mostrando video... Ctrl+C para salir.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Cerrando cámara...")

    picam2.stop()
    picam2.stop_preview()

if __name__ == "__main__":
    main()
