#include <Servo.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>

Servo miServo;
Adafruit_BMP280 bmp;

// --- PIN ESTÁNDAR ---
const int pinServo = 9;
const int pinHall = 11;   
const int pinLed = 8;   
const int pinRele = 12;   

// --- VARIABLES DE CONTROL ---
enum Estado {IDLE, BUSCANDO, ESPERANDO, REGRESANDO};
Estado estadoActual = IDLE;

bool moviendose = false;
int velActual = 90;
int velRegreso = 90;

unsigned long tiempoEsperaInicio = 0;
unsigned long tiempoAnteriorSensor = 0;
const long intervaloSensor = 2000; 

void setup() {
  Serial.begin(9600);
  miServo.attach(pinServo);
  miServo.write(90); 
  
  pinMode(pinHall, INPUT);
  pinMode(pinLed, OUTPUT);
  pinMode(pinRele, OUTPUT);
  digitalWrite(pinLed, LOW);
  digitalWrite(pinRele, HIGH); 

  bmp.begin(0x76, 0x60); // Tu sensor BME280
}

void loop() {
  // 1. GESTIÓN DE COMANDOS SERIAL
  if (Serial.available() > 0) {
    String comando = Serial.readStringUntil('\n');
    comando.trim(); 
    comando.toUpperCase(); 

    if (comando == "A") {
      ejecutarCiclo(100, 80); // Va a 100, regresa a 80
    } 
    else if (comando == "B") {
      ejecutarCiclo(80, 100); // Va a 80, regresa a 100
    }
    else if (comando == "LED ON")  digitalWrite(pinLed, HIGH);
    else if (comando == "LED OFF") digitalWrite(pinLed, LOW);
    else if (comando == "RELE ON")  digitalWrite(pinRele, LOW); 
    else if (comando == "RELE OFF") digitalWrite(pinRele, HIGH); 
  }

  // 2. LÓGICA DE MOVIMIENTO Y RETORNO (Máquina de Estados)
  switch (estadoActual) {
    case BUSCANDO:
      if (digitalRead(pinHall) == LOW) {
        miServo.write(90);
        estadoActual = ESPERANDO;
        tiempoEsperaInicio = millis(); // Empezamos a contar 2 seg
      }
      break;

    case ESPERANDO:
      if (millis() - tiempoEsperaInicio >= 2000) {
        // Terminó la espera, ahora regresamos
        iniciarGiro(velRegreso);
        estadoActual = REGRESANDO;
      }
      break;

    case REGRESANDO:
      if (digitalRead(pinHall) == LOW) {
        miServo.write(90);
        digitalWrite(pinRele, LOW);
        estadoActual = IDLE; // Fin de la secuencia
      }
      break;
      
    case IDLE:
      // No hace nada con el motor
      break;
  }

  // 3. IMPRESIÓN DEL SENSOR (Inalterada)
  if (millis() - tiempoAnteriorSensor >= intervaloSensor) {
    tiempoAnteriorSensor = millis();
    // Serial.print("Temp: ");
    Serial.print(bmp.readTemperature());
    // Serial.print(" *C  |  Presion: ");
    Serial.print(",");
    Serial.println(bmp.readPressure() / 100.0F);
    // Serial.println(" hPa");
  }
}

// Prepara las variables para el ciclo completo
void ejecutarCiclo(int vIda, int vVuelta) {
  velActual = vIda;
  velRegreso = vVuelta;
  digitalWrite(pinRele, HIGH);
  iniciarGiro(velActual);
  estadoActual = BUSCANDO;
}

// Arranca el motor y salta el imán actual
void iniciarGiro(int velocidad) {
  miServo.write(velocidad);
  delay(400); // Pequeño salto para no detectar el imán donde está parado
}