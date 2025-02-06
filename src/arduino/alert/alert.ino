#include <Arduino.h>
#include <HardwareSerial.h>  

#define BUZZER_PIN 6  // PWM 가능한 핀으로 변경
#define RED_PIN 11     // RGB LED - RED
#define GREEN_PIN 10  // RGB LED - GREEN
#define BLUE_PIN 9   // RGB LED - BLUE


bool isDrowsy = false;  // 졸음 감지 상태
bool isLaneChanging = false;  // 차선 변경 상태
unsigned long drowsyStartTime = 0;
int buzzerFrequency = 500;  // 초기 부저 주파수

// RGB LED 색상 설정 함수
void setColor(int red, int green, int blue) {
  analogWrite(RED_PIN, red);
  analogWrite(GREEN_PIN, green);
  analogWrite(BLUE_PIN, blue);
}

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);
  setColor(0, 0, 0);  // 초기 LED OFF
}

void loop() {
  if (Serial.available() > 0) {
    char ch = Serial.read();
    
    if (ch == 'B') {  
      // 🚨 졸음 감지 시작
      if (!isDrowsy) {
        isDrowsy = true;
        drowsyStartTime = millis();
        buzzerFrequency = 500;
      }
    }
    else if (ch == 'N') {
      // ✅ 졸음 종료 → 부저 중지
      isDrowsy = false;
      noTone(BUZZER_PIN);
    }
    else if (ch == 'L') {
      // ⚠️ 차선 변경 감지
      isLaneChanging = true;

      if (isDrowsy) {
        setColor(255, 0, 0); // 🚨 졸음 상태 → 빨간색
        Serial.println("⚠️ 졸음 상태에서 차선 변경! (RED LED ON)");
      } else {
        setColor(0, 255, 0); // ✅ 정상 상태 → 초록색
        Serial.println("🟢 정상 차선 변경 (GREEN LED ON)");
      }
    }
    else if (ch == 'l') {
      // ✅ 차선 변경 해제
      isLaneChanging = false;
      setColor(0, 0, 0); // LED OFF
      Serial.println("LED OFF (차선 변경 해제)");
    }
  }

  // 🚨 졸음 감지 중이면 시간에 따라 부저 주파수 증가
  if (isDrowsy) {
    unsigned long elapsedTime = millis() - drowsyStartTime;
    buzzerFrequency = 500 + (elapsedTime / 2000) * 500;
    if (buzzerFrequency > 2500) buzzerFrequency = 2500;
    tone(BUZZER_PIN, buzzerFrequency);
  } else {
    noTone(BUZZER_PIN);
  }
}
