#include <Arduino.h>
#include <HardwareSerial.h>  

#define BUZZER_PIN 9
#define LED_PIN 13

bool isDrowsy = false;  // 졸음 감지 상태 저장
unsigned long drowsyStartTime = 0;  // 졸음 시작 시간
int buzzerFrequency = 500;  // 초기 주파수 (500Hz)

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    char ch = Serial.read();
    
    if (ch == 'B') {  
      // 🚨 졸음 감지 시작
      if (!isDrowsy) {
        isDrowsy = true;
        drowsyStartTime = millis(); // 시작 시간 기록
        buzzerFrequency = 500; // 초기 주파수 설정
      }
    }
    else if (ch == 'N') {
      // ✅ 졸음 종료 → 부저 중지
      isDrowsy = false;
      noTone(BUZZER_PIN);
    }
    else if (ch == 'L') {
      // ⚠️ 차선 이탈 → LED ON
      digitalWrite(LED_PIN, HIGH);
      Serial.println("LED ON (Lane Departure)");
    }
    else if (ch == 'l') {
      // ✅ 차선 복귀 → LED OFF
      digitalWrite(LED_PIN, LOW);
      Serial.println("LED OFF");
    }
  }

  // 🚨 졸음 감지 중이면 시간에 따라 부저 주파수 증가
  if (isDrowsy) {
    unsigned long elapsedTime = millis() - drowsyStartTime;

    // 시간이 지남에 따라 주파수 증가 (500Hz → 2500Hz)
    buzzerFrequency = 500 + (elapsedTime / 2000) * 500;  // 2초마다 500Hz 증가
    if (buzzerFrequency > 2500) buzzerFrequency = 2500;  // 최대 2500Hz 제한

    tone(BUZZER_PIN, buzzerFrequency); // 부저 주파수 적용
  } else {
    noTone(BUZZER_PIN);
  }
}
