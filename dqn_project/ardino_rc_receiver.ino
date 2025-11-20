
// === Jetson과 통신하는 아두이노 RC카 코드 ===
// L298N + 서보모터 + 시리얼 통신


#include <Servo.h>
Servo servo;


// === 핀 설정 ===
const int servoPin = 8;        // 서보 핀
const int ENA = 3, IN1 = 4, IN2 = 11;
const int ENB = 5, IN3 = 7, IN4 = 6;


// === 조향 각도 ===
const int straightAngle = 90;
const int leftAngle     = 80;
const int rightAngle    = 100;


// === 속도 설정 ===
const int SPEED = 185;   // 0~255 (모든 동작 속도)


// === 모터 제어 ===
void forward() {
 // 왼쪽 모터 전진
 digitalWrite(IN1, HIGH);
 digitalWrite(IN2, LOW);
 analogWrite(ENA, SPEED);


 // 오른쪽 모터 전진
 digitalWrite(IN3, HIGH);
 digitalWrite(IN4, LOW);
 analogWrite(ENB, SPEED);
}


void stopMotors() {
 analogWrite(ENA, 0);
 analogWrite(ENB, 0);


 digitalWrite(IN1, LOW);
 digitalWrite(IN2, LOW);
 digitalWrite(IN3, LOW);
 digitalWrite(IN4, LOW);
}


// --- 명령 처리 함수들 ---
void processCommand(String command) {
 // 단순 명령 처리 (Jetson에서 오는 명령)
 if (command == "MOVE_FORWARD") {
   Serial.println("전진 명령 수신");
   forward();
   servo.write(straightAngle);
   return;
 }
 else if (command == "TURN_LEFT") {
   Serial.println("좌회전 명령 수신");
   forward();
   servo.write(leftAngle);
   return;
 }
 else if (command == "TURN_RIGHT") {
   Serial.println("우회전 명령 수신");
   forward();
   servo.write(rightAngle);
   return;
 }
 else if (command == "STOP") {
   Serial.println("정지 명령 수신");
   stopMotors();
   return;
 }
 else if (command == "STATUS") {
   Serial.println("상태 요청 수신");
   sendStatus();
   return;
 }
  // JSON 파싱 (기존 명령들)
 if (command.indexOf("move_forward") != -1) {
   forward();
   servo.write(straightAngle);
  
 } else if (command.indexOf("turn_left") != -1) {
   forward();
   servo.write(leftAngle);
  
 } else if (command.indexOf("turn_right") != -1) {
   forward();
   servo.write(rightAngle);
  
 } else if (command.indexOf("stop") != -1) {
   stopMotors();
  
 } else if (command.indexOf("get_status") != -1) {
   sendStatus();
 }
}


void sendStatus() {
 Serial.println("STATUS: Ready");
}


void setup() {
 Serial.begin(9600);


 pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
 pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);


 servo.attach(servoPin);
 servo.write(straightAngle);
 stopMotors();


 Serial.println("Ready: f=직진90, s=정지, l=좌80, r=우100");
}


void loop() {
 if (!Serial.available()) return;
 char cmd = Serial.read();


 if (cmd == 'f') {
   forward();
   servo.write(straightAngle);
 } else if (cmd == 's') {
   stopMotors();
 } else if (cmd == 'l') {
   forward();
   servo.write(leftAngle);
 } else if (cmd == 'r') {
   forward();
   servo.write(rightAngle);
 }
}










