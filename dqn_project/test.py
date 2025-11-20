import serial
import time

# ========================
# 아두이노 포트 및 Baudrate
# ========================
PORT = '/dev/ttyACM0'   # 아두이노 포트 확인
BAUDRATE = 9600

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"[SERIAL] Connected to {PORT} at {BAUDRATE} bps")
except Exception as e:
    print(f"[ERROR] Serial connection failed: {e}")
    exit(1)

time.sleep(2)  # 아두이노 리셋 대기

# ========================
# 명령 리스트 (테스트용)
# ========================
commands = [
    ('MOVE_FORWARD', 2),   # 전진 2초
    ('STOP', 1),           # 정지 1초
    ('MOVE_BACKWARD', 2),  # 후진 2초
    ('STOP', 1),
    ('TURN_LEFT', 1.5),    # 좌회전 1.5초
    ('STOP', 0.5),
    ('TURN_RIGHT', 1.5),   # 우회전 1.5초
    ('STOP', 0.5),
]

# ========================
# 반복 테스트
# ========================
for i in range(3):  # 3번 반복
    print(f"\n[TEST] Loop {i+1}")
    for cmd, duration in commands:
        try:
            ser.write(f"{cmd}\n".encode('utf-8'))
            print(f"[TX] {cmd} (for {duration}s)")
            start_time = time.time()
            
            # 시리얼로 응답 확인 (선택 사항)
            while time.time() - start_time < duration:
                if ser.in_waiting > 0:
                    resp = ser.readline().decode('utf-8', 'ignore').strip()
                    if resp:
                        print(f"[RX] {resp}")
                time.sleep(0.05)
        except Exception as e:
            print(f"[ERROR] Sending {cmd} failed: {e}")

# ========================
# 종료
# ========================
ser.write(b'STOP\n')
print("[TX] STOP")
ser.close()
print("[SERIAL] Connection closed")

