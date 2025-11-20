#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson과 아두이노 RC카 연결 코드 (DQN Replay 버전)


L298N 모터 드라이버와 엔코더를 사용하는 RC카를 Jetson에서 제어합니다.
dqn_Replay.py의 새로운 DQN 구조에 맞게 수정됨.
ENA, ENB 점퍼케이블 연결 버전 (PWM 제어 불가능)
"""


import serial
import time
import threading
import queue
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from collections import deque
import random
import psutil


# OpenCV GUI 비활성화
os.environ['OPENCV_VIDEOIO_PRIORITY_V4L2'] = '1'
cv2.setNumThreads(1)


# Jetson Orin에서 CUDA 메모리 최적화
if torch.cuda.is_available():
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.deterministic = False
   os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
   os.environ["CUDA_CACHE_DISABLE"] = "0"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# 새로운 DQN 구조 정의
class LaneDetector:
   def __init__(self):
       self.prev_lane = 1  # 초기값: 오른쪽 차선
       # 빨강 HSV 범위
       self.lower_red1 = np.array([0, 70, 50])
       self.upper_red1 = np.array([10, 255, 255])
       self.lower_red2 = np.array([170, 70, 50])
       self.upper_red2 = np.array([180, 255, 255])


   def process_frame(self, frame):
       height, width = frame.shape[:2]


       # ----------------------
       # HSV 변환 후 노랑+흰색 추출
       # ----------------------
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       mask_white = cv2.inRange(hsv, np.array([0, 0, 230]), np.array([180, 25, 255]))
       mask_yellow = cv2.inRange(
           hsv, np.array([15, 80, 100]), np.array([35, 255, 255])
       )
       lane_mask = cv2.bitwise_or(mask_white, mask_yellow)


       # ----------------------
       # 노란색 물체 기반 좌/우 판단
       # ----------------------
       yellow_mask = cv2.inRange(
           hsv, np.array([15, 80, 100]), np.array([35, 255, 255])
       )
       left_half = yellow_mask[:, : width // 2]
       right_half = yellow_mask[:, width // 2 :]
       left_count = cv2.countNonZero(left_half)
       right_count = cv2.countNonZero(right_half)


       # lane_state 결정: 오른쪽, 왼쪽
       if left_count > right_count and left_count > 0:
           lane_state = "right"
       elif right_count > left_count and right_count > 0:
           lane_state = "left"
       else:
           lane_state = self.prev_lane  # 이전 차선 유지
       self.prev_lane = lane_state


       # ----------------------
       # 빨강 물체 검출
       # ----------------------
       mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
       mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
       mask_red = cv2.bitwise_or(mask_red1, mask_red2)
       red_pixels = cv2.countNonZero(mask_red)
       red_ratio = red_pixels / (height * width)


       return lane_state, red_ratio


class DQN(nn.Module):
   def __init__(self, state_dim, action_dim):
       super().__init__()
       self.fc = nn.Sequential(
           nn.Linear(state_dim, 64),
           nn.ReLU(),
           nn.Dropout(0.2),  # 과적합 방지
           nn.Linear(64, 64),
           nn.ReLU(),
           nn.Dropout(0.2),
           nn.Linear(64, action_dim),
       )


   def forward(self, x):
       return self.fc(x)


class DQNAgent:
   def __init__(
       self, state_dim, action_dim=3, device="cpu", cql_alpha=1e-4, lr=1e-4, gamma=0.99
   ):
       if torch.cuda.is_available():
           device = "cuda"
       self.device = device
       self.state_dim = state_dim
       self.action_dim = action_dim
       self.gamma = gamma
       self.cql_alpha = cql_alpha


       # 네트워크 초기화
       self.policy_net = DQN(state_dim, action_dim).to(device)
       self.target_net = DQN(state_dim, action_dim).to(device)
       self.target_net.load_state_dict(self.policy_net.state_dict())


       self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)


       # 온라인 학습용 Replay Buffer
       self.replay_buffer = deque(maxlen=10000)
       self.batch_size = 32


       # 액션 출력 관련 상태 (초기값: 직진)
       self.prev_action = 1  # 1 = 직진
       self.same_count = 0
       self.last_output = None


   def train_online_step(self, state, reward, next_state, done):
       state_tensor = (
           torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
       )
       next_state_tensor = (
           torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
       )


       # Q값 계산
       self.policy_net.eval()
       q_values = self.policy_net(state_tensor)
       action = torch.argmax(q_values, dim=1).item()


       # --- 액션 출력 결정 (3개 액션: 0=좌회전, 1=직진, 2=우회전) ---
       if action == 0:
           output = "left"
       elif action == 1:
           output = "straight"
       elif action == 2:
           output = "right"
       else:
           output = "straight"  # 기본값


       print(f"Action: {action}, Output: {output}, Q-values: {q_values.cpu().detach().numpy()}")
       print(f"  → 실제 실행할 명령: {output}")
       self.prev_action = action


       # --- Replay Buffer에 저장 ---
       self.replay_buffer.append((state, action, reward, next_state, done))


       # --- 온라인 학습 ---
       if len(self.replay_buffer) >= self.batch_size:
           batch = random.sample(self.replay_buffer, self.batch_size)
           states = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(
               self.device
           )
           actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(
               self.device
           )
           rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(
               self.device
           )
           next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(
               self.device
           )
           dones = torch.tensor([b[4] for b in batch], dtype=torch.bool).to(
               self.device
           )


           current_q = (
               self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
           )
           with torch.no_grad():
               next_q = self.target_net(next_states).max(1)[0]
               target_q = rewards + self.gamma * (1 - dones.float()) * next_q


           all_q = self.policy_net(states)
           logsumexp_q = torch.logsumexp(all_q, dim=1)
           cql_penalty = (logsumexp_q - current_q).mean()


           loss = nn.MSELoss()(current_q, target_q) + self.cql_alpha * cql_penalty


           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()


           # 타겟 네트워크 업데이트 (확률적으로)
           if random.random() < 0.01:
               self.target_net.load_state_dict(self.policy_net.state_dict())


       return action, output


class ArduinoRCCarController:
   def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
       self.port = port
       self.baudrate = baudrate
       self.serial_conn = None
       self.connected = False


       # ENA, ENB 점퍼케이블 연결시 PWM 제어 불가능
       # 모터는 항상 최대 속도로 동작
       self.WHEEL_DIAMETER_CM = 6.5
       self.COUNTS_PER_REV = 20
       self.STOP_BRAKE_TIME_MS = 120


       self.command_queue = queue.Queue()
       self.response_queue = queue.Queue()


   def connect(self):
       try:
           self.serial_conn = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
           time.sleep(2)
           self.connected = True
           print(f"아두이노 연결 성공: {self.port}")
           self.start_communication_thread()
           return True
       except Exception as e:
           print(f"아두이노 연결 실패: {e}")
           return False


   def start_communication_thread(self):
       self.comm_thread = threading.Thread(target=self.communication_loop, daemon=True)
       self.comm_thread.start()


   def communication_loop(self):
       while self.connected:
           try:
               if not self.command_queue.empty():
                   cmd = self.command_queue.get_nowait()
                   self.serial_conn.write((cmd + '\n').encode('utf-8'))
               if self.serial_conn.in_waiting > 0:
                   resp = self.serial_conn.readline().decode('utf-8').strip()
                   if resp:
                       self.response_queue.put(resp)
           except Exception as e:
               print(f"통신 오류: {e}")
               break
           time.sleep(0.01)


   # === JSON 전송 공통 함수 ===
   def _send_json(self, type_, **kwargs):
       if not self.connected:
           return False
       try:
           payload = {'type': type_}
           payload.update(kwargs)
           self.command_queue.put(json.dumps(payload))
           return True
       except Exception as e:
           print(f"명령 전송 실패: {e}")
           return False


   # === 동작 명령 (ENA, ENB 점퍼케이블 연결 버전) ===
   def move_forward(self, distance_cm=None):
       """전진 (점퍼케이블 연결시 속도 제어 불가능)"""
       if distance_cm is not None:
           return self._send_json('move_forward', distance=distance_cm)
       return self._send_json('move_forward')


   def move_backward(self, distance_cm=None):
       """후진 (점퍼케이블 연결시 속도 제어 불가능)"""
       if distance_cm is not None:
           return self._send_json('move_backward', distance=distance_cm)
       return self._send_json('move_backward')


   def turn_left(self, angle_degrees=None):
       """좌회전"""
       if angle_degrees is not None:
           return self._send_json('turn_left', angle=angle_degrees)
       return self._send_json('turn_left')


   def turn_right(self, angle_degrees=None):
       """우회전"""
       if angle_degrees is not None:
           return self._send_json('turn_right', angle=angle_degrees)
       return self._send_json('turn_right')


   def stop(self):
       """정지"""
       return self._send_json('stop')


   def get_encoder_data(self):
       """엔코더 데이터 요청"""
       return self._send_json('get_encoders')


   def get_status(self):
       """상태 정보 요청"""
       return self._send_json('get_status')


class JetsonAutonomousController:
   def __init__(self, arduino_port='/dev/ttyUSB0'):
       """Jetson 자율주행 컨트롤러 (DQN Replay 버전)"""
       # 아두이노 RC카 컨트롤러
       self.rc_car = ArduinoRCCarController(port=arduino_port)
      
       # 새로운 구조의 LaneDetector
       self.lane_detector = LaneDetector()
      
       # 새로운 DQN Agent 생성 (3개 액션: 좌회전, 직진, 우회전)
       self.dqn_agent = DQNAgent(state_dim=4, action_dim=3, device=device)
      
       # See3CAM_CU27 카메라 설정 (GUI 없이 실행)
       self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
       self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
       self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
       self.cap.set(cv2.CAP_PROP_FPS, 30)
       self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
       self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
      
       # GUI 없이 실행을 위한 설정
       self.headless_mode = True
      
       # OpenCV 백엔드 설정 (GUI 완전 비활성화)
       cv2.setUseOptimized(True)
       cv2.setNumThreads(1)
      
       # 실제 설정값 확인
       actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       print(f"See3CAM_CU27 설정: {actual_width}x{actual_height}")


       # 실행 상태
       self.running = False
       self.autonomous_mode = False
      
       # 액션 매핑 (새로운 구조)
       self.action_map = {
           'left': self.action_left,      # 좌회전
           'straight': self.action_center,   # 직진
           'right': self.action_right     # 우회전
       }
      
       # 상태 추출 관련
       self.prev_clane = 1  # 이전 차선
      
   def action_left(self):
       """좌회전 액션"""
       print("🚗 실행: 좌회전")
       print("📤 아두이노에 전송: turn_left 명령")
       self.rc_car.turn_left()
      
   def action_center(self):
       """직진 액션"""
       print("🚗 실행: 직진")
       print("📤 아두이노에 전송: move_forward 명령")
       self.rc_car.move_forward()
      
   def action_right(self):
       """우회전 액션"""
       print("🚗 실행: 우회전")
       print("📤 아두이노에 전송: turn_right 명령")
       self.rc_car.turn_right()
      
   def get_state_replay_format(self, frame):
       """dqn_Replay.py 형식에 맞게 상태 추출"""
       height, width = frame.shape[:2]
      
       # 차선과 빨강 비율 계산
       lane_state, red_ratio = self.lane_detector.process_frame(frame)
      
       # over_line 계산 (바탕 픽셀 판단)
       bottom_center_pixel = frame[height - 1, width // 2]
       over_line = float(np.all(bottom_center_pixel > 240))
      
       # 이전 차선 숫자로 변환: left=0, right=1
       prev_lane_num = 0 if self.prev_clane == "left" else 1
      
       # 현재 차선을 저장
       self.prev_clane = lane_state
      
       print(f"Debug - lane_state: {lane_state}, red_ratio: {red_ratio:.4f}, over_line: {over_line}, prev_lane_num: {prev_lane_num}")
      
       # state = [red_ratio, over_line, prev_lane_num, move]
       # move는 DQNAgent에서 결정되므로 임시로 1 설정
       state = np.array([red_ratio, over_line, prev_lane_num, 1], dtype=np.float32)
       return state
      
   def calculate_reward(self, state, prev_state=None):
       """보상 계산"""
       reward = 3.0  # 기본 보상
       red_ratio = state[0]
      
       # 빨강 물체가 있을 때 추가 보상
       if red_ratio > 0.06:
           reward += 20
          
       return reward
      
   def autonomous_control_loop(self):
       """자율주행 제어 루프 (DQN Replay 버전)"""
       prev_state = None
      
       while self.running and self.autonomous_mode:
           try:
               ret, frame = self.cap.read()
               if ret:
                   # 상태 추출 (새로운 형식)
                   state = self.get_state_replay_format(frame)
                  
                   # 이전 상태가 있으면 보상 계산
                   if prev_state is not None:
                       reward = self.calculate_reward(state, prev_state)
                   else:
                       reward = 3.0  # 초기 보상
                  
                   # 종료 조건 확인
                   done = False
                   if state[1] > 0:  # over_line
                       print("종료 조건: 라인 넘어감")
                       done = True
                   if state[0] > 0.2:  # 빨강 물체 기준
                       print("종료 조건: 빨강 물체 접근")
                       done = True
                  
                   # DQN Agent로 액션 결정
                   if not done:
                       action, output = self.dqn_agent.train_online_step(state, reward, state, done)
                      
                       # 액션 실행
                       self.action_map[output]()
                   else:
                       self.rc_car.stop()
                       print("자율주행 종료")
                       break
                  
                   # 이전 상태 업데이트
                   prev_state = state.copy()
                  
               # GUI 없이 실행 - 디버그 정보만 출력
              
           except Exception as e:
               print(f"자율주행 제어 오류: {e}")
               self.rc_car.stop()
              
           time.sleep(0.1)  # Jetson Orin에 최적화된 간격
          
   def start_autonomous(self):
       """자율주행 시작"""
       if not self.rc_car.connected:
           print("아두이노가 연결되지 않았습니다.")
           return False
          
       print("DQN Replay 자율주행 모드 시작...")
       self.autonomous_mode = True
       self.running = True
      
       # 자율주행 스레드 시작
       self.autonomous_thread = threading.Thread(target=self.autonomous_control_loop)
       self.autonomous_thread.daemon = True
       self.autonomous_thread.start()
      
       return True
      
   def stop_autonomous(self):
       """자율주행 중지"""
       print("자율주행 모드 중지...")
       self.autonomous_mode = False
       self.running = False
       self.rc_car.stop()
      
   def manual_control(self):
       """수동 제어 모드"""
       print("수동 제어 모드 (키보드 입력)")
       print("w: 전진, s: 후진, a: 좌회전, d: 우회전, x: 정지, q: 종료")
      
       while self.running:
           key = input("명령 입력: ").lower()
          
           if key == 'w':
               self.rc_car.move_forward()
               print("전진")
           elif key == 's':
               self.rc_car.move_backward()
               print("후진")
           elif key == 'a':
               self.rc_car.turn_left()
               print("좌회전")
           elif key == 'd':
               self.rc_car.turn_right()
               print("우회전")
           elif key == 'x':
               self.rc_car.stop()
               print("정지")
           elif key == 'q':
               break
           elif key == 'auto':
               self.start_autonomous()
              
   def start(self):
       """시스템 시작"""
       print("Jetson 아두이노 RC카 제어 시스템 (DQN Replay 버전)")
       print("ENA, ENB 점퍼케이블 연결 - 모터는 항상 최대 속도로 동작")
      
       # 아두이노 연결
       if not self.rc_car.connect():
           print("아두이노 연결 실패. 수동 모드로 진행합니다.")
           return False
          
       self.running = True
      
       # 모드 선택
       print("모드를 선택하세요:")
       print("1. 수동 제어")
       print("2. DQN 자율주행 (Replay 버전)")
      
       choice = input("선택 (1/2): ")
      
       if choice == '1':
           self.manual_control()
       elif choice == '2':
           self.start_autonomous()
          
           # 메인 스레드에서 키보드 입력 대기
           try:
               while self.running:
                   key = input("Press 'q' to quit, 's' to stop: ")
                   if key == 'q':
                       break
                   elif key == 's':
                       self.stop_autonomous()
                      
           except KeyboardInterrupt:
               print("중단됨")
              
       return True
      
   def cleanup(self):
       """정리"""
       self.running = False
       self.autonomous_mode = False
       self.rc_car.stop()
       if self.rc_car.serial_conn:
           self.rc_car.serial_conn.close()
       if self.cap:
           self.cap.release()
       # Jetson Orin 메모리 정리
       if torch.cuda.is_available():
           torch.cuda.empty_cache()


def optimize_jetson_performance():
   """Jetson Orin 성능 최적화"""
   print("Jetson Orin 성능 최적화 적용 중...")
  
   # CUDA 설정 최적화
   if torch.cuda.is_available():
       torch.backends.cudnn.benchmark = True
       torch.backends.cudnn.deterministic = False
       torch.cuda.empty_cache()
      
       os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
       os.environ["CUDA_CACHE_DISABLE"] = "0"
      
       print("CUDA 최적화 완료")
  
   # OpenCV 최적화
   cv2.setNumThreads(4)
   print("OpenCV 최적화 완료")


def monitor_jetson_performance():
   """Jetson Orin 성능 모니터링"""
   print("=== Jetson Orin 성능 모니터링 ===")
  
   # CPU 정보
   cpu_percent = psutil.cpu_percent(interval=1)
   print(f"CPU 사용률: {cpu_percent}%")
  
   # 메모리 정보
   memory = psutil.virtual_memory()
   print(f"시스템 메모리 사용률: {memory.percent}%")
   print(f"사용 가능한 메모리: {memory.available / 1024**3:.1f} GB")
  
   # GPU 정보
   if torch.cuda.is_available():
       print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
       print(f"GPU 메모리 예약량: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")


def main():
   """메인 함수"""
   print("Jetson 아두이노 RC카 제어 시스템 (DQN Replay 버전)")
   print("ENA, ENB 점퍼케이블 연결 버전")
  
   # 성능 최적화
   optimize_jetson_performance()
   monitor_jetson_performance()
  
   # 아두이노 포트 확인
   import glob
   ports = glob.glob('/dev/tty[A-Za-z]*')
   print("사용 가능한 포트:")
   for port in ports:
       print(f"  {port}")
      
   # 포트 선택
   arduino_port = input("아두이노 포트를 입력하세요 (예: /dev/ttyUSB0): ")
   if not arduino_port:
       arduino_port = '/dev/ttyACM0'
      
   # 컨트롤러 초기화
   controller = JetsonAutonomousController(arduino_port)
  
   try:
       # 시스템 시작
       controller.start()
   except Exception as e:
       print(f"오류 발생: {e}")
   finally:
       controller.cleanup()


if __name__ == "__main__":
   main()




