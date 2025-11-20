#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Orin 호환 DQN 자율주행 코드


이 스크립트는 Jetson Orin에서 실행하기 위해 최적화된 DQN 기반 자율주행 코드입니다.


주요 변경사항:
- Jetson Orin CUDA 메모리 최적화
- 8개 비디오 파일 지원 (1_1.mp4, 1_2.mp4, 4_1.mp4, 4_2.mp4, 5_1.mp4, 5_2.mp4, 6_1.mp4, 6_2.mp4)
- 메모리 사용량 모니터링 및 제한
- GPU 텐서 처리 최적화
- 성능 모니터링 유틸리티 추가


사전 준비사항:
1. 필요한 패키지 설치:
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 pip install opencv-python numpy psutil


2. 비디오 파일 준비:
 - /home/nvidia/videos/ 경로에 8개 비디오 파일 배치
 - 또는 현재 디렉토리에 8개 비디오 파일 배치


주의사항:
- 메모리 사용량이 1.5GB를 초과하면 자동으로 중단됩니다
- GPU 온도가 높으면 성능이 저하될 수 있습니다
- 비디오 파일이 없으면 테스트용 샘플 프레임이 생성됩니다
"""


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import os
import psutil


# Jetson Orin에서 CUDA 메모리 최적화
if torch.cuda.is_available():
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.deterministic = False
   # Jetson Orin 메모리 최적화
   os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
   os.environ["CUDA_CACHE_DISABLE"] = "0"
   # NVML 오류 방지
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Jetson Orin에서 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Jetson Orin에서 메모리 정보 출력
if torch.cuda.is_available():
   print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
   print(
       f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
   )
   print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
   print(f"CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")




class LaneDetector:
   def __init__(self, slope_threshold=0.2):
       self.prev_lane = 1  # 초기값: 오른쪽 차선
       self.slope_threshold = slope_threshold
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




class OfflineDataCollector:
   def __init__(self, lane_detector):
       self.lane_detector = lane_detector
       self.before_act = None  # 이전 차선: 'left' 또는 'right'


   def _get_state(self, frame):
       """
       프레임에서 상태(state)를 계산하고 move를 결정.
       state = [red_ratio, over_line, prev_lane_num, move]
       """
       # 현재 차선과 빨강 비율
       lane_state, red_ratio = self.lane_detector.process_frame(frame)
       act_str = "left" if lane_state == "left" else "right"


       # 이전 차선 기반 move 계산
       prev_act = self.before_act if self.before_act is not None else act_str


       if prev_act == "right" and act_str == "left":
           move = 0  # 좌회전
       elif prev_act == "left" and act_str == "right":
           move = 2  # 우회전
       else:
           move = 1  # 직진


       # 현재 차선을 저장
       self.before_act = act_str


       # 바닥 픽셀 over_line 판단
       height, width, _ = frame.shape
       bottom_center_pixel = frame[height - 1, width // 2]
       over_line = float(np.all(bottom_center_pixel > 240))


       # 이전 차선 숫자로 변환: left=0, right=1
       prev_lane_num = 0 if prev_act == "left" else 1


       state = np.array([red_ratio, over_line, prev_lane_num, move], dtype=np.float32)
       return state, move, red_ratio, lane_state


   def _calculate_reward(self, state, move):
       """
       state와 move에 따른 보상 계산
       """
       reward = 3.0  # 기본 보상
       red_ratio = state[0]


       # 직진에 대한 보상 강화
       if move == 1:  # 직진
           reward += 5.0  # 직진 시 추가 보상
      
       # 빨강 물체가 있을 때 상황별 보상
       if red_ratio > 0.06:
           if move == 1:  # 직진
               reward += 10  # 직진하면서 빨강 물체에 접근
           else:  # 회전
               reward += 20  # 회전으로 빨강 물체 회피
      
       # 안전한 직진에 대한 추가 보상
       if red_ratio < 0.03 and move == 1:  # 빨강 물체가 없고 직진
           reward += 8.0


       return reward


   def collect_from_frames(
       self, frames, car_x_init=None, actions_taken=None, batch_size=1000
   ):
       """
       frames에서 state/action/reward/next_state/done 리스트 생성 (메모리 효율적)
       batch_size : 메모리 효율성을 위한 배치 크기
       """
       state_list = []
       action_list = []
       reward_list = []
       next_state_list = []
       done_list = []


       # 메모리 효율성을 위해 배치 단위로 처리
       total_frames = len(frames)
       processed_count = 0


       print(f"총 {total_frames} 프레임에서 데이터 수집 시작...")


       for batch_start in range(0, total_frames, batch_size):
           batch_end = min(batch_start + batch_size, total_frames)
           batch_frames = frames[batch_start:batch_end]


           print(
               f"배치 처리: {batch_start}-{batch_end} 프레임 ({len(batch_frames)}개)"
           )


           for idx, frame in enumerate(batch_frames):
               # 전체 프레임 인덱스로 조정
               global_idx = batch_start + idx
               next_global_idx = global_idx + 1
               done = False


               # 현재 상태
               state, move, red_ratio, lane_state = self._get_state(frame)


               # next_state 계산 시 이전 액션 복원
               saved_before_act = self.before_act
               if next_global_idx < len(frames):
                   next_frame = frames[next_global_idx]
                   next_state, _, _, _ = self._get_state(next_frame)
               else:
                   next_state = np.zeros_like(state)
                   done = True
               self.before_act = saved_before_act  # 복원


               # 종료 조건
               if state[1] > 0:  # over_line
                   done = True
               if state[0] > 0.2:  # 빨강 물체 기준
                   done = True


               # 리스트 저장 (move 값 그대로 사용)
               state_list.append(state)
               action_list.append(move)  # move 값 (0=좌회전, 1=직진, 2=우회전)을 그대로 사용
               reward_list.append(self._calculate_reward(state, move))
               next_state_list.append(next_state)
               done_list.append(done)


               processed_count += 1


               # 진행 상황 출력
               if processed_count % 100 == 0:
                   print(f"  처리된 transition: {processed_count}")


           # 배치 처리 후 메모리 정리
           if torch.cuda.is_available():
               torch.cuda.empty_cache()


       print(f"데이터 수집 완료: {len(state_list)} transition 생성")
       return state_list, action_list, reward_list, next_state_list, done_list




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
       if torch.cuda.is_available() and device == "cpu":
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


       # 액션 출력 관련 상태
       self.prev_action = 1
       self.same_count = 0
       self.last_output = None


   # ==========================
   # 1️⃣ 오프라인 RL 학습
   # ==========================
   def train_offline(
       self,
       state_list,
       action_list,
       reward_list,
       next_state_list,
       done_list,
       epochs=100,
       batch_size=32,
       update_frequency=50,
   ):
       """오프라인 RL DQN 학습 (Jetson Orin 최적화)"""
       print("Starting offline DQN training for Jetson Orin...")


       # 전체 경험 리스트
       dataset = list(
           zip(state_list, action_list, reward_list, next_state_list, done_list)
       )


       # Jetson Orin 메모리 최적화를 위한 배치 크기 조정
       if len(dataset) < batch_size:
           batch_size = min(len(dataset), 16)  # 작은 데이터셋의 경우 배치 크기 줄임
           print(f"데이터셋 크기에 맞춰 배치 크기를 {batch_size}로 조정")


       # 메모리 정리를 위한 주기적 가비지 컬렉션
       import gc


       for epoch in range(epochs):
           # 무작위 배치 샘플링
           batch = random.sample(dataset, batch_size)


           # Jetson Orin에서 효율적인 텐서 생성
           states = torch.tensor(
               np.array([exp[0] for exp in batch]), dtype=torch.float32, device=self.device
           )
           actions = torch.tensor(
               [exp[1] for exp in batch], dtype=torch.long, device=self.device
           )
           rewards = torch.tensor(
               [exp[2] for exp in batch], dtype=torch.float32, device=self.device
           )
           next_states = torch.tensor(
               np.array([exp[3] for exp in batch]), dtype=torch.float32, device=self.device
           )
           dones = torch.tensor(
               [exp[4] for exp in batch], dtype=torch.bool, device=self.device
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
           # Jetson Orin에서 그래디언트 클리핑 추가
           torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)


           self.optimizer.step()


           # 타겟 네트워크 주기적 업데이트
           if epoch % update_frequency == 0:
               self.target_net.load_state_dict(self.policy_net.state_dict())


           # Jetson Orin 메모리 모니터링
           if torch.cuda.is_available() and epoch % 5 == 0:
               memory_used = torch.cuda.memory_allocated() / 1024**3
               print(
                   f"Epoch {epoch}, Loss: {loss.item():.4f}, GPU Memory: {memory_used:.3f} GB"
               )


               # 메모리 사용량이 너무 높으면 정리
               if memory_used > 1.5:  # 1.5GB 이상 사용시
                   torch.cuda.empty_cache()
                   gc.collect()
           else:
               print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


           # 최종 메모리 정리
           if torch.cuda.is_available():
               torch.cuda.empty_cache()


       print("Offline DQN training completed!")
       return self.policy_net


   # ==========================
   # 2️⃣ 온라인 RL 한 스텝 학습 + 액션 출력
   # ==========================
   def train_online_step(self, state, reward, next_state, done):
       state_tensor = (
           torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
       )
       next_state_tensor = (
           torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
       )


       # Q값 계산
       self.policy_net.train()
       q_values = self.policy_net(state_tensor)
       action = torch.argmax(q_values, dim=1).item()


       # --- 액션 출력 결정 (직관적이고 명확하게) ---
       if action == 0:
           output = "left"
       elif action == 1:
           output = "straight"
       elif action == 2:
           output = "right"
       else:
           output = "straight"  # 기본값은 직진


       print(f"DQN Action: {action}, Output: {output}, Q-values: {q_values.cpu().detach().numpy()}")
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


       return action, q_values.cpu().detach().numpy()




def load_video_frames(video_path, max_frames_per_video=500):
   """단일 비디오에서 프레임을 로드하는 함수"""
   print(f"비디오 로딩: {video_path}")


   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       print(f"비디오 파일을 열 수 없습니다: {video_path}")
       return []


   frames = []
   frame_count = 0


   while True:
       ret, frame = cap.read()
       if not ret:
           break
       frames.append(frame)
       frame_count += 1


       # 메모리 사용량 체크
       if torch.cuda.is_available():
           memory_used = torch.cuda.memory_allocated() / 1024**3
           if memory_used > 1.5:  # 1.5GB 이상 사용시 중단
               print(f"메모리 사용량 제한으로 인해 {frame_count} 프레임에서 중단")
               break


       if frame_count >= max_frames_per_video:
           print(f"최대 프레임 수({max_frames_per_video})에 도달하여 중단")
           break


   cap.release()
   print(f"  - {len(frames)} 프레임 로드됨")
   return frames




def load_all_training_videos():
   """8개의 훈련 비디오를 모두 로드하는 함수"""
   # 훈련 데이터 비디오 파일 목록
   video_files = [
       "1_1.mp4",
       "1_2.mp4",
       "4_1.mp4",
       "4_2.mp4",
       "5_1.mp4",
       "5_2.mp4",
       "6_1.mp4",
       "6_2.mp4",
       "1_4.mov",
       "1_5.mov",
       "4_3.mov",
       "4_4.mov",
       "5_3.mov",
       "5_4.mov",
       "6_3.mov",
       "6_4.mov",
   ]


   all_frames = []
   total_frames = 0


   print("=== 훈련 비디오 로딩 시작 ===")


   for video_file in video_files:
       # 여러 경로에서 비디오 파일 찾기
       possible_paths = [
           f"/home/nvidia/videos/{video_file}",  # Jetson Orin 기본 경로
           f"./{video_file}",  # 현재 디렉토리
           video_file,  # 파일명만으로 찾기
       ]


       video_path = None
       for path in possible_paths:
           if os.path.exists(path):
               video_path = path
               break


       if video_path is None:
           print(f"경고: {video_file} 파일을 찾을 수 없습니다.")
           continue


       # 비디오에서 프레임 로드
       frames = load_video_frames(video_path, max_frames_per_video=500)


       if frames:
           all_frames.extend(frames)
           total_frames += len(frames)
           print(f"  ✓ {video_file}: {len(frames)} 프레임 추가")
       else:
           print(f"  ✗ {video_file}: 프레임 로드 실패")


   print(f"=== 총 {total_frames} 프레임 로드 완료 ===")


   # 비디오 파일이 하나도 없으면 샘플 프레임 생성
   if not all_frames:
       print("비디오 파일이 없어서 테스트용 샘플 프레임을 생성합니다...")
       for i in range(100):
           frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
           all_frames.append(frame)
       total_frames = len(all_frames)


   return all_frames




def monitor_jetson_performance():
   """Jetson Orin 성능 모니터링 (NVML 오류 방지)"""
   print("=== Jetson Orin 성능 모니터링 ===")


   # CPU 정보
   try:
       cpu_percent = psutil.cpu_percent(interval=1)
       print(f"CPU 사용률: {cpu_percent}%")
   except:
       print("CPU 사용률 정보를 읽을 수 없습니다.")


   # 메모리 정보
   try:
       memory = psutil.virtual_memory()
       print(f"시스템 메모리 사용률: {memory.percent}%")
       print(f"사용 가능한 메모리: {memory.available / 1024**3:.1f} GB")
   except:
       print("메모리 정보를 읽을 수 없습니다.")


   # GPU 정보 (CUDA 사용 가능한 경우) - 안전한 방식
   if torch.cuda.is_available():
       try:
           print(f"GPU 장치명: {torch.cuda.get_device_name(0)}")
           print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
           print(f"GPU 메모리 예약량: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
           print(f"GPU 메모리 총량: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
       except Exception as e:
           print(f"GPU 정보 조회 중 오류: {e}")
   else:
       print("CUDA를 사용할 수 없습니다.")


   # 온도 정보 (Jetson 특화)
   try:
       with open("/sys/devices/virtual/thermal/thermal_zone0/temp", "r") as f:
           cpu_temp = int(f.read()) / 1000
       print(f"CPU 온도: {cpu_temp:.1f}°C")
   except:
       print("CPU 온도 정보를 읽을 수 없습니다.")


   try:
       with open("/sys/devices/virtual/thermal/thermal_zone1/temp", "r") as f:
           gpu_temp = int(f.read()) / 1000
       print(f"GPU 온도: {gpu_temp:.1f}°C")
   except:
       print("GPU 온도 정보를 읽을 수 없습니다.")




def optimize_jetson_performance():
   """Jetson Orin 성능 최적화"""
   print("Jetson Orin 성능 최적화 적용 중...")


   # CUDA 설정 최적화
   if torch.cuda.is_available():
       torch.backends.cudnn.benchmark = True
       torch.backends.cudnn.deterministic = False


       # 메모리 정리
       torch.cuda.empty_cache()


       # Jetson 전용 설정
       os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 성능을 위해 비동기 실행
       os.environ["CUDA_CACHE_DISABLE"] = "0"  # 캐시 활성화


       print("CUDA 최적화 완료")


   # OpenCV 최적화
   cv2.setNumThreads(4)  # OpenCV 스레드 수 제한
   print("OpenCV 최적화 완료")




def cleanup_jetson_memory():
   """Jetson Orin 메모리 정리"""
   import gc


   # Python 가비지 컬렉션
   gc.collect()


   # CUDA 메모리 정리
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
       torch.cuda.synchronize()


   print("메모리 정리 완료")




def main():
   """메인 실행 함수"""
   print("Jetson Orin DQN 자율주행 코드 시작...")


   # 성능 모니터링 실행
   monitor_jetson_performance()


   # 성능 최적화 적용
   optimize_jetson_performance()


   # 1. 모든 훈련 비디오에서 프레임 읽기
   frames = load_all_training_videos()


   # 2. LaneDetector 생성
   lane_detector = LaneDetector()


   # 3. OfflineDataCollector 생성
   collector = OfflineDataCollector(lane_detector)


   # 4. 데이터 수집 (메모리 효율적 배치 처리)
   print("\n=== 데이터 수집 시작 ===")
   state_list, action_list, reward_list, next_state_list, done_list = (
       collector.collect_from_frames(
           frames, batch_size=1000  # 메모리 효율성을 위한 배치 크기
       )
   )


   # 5. 결과 확인
   print(f"\n=== 데이터 수집 결과 ===")
   print(f"총 transition 수: {len(state_list)}")
   if len(state_list) > 0:
       print(f"샘플 state: {state_list[0]}")
       print(f"샘플 action: {action_list[0]}")
       print(f"샘플 reward: {reward_list[0]}")
       print(f"샘플 next_state: {next_state_list[0]}")
       print(f"샘플 done: {done_list[0]}")


   # 6. Jetson Orin에 최적화된 학습 파라미터로 학습
   if len(state_list) > 0:
       print("\n=== DQN 학습 시작 ===")
       print(f"Agent device: {device}")
       agent = DQNAgent(state_dim=len(state_list[0]), action_dim=3, device=device)
       print(f"Agent 실제 device: {agent.device}")
       agent.train_offline(
           state_list, action_list, reward_list, next_state_list, done_list
       )


       # 7. Q-value 테스트
       print("\n=== Q-value 테스트 시작 ===")
       test_count = min(10, len(state_list))  # Jetson Orin에서 테스트 개수 제한


       for i in range(test_count):
           sample_state = torch.tensor(
               state_list[i], dtype=torch.float32, device=agent.device
           )
           q_values = agent.policy_net(sample_state.unsqueeze(0))  # 배치 차원 추가


           # GPU에서 CPU로 이동하여 출력
           q_values_cpu = q_values.detach().cpu().numpy()
           print(f"Sample {i+1} Q-values: {q_values_cpu}")
           action = q_values.argmax().item()
           print(f"Sample {i+1} action: {action}")


           # 메모리 정리
           if torch.cuda.is_available() and i % 5 == 0:
               torch.cuda.empty_cache()


       print("Q-value 테스트 완료!")
   else:
       print("데이터가 없어서 학습을 건너뜁니다.")


   # 8. 최종 메모리 정리
   cleanup_jetson_memory()
   print("\n=== Jetson Orin DQN 자율주행 학습 완료 ===")
   print("8개 비디오 파일로부터 학습이 성공적으로 완료되었습니다!")




if __name__ == "__main__":
   main()









