# -*- coding: utf-8 -*-
"""
Jetson용 DQN Lane-keeping / Obstacle Avoidance
순차 학습 (EWC 없음) + 데이터 캐싱 + 10회 반복 실험 버전
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import matplotlib.pyplot as plt
import time

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# -----------------------------
# LaneDetector
# -----------------------------
class LaneDetector:
    def __init__(self):
        # 빨강 HSV 범위
        self.lower_red1 = np.array([0,70,50])
        self.upper_red1 = np.array([10,255,255])
        self.lower_red2 = np.array([170,70,50])
        self.upper_red2 = np.array([180,255,255])

    def process_frame(self, frame):
      height, width = frame.shape[:2]
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      # 1. 노란색 마스크 생성
      mask_yellow = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))

      # 2. 흰색 마스크 생성 (시각화용)
      lower_white = np.array([0, 0, 180])
      upper_white = np.array([255, 30, 255])
      mask_white = cv2.inRange(hsv, lower_white, upper_white)

      # 3. 노란색과 흰색 마스크를 합침 (시각화용)
      lane_mask = cv2.bitwise_or(mask_yellow, mask_white)

      # 차선 방향 판단은 '노란색' 마스크 기준
      left_half = mask_yellow[:, :width // 2]
      right_half = mask_yellow[:, width // 2:]
      left_count = cv2.countNonZero(left_half)
      right_count = cv2.countNonZero(right_half)

      if left_count > right_count and left_count > 0:
          lane_state = 1 #"right"
      elif right_count > left_count and right_count > 0:
          lane_state = 0 #"left"
      else:
          lane_state = self.prev_lane

      self.prev_lane = lane_state

      # 빨간색 감지
      mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
      mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
      mask_red = cv2.bitwise_or(mask_red1, mask_red2)
      red_pixels = cv2.countNonZero(mask_red)
      red_ratio = red_pixels / float(height * width)

      # --- ### 1. 노란색 차선의 무게 중심 계산 ### ---
      yellow_center_x = -1 # 기본값은 -1
      M_yellow = cv2.moments(mask_yellow)
      if M_yellow["m00"] > 0:
          # 노란색 픽셀이 존재하면, 무게 중심 x좌표를 계산하여 기준선으로 사용
          yellow_center_x = int(M_yellow["m10"] / M_yellow["m00"])

      # 빨간색 객체 중심 계산
      red_center_x = -1
      red_center_y = -1
      if red_pixels > 0:
          M_red = cv2.moments(mask_red)
          if M_red["m00"] > 0:
              red_center_x = int(M_red["m10"] / M_red["m00"])
              red_center_y = int(M_red["m01"] / M_red["m00"])

      # --- ### 2. 노란색 중심을 기준으로 빨간색 위치 판단 ### ---
      red_side_relative_to_yellow = -1 # 기본값
      if red_pixels/(width*height) > 0.001 and yellow_center_x!=-1: # 노란색 차선이 있고 빨간색이 감지되었을 때만 판단
          if red_center_x < yellow_center_x:
              red_side_relative_to_yellow = 0 #"left"
          else:
              red_side_relative_to_yellow = 1 #"right"


      # 반환값에 새로 계산한 위치 정보 추가
      return lane_state, red_ratio, red_side_relative_to_yellow

# -----------------------------
# OfflineDataCollector
# -----------------------------
class OfflineDataCollector:
    def __init__(self, lane_detector):
        self.lane_detector = lane_detector
        self.before_act = None

    def _get_state(self, frame):
        lane_state, red_ratio, red_side = self.lane_detector.process_frame(frame)

        height, width, _ = frame.shape
        bottom_center_pixel = frame[height - 1, width // 2]
        over_line = float(np.all(bottom_center_pixel > 240))

        state = np.array([
            lane_state,
            over_line,
            red_side,
            red_ratio
        ], dtype=np.float32)

        return state

    def _calculate_reward(self, state, next_state=None, done=False):
      lane_state = int(state[0])    # 현재 레인
      over = float(state[1])
      red_side = int(state[2])    # 0:left, 1:right
      red_ratio = float(state[3])

      if next_state is None :
        return 0, True

      next_lane_state = int(next_state[0])
      next_over = float(next_state[1])
      next_red_side = int(next_state[2])
      next_red_ratio = float(next_state[3])

      # 기본 보상
      r=3

      # 🔹 빨간색과 레인 관계 강화
      if red_ratio > 0.06:
          if red_side == lane_state:
              # 빨간색이 내 레인에 있을 때 회피(move) 방향 맞으면 보상
              if (next_lane_state != lane_state):
                  r += 0.3  # 회피 성공 보상
              else:
                  r -= 0.5 * (red_ratio - 0.06) / 0.14  # 회피 실패 패널티

      # 🔹 차선 벗어남 패널티
      if over > 0.5:
          r -= 1.0
          done = True

      # 🔹 다음 상태 고려 (optional)
      if next_red_ratio < red_ratio :
          r += 0.01

      r = float(np.clip(r, -1.0, 1.0))
      return r, done


    def collect_from_frames(self, frames):
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
        for idx, frame in enumerate(frames):
            next_idx = idx + 1
            done = False
            state = self._get_state(frame)
            action = int(state[0])

            if next_idx < len(frames):
                next_state = self._get_state(frames[next_idx])
                reward, done = self._calculate_reward(state, next_state, done=done)
            else:
                reward = 0
                done = True

            if state[1] > 0.5 or state[3] > 0.20:
                done = True

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_list.append(done)

        return state_list, action_list, reward_list, next_state_list, done_list
# -----------------------------
# DQN 모델
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_dim)  # 각 action별 Q값 출력

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)  # [batch, action_dim]

# -----------------------------
# DQNAgent (EWC 없음)
# -----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim=2, device='cpu',
                 cql_alpha=10, lr=1e-5, gamma=0.1, batch_size=32,
                 target_update_freq=5):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=20000)

    def train_offline(self, state_list, action_list, reward_list, next_state_list, done_list, epochs=50, beta=5, keep_buffer=False):
        if not keep_buffer:
            self.replay_buffer.clear()
        
        for s, a, r, ns, d in zip(state_list, action_list, reward_list, next_state_list, done_list):
            self.replay_buffer.append((s, a, r, ns, d))

        if not self.replay_buffer:
            print("Replay buffer is empty. Skipping training.")
            return

        step_count = 0
        for epoch in range(epochs):
            batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
            
            states = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(self.device)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([b[4] for b in batch], dtype=torch.bool).to(self.device)

            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                target_q = rewards + self.gamma * (1 - dones.float()) * next_q_values

            all_q = self.policy_net(states)
            current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)
            logsumexp_q = torch.logsumexp(all_q, dim=1)
            mean_q = all_q.mean(dim=1)
            cql_penalty = (logsumexp_q - current_q).mean()
            entropy_penalty = all_q.var(dim=1).mean()
            td_loss = F.mse_loss(current_q, target_q)
            
            loss = td_loss + self.cql_alpha * cql_penalty + beta * entropy_penalty

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
            self.optimizer.step()

            step_count += 1
            if step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if epoch % 10 == 0:
                 print(f"[Offline] Epoch {epoch:03d}, Loss: {loss.item():.4f}")

# -----------------------------
# 헬퍼 함수 정의
# -----------------------------
def load_video_frames(video_path, max_frames=None):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    cap.release()
    return frames

def load_and_process_data(video_files):
    lane_detector = LaneDetector()
    collector = OfflineDataCollector(lane_detector)
    
    state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []

    for vf in video_files:
        print(f"Processing video: {vf}")
        frames = load_video_frames(vf)
        if not frames:
            continue
        s, a, r, ns, d = collector.collect_from_frames(frames)
        state_list.extend(s)
        action_list.extend(a)
        reward_list.extend(r)
        next_state_list.extend(ns)
        done_list.extend(d)
        
    print(f"Generated {len(state_list)} transitions.")
    return state_list, action_list, reward_list, next_state_list, done_list


def evaluate_agent(agent, eval_video_file, run_index=0):
    print(f"\n--- [Run {run_index+1}] Evaluation ---")
    
    eval_frames = load_video_frames(eval_video_file)
    # Ensure there are at least two frames to form a state-next_state pair
    if not eval_frames or len(eval_frames) < 2:
        print("Could not load evaluation video or not enough frames to evaluate.")
        return 0.0, 0.0

    collector_eval = OfflineDataCollector(LaneDetector())
    total_reward = 0.0
    agent_actions = []
    offline_actions = []

    # Iterate until the second-to-last frame to have a 'next_state'
    for i in range(len(eval_frames) - 1):
        frame = eval_frames[i]
        next_frame = eval_frames[i+1]
        
        # 1. Correctly get the full state array for the current and next frame
        state = collector_eval._get_state(frame)
        next_state = collector_eval._get_state(next_frame)

        # 2. The "correct" offline action is the lane state from the CURRENT frame
        #    This is what the agent should ideally predict.
        correct_action = int(state[0])
        offline_actions.append(correct_action)

        # 3. Calculate the reward for the transition from state to next_state
        reward, _ = collector_eval._calculate_reward(state, next_state)
        total_reward += reward

        # 4. Get the agent's predicted action for the current state
        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = agent.policy_net(s_tensor)
            action = q_values.argmax().item()
            agent_actions.append(action)

    # --- Calculate Final Metrics ---
    offline_actions = np.array(offline_actions)
    agent_actions = np.array(agent_actions)
    
    matches = np.sum(offline_actions == agent_actions)
    accuracy = matches / len(offline_actions) if len(offline_actions) > 0 else 0

    print(f"=== [Run {run_index+1}] Result ===")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%), Total Reward: {total_reward:.4f}")

    return accuracy, total_reward

# ==================================
# 메인 실행 로직
# ==================================
start_time = time.time()
DATA_FILE_T1 = "task1_data.npz"
DATA_FILE_T4 = "task4_data.npz"
DATA_FILE_T5 = "task5_data.npz"
DATA_FILE_T6 = "task6_data.npz"

# ❗️ 경로를 실제 환경에 맞게 수정해주세요
video_files_t1 = ["/home/jieun/seohee/1_1.mov", "/home/jieun/seohee/1_2.mov","/home/jieun/seohee/1_3.mov", "/home/jieun/seohee/1_4.mov"]
video_files_t4 = ["/home/jieun/seohee/4_1.mov", "/home/jieun/seohee/4_2.mov","/home/jieun/seohee/4_3.mov", "/home/jieun/seohee/4_4.mov"]
video_files_t5 = ["/home/jieun/seohee/5_1.mov", "/home/jieun/seohee/5_2.mov","/home/jieun/seohee/5_3.mov", "/home/jieun/seohee/5_4.mov"]
video_files_t6 = ["/home/jieun/seohee/6_1.mov", "/home/jieun/seohee/6_2.mov","/home/jieun/seohee/6_3.mov", "/home/jieun/seohee/6_4.mov"]
eval_video_file = "/home/jieun/seohee/3_2.mov"
# Task 1 데이터 준비
if os.path.exists(DATA_FILE_T1):
    print(f"Loading data for Task 1 from '{DATA_FILE_T1}'")
    data = np.load(DATA_FILE_T1, allow_pickle=True)
    s1, a1, r1, ns1, d1 = data['s'], data['a'], data['r'], data['ns'], data['d']
else:
    print(f"Creating and saving data for Task 1 to '{DATA_FILE_T1}'")
    s1, a1, r1, ns1, d1 = load_and_process_data(video_files_t1)
    np.savez(DATA_FILE_T1, s=s1, a=a1, r=r1, ns=ns1, d=d1)

# Task 4 데이터 준비
if os.path.exists(DATA_FILE_T4):
    print(f"Loading data for Task 4 from '{DATA_FILE_T4}'")
    data = np.load(DATA_FILE_T4, allow_pickle=True)
    s4, a4, r4, ns4, d4 = data['s'], data['a'], data['r'], data['ns'], data['d']
else:
    print(f"Creating and saving data for Task 4 to '{DATA_FILE_T4}'")
    s4, a4, r4, ns4, d4 = load_and_process_data(video_files_t4)
    np.savez(DATA_FILE_T4, s=s4, a=a4, r=r4, ns=ns4, d=d4)
    
# Task 5 데이터 준비
if os.path.exists(DATA_FILE_T5):
    print(f"Loading data for Task 5 from '{DATA_FILE_T5}'")
    data = np.load(DATA_FILE_T5, allow_pickle=True)
    s5, a5, r5, ns5, d5 = data['s'], data['a'], data['r'], data['ns'], data['d']
else:
    print(f"Creating and saving data for Task 5 to '{DATA_FILE_T5}'")
    s5, a5, r5, ns5, d5 = load_and_process_data(video_files_t5)
    np.savez(DATA_FILE_T5, s=s5, a=a5, r=r5, ns=ns5, d=d5)

# Task 6 데이터 준비
if os.path.exists(DATA_FILE_T6):
    print(f"Loading data for Task 6 from '{DATA_FILE_T6}'")
    data = np.load(DATA_FILE_T6, allow_pickle=True)
    s6, a6, r6, ns6, d6 = data['s'], data['a'], data['r'], data['ns'], data['d']
else:
    print(f"Creating and saving data for Task 6 to '{DATA_FILE_T6}'")
    s6, a6, r6, ns6, d6 = load_and_process_data(video_files_t6)
    np.savez(DATA_FILE_T6, s=s6, a=a6, r=r6, ns=ns6, d=d6)

data_prep_time = time.time() - start_time
print(f"Data preparation finished. (Took {data_prep_time:.2f} seconds)")

# --- 10회 실험 반복 ---
num_runs = 10
all_accuracies = []
all_rewards = []

for i in range(num_runs):
    print(f"\n{'='*20} Experiment {i+1}/{num_runs} Start {'='*20}")
            
    agent = DQNAgent(state_dim=4, action_dim=2, device=device, cql_alpha=30,lr=1e-5,gamma=0.99, batch_size=32, target_update_freq=10)

    # Step 1: Task 1 학습
    print("\n--- Training on Task 1 ---")
    agent.train_offline(s1, a1, r1, ns1, d1, epochs=25, beta=5)

    # Step 2: Task 4 학습
    print("\n--- Training on Task 4 ---")
    agent.train_offline(s4, a4, r4, ns4, d4, epochs=25, beta=5)
    
    # Step 3: Task 5 추가 학습
    print("\n--- Training on Task 5 (Continual) ---")
    agent.train_offline(s5, a5, r5, ns5, d5, epochs=25, beta=5, keep_buffer=True)

    # Step 4: Task 6 학습
    print("\n--- Training on Task 6 ---")
    agent.train_offline(s6, a6, r6, ns6, d6, epochs=25, beta=5)

                                                                
    # Step 3: 최종 평가
    accuracy, total_reward = evaluate_agent(agent, eval_video_file, run_index=i)
    all_accuracies.append(accuracy)
    all_rewards.append(total_reward)

# --- 최종 결과 요약 ---
total_run_time = time.time() - start_time
print(f"\n\n{'='*20} Final Summary of {num_runs} Experiments {'='*20}")
print(f"Total time: {total_run_time:.2f} seconds")

if all_accuracies:
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    print(f"Average Accuracy: {mean_accuracy:.3f} (Std Dev: {std_accuracy:.3f})")
    print(f"Average Total Reward: {mean_reward:.3f} (Std Dev: {std_reward:.3f})")
    print("\nIndividual Accuracies:", [f"{acc:.3f}" for acc in all_accuracies])


