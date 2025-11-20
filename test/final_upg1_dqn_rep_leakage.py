# -*- coding: utf-8 -*-
"""
Jetson용 DQN Lane-keeping / Obstacle Avoidance
CaSSLe (Contrastive Self-Supervised Learning) 기반 Continual Learning
10회 반복 실험 버전
알고리즘 버전1: td_loss의 값에 따라 augmentation 레벨 조절(td_loss가 클수록 augmentation 레벨 낮춤)
+ Continual Learning 지표(FWT, BWT, IM) 추가
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import os
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
        self.prev_lane = 1  # 초기값: 오른쪽 차선
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
            lane_state = 1  # "right"
        elif right_count > left_count and right_count > 0:
            lane_state = 0  # "left"
        else:
            lane_state = self.prev_lane

        self.prev_lane = lane_state

        # 빨간색 감지
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_pixels = cv2.countNonZero(mask_red)
        red_ratio = red_pixels / float(height * width)

        # --- 노란색 중심 계산 ---
        yellow_center_x = -1
        M_yellow = cv2.moments(mask_yellow)
        if M_yellow["m00"] > 0:
            yellow_center_x = int(M_yellow["m10"] / M_yellow["m00"])

        # 빨간색 중심 계산
        red_center_x = -1
        red_center_y = -1
        if red_pixels > 0:
            M_red = cv2.moments(mask_red)
            if M_red["m00"] > 0:
                red_center_x = int(M_red["m10"] / M_red["m00"])
                red_center_y = int(M_red["m01"] / M_red["m00"])

        # --- 노란색 기준 빨간색 위치 ---
        red_side_relative_to_yellow = -1
        if red_pixels/(width*height) > 0.001 and yellow_center_x != -1:
            if red_center_x < yellow_center_x:
                red_side_relative_to_yellow = 0  # "left"
            else:
                red_side_relative_to_yellow = 1  # "right"

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
            over_line,  # 0
            red_side,   # 1  (-1/0/1)
            red_ratio   # 2
        ], dtype=np.float32)

        return state, lane_state

    def _calculate_reward(self, state, current_lane_state, next_state, next_lane_state, done=False):
        over = float(state[0])
        red_side = int(state[1])
        red_ratio = float(state[2])

        if next_state is None:
            return 0.0, True

        next_red_ratio = float(next_state[2])
        r = 3.0

        if red_ratio > 0.06:
            if red_side == current_lane_state:
                if next_lane_state != current_lane_state:
                    r += 0.3
                else:
                    r -= 0.5 * (red_ratio - 0.06) / 0.14

        if over > 0.5:
            r -= 1.0
            done = True

        if next_red_ratio < red_ratio:
            r += 0.01

        r = float(np.clip(r, -1.0, 1.0))
        return r, done

    def collect_from_frames(self, frames):
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
        for idx in range(len(frames) - 1):
            frame = frames[idx]
            next_frame = frames[idx + 1]
            done = False

            state, current_lane_state = self._get_state(frame)
            next_state, next_lane_state = self._get_state(next_frame)

            action = next_lane_state  # 예측 타깃
            reward, done = self._calculate_reward(state, current_lane_state, next_state, next_lane_state, done=done)

            if state[0] > 0.5 or state[2] > 0.20:
                done = True

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_list.append(done)

        return state_list, action_list, reward_list, next_state_list, done_list

# -----------------------------
# DQN 모델 (CaSSLe용 - Feature 추출 가능)
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1=nn.Linear(state_dim, 256)
        self.ln1=nn.LayerNorm(256)
        self.fc2=nn.Linear(256, 256)
        self.ln2=nn.LayerNorm(256)
        self.q_head = nn.Linear(256, action_dim)

    def forward(self, state, return_features=False):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        if return_features:
            return x
        return self.q_head(x)

# -----------------------------
# DQNAgentCaSSLe (Representation Learning)
# -----------------------------
class DQNAgentCaSSLe:
    def __init__(self, state_dim, action_dim=2, device='cpu',
                 cql_alpha=10, lr=1e-5, gamma=0.1, batch_size=32,
                 target_update_freq=5, rep_lambda=0.1):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rep_lambda = rep_lambda

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=20000)

    def aug(self, x, level=0.05):
        """Data Augmentation: 가우시안 노이즈 추가"""
        return x + level * torch.randn_like(x)
    
    def adaptive_aug_level(self, rep_loss, min_level=0.05, max_level=0.5):
        rep_norm = torch.clamp(rep_loss.detach(), 0, 1)
        return min_level + (max_level - min_level) * (1 - rep_norm)

    def train_offline(self, state_list, action_list, reward_list, next_state_list, done_list, 
                      epochs=50, beta=5, keep_buffer=False):
        if not keep_buffer:
            self.replay_buffer.clear()
        
        for s, a, r, ns, d in zip(state_list, action_list, reward_list, next_state_list, done_list):
            self.replay_buffer.append((s, a, r, ns, d))

        if not self.replay_buffer:
            print("Replay buffer is empty. Skipping training.")
            return

        cos = nn.CosineSimilarity(dim=-1)
        step_count = 0
        
        for epoch in range(epochs):
            batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
            
            states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(self.device)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
            next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(self.device)
            dones = torch.tensor([b[4] for b in batch], dtype=torch.bool).to(self.device)

            # --- Q-Learning Loss ---
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                target_q = rewards + self.gamma * (1 - dones.float()) * next_q_values

            all_q = self.policy_net(states)
            current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)
            logsumexp_q = torch.logsumexp(all_q, dim=1)
            cql_penalty = (logsumexp_q - current_q).mean()
            entropy_penalty = all_q.var(dim=1).mean()
            td_loss = F.mse_loss(current_q, target_q)
            
            # --- CaSSLe Representation Loss ---
            level = self.adaptive_aug_level(td_loss)
            aug1 = self.aug(states, level=level)
            aug2 = self.aug(states, level=level)
            z_online = self.policy_net(aug1, return_features=True)
            with torch.no_grad():
                z_target = self.target_net(aug2, return_features=True)
            rep_loss = 1 - cos(z_online, z_target).mean()
            
            # --- Total Loss ---
            loss = td_loss + self.cql_alpha * cql_penalty + beta * entropy_penalty + self.rep_lambda * rep_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
            self.optimizer.step()

            step_count += 1
            if step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if epoch % 10 == 0:
                print(f"[Offline] Epoch {epoch:03d}, Loss: {loss.item():.4f} (TD: {td_loss.item():.2f}, Rep: {rep_loss.item():.4f})")

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
    if not eval_frames or len(eval_frames) < 2:
        print("Could not load evaluation video or not enough frames to evaluate.")
        return 0.0, 0.0

    collector_eval = OfflineDataCollector(LaneDetector())
    total_reward = 0.0
    agent_actions = []
    offline_actions = []

    for i in range(len(eval_frames) - 1):
        frame = eval_frames[i]
        next_frame = eval_frames[i+1]
        state, current_lane_state = collector_eval._get_state(frame)
        next_state, next_lane_state = collector_eval._get_state(next_frame)

        correct_action = next_lane_state
        offline_actions.append(correct_action)

        reward, _ = collector_eval._calculate_reward(state, current_lane_state, next_state, next_lane_state)
        total_reward += reward

        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = agent.policy_net(s_tensor)
            action = q_values.argmax().item()
            agent_actions.append(action)

    offline_actions = np.array(offline_actions)
    agent_actions = np.array(agent_actions)
    matches = np.sum(offline_actions == agent_actions)
    accuracy = matches / len(offline_actions) if len(offline_actions) > 0 else 0.0

    print(f"=== [Run {run_index+1}] Result ===")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%), Total Reward: {total_reward:.4f}")

    return accuracy, total_reward

# ==============================
# CL 지표 계산용 공통 유틸 (CaSSLe 포함 공용)
# ==============================
def evaluate_on_frames(agent, frames, device):
    """작업 단위(프레임 리스트) 정확도/보상 평가"""
    if not frames or len(frames) < 2:
        return 0.0, 0.0
    collector_eval = OfflineDataCollector(LaneDetector())
    total_reward = 0.0
    agent_actions, offline_actions = [], []

    for i in range(len(frames) - 1):
        frame, next_frame = frames[i], frames[i+1]
        state, current_lane_state = collector_eval._get_state(frame)
        next_state, next_lane_state = collector_eval._get_state(next_frame)

        correct_action = next_lane_state
        offline_actions.append(correct_action)

        r, _ = collector_eval._calculate_reward(state, current_lane_state, next_state, next_lane_state)
        total_reward += r

        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = agent.policy_net(s_tensor)  # CaSSLe도 동일
            a_hat = q_values.argmax().item()
            agent_actions.append(a_hat)

    offline_actions = np.array(offline_actions)
    agent_actions = np.array(agent_actions)
    acc = float((offline_actions == agent_actions).mean()) if len(offline_actions) > 0 else 0.0
    return acc, float(total_reward)

def frames_for_task(video_list, max_frames=None):
    """각 작업을 대표하는 평가 프레임(간단히 리스트 첫 파일)"""
    if not video_list:
        return []
    return load_video_frames(video_list[0], max_frames=max_frames)

def build_performance_matrix(agent_factory, train_order, task_frames_map, train_datasets, device, epochs_per_task=25):
    """
    R[i, j] = i단계(작업 i까지 학습) 후 작업 j의 정확도
    i=0은 학습 전
    """
    T = len(train_order)
    R = np.zeros((T+1, T), dtype=np.float32)

    # 0행: 학습 전 성능 (새 에이전트)
    agent = agent_factory()
    for j, tj in enumerate(train_order):
        acc, _ = evaluate_on_frames(agent, task_frames_map[tj], device)
        R[0, j] = acc

    # 하나의 에이전트로 순차 학습 진행(버퍼 누적 권장)
    for i, ti in enumerate(train_order, start=1):
        s, a, r, ns, d = train_datasets[ti]
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.train_offline(s, a, r, ns, d, epochs=epochs_per_task, beta=5, keep_buffer=True)
        for j, tj in enumerate(train_order):
            acc, _ = evaluate_on_frames(agent, task_frames_map[tj], device)
            R[i, j] = acc

    return R

def single_task_best_performance(agent_factory, task_name, train_datasets, task_frames_map, device, epochs=25):
    """작업 j만 단독 학습했을 때의 최종 정확도 R*_j"""
    agent = agent_factory()
    s, a, r, ns, d = train_datasets[task_name]
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.train_offline(s, a, r, ns, d, epochs=epochs, beta=5, keep_buffer=False)
    acc, _ = evaluate_on_frames(agent, task_frames_map[task_name], device)
    return acc

def compute_cl_metrics(R, R_star):
    """
    R: (T+1, T) — i단계 후 j작업 정확도, i=0은 학습 전
    R_star: 길이 T — 단일작업 최종 정확도
    FWT = mean_{j>=1} (R[j-1, j] - R[0, j])
    BWT = mean_{j<=T-2} (R[T, j] - R[j, j])
    IM  = mean_{j>=1} (R*_j - R[j, j])
    ACC = mean(R[T, :])
    """
    T = R.shape[1]
    fwt_terms = [R[j-1, j] - R[0, j] for j in range(1, T)] if T > 1 else []
    FWT = float(np.mean(fwt_terms)) if fwt_terms else 0.0
    bwt_terms = [R[T, j] - R[j, j] for j in range(T-1)] if T > 1 else []
    BWT = float(np.mean(bwt_terms)) if bwt_terms else 0.0
    im_terms = [R_star[j] - R[j, j] for j in range(1, T)] if T > 1 else []
    IM = float(np.mean(im_terms)) if im_terms else 0.0
    ACC = float(np.mean(R[-1, :])) if T > 0 else 0.0
    return {"ACC": ACC, "BWT": BWT, "FWT": FWT, "IM": IM}

# ==================================
# 메인 실행 로직
# ==================================
start_time = time.time()
DATA_FILE_T1 = "task1_data.npz"
DATA_FILE_T4 = "task4_data.npz"
DATA_FILE_T5 = "task5_data.npz"
DATA_FILE_T3 = "task3_data.npz"

# ❗️ 경로를 실제 환경에 맞게 수정해주세요
video_files_t1 = ["/home/jieun/seohee/1_1.mov", "/home/jieun/seohee/1_2.mov","/home/jieun/seohee/1_3.mov", "/home/jieun/seohee/1_4.mov"]
video_files_t4 = ["/home/jieun/seohee/4_1.mov", "/home/jieun/seohee/4_2.mov","/home/jieun/seohee/4_3.mov", "/home/jieun/seohee/4_4.mov"]
video_files_t5 = ["/home/jieun/seohee/5_1.mov", "/home/jieun/seohee/5_2.mov","/home/jieun/seohee/5_3.mov", "/home/jieun/seohee/5_4.mov"]
video_files_t3 = ["/home/jieun/seohee/3_1.mov", "/home/jieun/seohee/3_2.mov","/home/jieun/seohee/3_3.mov", "/home/jieun/seohee/3_4.mov"]
eval_video_file = "/home/jieun/seohee/test.mov"

# Task 데이터 준비(캐시)
def load_or_create(npz_path, video_files):
    if os.path.exists(npz_path):
        print(f"Loading data from '{npz_path}'")
        data = np.load(npz_path, allow_pickle=True)
        return data['s'], data['a'], data['r'], data['ns'], data['d']
    else:
        print(f"Creating and saving data to '{npz_path}'")
        s, a, r, ns, d = load_and_process_data(video_files)
        np.savez(npz_path, s=s, a=a, r=r, ns=ns, d=d)
        return s, a, r, ns, d

s1, a1, r1, ns1, d1 = load_or_create(DATA_FILE_T1, video_files_t1)
s4, a4, r4, ns4, d4 = load_or_create(DATA_FILE_T4, video_files_t4)
s5, a5, r5, ns5, d5 = load_or_create(DATA_FILE_T5, video_files_t5)
s3, a3, r3, ns3, d3 = load_or_create(DATA_FILE_T3, video_files_t3)

data_prep_time = time.time() - start_time
print(f"Data preparation finished. (Took {data_prep_time:.2f} seconds)")

# --- 10회 실험 반복 ---
num_runs = 10
all_accuracies = []
all_rewards = []

for i in range(num_runs):
    print(f"\n{'='*20} Experiment {i+1}/{num_runs} Start {'='*20}")
    
    agent = DQNAgentCaSSLe(
        state_dim=3, action_dim=2, device=device, cql_alpha=30,
        lr=1e-5, gamma=0.99, batch_size=32, target_update_freq=10, rep_lambda=0.1
    )

    # Step 1: Task 1 학습
    print("\n--- Training on Task 1 ---")
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.train_offline(s1, a1, r1, ns1, d1, epochs=25, beta=5)

    # Step 2: Task 4 학습
    print("\n--- Training on Task 4 ---")
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.train_offline(s4, a4, r4, ns4, d4, epochs=25, beta=5)

    # Step 3: Task 5 추가 학습
    print("\n--- Training on Task 5 (Continual) ---")
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.train_offline(s5, a5, r5, ns5, d5, epochs=25, beta=5, keep_buffer=True)

    # Step 4: Task 3 학습
    print("\n--- Training on Task 3 ---")
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.train_offline(s3, a3, r3, ns3, d3, epochs=25, beta=5)

    # Step 5: 최종 평가
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

# ===============================
# Continual Learning 지표(FWT/BWT/IM)
# ===============================
print("\n\n" + "="*20 + " Continual Learning Metrics (CaSSLe) " + "="*20)

# 작업 정의
train_order = ["task1", "task4", "task5", "task3"]
train_datasets = {
    "task1": (s1, a1, r1, ns1, d1),
    "task4": (s4, a4, r4, ns4, d4),
    "task5": (s5, a5, r5, ns5, d5),
    "task3": (s3, a3, r3, ns3, d3),
}
task_frames_map = {
    "task1": frames_for_task(video_files_t1, max_frames=None),
    "task4": frames_for_task(video_files_t4, max_frames=None),
    "task5": frames_for_task(video_files_t5, max_frames=None),
    "task3": frames_for_task(video_files_t3, max_frames=None),
}

def agent_factory():
    return DQNAgentCaSSLe(
        state_dim=3, action_dim=2, device=device, cql_alpha=30,
        lr=1e-5, gamma=0.99, batch_size=32, target_update_freq=10, rep_lambda=0.1
    )

# 1) 성능 행렬 R
R = build_performance_matrix(
    agent_factory=agent_factory,
    train_order=train_order,
    task_frames_map=task_frames_map,
    train_datasets=train_datasets,
    device=device,
    epochs_per_task=25
)
print("\nR (rows: stage 0..T, cols: tasks in order):")
print(np.array2string(R, formatter={'float_kind': lambda x: f"{x:0.3f}"}))

# 2) 단일 작업 기준 성능 R*
R_star = []
for tj in train_order:
    acc_star = single_task_best_performance(
        agent_factory=agent_factory,
        task_name=tj,
        train_datasets=train_datasets,
        task_frames_map=task_frames_map,
        device=device,
        epochs=25
    )
    R_star.append(acc_star)
print("\nR* (single-task final acc per task):")
print([f"{v:.3f}" for v in R_star])

# 3) 지표 계산
metrics = compute_cl_metrics(R, R_star)
print("\n--- CL Metrics (CaSSLe) ---")
print(f"ACC: {metrics['ACC']:.3f}")
print(f"BWT: {metrics['BWT']:.3f}")
print(f"FWT: {metrics['FWT']:.3f}")
print(f"IM : {metrics['IM']:.3f}")

