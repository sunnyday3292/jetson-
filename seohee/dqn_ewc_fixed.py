# -*- coding: utf-8 -*-
"""
Jetson용 DQN Lane-keeping / Obstacle Avoidance
순차 학습 + (선택)EWC + 데이터 캐싱 + 10회 반복 실험 버전
+ Continual Learning 지표: FWT, BWT, IM
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
        self.prev_lane = 1  # 안전한 초기값(오른쪽)
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

        # --- ### 1. 노란색 차선의 무게 중심 계산 ### ---
        yellow_center_x = -1
        M_yellow = cv2.moments(mask_yellow)
        if M_yellow["m00"] > 0:
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
        red_side_relative_to_yellow = -1
        if red_pixels/(width*height) > 0.001 and yellow_center_x != -1:
            if red_center_x < yellow_center_x:
                red_side_relative_to_yellow = 0  # "left"
            else:
                red_side_relative_to_yellow = 1  # "right"

        # 반환값
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

        # 상태: lane_state 제거(누수 방지)
        state = np.array([
            over_line,  # 0
            red_side,   # 1
            red_ratio   # 2
        ], dtype=np.float32)

        return state, lane_state  # lane_state는 별도 반환

    def _calculate_reward(self, state, current_lane_state, next_state, next_lane_state, done=False):
        over = float(state[0])
        red_side = int(state[1])
        red_ratio = float(state[2])

        if next_state is None:
            return 0.0, True

        next_red_ratio = float(next_state[2])
        r = 3.0

        # 장애물이 '현재 차선'에 있을 때 회피 못하면 페널티
        if red_ratio > 0.06:
            if red_side == current_lane_state:
                if next_lane_state != current_lane_state:
                    r += 0.3  # 회피 성공 보상
                else:
                    r -= 0.5 * (red_ratio - 0.06) / 0.14  # 실패 패널티

        # 차선 이탈 패널티
        if over > 0.5:
            r -= 1.0
            done = True

        # 장애물 비율 감소를 약하게 장려
        if next_red_ratio < red_ratio:
            r += 0.01

        r = float(np.clip(r, -1.0, 1.0))
        return r, done

    def collect_from_frames(self, frames):
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
        # 마지막 프레임은 next가 없으니 len-1까지만
        for idx in range(len(frames) - 1):
            frame = frames[idx]
            next_frame = frames[idx + 1]
            done = False

            state, current_lane_state = self._get_state(frame)
            next_state, next_lane_state = self._get_state(next_frame)

            # 정답 액션: 다음 프레임의 lane_state (예측 타깃)
            action = next_lane_state

            reward, done = self._calculate_reward(state, current_lane_state, next_state, next_lane_state, done=done)

            if state[0] > 0.5 or state[2] > 0.20:  # over_line or red_ratio
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
# DQNAgent with optional EWC
# -----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim=2, device='cpu',
                 cql_alpha=10, lr=1e-5, gamma=0.1, batch_size=32,
                 target_update_freq=5, ewc_lambda=0):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.ewc_lambda = ewc_lambda

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=20000)

        self.task_params = {}
        self.task_fisher = {}

    def calculate_fisher_information(self, states, actions):
        self.policy_net.eval()
        self.optimizer.zero_grad()
        q_values = self.policy_net(states)
        selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Fisher 근사: 선택 Q의 평균을 최대화하는 그라드의 제곱
        loss = selected_q_values.mean()
        loss.backward()
        fisher_information = {}
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                fisher_information[name] = (param.grad.data.clone() ** 2)
        return fisher_information

    def save_params_and_fisher(self, dataset, task_name, sample_max=1024):
        print(f"[EWC] Saving parameters and Fisher for {task_name}...")
        # dataset: list of tuples (s,a,r,ns,d) — 여기서 s,a만 샘플
        if len(dataset) == 0:
            return
        # 샘플 다운샘플링(너무 크면)
        if len(dataset) > sample_max:
            dataset = random.sample(dataset, sample_max)

        states_sample = torch.tensor([x[0] for x in dataset], dtype=torch.float32).to(self.device)
        actions_sample = torch.tensor([x[1] for x in dataset], dtype=torch.long).to(self.device)

        self.task_params[task_name] = {name: p.clone().detach() for name, p in self.policy_net.named_parameters()}
        self.task_fisher[task_name] = self.calculate_fisher_information(states_sample, actions_sample)

    def ewc_loss(self):
        if (self.ewc_lambda is None) or (self.ewc_lambda <= 0) or (len(self.task_params) == 0):
            return torch.tensor(0.0, device=self.device)
        ewc_loss_val = torch.tensor(0.0, device=self.device)
        for task_name in self.task_params:
            for name, param in self.policy_net.named_parameters():
                if name in self.task_params[task_name]:
                    saved_param = self.task_params[task_name][name]
                    fisher = self.task_fisher[task_name].get(name, None)
                    if fisher is not None:
                        ewc_loss_val = ewc_loss_val + (fisher * (param - saved_param) ** 2).sum()
        return self.ewc_lambda * ewc_loss_val

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
            cql_penalty = (logsumexp_q - current_q).mean()

            entropy_penalty = all_q.var(dim=1).mean()
            td_loss = F.mse_loss(current_q, target_q)

            loss = td_loss + self.cql_alpha * cql_penalty + beta * entropy_penalty + self.ewc_loss()

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
# 헬퍼 함수: 비디오/데이터 로딩
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
    state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
    for vf in video_files:
        print(f"Processing video: {vf}")
        frames = load_video_frames(vf)
        if not frames:
            continue
        # 비디오마다 detector/collector 초기화 (prev_lane 누적 방지)
        lane_detector = LaneDetector()
        collector = OfflineDataCollector(lane_detector)
        s, a, r, ns, d = collector.collect_from_frames(frames)
        state_list.extend(s)
        action_list.extend(a)
        reward_list.extend(r)
        next_state_list.extend(ns)
        done_list.extend(d)
    print(f"Generated {len(state_list)} transitions.")
    return state_list, action_list, reward_list, next_state_list, done_list

# -----------------------------
# 평가 유틸
# -----------------------------
def evaluate_agent(agent, eval_video_file, run_index=0):
    print(f"\n--- [Run {run_index+1}] Evaluation (single file) ---")
    eval_frames = load_video_frames(eval_video_file)
    return evaluate_on_frames(agent, eval_frames, device)

def evaluate_on_frames(agent, frames, device):
    if not frames or len(frames) < 2:
        print("Could not load evaluation frames or not enough frames to evaluate.")
        return 0.0, 0.0
    collector_eval = OfflineDataCollector(LaneDetector())
    total_reward = 0.0
    agent_actions = []
    offline_actions = []

    for i in range(len(frames) - 1):
        frame = frames[i]
        next_frame = frames[i+1]

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
    acc = float((offline_actions == agent_actions).mean()) if len(offline_actions) > 0 else 0.0
    return acc, float(total_reward)

def frames_for_task(video_list, max_frames=None):
    if not video_list:
        return []
    return load_video_frames(video_list[0], max_frames=max_frames)

# -----------------------------
# CL 성능 행렬 R / R* 및 지표
# -----------------------------
def build_performance_matrix(agent_factory, train_order, task_frames_map, train_datasets, device, epochs_per_task=25, use_ewc=True):
    """
    R[i, j] = i단계(작업 i까지 학습) 후 작업 j의 정확도
    i=0은 학습 전
    """
    T = len(train_order)
    R = np.zeros((T+1, T), dtype=np.float32)

    # 0행: 학습 전 성능
    agent = agent_factory()
    for j, tj in enumerate(train_order):
        acc, _ = evaluate_on_frames(agent, task_frames_map[tj], device)
        R[0, j] = acc

    # 순차 학습
    for i, ti in enumerate(train_order, start=1):
        s, a, r, ns, d = train_datasets[ti]
        # 누적 학습 권장
        agent.train_offline(s, a, r, ns, d, epochs=epochs_per_task, beta=5, keep_buffer=True)

        # EWC 파라미터 저장 → 다음 작업부터 ewc_loss 적용
        if use_ewc and hasattr(agent, "save_params_and_fisher"):
            dataset_for_fisher = list(zip(s, a, r, ns, d))
            agent.save_params_and_fisher(dataset_for_fisher, ti)

        # i단계 후 전체 작업 평가
        for j, tj in enumerate(train_order):
            acc, _ = evaluate_on_frames(agent, task_frames_map[tj], device)
            R[i, j] = acc

    return R

def single_task_best_performance(agent_factory, task_name, train_datasets, task_frames_map, device, epochs=25):
    agent = agent_factory()
    s, a, r, ns, d = train_datasets[task_name]
    agent.train_offline(s, a, r, ns, d, epochs=epochs, beta=5, keep_buffer=False)
    acc, _ = evaluate_on_frames(agent, task_frames_map[task_name], device)
    return acc

def compute_cl_metrics(R, R_star):
    """
    R: (T+1, T)
    R_star: 길이 T
    BWT = mean_j<T (R[T,j] - R[j,j])
    FWT = mean_j>0 (R[j-1,j] - R[0,j])    # Lopez-Paz & Ranzato(2017) 정의
    IM  = mean_j>=1 (R*_j - R[j,j])       # 단독학습 대비 순차학습의 비학습성
    ACC = mean(R[T,:])
    """
    T = R.shape[1]
    # BWT
    bwt_terms = [R[T, j] - R[j, j] for j in range(T-1)] if T > 1 else []
    BWT = float(np.mean(bwt_terms)) if bwt_terms else 0.0
    # FWT
    fwt_terms = [R[j-1, j] - R[0, j] for j in range(1, T)] if T > 1 else []
    FWT = float(np.mean(fwt_terms)) if fwt_terms else 0.0
    # IM
    im_terms = [R_star[j] - R[j, j] for j in range(1, T)] if T > 1 else []
    IM = float(np.mean(im_terms)) if im_terms else 0.0
    # ACC
    ACC = float(np.mean(R[-1, :])) if T > 0 else 0.0
    return {"ACC": ACC, "BWT": BWT, "FWT": FWT, "IM": IM}

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
eval_video_file = "/home/jieun/seohee/2_2.mov"

# --- 데이터 준비 (캐시 사용) ---
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
s6, a6, r6, ns6, d6 = load_or_create(DATA_FILE_T6, video_files_t6)

data_prep_time = time.time() - start_time
print(f"Data preparation finished. (Took {data_prep_time:.2f} seconds)")

# --- 반복 실험 ---
num_runs = 10
all_accuracies = []
all_rewards = []

# 작업 정의
train_order = ["task1", "task4", "task5", "task6"]
train_datasets = {
    "task1": (s1, a1, r1, ns1, d1),
    "task4": (s4, a4, r4, ns4, d4),
    "task5": (s5, a5, r5, ns5, d5),
    "task6": (s6, a6, r6, ns6, d6),
}
task_frames_map = {
    "task1": frames_for_task(video_files_t1, max_frames=None),
    "task4": frames_for_task(video_files_t4, max_frames=None),
    "task5": frames_for_task(video_files_t5, max_frames=None),
    "task6": frames_for_task(video_files_t6, max_frames=None),
}

def agent_factory():
    return DQNAgent(
        state_dim=3, action_dim=2, device=device,
        cql_alpha=30, lr=1e-5, gamma=0.99,
        batch_size=32, target_update_freq=10, ewc_lambda=10  # EWC 강도 조절 가능
    )

# (선택) 기존 단일 파일 평가 흐름 예시
for i in range(num_runs):
    print(f"\n{'='*20} Experiment {i+1}/{num_runs} Start {'='*20}")
    agent = agent_factory()

    # 순차 학습(버퍼 누적 권장)
    print("\n--- Training on Task 1 ---")
    agent.train_offline(s1, a1, r1, ns1, d1, epochs=25, beta=5, keep_buffer=False)
    agent.save_params_and_fisher(list(zip(s1, a1, r1, ns1, d1)), "task1")

    print("\n--- Training on Task 4 ---")
    agent.train_offline(s4, a4, r4, ns4, d4, epochs=25, beta=5, keep_buffer=True)
    agent.save_params_and_fisher(list(zip(s4, a4, r4, ns4, d4)), "task4")

    print("\n--- Training on Task 5 ---")
    agent.train_offline(s5, a5, r5, ns5, d5, epochs=25, beta=5, keep_buffer=True)
    agent.save_params_and_fisher(list(zip(s5, a5, r5, ns5, d5)), "task5")

    print("\n--- Training on Task 6 ---")
    agent.train_offline(s6, a6, r6, ns6, d6, epochs=25, beta=5, keep_buffer=True)
    agent.save_params_and_fisher(list(zip(s6, a6, r6, ns6, d6)), "task6")

    # 최종 평가 (단일 파일)
    acc, total_reward = evaluate_agent(agent, eval_video_file, run_index=i)
    print(f"=== [Run {i+1}] Result ===")
    print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%), Total Reward: {total_reward:.4f}")
    all_accuracies.append(acc)
    all_rewards.append(total_reward)

# --- 최종 결과 요약(단일 파일 평가) ---
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
print("\n\n" + "="*20 + " Continual Learning Metrics " + "="*20)
# 1) 성능 행렬 R
R = build_performance_matrix(
    agent_factory=agent_factory,
    train_order=train_order,
    task_frames_map=task_frames_map,
    train_datasets=train_datasets,
    device=device,
    epochs_per_task=25,
    use_ewc=True
)
print("\nR (rows: stage 0..T, cols: tasks in order):")
print(np.array2string(R, formatter={'float_kind':lambda x: f"{x:0.3f}"}))

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
print("\n--- CL Metrics ---")
print(f"ACC: {metrics['ACC']:.3f}")
print(f"BWT: {metrics['BWT']:.3f}")
print(f"FWT: {metrics['FWT']:.3f}")
print(f"IM : {metrics['IM']:.3f}")
