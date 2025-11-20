# -*- coding: utf-8 -*-
"""
Jetson용 DQN Lane-keeping / Obstacle Avoidance
순차 학습 (EWC 없음) + 데이터 캐싱 + 10회 반복 실험 버전
각 작업 종료 시마다 Continual Learning 지표(FWT, BWT, IM) 출력
test.mov는 Accuracy만 측정
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
        # 빨강 HSV 범위
        self.lower_red1 = np.array([0,70,50])
        self.upper_red1 = np.array([10,255,255])
        self.lower_red2 = np.array([170,70,50])
        self.upper_red2 = np.array([180,255,255])
        # 직전 차선 상태 기본값
        self.prev_lane = 1  # 1="right", 0="left"

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1) 노란색 마스크
        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
        # 2) 흰색 마스크 (시각화용)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([255, 30, 255]))

        # 차선 방향 판단은 '노란색' 기준
        left_half = mask_yellow[:, :width // 2]
        right_half = mask_yellow[:, width // 2:]
        left_count = cv2.countNonZero(left_half)
        right_count = cv2.countNonZero(right_half)

        if left_count > right_count and left_count > 0:
            lane_state = 1  # "right"로 가야 (노란선이 왼쪽)
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

        # 노란색 무게중심
        yellow_center_x = -1
        M_yellow = cv2.moments(mask_yellow)
        if M_yellow["m00"] > 0:
            yellow_center_x = int(M_yellow["m10"] / M_yellow["m00"])

        # 빨간색 무게중심
        red_center_x = -1
        if red_pixels > 0:
            M_red = cv2.moments(mask_red)
            if M_red["m00"] > 0:
                red_center_x = int(M_red["m10"] / M_red["m00"])

        # 빨간색의 노란선 기준 좌/우
        red_side_relative_to_yellow = -1
        if red_pixels/(width*height) > 0.001 and yellow_center_x != -1:
            red_side_relative_to_yellow = 0 if red_center_x < yellow_center_x else 1

        return lane_state, red_ratio, red_side_relative_to_yellow

# -----------------------------
# OfflineDataCollector
# -----------------------------
class OfflineDataCollector:
    def __init__(self, lane_detector):
        self.lane_detector = lane_detector

    def _get_state(self, frame):
        lane_state, red_ratio, red_side = self.lane_detector.process_frame(frame)
        height, width, _ = frame.shape
        bottom_center_pixel = frame[height - 1, width // 2]
        over_line = float(np.all(bottom_center_pixel > 240))
        state = np.array([over_line, red_side, red_ratio], dtype=np.float32)
        return state, lane_state

    def _calculate_reward(self, state, current_lane_state, next_state, next_lane_state, done=False):
        over = float(state[0])
        red_side = int(state[1])
        red_ratio = float(state[2])
        if next_state is None:
            return 0.0, True

        next_red_ratio = float(next_state[2])
        r = 3.0
        # 장애물이 '현재 차선'에 있고 회피 못하면 페널티
        if red_ratio > 0.06:
            if red_side == current_lane_state:
                if next_lane_state != current_lane_state:
                    r += 0.3  # 회피 성공 보상(옵션)
                else:
                    r -= 0.5 * (red_ratio - 0.06) / 0.14

        # 차선 이탈 페널티
        if over > 0.5:
            r -= 1.0
            done = True

        # 장애물에서 멀어지면 소폭 보상
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
            action = next_lane_state
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
# DQN 모델
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

# -----------------------------
# DQNAgent (naive / EWC 없음)
# -----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim=2, device='cpu',
                 cql_alpha=30, lr=1e-5, gamma=0.99, batch_size=32,
                 target_update_freq=10):
        self.device = device
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

    def train_offline(self, state_list, action_list, reward_list, next_state_list, done_list,
                      epochs=20, beta=5, keep_buffer=False):
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
# 데이터/평가 헬퍼
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
    # 비디오마다 detector/collector 새로 만들어 prev_lane 누적 방지
    state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
    for vf in video_files:
        print(f"Processing video: {vf}")
        frames = load_video_frames(vf)
        if not frames:
            continue
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

def evaluate_agent_on_frames(agent, frames):
    """주어진 프레임 시퀀스(영상 1개)에 대해 정확도"""
    if not frames or len(frames) < 2:
        return 0.0
    collector_eval = OfflineDataCollector(LaneDetector())
    agent_actions, offline_actions = [], []
    for i in range(len(frames) - 1):
        frame = frames[i]
        next_frame = frames[i + 1]
        state, _ = collector_eval._get_state(frame)
        _, next_lane_state = collector_eval._get_state(next_frame)

        offline_actions.append(next_lane_state)
        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = agent.policy_net(s_tensor)
            action = q_values.argmax().item()
            agent_actions.append(action)

    offline_actions = np.array(offline_actions)
    agent_actions = np.array(agent_actions)
    if len(offline_actions) == 0:
        return 0.0
    return float((offline_actions == agent_actions).mean())

def evaluate_on_video(agent, video_file):
    frames = load_video_frames(video_file)
    return evaluate_agent_on_frames(agent, frames)

def frames_for_task(video_list, max_frames=None):
    return load_video_frames(video_list[0], max_frames=max_frames) if video_list else []

# -----------------------------
# 새 헬퍼: 모든 작업 평가(R의 한 행) + 부분 지표 계산
# -----------------------------
def evaluate_all_tasks(agent, task_eval_frames_list):
    """
    task_eval_frames_list: 각 작업의 평가용 프레임 목록 리스트
    반환: np.array shape (T,)
    """
    accs = np.zeros(len(task_eval_frames_list), dtype=np.float32)
    for j, frames in enumerate(task_eval_frames_list):
        accs[j] = evaluate_agent_on_frames(agent, frames)
    return accs

def compute_cl_metrics_partial(R_rows, R_star, upto_i):
    """
    R_rows: [R[0,:], R[1,:], ..., R[i,:]] (list of 1D arrays)
    upto_i: 현재 단계 i (작업 i까지 학습 완료 후), i in [1..T]
    R_star: 단일작업 기준 정확도 리스트 길이 T

    정의 (Lopez-Paz & Ranzato, 2017)
      BWT = mean_{j<i} (R[i,j] - R[j,j])
      FWT = mean_{j<=i, j>0} (R[j-1, j] - R[0, j])
      IM  = mean_{j<=i, j>0} (R*_j - R[j,j])
    """
    R = np.vstack(R_rows[:upto_i+1])  # (upto_i+1, T)
    T = R.shape[1]
    i = upto_i

    # BWT
    bwt_terms = [R[i, j] - R[j, j] for j in range(0, min(i, T))]
    BWT = float(np.mean(bwt_terms)) if bwt_terms else 0.0

    # FWT
    fwt_terms = [R[j-1, j] - R[0, j] for j in range(1, min(i, T-1)+1)]
    FWT = float(np.mean(fwt_terms)) if fwt_terms else 0.0

    # IM
    im_terms = [R_star[j] - R[j, j] for j in range(1, min(i, T-1)+1)]
    IM = float(np.mean(im_terms)) if im_terms else 0.0

    return {"FWT": FWT, "BWT": BWT, "IM": IM}

# -----------------------------
# 단일 작업 기준 성능 (R*) 계산용
# -----------------------------
def single_task_best_performance(task_data, eval_frames,
                                 lr=1e-5, gamma=0.99, cql_alpha=30,
                                 batch_size=32, target_update_freq=10,
                                 epochs=20):
    """해당 태스크 데이터만으로 새 에이전트 학습 후 그 태스크 프레임으로 정확도 평가"""
    s, a, r, ns, d = task_data
    agent = DQNAgent(state_dim=3, action_dim=2, device=device,
                     cql_alpha=cql_alpha, lr=lr, gamma=gamma,
                     batch_size=batch_size, target_update_freq=target_update_freq)
    agent.train_offline(s, a, r, ns, d, epochs=epochs, beta=5, keep_buffer=False)
    return evaluate_agent_on_frames(agent, eval_frames)

# ==================================
# 메인 실행 로직
# ==================================
start_time = time.time()
DATA_FILE_T1 = "task1_data.npz"
DATA_FILE_T4 = "task4_data.npz"
DATA_FILE_T5 = "task5_data.npz"
DATA_FILE_T3 = "task3_data.npz"

# ❗️ 경로를 실제 환경에 맞게 수정
video_files_t1 = ["/home/jieun/test/1_1.mov", "/home/jieun/test/1_2.mov","/home/jieun/test/1_3.mov", "/home/jieun/test/1_4.mov"]
video_files_t4 = ["/home/jieun/test/4_1.mov", "/home/jieun/test/4_2.mov","/home/jieun/test/4_3.mov", "/home/jieun/test/4_4.mov"]
video_files_t5 = ["/home/jieun/test/5_1.mov", "/home/jieun/test/5_2.mov","/home/jieun/test/5_3.mov", "/home/jieun/test/5_4.mov"]
video_files_t3 = ["/home/jieun/test/3_1.mov", "/home/jieun/test/3_2.mov","/home/jieun/test/3_3.mov", "/home/jieun/test/3_4.mov"]
eval_video_file = "/home/jieun/test/test.mov"

# 데이터 로딩/생성
def load_or_create(npz_file, video_files):
    if os.path.exists(npz_file):
        print(f"Loading data from '{npz_file}'")
        data = np.load(npz_file, allow_pickle=True)
        return data['s'], data['a'], data['r'], data['ns'], data['d']
    else:
        print(f"Creating and saving data to '{npz_file}'")
        s, a, r, ns, d = load_and_process_data(video_files)
        np.savez(npz_file, s=s, a=a, r=r, ns=ns, d=d)
        return s, a, r, ns, d

s1, a1, r1, ns1, d1 = load_or_create(DATA_FILE_T1, video_files_t1)
s4, a4, r4, ns4, d4 = load_or_create(DATA_FILE_T4, video_files_t4)
s5, a5, r5, ns5, d5 = load_or_create(DATA_FILE_T5, video_files_t5)
s3, a3, r3, ns3, d3 = load_or_create(DATA_FILE_T3, video_files_t3)

data_prep_time = time.time() - start_time
print(f"Data preparation finished. (Took {data_prep_time:.2f} seconds)")

# 태스크 구성 (순서 중요)
tasks = [
    ("task1", (s1, a1, r1, ns1, d1), video_files_t1),
    ("task4", (s4, a4, r4, ns4, d4), video_files_t4),
    ("task5", (s5, a5, r5, ns5, d5), video_files_t5),
    ("task3", (s3, a3, r3, ns3, d3), video_files_t3),
]
T = len(tasks)

# 각 태스크 평가 프레임(각 태스크 첫 영상 사용)
task_eval_frames_list = []
for name, _, vids in tasks:
    task_eval_frames_list.append(frames_for_task(vids, max_frames=None))

# ----- R* (단일작업 기준) 한 번만 선계산 -----
print("\n[Compute R*] single-task best accuracies per task ...")
R_star = []
for ti, (name, data_i, vids_i) in enumerate(tasks):
    stl_acc = single_task_best_performance(
        task_data=data_i,
        eval_frames=task_eval_frames_list[ti],
        lr=1e-5, gamma=0.99, cql_alpha=30,
        batch_size=32, target_update_freq=10, epochs=20
    )
    R_star.append(stl_acc)
print("[R*] =", [f"{v:.3f}" for v in R_star])

# --- 반복 실험 ---
num_runs = 3
all_test_acc = []   # test.mov accuracy 기록

for run in range(num_runs):
    print(f"\n{'='*20} Experiment {run+1}/{num_runs} Start {'='*20}")
    agent = DQNAgent(state_dim=3, action_dim=2, device=device,
                     cql_alpha=30, lr=1e-5, gamma=0.99,
                     batch_size=32, target_update_freq=10)

    # R 행렬(진행형) 저장용: 리스트로 관리 후 필요 시 stack
    R_rows = []

    # 0단계(무학습) 성능 R[0,:]
    R0 = evaluate_all_tasks(agent, task_eval_frames_list)
    R_rows.append(R0)
    print(f"[Stage 0] Acc per task: {[f'{x:.3f}' for x in R0]}")

    # 연속 학습 진행 + 각 작업 종료 시 지표 출력
    for i, (name_i, data_i, vids_i) in enumerate(tasks, start=1):
        print(f"\n--- Training on {name_i} ---")
        s_i, a_i, r_i, ns_i, d_i = data_i
        agent.train_offline(s_i, a_i, r_i, ns_i, d_i, epochs=20, beta=5, keep_buffer=False)

        # i단계 후 R[i,:] 계산/저장
        Ri = evaluate_all_tasks(agent, task_eval_frames_list)
        R_rows.append(Ri)
        print(f"[Stage {i}] Acc per task: {[f'{x:.3f}' for x in Ri]}")

        # 부분 지표(FWT/BWT/IM) 계산/출력
        metrics_i = compute_cl_metrics_partial(R_rows, R_star, upto_i=i)
        print(f"[After {name_i}] FWT={metrics_i['FWT']:.4f}, BWT={metrics_i['BWT']:.4f}, IM={metrics_i['IM']:.4f}")

    # test.mov: Accuracy만
    acc_test = evaluate_on_video(agent, eval_video_file)
    print(f"\n[Run {run+1}] test.mov Accuracy: {acc_test:.3f}")
    all_test_acc.append(acc_test)

# --- 최종 결과 요약(Accuracy만) ---
total_run_time = time.time() - start_time
print(f"\n\n{'='*20} Final Summary of {num_runs} Experiments {'='*20}")
print(f"Total time: {total_run_time:.2f} seconds")
if all_test_acc:
    mean_acc = np.mean(all_test_acc)
    std_acc = np.std(all_test_acc)
    print(f"Average test.mov Accuracy: {mean_acc:.3f} (Std: {std_acc:.3f})")
    print("Individual Accuracies:", [f"{a:.3f}" for a in all_test_acc])



