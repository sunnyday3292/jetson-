# -*- coding: utf-8 -*-
"""
Jetson용 DQN Lane-keeping / Obstacle Avoidance
순차 학습 + (선택)EWC + 데이터 캐싱 + 10회 반복 실험 버전
+ Continual Learning 지표: FWT, BWT, IM (각 작업 종료 시마다 계산)

[수정 사항]
- 상태 벡터(State Vector)에 Z-점수 표준화 적용
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
# LaneDetector (변경 없음)
# -----------------------------
class LaneDetector:
    def __init__(self):
        self.prev_lane = 1  # 안전한 초기값(오른쪽)
        # 빨강 HSV 범위
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(
            hsv, np.array([15, 80, 100]), np.array([35, 255, 255])
        )
        left_half = mask_yellow[:, : width // 2]
        right_half = mask_yellow[:, width // 2 :]
        left_count = cv2.countNonZero(left_half)
        right_count = cv2.countNonZero(right_half)
        if left_count > right_count and left_count > 0:
            lane_state = 1
        elif right_count > left_count and right_count > 0:
            lane_state = 0
        else:
            lane_state = self.prev_lane
        self.prev_lane = lane_state
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_pixels = cv2.countNonZero(mask_red)
        red_ratio = red_pixels / float(height * width)
        yellow_center_x = -1
        M_yellow = cv2.moments(mask_yellow)
        if M_yellow["m00"] > 0:
            yellow_center_x = int(M_yellow["m10"] / M_yellow["m00"])
        red_center_x = -1
        if red_pixels > 0:
            M_red = cv2.moments(mask_red)
            if M_red["m00"] > 0:
                red_center_x = int(M_red["m10"] / M_red["m00"])
        red_side_relative_to_yellow = -1
        if red_pixels / (width * height) > 0.001 and yellow_center_x != -1:
            if red_center_x < yellow_center_x:
                red_side_relative_to_yellow = 0
            else:
                red_side_relative_to_yellow = 1
        return lane_state, red_ratio, red_side_relative_to_yellow


# -----------------------------
# OfflineDataCollector (변경 없음)
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
        
        # Z-점수 표준화는 이 함수 밖에서 일괄적으로 적용하므로, 여기서는 Min-Max 정규화만 유지
        red_side = (float(red_side) + 1.0) / 2.0
        
        state = np.array(
            [over_line, red_side, red_ratio], dtype=np.float32
        )
        return state, lane_state

    def _calculate_reward(
        self, state, current_lane_state, next_state, next_lane_state, done=False
    ):
        over = float(state[0])
        # Min-Max 정규화된 red_side를 원래 값(0 또는 1)으로 복원
        red_side = int(round(state[1])) if state[1] != 0.5 else current_lane_state
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
        state_list, action_list, reward_list, next_state_list, done_list = (
            [], [], [], [], [],
        )
        for idx in range(len(frames) - 1):
            frame = frames[idx]
            next_frame = frames[idx + 1]
            done = False
            state, current_lane_state = self._get_state(frame)
            next_state, next_lane_state = self._get_state(next_frame)
            action = next_lane_state
            reward, done = self._calculate_reward(
                state, current_lane_state, next_state, next_lane_state, done=done
            )
            if state[0] > 0.5 or state[2] > 0.20:
                done = True
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_list.append(done)
        return state_list, action_list, reward_list, next_state_list, done_list


# -----------------------------
# DQN 모델 및 DQNAgent (변경 없음)
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

class DQNAgent:
    def __init__( self, state_dim, action_dim=2, device="cpu", cql_alpha=30, lr=1e-5,
        gamma=0.99, batch_size=32, target_update_freq=10, ewc_lambda=10, ):
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
        loss = selected_q_values.mean()
        loss.backward()
        fisher_information = {}
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                fisher_information[name] = param.grad.data.clone() ** 2
        return fisher_information

    def save_params_and_fisher(self, dataset, task_name, sample_max=1024):
        print(f"[EWC] Saving parameters and Fisher for {task_name}...")
        if len(dataset) == 0: return
        if len(dataset) > sample_max:
            dataset = random.sample(dataset, sample_max)
        states_np = np.stack([x[0] for x in dataset], axis=0).astype(np.float32)
        actions_np = np.array([x[1] for x in dataset], dtype=np.int64)
        states_sample = torch.from_numpy(states_np).to(self.device)
        actions_sample = torch.from_numpy(actions_np).to(self.device)
        self.task_params[task_name] = {
            name: p.clone().detach() for name, p in self.policy_net.named_parameters()
        }
        self.task_fisher[task_name] = self.calculate_fisher_information(
            states_sample, actions_sample
        )

    def ewc_loss(self):
        if ( (self.ewc_lambda is None) or (self.ewc_lambda <= 0) or (len(self.task_params) == 0) ):
            return torch.tensor(0.0, device=self.device)
        ewc_loss_val = torch.tensor(0.0, device=self.device)
        for task_name in self.task_params:
            for name, param in self.policy_net.named_parameters():
                if name in self.task_params[task_name]:
                    saved_param = self.task_params[task_name][name]
                    fisher = self.task_fisher[task_name].get(name, None)
                    if fisher is not None:
                        ewc_loss_val = ( ewc_loss_val + (fisher * (param - saved_param) ** 2).sum() )
        return self.ewc_lambda * ewc_loss_val

    def train_offline( self, state_list, action_list, reward_list, next_state_list,
        done_list, epochs=20, beta=5, keep_buffer=False, ):
        if not keep_buffer:
            self.replay_buffer.clear()
        for s, a, r, ns, d in zip(
            state_list, action_list, reward_list, next_state_list, done_list
        ):
            self.replay_buffer.append((s, a, r, ns, d))
        if not self.replay_buffer:
            print("Replay buffer is empty. Skipping training.")
            return
        step_count = 0
        for epoch in range(epochs):
            batch = random.sample(
                self.replay_buffer, min(len(self.replay_buffer), self.batch_size)
            )
            states_np = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
            actions_np = np.array([b[1] for b in batch], dtype=np.int64)
            rewards_np = np.array([b[2] for b in batch], dtype=np.float32)
            next_states_np = np.stack([b[3] for b in batch], axis=0).astype(np.float32)
            dones_np = np.array([bool(b[4]) for b in batch], dtype=np.float32)
            states = torch.from_numpy(states_np).to(self.device)
            actions = torch.from_numpy(actions_np).to(self.device)
            rewards = torch.from_numpy(rewards_np).to(self.device)
            next_states = torch.from_numpy(next_states_np).to(self.device)
            dones = torch.from_numpy(dones_np).to(self.device)
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = (
                    self.target_net(next_states).gather(1, next_actions).squeeze(1)
                )
                target_q = rewards + self.gamma * (1.0 - dones) * next_q_values
            all_q = self.policy_net(states)
            current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)
            logsumexp_q = torch.logsumexp(all_q, dim=1)
            cql_penalty = (logsumexp_q - current_q).mean()
            entropy_penalty = all_q.var(dim=1).mean()
            td_loss = F.mse_loss(current_q, target_q)
            loss = ( td_loss + self.cql_alpha * cql_penalty + beta * entropy_penalty + self.ewc_loss() )
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
# 헬퍼 함수: 비디오/데이터 로딩 (변경 없음)
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
        if not ret: break
        frames.append(frame)
        frame_count += 1
        if max_frames and frame_count >= max_frames: break
    cap.release()
    return frames

def load_and_process_data(video_files):
    state_list, action_list, reward_list, next_state_list, done_list = ( [], [], [], [], [], )
    for vf in video_files:
        print(f"Processing video: {vf}")
        frames = load_video_frames(vf)
        if not frames: continue
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
# === Z-점수 표준화: 평가 유틸 수정 ===
# -----------------------------
def evaluate_agent(agent, eval_video_file, run_index=0, state_mean=None, state_std=None):
    print(f"\n--- [Run {run_index+1}] Evaluation (test.mov, accuracy only) ---")
    eval_frames = load_video_frames(eval_video_file)
    return evaluate_on_frames(agent, eval_frames, device, state_mean, state_std)

def evaluate_on_frames(agent, frames, device, state_mean, state_std):
    if not frames or len(frames) < 2:
        print("Could not load evaluation frames or not enough frames to evaluate.")
        return 0.0, 0.0
    collector_eval = OfflineDataCollector(LaneDetector())
    total_reward = 0.0
    agent_actions, offline_actions = [], []

    for i in range(len(frames) - 1):
        frame = frames[i]
        next_frame = frames[i + 1]
        state, current_lane_state = collector_eval._get_state(frame)
        next_state, next_lane_state = collector_eval._get_state(next_frame)
        
        # === Z-점수 표준화: 평가 시에도 동일한 통계량으로 표준화 적용 ===
        state = (state - state_mean) / state_std

        correct_action = next_lane_state
        offline_actions.append(correct_action)
        reward, _ = collector_eval._calculate_reward(
            state, current_lane_state, next_state, next_lane_state
        )
        total_reward += reward
        with torch.no_grad():
            s_tensor = torch.from_numpy(
                np.asarray(state, dtype=np.float32)
            ).unsqueeze(0).to(device)
            q_values = agent.policy_net(s_tensor)
            action = q_values.argmax().item()
            agent_actions.append(action)
    offline_actions = np.array(offline_actions)
    agent_actions = np.array(agent_actions)
    acc = ( float((offline_actions == agent_actions).mean()) if len(offline_actions) > 0 else 0.0 )
    return acc, float(total_reward)

def frames_for_task(video_list, max_frames=None):
    if not video_list: return []
    return load_video_frames(video_list[0], max_frames=max_frames)

# -----------------------------
# === Z-점수 표준화: CL 지표 계산 함수 수정 ===
# -----------------------------
def evaluate_all_tasks(agent, train_order, task_frames_map, device, state_mean, state_std):
    row = np.zeros(len(train_order), dtype=np.float32)
    for j, tj in enumerate(train_order):
        acc, _ = evaluate_on_frames(agent, task_frames_map[tj], device, state_mean, state_std)
        row[j] = acc
    return row

def compute_cl_metrics_partial(R_rows, R_star, upto_i):
    R = np.vstack(R_rows[: upto_i + 1])
    T = R.shape[1]
    i = upto_i
    bwt_terms = [R[i, j] - R[j, j] for j in range(0, min(i, T))]
    BWT = float(np.mean(bwt_terms)) if bwt_terms else 0.0
    fwt_terms = [R[j - 1, j] - R[0, j] for j in range(1, min(i, T - 1) + 1)]
    FWT = float(np.mean(fwt_terms)) if fwt_terms else 0.0
    im_terms = [R_star[j] - R[j, j] for j in range(1, min(i, T - 1) + 1)]
    IM = float(np.mean(im_terms)) if im_terms else 0.0
    return {"BWT": BWT, "FWT": FWT, "IM": IM}

# -----------------------------
# === Z-점수 표준화: 단일 작업 성능 계산 함수 수정 ===
# -----------------------------
def single_task_best_performance(
    agent_factory, task_name, train_datasets, task_frames_map, device, epochs=20, state_mean=None, state_std=None
):
    agent = agent_factory()
    s, a, r, ns, d = train_datasets[task_name]
    
    # === Z-점수 표준화: R* 계산 시에도 표준화된 데이터 사용 ===
    s_std = (s - state_mean) / state_std
    
    agent.train_offline(s_std, a, r, ns, d, epochs=epochs, beta=5, keep_buffer=False)
    acc, _ = evaluate_on_frames(agent, task_frames_map[task_name], device, state_mean, state_std)
    return acc


# ==================================
# 메인 실행 로직
# ==================================
start_time = time.time()
DATA_FILE_T1, DATA_FILE_T6 = "task1_data.npz", "task6_data.npz"
DATA_FILE_T2, DATA_FILE_T3 = "task2_data.npz", "task3_data.npz"

video_files_t1 = ["/home/jieun/test/1_1.mov", "/home/jieun/test/1_2.mov", "/home/jieun/test/1_3.mov", "/home/jieun/test/1_4.mov",]
video_files_t6 = ["/home/jieun/test/6_1.mov", "/home/jieun/test/6_2.mov", "/home/jieun/test/6_3.mov", "/home/jieun/test/6_4.mov",]
video_files_t2 = ["/home/jieun/test/2_1.mov", "/home/jieun/test/2_2.mov", "/home/jieun/test/2_3.mov", "/home/jieun/test/2_4.mov",]
video_files_t3 = ["/home/jieun/test/3_1.mov", "/home/jieun/test/3_2.mov", "/home/jieun/test/3_3.mov", "/home/jieun/test/3_4.mov",]
eval_video_file = "/home/jieun/test/test.mov"

def load_or_create(npz_path, video_files):
    if os.path.exists(npz_path):
        print(f"Loading data from '{npz_path}'")
        data = np.load(npz_path, allow_pickle=True)
        return data["s"], data["a"], data["r"], data["ns"], data["d"]
    else:
        print(f"Creating and saving data to '{npz_path}'")
        s, a, r, ns, d = load_and_process_data(video_files)
        np.savez(npz_path, s=s, a=a, r=r, ns=ns, d=d)
        return s, a, r, ns, d

s1, a1, r1, ns1, d1 = load_or_create(DATA_FILE_T1, video_files_t1)
s6, a6, r6, ns6, d6 = load_or_create(DATA_FILE_T6, video_files_t6)
s2, a2, r2, ns2, d2 = load_or_create(DATA_FILE_T2, video_files_t2)
s3, a3, r3, ns3, d3 = load_or_create(DATA_FILE_T3, video_files_t3)
data_prep_time = time.time() - start_time
print(f"Data preparation finished. (Took {data_prep_time:.2f} seconds)")

# === Z-점수 표준화 1단계: 전체 데이터셋의 평균, 표준편차 계산 ===
print("\nCalculating statistics for Z-score standardization...")
all_states_for_stats = np.vstack([s1, s6, s2, s3])
state_mean = all_states_for_stats.mean(axis=0, dtype=np.float32)
state_std = all_states_for_stats.std(axis=0, dtype=np.float32)
# 표준편차가 0인 경우(모든 값이 같을 때) 분모가 0이 되는 것을 방지하기 위해 1로 설정
state_std[state_std == 0] = 1.0
print(f"State Mean (μ): {state_mean}")
print(f"State Std Dev (σ): {state_std}")

# --- 작업 정의 ---
train_order = ["task1", "task6", "task2", "task3"]
train_datasets = {
    "task1": (s1, a1, r1, ns1, d1), "task6": (s6, a6, r6, ns6, d6),
    "task2": (s2, a2, r2, ns2, d2), "task3": (s3, a3, r3, ns3, d3),
}
task_frames_map = {
    "task1": frames_for_task(video_files_t1), "task6": frames_for_task(video_files_t6),
    "task2": frames_for_task(video_files_t2), "task3": frames_for_task(video_files_t3),
}

def agent_factory():
    return DQNAgent( state_dim=3, action_dim=2, device=device, cql_alpha=30, lr=1e-5,
        gamma=0.99, batch_size=32, target_update_freq=10, ewc_lambda=10, )

# ----- R* 계산 (표준화 통계량 전달) -----
R_star = []
for tj in train_order:
    acc_star = single_task_best_performance(
        agent_factory=agent_factory, task_name=tj, train_datasets=train_datasets,
        task_frames_map=task_frames_map, device=device, epochs=20,
        state_mean=state_mean, state_std=state_std  # === Z-점수 표준화: 통계량 전달 ===
    )
    R_star.append(acc_star)
print("\n[R* ready] single-task best accuracies:", [f"{v:.3f}" for v in R_star])

# --- 반복 실험 ---
num_runs = 3
all_accuracies = []

for run_idx in range(num_runs):
    print(f"\n{'='*20} Experiment {run_idx+1}/{num_runs} Start {'='*20}")
    agent = agent_factory()
    R_rows = []
    
    # 미학습 성능 평가 (표준화 통계량 전달)
    R0 = evaluate_all_tasks(agent, train_order, task_frames_map, device, state_mean, state_std)
    R_rows.append(R0)
    print(f"[Stage 0] Untrained Acc per task: {[f'{v:.3f}' for v in R0]}")

    # 순차 학습 루프
    for i, task_name in enumerate(train_order):
        print(f"\n--- Training on {task_name.capitalize()} ---")
        s, a, r, ns, d = train_datasets[task_name]
        
        # === Z-점수 표준화 2단계: 학습 데이터 표준화 ===
        s_std = (s - state_mean) / state_std
        # next_state도 표준화 (참고: 여기서는 TD 타겟 계산에 사용되지 않아 영향은 없음)
        ns_std = (ns - state_mean) / state_std

        # 표준화된 데이터로 학습
        agent.train_offline(s_std, a, r, ns_std, d, epochs=20, beta=5, keep_buffer=False)
        
        # EWC를 위해 Fisher 정보 계산 (표준화된 데이터 사용)
        agent.save_params_and_fisher(list(zip(s_std, a, r, ns_std, d)), task_name)
        
        # 학습 후 성능 평가 (표준화 통계량 전달)
        Ri = evaluate_all_tasks(agent, train_order, task_frames_map, device, state_mean, state_std)
        R_rows.append(Ri)
        metrics = compute_cl_metrics_partial(R_rows, R_star, upto_i=i + 1)
        print(
            f"[After {task_name.capitalize()}] Acc per task: {[f'{v:.3f}' for v in Ri]} | "
            f"FWT={metrics['FWT']:.3f}, BWT={metrics['BWT']:.3f}, IM={metrics['IM']:.3f}"
        )

    # 최종 test.mov 평가 (표준화 통계량 전달)
    acc_only, _ = evaluate_agent(agent, eval_video_file, run_index=run_idx, state_mean=state_mean, state_std=state_std)
    print(f"=== [Run {run_idx+1}] test.mov Accuracy: {acc_only:.3f} ({acc_only*100:.1f}%) ===")
    all_accuracies.append(acc_only)

# --- 최종 결과 요약 ---
total_run_time = time.time() - start_time
print(f"\n\n{'='*20} Final Summary of {num_runs} Experiments {'='*20}")
print(f"Total time: {total_run_time:.2f} seconds")
if all_accuracies:
    mean_accuracy, std_accuracy = np.mean(all_accuracies), np.std(all_accuracies)
    print(f"Average test.mov Accuracy: {mean_accuracy:.3f} (Std Dev: {std_accuracy:.3f})")
    print("\nIndividual Accuracies:", [f"{acc:.3f}" for acc in all_accuracies])