# -*- coding: utf-8 -*-
"""
Jetson용 DQN Lane-keeping / Obstacle Avoidance
순차 학습 (EWC 없음) + 데이터 캐싱 + 10회 반복 실험 버전
[수정 사항]
- Z-점수 표준화 적용
- Task 1+6으로 Base 모델 학습 후, Task 2, 3에 대해서만 CL 지표 계산
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
# LaneDetector (변경 없음)
# -----------------------------
class LaneDetector:
    def __init__(self):
        self.prev_lane = 1
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
        left_half, right_half = mask_yellow[:, : width // 2], mask_yellow[:, width // 2 :]
        left_count, right_count = cv2.countNonZero(left_half), cv2.countNonZero(right_half)
        if left_count > right_count and left_count > 0: lane_state = 1
        elif right_count > left_count and right_count > 0: lane_state = 0
        else: lane_state = self.prev_lane
        self.prev_lane = lane_state
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_pixels = cv2.countNonZero(mask_red)
        red_ratio = red_pixels / float(height * width)
        M_yellow = cv2.moments(mask_yellow)
        yellow_center_x = int(M_yellow["m10"] / M_yellow["m00"]) if M_yellow["m00"] > 0 else -1
        M_red = cv2.moments(mask_red)
        red_center_x = int(M_red["m10"] / M_red["m00"]) if red_pixels > 0 and M_red["m00"] > 0 else -1
        red_side_relative_to_yellow = -1
        if red_ratio > 0.001 and yellow_center_x != -1:
            red_side_relative_to_yellow = 0 if red_center_x < yellow_center_x else 1
        return lane_state, red_ratio, red_side_relative_to_yellow

# -----------------------------
# OfflineDataCollector (Z-score 위해 state 원본 값 반환)
# -----------------------------
class OfflineDataCollector:
    def __init__(self, lane_detector):
        self.lane_detector = lane_detector

    def _get_state(self, frame):
        lane_state, red_ratio, red_side = self.lane_detector.process_frame(frame)
        height, width, _ = frame.shape
        bottom_center_pixel = frame[height - 1, width // 2]
        over_line = float(np.all(bottom_center_pixel > 240))
        # Z-score를 외부에서 일괄 적용하기 위해 원본 값(-1, 0, 1)을 그대로 사용
        state = np.array([over_line, float(red_side), red_ratio], dtype=np.float32)
        return state, lane_state

    def _calculate_reward(self, state, current_lane_state, next_state, next_lane_state, done=False):
        over, red_side, red_ratio = float(state[0]), int(state[1]), float(state[2])
        if next_state is None: return 0.0, True
        next_red_ratio = float(next_state[2])
        r = 3.0
        if red_ratio > 0.06 and red_side == current_lane_state:
            if next_lane_state != current_lane_state: r += 0.3
            else: r -= 0.5 * (red_ratio - 0.06) / 0.14
        if over > 0.5: r -= 1.0; done = True
        if next_red_ratio < red_ratio: r += 0.01
        return float(np.clip(r, -1.0, 1.0)), done

    def collect_from_frames(self, frames):
        data = ([], [], [], [], [])
        for idx in range(len(frames) - 1):
            state, cur_lane = self._get_state(frames[idx])
            next_state, next_lane = self._get_state(frames[idx + 1])
            reward, done = self._calculate_reward(state, cur_lane, next_state, next_lane)
            if state[0] > 0.5 or state[2] > 0.20: done = True
            for lst, val in zip(data, [state, next_lane, reward, next_state, done]):
                lst.append(val)
        return [np.array(item) for item in data]


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

    def forward(self, state,return_features=False):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        if return_features:
            return x  # CaSSLe 등 representation 용
        # Q값 계산
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.gamma = kwargs.get("gamma", 0.99)
        self.cql_alpha = kwargs.get("cql_alpha", 30)
        self.batch_size = kwargs.get("batch_size", 32)
        self.target_update_freq = kwargs.get("target_update_freq", 10)
        self.policy_net = DQN(state_dim, 2).to(self.device)
        self.target_net = DQN(state_dim, 2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=kwargs.get("lr", 1e-5))
        self.replay_buffer = deque(maxlen=20000)
    def aug(self, x, level=0.05):
        """Data Augmentation: 가우시안 노이즈 추가"""
        return x + level * torch.randn_like(x)
    def hybrid_aug_level(self, epoch, total_epochs, rep_loss, min_level=0.05, max_level=0.5):
        # ① progressive (epoch 기반)
        progress = min(epoch / (total_epochs * 0.8), 1.0) #0.7에서 0.8로 수정

        # ② adaptive (rep_loss 기반)
        rep_norm = torch.clamp(rep_loss.detach()*10, 0, 1)
        adapt_factor = 1 - rep_norm  # rep_loss가 작을수록 더 강한 augmentation

        # ③ 두 요소 결합 (곱)
        level = min_level + (max_level - min_level) * progress * adapt_factor
        return torch.clamp(level, min_level, max_level)
    def train_offline(self, s_list, a_list, r_list, ns_list, d_list, epochs=20, beta=5, keep_buffer=False):
        if not keep_buffer: self.replay_buffer.clear()
        for t in zip(s_list, a_list, r_list, ns_list, d_list): self.replay_buffer.append(t)
        if not self.replay_buffer: return
        cos = nn.CosineSimilarity(dim=-1)
        for epoch in range(epochs):
            batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
            states, actions, rewards, next_states, dones = (torch.from_numpy(np.array(item)).to(self.device) for item in zip(*batch))
            states, actions, rewards, next_states, dones = states.float(), actions.long(), rewards.float(), next_states.float(), dones.bool()
            with torch.no_grad():
                next_q = self.target_net(next_states).gather(1, self.policy_net(next_states).argmax(1, keepdim=True)).squeeze()
                target_q = rewards + self.gamma * (~dones) * next_q
            all_q = self.policy_net(states)
            current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze()
            td_loss = F.mse_loss(current_q, target_q)
            temp_level = 0.05  # 기본 노이즈 수준
            aug1_temp = self.aug(states, level=temp_level)
            aug2_temp = self.aug(states, level=temp_level)
            # 임시 feature로 rep_loss 계산
            with torch.no_grad():
                z_online_temp = self.policy_net(aug1_temp, return_features=True)
                z_target_temp = self.target_net(aug2_temp, return_features=True)
                rep_loss_temp = 1 - cos(z_online_temp, z_target_temp).mean()
            # Augmentation 2개 생성
            level = self.hybrid_aug_level(epoch, epochs, rep_loss_temp)
            aug1 = self.aug(states, level=level)
            aug2 = self.aug(states, level=level)
            z_online = self.policy_net(aug1, return_features=True)
            with torch.no_grad():
                z_target = self.target_net(aug2, return_features=True)
            rep_loss = 1 - cos(z_online, z_target).mean()
            cql_penalty = (torch.logsumexp(all_q, dim=1) - current_q).mean()
            entropy_penalty = all_q.var(dim=1).mean()
            loss = td_loss + self.cql_alpha * cql_penalty + beta * entropy_penalty + 0.05 * rep_loss #0.1에서 0.05로 수정
            self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0); self.optimizer.step()
            # if (epoch+1) % self.target_update_freq == 0: self.target_net.load_state_dict(self.policy_net.state_dict())
            if epoch % 10 == 0: print(f"[Offline] Epoch {epoch:03d}, Loss: {loss.item():.4f}")


# -----------------------------
# 데이터/평가 헬퍼 (Z-score 적용)
# -----------------------------
def load_video_frames(video_path, max_frames=None):
    if not os.path.exists(video_path): return []
    cap, frames, n = cv2.VideoCapture(video_path), [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame); n += 1
        if max_frames and n >= max_frames: break
    cap.release()
    return frames

def load_and_process_data(video_files):
    s, a, r, ns, d = ([], [], [], [], [])
    for vf in video_files:
        print(f"Processing video: {vf}")
        frames = load_video_frames(vf)
        if frames:
            s_i, a_i, r_i, ns_i, d_i = OfflineDataCollector(LaneDetector()).collect_from_frames(frames)
            s.extend(s_i); a.extend(a_i); r.extend(r_i); ns.extend(ns_i); d.extend(d_i)
    print(f"Generated {len(s)} transitions.")
    return s, a, r, ns, d

def load_or_create(npz_path, video_files):
    if os.path.exists(npz_path):
        print(f"Loading data from '{npz_path}'")
        data = np.load(npz_path, allow_pickle=True)
        return {k: data[k] for k in ['s', 'a', 'r', 'ns', 'd']}
    else:
        print(f"Creating and saving data to '{npz_path}'")
        s, a, r, ns, d = load_and_process_data(video_files)
        np.savez(npz_path, s=s, a=a, r=r, ns=ns, d=d)
        return {'s': np.array(s), 'a': np.array(a), 'r': np.array(r), 'ns': np.array(ns), 'd': np.array(d)}

def evaluate_on_frames(agent, frames, device, state_mean, state_std):
    if not frames or len(frames) < 2: return 0.0
    collector = OfflineDataCollector(LaneDetector())
    agent_actions, offline_actions = [], []
    for i in range(len(frames) - 1):
        state, _ = collector._get_state(frames[i])
        _, next_lane = collector._get_state(frames[i+1])
        state = (state - state_mean) / state_std # Z-Score 적용
        offline_actions.append(next_lane)
        with torch.no_grad():
            action = agent.policy_net(torch.from_numpy(state).float().unsqueeze(0).to(device)).argmax().item()
            agent_actions.append(action)
    return np.mean(np.array(agent_actions) == np.array(offline_actions))

def evaluate_all_tasks(agent, eval_order, frames_map, device, mean, std):
    return np.array([evaluate_on_frames(agent, frames_map[task], device, mean, std) for task in eval_order])

def compute_cl_metrics_partial(R_rows, R_star, upto_i):
    R = np.vstack(R_rows[:upto_i + 1]); i = upto_i
    bwt = np.mean([R[i, j] - R[j, j] for j in range(i)]) if i > 0 else 0.0
    fwt = np.mean([R[j-1, j] - R[0, j] for j in range(1, i + 1)]) if i > 0 else 0.0
    im = np.mean([R_star[j] - R[j, j] for j in range(i + 1)])
    return {"BWT": bwt, "FWT": fwt, "IM": im}

def single_task_best_performance(factory, task, datasets, frames_map, device, mean, std, epochs=20):
    agent = factory()
    s, a, r, ns, d = datasets[task].values()
    s_std = (s - mean) / std # Z-Score 적용
    agent.train_offline(s_std, a, r, ns, d, epochs=epochs, keep_buffer=False)
    return evaluate_on_frames(agent, frames_map[task], device, mean, std)

# ==================================
# 메인 실행 로직
# ==================================
start_time = time.time()
video_files = {
    "task1": ["/home/jieun/test/1_1.mov", "/home/jieun/test/1_2.mov", "/home/jieun/test/1_3.mov", "/home/jieun/test/1_4.mov"],
    "task2": ["/home/jieun/test/2_1.mov", "/home/jieun/test/2_2.mov", "/home/jieun/test/2_3.mov", "/home/jieun/test/2_4.mov"],
    "task3": ["/home/jieun/test/3_1.mov", "/home/jieun/test/3_2.mov", "/home/jieun/test/3_3.mov", "/home/jieun/test/3_4.mov"],
    "task6": ["/home/jieun/test/6_1.mov", "/home/jieun/test/6_2.mov", "/home/jieun/test/6_3.mov", "/home/jieun/test/6_4.mov"],
}
eval_video_file = "/home/jieun/test/test.mov"
datasets = {name: load_or_create(f"{name}_data.npz", files) for name, files in video_files.items()}
print(f"Data prep finished in {time.time() - start_time:.2f}s")

# --- Z-점수 표준화 통계량 계산 ---
all_states = np.vstack([datasets[name]['s'] for name in video_files])
state_mean = all_states.mean(axis=0, dtype=np.float32)
state_std = all_states.std(axis=0, dtype=np.float32)
state_std[state_std == 0] = 1.0
print(f"\nState Mean (μ): {state_mean}\nState Std Dev (σ): {state_std}")

# --- 실험 설정 변경 ---
EVALUATION_ORDER = ["task1", "task6", "task2", "task3"]
SEQUENTIAL_TASKS = ["task2", "task3"]
base_task_name = "base_1_6"

# 기반(Base) 태스크 데이터 통합
train_datasets = {
    base_task_name: {k: np.concatenate([datasets["task1"][k], datasets["task6"][k]]) for k in ['s','a','r','ns','d']},
    "task2": datasets["task2"], "task3": datasets["task3"],
}
task_frames_map = {name: load_video_frames(files[0]) for name, files in video_files.items() if name in EVALUATION_ORDER}
agent_factory = lambda: DQNAgent(state_dim=3, device=device, lr=1e-5)

# --- R* 계산 (평가 대상 태스크에 대해서만) ---
R_star = [single_task_best_performance(agent_factory, task, datasets, task_frames_map, device, state_mean, state_std) for task in EVALUATION_ORDER]
print("\n[R* ready] Single-task best accuracies:", [f"{v:.3f}" for v in R_star])

# --- 반복 실험 ---
num_runs = 3
all_accuracies = []
for run_idx in range(num_runs):
    print(f"\n{'='*20} Experiment {run_idx+1}/{num_runs} Start {'='*20}")
    agent = agent_factory()
    
    R_rows = [evaluate_all_tasks(agent, EVALUATION_ORDER, task_frames_map, device, state_mean, state_std)]
    print(f"[Stage 0] Untrained Acc: {[f'{v:.3f}' for v in R_rows[0]]}")

    # 1. 기반(Base) 모델 학습
    print(f"\n--- Training on Base Task ({base_task_name}) ---")
    s, a, r, ns, d = train_datasets[base_task_name].values()
    s_std = (s - state_mean) / state_std
    ns_std = (ns - state_mean) / state_std # next_state도 표준화
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.train_offline(s_std, a, r, ns_std, d, epochs=20, keep_buffer=False)
    
    R_rows.append(evaluate_all_tasks(agent, EVALUATION_ORDER, task_frames_map, device, state_mean, state_std))
    print(f"[After Base] Acc: {[f'{v:.3f}' for v in R_rows[-1]]}")
    
    # 2. Continual Learning
    for i, task_name in enumerate(SEQUENTIAL_TASKS):
        print(f"\n--- Continual Learning on {task_name.capitalize()} ---")
        s, a, r, ns, d = train_datasets[task_name].values()
        s_std = (s - state_mean) / state_std
        ns_std = (ns - state_mean) / state_std # next_state도 표준화
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.train_offline(s_std, a, r, ns_std, d, epochs=20, keep_buffer=False)
        
        R_rows.append(evaluate_all_tasks(agent, EVALUATION_ORDER, task_frames_map, device, state_mean, state_std))
        metrics = compute_cl_metrics_partial(R_rows, R_star, upto_i=i + 2) # Stage: Base(1) + CL(i+1)
        print(f"[After {task_name.capitalize()}] Acc: {[f'{v:.3f}' for v in R_rows[-1]]} | FWT={metrics['FWT']:.3f}, BWT={metrics['BWT']:.3f}, IM={metrics['IM']:.3f}")

    # 최종 평가
    acc = evaluate_on_frames(agent, load_video_frames(eval_video_file), device, state_mean, state_std)
    print(f"\n=== [Run {run_idx+1}] test.mov Accuracy: {acc:.3f} ({acc*100:.1f}%) ===")
    all_accuracies.append(acc)

# --- 최종 결과 요약 ---
print(f"\n\n{'='*20} Final Summary of {num_runs} Experiments {'='*20}")
print(f"Total time: {time.time() - start_time:.2f} seconds")
if all_accuracies:
    print(f"Average test.mov Accuracy: {np.mean(all_accuracies):.3f} (Std Dev: {np.std(all_accuracies):.3f})")
    print("Individual Accuracies:", [f"{acc:.3f}" for acc in all_accuracies])

