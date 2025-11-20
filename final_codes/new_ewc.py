# -*- coding: utf-8 -*-
"""
Jetson용 DQN Lane-keeping / Obstacle Avoidance

[최종 구현 버전]
- CNN 기반 Feature Extractor
- Self-Supervised (Contrastive) Reward 방식 적용
- 순차 학습 (EWC 없음) + 데이터 캐싱 + 3회 반복 실험
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
# Device 설정
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# -----------------------------
# LaneDetector (정보 추출용)
# -----------------------------
class LaneDetector:
    def __init__(self):
        self.prev_lane = 1  # 0: Right, 1: Left
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

    def get_info(self, frame):
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
        left_half, right_half = mask_yellow[:, : width // 2], mask_yellow[:, width // 2 :]
        left_count, right_count = cv2.countNonZero(left_half), cv2.countNonZero(right_half)
        
        if left_count > right_count and left_count > 50: lane_state = 1
        elif right_count > left_count and right_count > 50: lane_state = 0
        else: lane_state = self.prev_lane
        self.prev_lane = lane_state
        
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_pixels = cv2.countNonZero(mask_red)
        red_ratio = red_pixels / float(height * width)
        
        M_red = cv2.moments(mask_red)
        red_center_x = int(M_red["m10"] / M_red["m00"]) if red_pixels > 0 and M_red["m00"] > 0 else -1
        
        bottom_center_pixel = frame[height - 1, width // 2]
        over_line = float(np.all(bottom_center_pixel > 240))
        
        return {"lane_state": lane_state, "red_ratio": red_ratio, "red_center_x": red_center_x, "over_line": over_line, "width": width}

# -----------------------------
# CNN 기반 DQN 모델
# -----------------------------
class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)

    def forward(self, x):
        # 만약 입력이 (N, H, W)라면 채널 차원 추가
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 이제 (N, 64, 7, 7)
        x = x.view(x.size(0), -1)  # (N, 3136)
        return F.relu(self.fc(x))

class DQN(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 256); self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# DQNAgent 클래스를 아래 코드로 교체해주세요.

class DQNAgent:
    def __init__(self, feature_dim, **kwargs):
        # --- 기존 파라미터는 동일 ---
        self.device = kwargs.get("device", "cpu")
        self.gamma = kwargs.get("gamma", 0.99)
        self.cql_alpha = kwargs.get("cql_alpha", 5.0)
        self.batch_size = kwargs.get("batch_size", 32)
        self.target_update_freq = kwargs.get("target_update_freq", 10)
        
        # --- EWC를 위한 파라미터 추가 ---
        self.ewc_lambda = kwargs.get("ewc_lambda", 5000.0) # EWC 페널티 강도
        self.ewc_fisher_matrix = {} # 이전 태스크 가중치의 중요도(Fisher 정보)
        self.ewc_optimal_params = {} # 이전 태스크 학습 후의 최적 가중치

        # --- 모델 및 옵티마이저는 동일 ---
        self.policy_encoder = ImageEncoder(feature_dim).to(self.device)
        self.policy_net = DQN(feature_dim, 2).to(self.device)
        self.target_encoder = ImageEncoder(feature_dim).to(self.device)
        self.target_net = DQN(feature_dim, 2).to(self.device)
        self.target_encoder.load_state_dict(self.policy_encoder.state_dict()); self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_encoder.eval(); self.target_net.eval()
        self.optimizer = torch.optim.Adam(list(self.policy_encoder.parameters()) + list(self.policy_net.parameters()), lr=kwargs.get("lr", 5e-5)) # 학습 안정성을 위해 LR 약간 감소
        self.replay_buffer = deque(maxlen=20000)

    # _update_target_net, get_action 함수는 기존과 동일

    def _update_target_net(self):
        self.target_encoder.load_state_dict(self.policy_encoder.state_dict())
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, frame, collector):
        state_tensor = torch.from_numpy(collector._preprocess_frame(frame)).float().to(self.device)
        with torch.no_grad():
            feature = self.policy_encoder(state_tensor)
            action = self.policy_net(feature).argmax().item()
        return action

    def _compute_ewc_fisher(self, dataset, num_samples=200):
        """EWC의 핵심: 이전 태스크에 대한 가중치 중요도(Fisher 정보)를 계산"""
        s_list, a_list, _, _, _ = dataset
        if len(s_list) == 0: return

        # 1. 현재 모델 가중치를 '이전 태스크의 최적 가중치'로 저장
        for n, p in self.policy_encoder.named_parameters():
            if p.requires_grad: self.ewc_optimal_params[f"enc_{n}"] = p.clone().detach()
        for n, p in self.policy_net.named_parameters():
            if p.requires_grad: self.ewc_optimal_params[f"net_{n}"] = p.clone().detach()

        # 2. Fisher 행렬 초기화
        self.ewc_fisher_matrix = {}
        for n, p in self.policy_encoder.named_parameters():
            if p.requires_grad: self.ewc_fisher_matrix[f"enc_{n}"] = torch.zeros_like(p)
        for n, p in self.policy_net.named_parameters():
            if p.requires_grad: self.ewc_fisher_matrix[f"net_{n}"] = torch.zeros_like(p)
            
        self.policy_encoder.train(); self.policy_net.train()
        self.optimizer.zero_grad()
        
        # 3. 데이터셋의 일부 샘플에 대한 그래디언트 제곱의 평균을 계산
        indices = random.sample(range(len(s_list)), min(num_samples, len(s_list)))
        for idx in indices:
            state = torch.from_numpy(s_list[idx]).float().to(self.device)
            action = torch.from_numpy(np.array(a_list[idx])).long().to(self.device)

            feature = self.policy_encoder(state)
            q_values = self.policy_net(feature)
            
            # log-softmax를 사용하여 확률 분포로 변환 후, 해당 액션의 로그 확률에 대한 Loss 계산
            log_probs = F.log_softmax(q_values, dim=-1)
            log_prob_action = log_probs[range(len(action)), action]
            
            # 그래디언트 계산
            log_prob_action.backward()

            # 그래디언트의 제곱을 누적
            for n, p in self.policy_encoder.named_parameters():
                if p.grad is not None: self.ewc_fisher_matrix[f"enc_{n}"] += p.grad.clone().pow(2)
            for n, p in self.policy_net.named_parameters():
                if p.grad is not None: self.ewc_fisher_matrix[f"net_{n}"] += p.grad.clone().pow(2)
            
            self.optimizer.zero_grad()

        # 4. 샘플 수로 나누어 평균 Fisher 정보 계산
        for n in self.ewc_fisher_matrix:
            self.ewc_fisher_matrix[n] /= len(indices)

        print("EWC Fisher matrix computed.")


    def train_offline(self, s_list, a_list, r_list, ns_list, d_list, epochs=20, beta=1.0, keep_buffer=False):
        if not keep_buffer: self.replay_buffer.clear()
        for t in zip(s_list, a_list, r_list, ns_list, d_list): self.replay_buffer.append(t)
        if not self.replay_buffer: return
        
        self.policy_encoder.train(); self.policy_net.train()
        for epoch in range(epochs):
            batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
            states, actions, rewards, next_states, dones = (torch.from_numpy(np.array(item)).to(self.device) for item in zip(*batch))
            states, actions, rewards, next_states, dones = states.float(), actions.long(), rewards.float(), next_states.float(), dones.bool()
            
            # --- [기존 Loss 계산은 동일] ---
            state_features = self.policy_encoder(states)
            with torch.no_grad():
                next_state_features = self.target_encoder(next_states)
                next_policy_features = self.policy_encoder(next_states)
                next_actions = self.policy_net(next_policy_features).argmax(1, keepdim=True)
                next_q = self.target_net(next_state_features).gather(1, next_actions).squeeze()
                target_q = rewards + self.gamma * (~dones) * next_q
            
            all_q = self.policy_net(state_features)
            current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze()
            td_loss = F.mse_loss(current_q, target_q)
            cql_penalty = (torch.logsumexp(all_q, dim=1) - current_q).mean()
            entropy_penalty = -all_q.var(dim=1).mean()
            base_loss = td_loss + self.cql_alpha * cql_penalty + beta * entropy_penalty
            
            # --- [EWC 페널티 계산 및 추가] ---
            ewc_loss = 0.0
            # ewc_fisher_matrix가 비어있지 않다면 (즉, 두 번째 태스크부터)
            if self.ewc_fisher_matrix:
                for n, p in self.policy_encoder.named_parameters():
                    if p.requires_grad:
                        fisher = self.ewc_fisher_matrix[f"enc_{n}"]
                        opt_param = self.ewc_optimal_params[f"enc_{n}"]
                        ewc_loss += (fisher * (p - opt_param).pow(2)).sum()
                for n, p in self.policy_net.named_parameters():
                    if p.requires_grad:
                        fisher = self.ewc_fisher_matrix[f"net_{n}"]
                        opt_param = self.ewc_optimal_params[f"net_{n}"]
                        ewc_loss += (fisher * (p - opt_param).pow(2)).sum()
            
            loss = base_loss + self.ewc_lambda * ewc_loss

            # --- [옵티마이저 업데이트는 동일] ---
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy_encoder.parameters()) + list(self.policy_net.parameters()), 5.0)
            self.optimizer.step()
            
            if (epoch + 1) % self.target_update_freq == 0: self._update_target_net()
            if epoch % 10 == 0: print(f"[Offline] Epoch {epoch:03d}, Loss: {loss.item():.4f}")
        
        self.policy_encoder.eval(); self.policy_net.eval()

# -----------------------------
# 데이터 수집기 (Contrastive Reward 방식)
# -----------------------------
class ContrastiveDataCollector:
    def __init__(self, lane_detector, image_size=(84, 84)):
        self.lane_detector = lane_detector
        self.image_size = image_size

    def _preprocess_frame(self, frame):
        frame = cv2.resize(frame, self.image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        return np.expand_dims(frame, axis=0)

    def _calculate_contrastive_reward(self, next_state_feature, ideal_feature, danger_feature):
        if ideal_feature is None or danger_feature is None: return 0.0
        sim_to_ideal = F.cosine_similarity(next_state_feature, ideal_feature)
        sim_to_danger = F.cosine_similarity(next_state_feature, danger_feature)
        reward = sim_to_ideal - sim_to_danger
        return reward.item()

    def collect_from_frames(self, frames, agent, ideal_feature, danger_feature):
        data = ([], [], [], [], [])
        for idx in range(len(frames) - 1):
            current_frame, next_frame = frames[idx], frames[idx + 1]
            state = self._preprocess_frame(current_frame)
            next_state = self._preprocess_frame(next_frame)
            current_info = self.lane_detector.get_info(current_frame)
            next_info = self.lane_detector.get_info(next_frame)
            action = next_info["lane_state"]
            
            with torch.no_grad():
                next_state_tensor = torch.from_numpy(next_state).float().to(device)
                next_state_feature = agent.policy_encoder(next_state_tensor)
                reward = self._calculate_contrastive_reward(next_state_feature, ideal_feature, danger_feature)

            done = False
            if next_info["over_line"] > 0.5 or next_info["red_ratio"] > 0.20:
                reward = -2.0  # 종료 시에는 코사인 유사도 범위를 벗어나는 강한 페널티
                done = True

            for lst, val in zip(data, [state, action, reward, next_state, done]):
                lst.append(val)
        return [np.array(item) for item in data]

# -----------------------------
# 데이터/평가 헬퍼
# -----------------------------
def load_video_frames(video_path, max_frames=None):
    if not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    frames, n = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame); n += 1
        if max_frames and n >= max_frames: break
    cap.release()
    return frames

def get_anchor_features(agent, paths_input, device, max_frames_per_video=100):
    """
    [Corrected Version]
    Accepts 'paths_input' (a string or a list of strings) and handles all logic internally.
    """
    paths_to_process = []

    # Check if the input is a single path (string) or a list of paths
    if isinstance(paths_input, str):
        paths_to_process = [paths_input]
    elif isinstance(paths_input, list):
        paths_to_process = paths_input

    # Load frames from all provided paths
    all_frames = []
    for path in paths_to_process:
        frames = load_video_frames(path, max_frames=max_frames_per_video)
        if frames:
            all_frames.extend(frames)

    if not all_frames:
        print(f"Warning: No frames were loaded from the provided paths: {paths_input}")
        return None
    
    # Extract features from the loaded frames
    features = []
    collector = ContrastiveDataCollector(LaneDetector())
    with torch.no_grad():
        for frame in all_frames:
            state_tensor = torch.from_numpy(collector._preprocess_frame(frame)).float().to(device)
            feature = agent.policy_encoder(state_tensor)
            features.append(feature)

    if features:
        return torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)
    
    return None

def load_or_create_contrastive_data(npz_path, video_files, agent, ideal_feature, danger_feature):
    if os.path.exists(npz_path):
        print(f"Loading data from '{npz_path}'")
        data = np.load(npz_path, allow_pickle=True)
        return {k: data[k] for k in ['s', 'a', 'r', 'ns', 'd']}
    else:
        print(f"Creating and saving data to '{npz_path}'")
        s, a, r, ns, d = [], [], [], [], []
        collector = ContrastiveDataCollector(LaneDetector())
        for vf in video_files:
            print(f"Processing video: {vf}")
            frames = load_video_frames(vf)
            if frames:
                s_i, a_i, r_i, ns_i, d_i = collector.collect_from_frames(frames, agent, ideal_feature, danger_feature)
                s.extend(s_i); a.extend(a_i); r.extend(r_i); ns.extend(ns_i); d.extend(d_i)
        print(f"Generated {len(s)} transitions.")
        np.savez(npz_path, s=s, a=a, r=r, ns=ns, d=d)
        return {'s': np.array(s), 'a': np.array(a), 'r': np.array(r), 'ns': np.array(ns), 'd': np.array(d)}

def evaluate_on_frames(agent, frames, device):
    if not frames or len(frames) < 2: return 0.0
    collector = ContrastiveDataCollector(LaneDetector())
    agent_actions, expert_actions = [], []
    for i in range(len(frames) - 1):
        next_info = collector.lane_detector.get_info(frames[i+1])
        expert_actions.append(next_info['lane_state'])
        agent_actions.append(agent.get_action(frames[i], collector))
    return np.mean(np.array(agent_actions) == np.array(expert_actions))

def evaluate_all_tasks(agent, eval_order, frames_map, device):
    return np.array([evaluate_on_frames(agent, frames_map[task], device) for task in eval_order])

def single_task_best_performance(factory, task_name, datasets, frames_map, device, epochs=40):
    agent = factory()
    s, a, r, ns, d = datasets[task_name].values()
    agent.train_offline(s, a, r, ns, d, epochs=epochs, keep_buffer=False)
    return evaluate_on_frames(agent, frames_map[task_name], device)

def compute_cl_metrics_partial(R_rows, R_star, upto_i):
    R = np.array(R_rows)
    # shape: (num_stages, num_tasks)
    n_rows, n_cols = R.shape
    n_ref = len(R_star)
    
    # 안전하게 최소 범위만큼만 계산
    n_valid = min(upto_i, n_ref, n_rows, n_cols)
    if n_valid == 0:
        return {'IM': 0.0, 'FWT': 0.0, 'BWT': 0.0}

    im_range = range(n_valid)

    # IM (Intransigence): 성능 손실 측정
    im = np.mean([R_star[j] - R[j, j] for j in im_range])

    # FWT (Forward Transfer)
    if n_valid > 1:
        fwt = np.mean([R[j - 1, j] - R_star[j] for j in range(1, n_valid)])
    else:
        fwt = 0.0

    # BWT (Backward Transfer)
    bwt = np.mean([R[-1, j] - R_star[j] for j in im_range])

    return {'IM': im, 'FWT': fwt, 'BWT': bwt}



# ==================================
# 메인 실행 로직 (EWC 적용)
# ==================================
if __name__ == "__main__":
    # ... [기존 설정은 동일: start_time, FEATURE_DIM, 파일 경로 등] ...
    start_time = time.time()
    FEATURE_DIM = 256
    base_path = "/home/jieun/test/"
    IDEAL_ANCHOR_VIDEO = os.path.join(base_path, "6_1.mov") 
    DANGER_ANCHOR_VIDEO = [
        os.path.join(base_path, "4_1.mov"), # 장애물 1개
        os.path.join(base_path, "5_1.mov") # 장애물 2개
    ]
    video_files = {
        "task1": [os.path.join(base_path, f"1_{i}.mov") for i in range(1, 3)],
        "task2": [os.path.join(base_path, f"2_{i}.mov") for i in range(1, 3)],
        "task3": [os.path.join(base_path, f"3_{i}.mov") for i in range(1, 3)]
    }
    eval_video_file = os.path.join(base_path, "test.mov")
     # --- 1. 기준(Anchor) 피쳐 생성 ---
    print("--- Generating Anchor Features ---")
    temp_agent = DQNAgent(feature_dim=FEATURE_DIM, device=device)

    ideal_feature_anchor = get_anchor_features(temp_agent, IDEAL_ANCHOR_VIDEO, device)
    danger_feature_anchor = get_anchor_features(temp_agent, DANGER_ANCHOR_VIDEO, device)

    print("Anchor features generated.")
    
    # --- 2. Contrastive Reward 기반 데이터셋 생성/로드 ---
    datasets = {name: load_or_create_contrastive_data(
        f"{name}_data_contrastive.npz", files, temp_agent, ideal_feature_anchor, danger_feature_anchor
    ) for name, files in video_files.items()}
    print(f"Data prep finished in {time.time() - start_time:.2f}s")
    del temp_agent

    
    # --- 실험 설정 (EWC 람다 추가) ---
    EVALUATION_ORDER = ["task1", "task2", "task3"]
    SEQUENTIAL_TASKS = ["task2", "task3"]
    base_task_name = "base_1"
    
    # train_datasets 생성 시 작은 오류 수정
    train_datasets = {
        base_task_name: datasets["task1"], # np.concatenate 불필요
        "task2": datasets["task2"], "task3": datasets["task3"],
    }
    task_frames_map = {name: load_video_frames(files[0]) for name, files in video_files.items() if name in EVALUATION_ORDER}
    
    # 에이전트 생성 시 ewc_lambda 전달
    agent_factory = lambda: DQNAgent(feature_dim=FEATURE_DIM, device=device, lr=5e-5, ewc_lambda=5000.0)

    # ... [R* 계산은 동일] ...
    R_star = [single_task_best_performance(agent_factory, task, datasets, task_frames_map, device, epochs=40) for task in EVALUATION_ORDER]
    print("\n[R* ready] Single-task best accuracies:", [f"{v:.3f}" for v in R_star])

    # --- 반복 실험 (EWC 적용) ---
    num_runs = 3
    all_accuracies = []
    for run_idx in range(num_runs):
        print(f"\n{'='*20} Experiment {run_idx+1}/{num_runs} Start {'='*20}")
        agent = agent_factory()
        
        R_rows = [evaluate_all_tasks(agent, EVALUATION_ORDER, task_frames_map, device)]
        print(f"[Stage 0] Untrained Acc: {[f'{v:.3f}' for v in R_rows[0]]}")

        # 5-1. 기반(Base) 모델 학습
        print(f"\n--- Training on Base Task ({base_task_name}) ---")
        s, a, r, ns, d = train_datasets[base_task_name].values()
        agent.train_offline(s, a, r, ns, d, epochs=40, keep_buffer=False)
        R_rows.append(evaluate_all_tasks(agent, EVALUATION_ORDER, task_frames_map, device))
        print(f"[After Base] Acc: {[f'{v:.3f}' for v in R_rows[-1]]}")
        
        # ✅ EWC: Base 태스크 학습 후, 가중치 중요도 계산
        agent._compute_ewc_fisher(list(train_datasets[base_task_name].values()))
        
        # 5-2. 순차 학습 (Continual Learning)
        for i, task_name in enumerate(SEQUENTIAL_TASKS):
            print(f"\n--- Continual Learning on {task_name.capitalize()} ---")
            s, a, r, ns, d = train_datasets[task_name].values()
            agent.train_offline(s, a, r, ns, d, epochs=40, keep_buffer=False)
            R_rows.append(evaluate_all_tasks(agent, EVALUATION_ORDER, task_frames_map, device))
            metrics = compute_cl_metrics_partial(R_rows, R_star, upto_i=min(i + 2, len(R_star)))
            print(f"[After {task_name.capitalize()}] Acc: {[f'{v:.3f}' for v in R_rows[-1]]} | FWT={metrics['FWT']:.3f}, BWT={metrics['BWT']:.3f}, IM={metrics['IM']:.3f}")

            # ✅ EWC: 다음 태스크를 위해 현재 태스크의 가중치 중요도 계산
            # 마지막 태스크 후에는 계산할 필요 없음
            if i < len(SEQUENTIAL_TASKS) - 1:
                agent._compute_ewc_fisher(list(train_datasets[task_name].values()))

        # ... [최종 평가 및 요약은 동일] ...

    # --- 6. 최종 결과 요약 ---
    print(f"\n\n{'='*20} Final Summary of {num_runs} Experiments {'='*20}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    if all_accuracies:
        print(f"Average test.mov Accuracy: {np.mean(all_accuracies):.3f} (Std Dev: {np.std(all_accuracies):.3f})")
        print("Individual Accuracies:", [f"{acc:.3f}" for acc in all_accuracies])