#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Orin 호환 DQN 자율주행 코드 (수정판)

- CUDA/메모리 최적화
- 8개 비디오 파일 지원 (1_1.mp4, 1_2.mp4, 4_1.mp4, 4_2.mp4, 5_1.mp4, 5_2.mp4, 6_1.mp4, 6_2.mp4)
- 메모리 사용량 모니터링 및 제한
- GPU 텐서 처리 최적화
- 성능 모니터링 유틸리티
"""

import os
import cv2
import time
import psutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# CUDA & Device
# ----------------------------
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_CACHE_DISABLE"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    prop = torch.cuda.get_device_properties(0)
    print(f"CUDA Memory (total): {prop.total_memory / 1024**3:.1f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
    print(f"Reserved : {torch.cuda.memory_reserved() / 1024**3:.3f} GB")


# ----------------------------
# Vision utils
# ----------------------------
class ObstacleDetector:
    def find_red_area(self, frame):
        """빨간색 물체 넓이(픽셀수). 없으면 0"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0

        c = max(contours, key=cv2.contourArea)
        return float(cv2.contourArea(c))


class LaneDetector:
    def __init__(self):
        self.prev_lanes = [None, None]  # [left, right]
        self.img_center = None
        self.margin = 50

    def process_frame(self, frame):
        """lanes(list[2]), lane_state(int: 0=left,1=center,2=right)"""
        h, w = frame.shape[:2]
        self.img_center = w // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 흰선/노란선 마스크
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_yellow = np.array([15, 80, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask = cv2.bitwise_or(mask_white, mask_yellow)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # ROI
        roi_mask = np.zeros_like(edges)
        roi = np.array([[(0, h), (0, int(h * 0.6)), (w, int(h * 0.6)), (w, h)]]).astype(np.int32)
        cv2.fillPoly(roi_mask, roi, 255)
        edges_roi = cv2.bitwise_and(edges, roi_mask)

        # Hough
        lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=100)

        left_lines, right_lines = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < -0.5:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0.5:
                    right_lines.append((x1, y1, x2, y2))

        # 안쪽선 선택
        left_inner = max(left_lines, key=lambda l: (l[0] + l[2]) / 2) if left_lines else None
        right_inner = min(right_lines, key=lambda l: (l[0] + l[2]) / 2) if right_lines else None

        lanes = [None, None]
        for inner, idx in [(left_inner, 0), (right_inner, 1)]:
            if inner is not None:
                x1, y1, x2, y2 = inner
                lane = [(x1, y1), (x2, y2)]
                if self.prev_lanes[idx] is not None:
                    lane = [
                        ((lane[i][0] + self.prev_lanes[idx][i][0]) // 2,
                         (lane[i][1] + self.prev_lanes[idx][i][1]) // 2)
                        for i in range(2)
                    ]
                self.prev_lanes[idx] = lane
                lanes[idx] = lane

        lane_state = 1  # center
        if lanes[0] is not None and lanes[1] is not None:
            left_center_x = (lanes[0][0][0] + lanes[0][1][0]) // 2
            right_center_x = (lanes[1][0][0] + lanes[1][1][0]) // 2
            lane_center = (left_center_x + right_center_x) // 2

            if abs(lane_center - self.img_center) < self.margin:
                lane_state = 1
            elif lane_center < self.img_center:
                lane_state = 0
            else:
                lane_state = 2

        return lanes, lane_state


# ----------------------------
# Dataset builder
# ----------------------------
class OfflineDataCollector:
    def __init__(self, lane_detector, obstacle_detector):
        self.lane_detector = lane_detector
        self.obstacle_detector = obstacle_detector

    def _get_state(self, frame, car_x):
        """state: [left_x, right_x, lane_center, car_x, red_area_ratio]"""
        area = self.obstacle_detector.find_red_area(frame)
        lanes, act = self.lane_detector.process_frame(frame)
        lanes = np.array(lanes, dtype=object)
        left_lane, right_lane = lanes

        h, w = frame.shape[:2]
        left_x = min(left_lane[0][0], left_lane[1][0]) if left_lane is not None else 0
        right_x = max(right_lane[0][0], right_lane[1][0]) if right_lane is not None else w

        state = np.array(
            [
                left_x / w,
                right_x / w,
                (left_x + right_x) / (2 * w),
                car_x / w,
                area / float(h * w),
            ],
            dtype=np.float32,
        )
        return state, int(act)

    def _calculate_reward(self, state):
        reward = 0.0
        lane_center = state[2]
        car_position = state[3]
        distance = abs(car_position - lane_center)

        # 차선 중심 유지
        if distance < 0.1:
            reward += 10.0
        elif distance < 0.2:
            reward += 5.0
        else:
            reward -= 5.0

        # 이탈 페널티
        if distance > 0.4:
            reward -= 20.0

        reward += 5.0  # 기본 주행 보상

        # 장애물(빨간 면적) 가까우면 페널티
        if state[4] > 0.7:
            reward -= 70.0

        return float(reward)

    def collect_from_frames(self, frames, car_x_init=None, actions_taken=None, batch_size=1000):
        """
        frames -> (s,a,r,s',done) list (메모리 효율 배치 처리)
        """
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []

        car_x = frames[0].shape[1] // 2 if frames else 320
        total_frames = len(frames)
        processed = 0

        print(f"총 {total_frames} 프레임에서 데이터 수집 시작...")

        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_frames = frames[batch_start:batch_end]
            print(f"배치 처리: {batch_start}-{batch_end} 프레임 ({len(batch_frames)}개)")

            # 4프레임 간격 샘플링
            valid_local_idx = [i for i in range(len(batch_frames) - 1) if i % 4 == 0]

            for local_idx in valid_local_idx:
                global_idx = batch_start + local_idx
                if global_idx + 4 >= total_frames:
                    break

                frame = batch_frames[local_idx]
                next_frame = frames[global_idx + 4]

                state, act = self._get_state(frame, car_x)
                next_state, _ = self._get_state(next_frame, car_x)
                reward = self._calculate_reward(state)

                # ---------- done 판정(수정 포인트) ----------
                done = False

                # 프레임 하단 중앙 픽셀의 밝기가 아주 밝으면(하얀 화면 등) 종료 신호로 간주
                h, w, _ = frame.shape
                bottom_center_pixel = frame[h - 1, w // 2]  # BGR 벡터 (np.array shape=(3,))
                if np.all(bottom_center_pixel > 240):
                    done = True

                # 장애물 충돌(빨간 면적 비율 매우 큼)
                if next_state[4] > 0.7:
                    done = True

                # 마지막 구간
                if global_idx + 4 >= len(frames):
                    done = True
                # --------------------------------------------

                state_list.append(state)
                action_list.append(act)         # 0/1/2
                reward_list.append(reward)
                next_state_list.append(next_state)
                done_list.append(done)

                processed += 1
                if processed % 100 == 0:
                    print(f"  처리된 transition: {processed}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"데이터 수집 완료: {len(state_list)} transition 생성")
        return state_list, action_list, reward_list, next_state_list, done_list


# ----------------------------
# DQN
# ----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


def train_offline_dqn(state_list, action_list, reward_list, next_state_list, done_list,
                      epochs=100, batch_size=32):
        print("Starting offline DQN training for Jetson Orin...")

        state_dim = len(state_list[0])
        action_dim = 3

        policy_net = DQN(state_dim, action_dim).to(device)
        target_net = DQN(state_dim, action_dim).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.Adam(policy_net.parameters(), lr=5e-4, weight_decay=1e-5)
        gamma = 0.99
        update_frequency = 10

        dataset = list(zip(state_list, action_list, reward_list, next_state_list, done_list))
        if len(dataset) < batch_size:
            batch_size = min(len(dataset), 16)
            print(f"데이터셋 크기에 맞춰 배치 크기를 {batch_size}로 조정")

        import gc

        for epoch in range(epochs):
            batch = random.sample(dataset, batch_size)

            states = torch.tensor(np.array([e[0] for e in batch]), dtype=torch.float32, device=device)
            actions = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)  # 0/1/2
            rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=device)
            next_states = torch.tensor(np.array([e[3] for e in batch]), dtype=torch.float32, device=device)
            dones = torch.tensor([e[4] for e in batch], dtype=torch.bool, device=device)

            q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target = rewards + gamma * (1 - dones.float()) * next_q

            loss = nn.MSELoss()(q, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

            if epoch % update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if torch.cuda.is_available() and epoch % 5 == 0:
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | GPU Mem {mem:.3f} GB")
                if mem > 1.5:
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Offline DQN training completed!")
        return policy_net


# ----------------------------
# Video loading
# ----------------------------
def load_video_frames(video_path, max_frames_per_video=500):
    print(f"비디오 로딩: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return []

    frames, cnt = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cnt += 1

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**3
            if mem > 1.5:
                print(f"메모리 제한으로 {cnt}프레임에서 중단")
                break

        if cnt >= max_frames_per_video:
            print(f"최대 프레임 {max_frames_per_video} 도달, 중단")
            break

    cap.release()
    print(f"  - {len(frames)} 프레임 로드됨")
    return frames


def load_all_training_videos():
    video_files = [
        "1_1.mp4", "1_2.mp4",
        "4_1.mp4", "4_2.mp4",
        "5_1.mp4", "5_2.mp4",
        "6_1.mp4", "6_2.mp4",
        "7_1.mov", "7_2.mov",
        "8_1.mov", "8_2.mov"
      
    ]

    all_frames = []
    total = 0
    print("=== 훈련 비디오 로딩 시작 ===")
    for vf in video_files:
        possible = [f"/home/nvidia/videos/{vf}", f"./{vf}", vf]
        path = next((p for p in possible if os.path.exists(p)), None)
        if path is None:
            print(f"경고: {vf} 파일을 찾을 수 없습니다.")
            continue

        frames = load_video_frames(path, max_frames_per_video=500)
        if frames:
            all_frames.extend(frames)
            total += len(frames)
            print(f"  ✓ {vf}: {len(frames)} 프레임 추가")
        else:
            print(f"  ✗ {vf}: 프레임 로드 실패")

    print(f"=== 총 {total} 프레임 로드 완료 ===")

    if not all_frames:
        print("비디오가 없어 테스트용 샘플 프레임 생성...")
        for _ in range(100):
            all_frames.append(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    return all_frames


# ----------------------------
# Perf utils
# ----------------------------
def monitor_jetson_performance():
    print("=== Jetson Orin 성능 모니터링 ===")
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU 사용률: {cpu_percent}%")

    mem = psutil.virtual_memory()
    print(f"시스템 메모리 사용률: {mem.percent}%")
    print(f"사용 가능한 메모리: {mem.available / 1024**3:.1f} GB")

    if torch.cuda.is_available():
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"GPU 메모리 예약량: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")

    # 온도(가능한 경우)
    def _read_temp(path):
        try:
            with open(path, "r") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return None

    cpu_t = _read_temp("/sys/devices/virtual/thermal/thermal_zone0/temp")
    if cpu_t is not None:
        print(f"CPU 온도: {cpu_t:.1f}°C")
    else:
        print("CPU 온도 정보를 읽을 수 없습니다.")

    gpu_t = _read_temp("/sys/devices/virtual/thermal/thermal_zone1/temp")
    if gpu_t is not None:
        print(f"GPU 온도: {gpu_t:.1f}°C")
    else:
        print("GPU 온도 정보를 읽을 수 없습니다.")


def optimize_jetson_performance():
    print("Jetson Orin 성능 최적화 적용 중...")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        print("CUDA 최적화 완료")

    cv2.setNumThreads(4)
    print("OpenCV 최적화 완료")


def cleanup_jetson_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("메모리 정리 완료")


# ----------------------------
# Main
# ----------------------------
def main():
    print("Jetson Orin DQN 자율주행 코드 시작...")

    monitor_jetson_performance()
    optimize_jetson_performance()

    # 1) 프레임 로드
    frames = load_all_training_videos()

    # 2) Detector
    lane_detector = LaneDetector()
    obstacle_detector = ObstacleDetector()

    # 3) Collector
    collector = OfflineDataCollector(lane_detector, obstacle_detector)

    # 4) 수집
    print("\n=== 데이터 수집 시작 ===")
    state_list, action_list, reward_list, next_state_list, done_list = collector.collect_from_frames(
        frames, batch_size=1000
    )

    # 5) 결과 확인
    print("\n=== 데이터 수집 결과 ===")
    print(f"총 transition 수: {len(state_list)}")
    if state_list:
        print(f"샘플 state: {state_list[0]}")
        print(f"샘플 action: {action_list[0]}")
        print(f"샘플 reward: {reward_list[0]}")
        print(f"샘플 next_state: {next_state_list[0]}")
        print(f"샘플 done: {done_list[0]}")

    # 6) 학습
    if state_list:
        print("\n=== DQN 학습 시작 ===")
        policy_net = train_offline_dqn(
            state_list, action_list, reward_list, next_state_list, done_list,
            epochs=20, batch_size=32
        )

        # 7) Q-value 테스트
        print("\n=== Q-value 테스트 시작 ===")
        test_n = min(10, len(state_list))
        for i in range(test_n):
            s = torch.tensor(state_list[i], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy_net(s).cpu().numpy()
            print(f"Sample {i+1} Q-values: {q}")
            print(f"Sample {i+1} action: {int(np.argmax(q))}")
            if torch.cuda.is_available() and i % 5 == 0:
                torch.cuda.empty_cache()
        print("Q-value 테스트 완료!")
    else:
        print("데이터가 없어 학습을 건너뜁니다.")

    # 8) 마무리
    cleanup_jetson_memory()
    print("\n=== Jetson Orin DQN 자율주행 학습 완료 ===")
    print("8개 비디오 파일로부터 학습을 완료했습니다!")


if __name__ == "__main__":
    main()



