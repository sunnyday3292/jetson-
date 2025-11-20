#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved DQN Lane-keeping / Obstacle Avoidance
Jetson Orin 버전 - 학습 + 평가 + 실시간 추론
"""


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import os
import argparse
import glob
import fnmatch
from dataclasses import dataclass
from typing import Tuple, List, Optional
import time


# -----------------------------
# Hyperparameter Configuration
# -----------------------------
@dataclass
class Config:
   # Model
   img_height: int = 84
   img_width: int = 84
   frame_stack: int = 4
   state_dim: int = 7
   action_dim: int = 3  # left, straight, right
  
   # Training
   lr: float = 3e-4
   gamma: float = 0.99
   batch_size: int = 128
   buffer_size: int = 50000
   epochs: int = 100
   updates_per_epoch: int = 50
  
   # DQN specific
   cql_alpha: float = 5.0
   target_update_freq: int = 100
   tau: float = 0.005  # Soft update
  
   # Reward weights (normalized)
   w_collision: float = -10.0
   w_lane_deviation: float = -5.0
   w_safe_distance: float = 2.0
   w_lane_center: float = 1.0
   w_smooth_steering: float = 0.5
   w_survival: float = 0.1
  
   # Data collection
   red_threshold_danger: float = 0.10
   red_threshold_warning: float = 0.05
   over_line_threshold: float = 0.5
  
   # Sampling
   prioritized_sampling: bool = True
   temperature: float = 2.0




# -----------------------------
# LaneDetector
# -----------------------------
class LaneDetector:
   def __init__(self):
       self.prev_lane = "right"  # 원본: right로 초기화
       self.lower_red1 = np.array([0, 70, 50])
       self.upper_red1 = np.array([10, 255, 255])
       self.lower_red2 = np.array([170, 70, 50])
       self.upper_red2 = np.array([180, 255, 255])


   def process_frame(self, frame):
       """원본 노트북의 차선 감지 방식 (2-lane: left/right만)"""
       height, width = frame.shape[:2]
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      
       # 노란색 차선 검출
       mask_yellow = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
       left_half = mask_yellow[:, :width // 2]
       right_half = mask_yellow[:, width // 2:]
       left_count = cv2.countNonZero(left_half)
       right_count = cv2.countNonZero(right_half)


       # 원본 방식: left vs right 비교 (center 없음)
       if left_count > right_count and left_count > 0:
           lane_state = "left"
       elif right_count > left_count and right_count > 0:
           lane_state = "right"
       else:
           lane_state = self.prev_lane  # 판단 불가 시 이전 상태 유지
       self.prev_lane = lane_state


       # 빨간색 장애물 검출
       mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
       mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
       mask_red = cv2.bitwise_or(mask_red1, mask_red2)
       red_pixels = cv2.countNonZero(mask_red)
       red_ratio = red_pixels / float(height * width)


       # 빨간색 중심 및 위치 (원본 방식: 0=left, 1=right)
       red_center_x = width // 2
       red_side = -1  # 원본: -1로 초기화
      
       if red_pixels > 0:
           M = cv2.moments(mask_red)
           if M["m00"] > 0:
               red_center_x = int(M["m10"] / M["m00"])
               if red_center_x < width // 2:
                   red_side = 0  # left
               else:
                   red_side = 1  # right
       else:
           red_center_x = width // 2  # 원본: 없으면 중앙


       # 차선 이탈 체크
       bottom_center_pixel = frame[height - 1, width // 2]
       over_line = float(np.all(bottom_center_pixel > 240))
      
       # 차선 밸런스 계산
       total = left_count + right_count + 1e-6
       lane_balance = (right_count - left_count) / total
       lane_deviation = float(abs(lane_balance))


       return {
           'lane_state': lane_state,
           'red_ratio': red_ratio,
           'red_side': red_side,
           'red_center_x': red_center_x,
           'over_line': over_line,
           'lane_deviation': lane_deviation,
           'lane_balance': lane_balance
       }




# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_frame(frame, height=84, width=84):
   """프레임을 grayscale로 변환하고 리사이즈"""
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   resized = cv2.resize(gray, (width, height))
   normalized = resized.astype(np.float32) / 255.0
   return normalized




class FrameStack:
   """프레임을 스택하여 시간적 정보 제공"""
   def __init__(self, stack_size=4, img_height=84, img_width=84):
       self.stack_size = stack_size
       self.img_height = img_height
       self.img_width = img_width
       self.frames = deque(maxlen=stack_size)
  
   def reset(self, frame):
       processed = preprocess_frame(frame, self.img_height, self.img_width)
       for _ in range(self.stack_size):
           self.frames.append(processed)
       return self.get_state()
  
   def append(self, frame):
       processed = preprocess_frame(frame, self.img_height, self.img_width)
       self.frames.append(processed)
       return self.get_state()
  
   def get_state(self):
       return np.stack(self.frames, axis=0)  # [stack_size, H, W]




# -----------------------------
# Improved Data Collector
# -----------------------------
class ImprovedDataCollector:
   def __init__(self, lane_detector, config):
       self.lane_detector = lane_detector
       self.config = config
       self.frame_stack = FrameStack(config.frame_stack, config.img_height, config.img_width)
       self.prev_action = 1  # straight


   def extract_action_from_needed_steering(self, detection_info):
       """필요한 조향 판단 (차선 편차 + 장애물 기반)
      
       전략:
       1. 장애물이 크면 회피 행동 (Left/Right)
       2. 차선 편차가 크면 보정 행동 (Left/Right)
       3. 그 외에는 직진 (Straight)
       """
       red_ratio = detection_info['red_ratio']
       lane_balance = detection_info['lane_balance']
       lane_deviation = detection_info['lane_deviation']
       red_side = detection_info['red_side']
      
       # 1. 장애물 회피 (최우선)
       if red_ratio > 0.08:  # 8% 이상 장애물
           if red_side == 0:  # 장애물이 왼쪽
               return 2  # Right로 회피
           elif red_side == 1:  # 장애물이 오른쪽
               return 0  # Left로 회피
      
       # 2. 차선 편차 보정
       # lane_balance: 음수=왼쪽 치우침, 양수=오른쪽 치우침
       if lane_deviation > 0.4:  # 40% 이상 편차
           if lane_balance < -0.3:  # 왼쪽으로 많이 치우침
               return 2  # Right로 보정
           elif lane_balance > 0.3:  # 오른쪽으로 많이 치우침
               return 0  # Left로 보정
      
       # 3. 기본: 직진
       return 1  # Straight


   def get_state(self, frame, detection_info):
       """통합 상태 추출: 이미지 + 벡터"""
       # 이미지 상태 (frame stack)
       img_state = self.frame_stack.append(frame)
      
       # 벡터 상태 (원본 방식: left=0, right=1)
       lane_to_num = {'left': 0, 'right': 1}
       lane_num = lane_to_num.get(detection_info['lane_state'], 1)  # 기본값 right=1
      
       # 정규화된 red_center_x
       red_x_norm = detection_info['red_center_x'] / frame.shape[1]
      
       vector_state = np.array([
           detection_info['red_ratio'],
           detection_info['over_line'],
           float(lane_num),  # 0 or 1 (원본 방식)
           float(self.prev_action) / 2.0,  # normalize to [0, 1]
           detection_info['lane_deviation'],
           detection_info['lane_balance'] / 2.0 + 0.5,  # normalize to [0, 1]
           float(detection_info['red_side']) if detection_info['red_side'] >= 0 else 0.5  # 0, 1, or 0.5(없음)
       ], dtype=np.float32)
      
       return img_state, vector_state


   def calculate_reward(self, detection_info, action, next_detection_info=None):
       """개선된 보상 함수: 명확한 스케일과 안전성 중심"""
       reward = 0.0
       done = False
      
       red_ratio = detection_info['red_ratio']
       over_line = detection_info['over_line']
       lane_dev = detection_info['lane_deviation']
       red_side = detection_info['red_side']
       lane_state = detection_info['lane_state']
      
       # 1. 충돌 패널티 (가장 중요)
       if red_ratio > self.config.red_threshold_danger:
           reward += self.config.w_collision * (red_ratio - self.config.red_threshold_danger)
           done = True
      
       # 2. 차선 이탈 패널티
       if over_line > self.config.over_line_threshold:
           reward += self.config.w_lane_deviation
           done = True
      
       # 3. 안전 거리 유지 보상
       if self.config.red_threshold_warning < red_ratio <= self.config.red_threshold_danger:
           # red_side: 0=left, 1=right (원본 방식)
           # 장애물과 반대 방향으로 회피하면 보상
           if red_side == 0 and action == 2:  # 장애물 왼쪽, 오른쪽으로 회피
               reward += self.config.w_safe_distance
           elif red_side == 1 and action == 0:  # 장애물 오른쪽, 왼쪽으로 회피
               reward += self.config.w_safe_distance
      
       # 4. 차선 중심 유지 보상
       reward += self.config.w_lane_center * (1.0 - lane_dev)
      
       # 5. 부드러운 조향 보상
       steering_change = abs(action - self.prev_action)
       reward += self.config.w_smooth_steering * (1.0 - steering_change / 2.0)
      
       # 6. 생존 보상
       if not done:
           reward += self.config.w_survival
      
       # 보상 클리핑
       reward = np.clip(reward, -15.0, 5.0)
      
       return reward, done


   def collect_from_frames(self, frames):
       """프레임들로부터 경험 수집"""
       experiences = []
      
       if len(frames) == 0:
           return experiences
      
       self.frame_stack.reset(frames[0])
      
       for idx in range(len(frames) - 1):
           frame = frames[idx]
           next_frame = frames[idx + 1]
          
           # 현재 프레임 정보
           detection_info = self.lane_detector.process_frame(frame)
           img_state, vec_state = self.get_state(frame, detection_info)
          
           # 다음 프레임 정보
           next_detection_info = self.lane_detector.process_frame(next_frame)
           next_img_state, next_vec_state = self.get_state(next_frame, next_detection_info)
          
           # 필요한 조향 판단 (차선 편차 + 장애물 기반)
           action = self.extract_action_from_needed_steering(detection_info)
          
           # 보상 계산
           reward, done = self.calculate_reward(
               detection_info, action, next_detection_info
           )
          
           experiences.append({
               'img_state': img_state,
               'vec_state': vec_state,
               'action': action,
               'reward': reward,
               'next_img_state': next_img_state,
               'next_vec_state': next_vec_state,
               'done': done
           })
          
           self.prev_action = action
          
           if done:
               if idx + 2 < len(frames):
                   self.frame_stack.reset(frames[idx + 2])
      
       return experiences




# -----------------------------
# 비디오 프레임 로드
# -----------------------------
def load_video_frames(video_path, max_frames=None):
   """비디오에서 프레임 추출"""
   if not os.path.exists(video_path):
       print(f"파일 없음: {video_path}")
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
   print(f"로드된 프레임: {len(frames)} from {os.path.basename(video_path)}")
   return frames




# -----------------------------
# Improved DQN Network
# -----------------------------
class ImprovedDQN(nn.Module):
   """CNN + FC를 결합한 DQN"""
   def __init__(self, img_channels, vector_dim, action_dim):
       super().__init__()
      
       # CNN for image features
       self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=8, stride=4)
       self.bn1 = nn.BatchNorm2d(32)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
       self.bn2 = nn.BatchNorm2d(64)
       self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
       self.bn3 = nn.BatchNorm2d(64)
      
       # CNN output size 계산
       def conv2d_size_out(size, kernel_size, stride):
           return (size - kernel_size) // stride + 1
      
       convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
       convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
       cnn_output_size = convw * convh * 64
      
       # FC layers for combined features
       self.fc1 = nn.Linear(cnn_output_size + vector_dim, 512)
       self.ln1 = nn.LayerNorm(512)
       self.fc2 = nn.Linear(512, 256)
       self.ln2 = nn.LayerNorm(256)
       self.fc3 = nn.Linear(256, action_dim)
      
   def forward(self, img_state, vec_state):
       # CNN forward
       x = F.relu(self.bn1(self.conv1(img_state)))
       x = F.relu(self.bn2(self.conv2(x)))
       x = F.relu(self.bn3(self.conv3(x)))
       x = x.reshape(x.size(0), -1)  # Flatten
      
       # Combine with vector state
       combined = torch.cat([x, vec_state], dim=1)
      
       # FC forward
       x = F.relu(self.ln1(self.fc1(combined)))
       x = F.relu(self.ln2(self.fc2(x)))
       q_values = self.fc3(x)
      
       return q_values




# -----------------------------
# Prioritized Replay Buffer (Class-Balanced)
# -----------------------------
class PrioritizedReplayBuffer:
   def __init__(self, capacity, alpha=0.6, num_actions=3):
       self.capacity = capacity
       self.alpha = alpha
       self.num_actions = num_actions
       self.buffer = []
       self.priorities = []
       # 클래스별로 인덱스 관리
       self.class_indices = {i: [] for i in range(num_actions)}
  
   def add(self, experience, priority=None):
       if priority is None:
           priority = max(self.priorities) if self.priorities else 1.0
      
       if len(self.buffer) >= self.capacity:
           min_idx = np.argmin(self.priorities)
           removed_action = self.buffer[min_idx]['action']
           self.class_indices[removed_action].remove(min_idx)
           self.buffer.pop(min_idx)
           self.priorities.pop(min_idx)
           # 인덱스 재정렬
           for action in range(self.num_actions):
               self.class_indices[action] = [i if i < min_idx else i-1
                                              for i in self.class_indices[action]]
      
       idx = len(self.buffer)
       self.buffer.append(experience)
       self.priorities.append(priority)
       self.class_indices[experience['action']].append(idx)
  
   def sample(self, batch_size, beta=0.4, class_balanced=True, balance_ratio=0.5):
       """
       class_balanced: True이면 클래스 균형 샘플링 적용
       balance_ratio: 균형 샘플링 비율 (0.0=완전 우선순위, 1.0=완전 균형)
       """
       if class_balanced and balance_ratio > 0:
           # 클래스별 샘플 수 계산
           class_counts = {a: len(self.class_indices[a]) for a in range(self.num_actions)}
           total_samples = sum(class_counts.values())
          
           if total_samples == 0:
               return [], [], []
          
           # 균형 샘플링 비율 적용
           balanced_samples_per_class = batch_size // self.num_actions
           priority_samples = int(batch_size * (1 - balance_ratio))
           balanced_samples = batch_size - priority_samples
          
           indices = []
          
           # 1. 우선순위 기반 샘플링
           if priority_samples > 0:
               priorities = np.array(self.priorities)
               probs = priorities ** self.alpha
               probs /= probs.sum()
               priority_indices = np.random.choice(len(self.buffer),
                                                  min(priority_samples, len(self.buffer)),
                                                  p=probs, replace=False)
               indices.extend(priority_indices.tolist())
          
           # 2. 클래스 균형 샘플링
           if balanced_samples > 0:
               samples_per_class = max(1, balanced_samples // self.num_actions)
               for action in range(self.num_actions):
                   if len(self.class_indices[action]) > 0:
                       n_samples = min(samples_per_class, len(self.class_indices[action]))
                       action_indices = np.random.choice(self.class_indices[action],
                                                        n_samples, replace=False)
                       indices.extend(action_indices.tolist())
          
           # 중복 제거
           indices = list(set(indices))[:batch_size]
          
       else:
           # 기존 우선순위 기반 샘플링
           priorities = np.array(self.priorities)
           probs = priorities ** self.alpha
           probs /= probs.sum()
           indices = np.random.choice(len(self.buffer),
                                     min(batch_size, len(self.buffer)),
                                     p=probs, replace=False)
      
       if len(indices) == 0:
           return [], [], []
      
       indices = np.array(indices)
       samples = [self.buffer[idx] for idx in indices]
      
       # Importance sampling weights
       priorities = np.array(self.priorities)
       probs = priorities ** self.alpha
       probs /= probs.sum()
       weights = (len(self.buffer) * probs[indices]) ** (-beta)
       weights /= weights.max()
      
       return samples, indices, weights
  
   def update_priorities(self, indices, priorities):
       for idx, priority in zip(indices, priorities):
           self.priorities[idx] = priority
  
   def get_class_distribution(self):
       """클래스 분포 반환"""
       return {a: len(self.class_indices[a]) for a in range(self.num_actions)}
  
   def __len__(self):
       return len(self.buffer)




# -----------------------------
# Improved DQN Agent
# -----------------------------
class ImprovedDQNAgent:
   def __init__(self, config, device):
       self.config = config
       self.device = device
      
       # Networks
       self.policy_net = ImprovedDQN(
           img_channels=config.frame_stack,
           vector_dim=config.state_dim,
           action_dim=config.action_dim
       ).to(device)
      
       self.target_net = ImprovedDQN(
           img_channels=config.frame_stack,
           vector_dim=config.state_dim,
           action_dim=config.action_dim
       ).to(device)
      
       self.target_net.load_state_dict(self.policy_net.state_dict())
       self.target_net.eval()
      
       # Optimizer
       self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
      
       # Replay buffer
       self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size, num_actions=config.action_dim)
      
       # Training stats
       self.update_counter = 0
       self.loss_history = []
      
       # 클래스 가중치 (초기값, 나중에 데이터 기반으로 업데이트)
       self.class_weights = torch.ones(config.action_dim).to(device)
  
   def add_experiences(self, experiences):
       """경험들을 버퍼에 추가 및 클래스 가중치 계산"""
       # 버퍼에 추가
       for exp in experiences:
           priority = abs(exp['reward']) + 0.1
           self.replay_buffer.add(exp, priority)
      
       # 클래스 가중치 계산 (Inverse Frequency)
       class_dist = self.replay_buffer.get_class_distribution()
       total_samples = sum(class_dist.values())
      
       if total_samples > 0:
           weights = []
           for action in range(self.config.action_dim):
               count = class_dist.get(action, 1)  # 0으로 나누기 방지
               # Inverse frequency weighting
               weight = total_samples / (self.config.action_dim * count)
               weights.append(weight)
          
           self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
          
           # 출력
           print("\n클래스 가중치 업데이트:")
           action_names = ['Left', 'Straight', 'Right']
           for i, (name, w) in enumerate(zip(action_names, weights)):
               print(f"  {name}: {w:.4f} (샘플 수: {class_dist.get(i, 0)})")


  
   def train_step(self, batch_size, beta=0.4, class_balanced=True, balance_ratio=0.5):
       """단일 학습 스텝"""
       if len(self.replay_buffer) < batch_size:
           return None
      
       # Sample batch (클래스 균형 샘플링 적용)
       batch, indices, weights = self.replay_buffer.sample(
           batch_size, beta,
           class_balanced=class_balanced,
           balance_ratio=balance_ratio
       )
      
       # 배치가 비어있으면 None 반환
       if len(batch) == 0:
           return None
      
       # Prepare tensors
       img_states = torch.tensor(
           np.array([b['img_state'] for b in batch]), dtype=torch.float32
       ).to(self.device)
      
       vec_states = torch.tensor(
           np.array([b['vec_state'] for b in batch]), dtype=torch.float32
       ).to(self.device)
      
       actions = torch.tensor(
           [b['action'] for b in batch], dtype=torch.long
       ).to(self.device)
      
       rewards = torch.tensor(
           [b['reward'] for b in batch], dtype=torch.float32
       ).to(self.device)
      
       next_img_states = torch.tensor(
           np.array([b['next_img_state'] for b in batch]), dtype=torch.float32
       ).to(self.device)
      
       next_vec_states = torch.tensor(
           np.array([b['next_vec_state'] for b in batch]), dtype=torch.float32
       ).to(self.device)
      
       dones = torch.tensor(
           [b['done'] for b in batch], dtype=torch.bool
       ).to(self.device)
      
       weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
      
       # Current Q values
       current_q = self.policy_net(img_states, vec_states).gather(1, actions.unsqueeze(1)).squeeze(1)
      
       # Double DQN target
       with torch.no_grad():
           next_actions = self.policy_net(next_img_states, next_vec_states).argmax(dim=1, keepdim=True)
           next_q = self.target_net(next_img_states, next_vec_states).gather(1, next_actions).squeeze(1)
           target_q = rewards + self.config.gamma * (1 - dones.float()) * next_q
      
       # TD loss with importance sampling and class weights
       td_errors = (current_q - target_q).abs()
      
       # 클래스 가중치 적용
       action_weights = self.class_weights[actions]
       weighted_td_loss = weights_tensor * action_weights * F.mse_loss(current_q, target_q, reduction='none')
       td_loss = weighted_td_loss.mean()
      
       # CQL penalty
       all_q = self.policy_net(img_states, vec_states)
       logsumexp_q = torch.logsumexp(all_q, dim=1)
       cql_penalty = (logsumexp_q - current_q).mean()
      
       # Total loss
       loss = td_loss + self.config.cql_alpha * cql_penalty
      
       # Optimize
       self.optimizer.zero_grad()
       loss.backward()
       torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
       self.optimizer.step()
      
       # Update priorities
       new_priorities = td_errors.detach().cpu().numpy() + 1e-6
       self.replay_buffer.update_priorities(indices, new_priorities)
      
       # Soft update target network
       self.update_counter += 1
       if self.update_counter % self.config.target_update_freq == 0:
           self.soft_update_target()
      
       self.loss_history.append(loss.item())
      
       return {
           'loss': loss.item(),
           'td_loss': td_loss.item(),
           'cql_penalty': cql_penalty.item(),
           'mean_q': current_q.mean().item()
       }
  
   def soft_update_target(self):
       """Soft update of target network"""
       for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
           target_param.data.copy_(
               self.config.tau * policy_param.data + (1 - self.config.tau) * target_param.data
           )
  
   def train_offline(self, epochs, updates_per_epoch, class_balanced=True, balance_ratio=0.7):
       """Offline 학습 루프
      
       Args:
           class_balanced: 클래스 균형 샘플링 사용 여부
           balance_ratio: 균형 샘플링 비율 (0.0=완전 우선순위, 1.0=완전 균형)
       """
       self.policy_net.train()
      
       for epoch in range(epochs):
           epoch_stats = {
               'loss': [],
               'td_loss': [],
               'cql_penalty': [],
               'mean_q': []
           }
          
           # Beta annealing
           beta = min(1.0, 0.4 + epoch / epochs * 0.6)
          
           for _ in range(updates_per_epoch):
               stats = self.train_step(self.config.batch_size, beta, class_balanced, balance_ratio)
               if stats:
                   for key in epoch_stats:
                       epoch_stats[key].append(stats[key])
          
           # Print epoch stats
           if epoch % 5 == 0 or epoch == epochs - 1:
               print(f"[Epoch {epoch:03d}] "
                     f"Loss: {np.mean(epoch_stats['loss']):.4f}, "
                     f"TD: {np.mean(epoch_stats['td_loss']):.4f}, "
                     f"CQL: {np.mean(epoch_stats['cql_penalty']):.4f}, "
                     f"Q: {np.mean(epoch_stats['mean_q']):.4f}")
  
   def select_action(self, img_state, vec_state):
       """행동 선택 (greedy)"""
       self.policy_net.eval()
       with torch.no_grad():
           img_tensor = torch.tensor(img_state, dtype=torch.float32).unsqueeze(0).to(self.device)
           vec_tensor = torch.tensor(vec_state, dtype=torch.float32).unsqueeze(0).to(self.device)
           q_values = self.policy_net(img_tensor, vec_tensor)
           action = q_values.argmax(dim=1).item()
       return action, q_values.cpu().numpy().flatten()
  
   def save_model(self, path):
       """모델 저장"""
       torch.save({
           'policy_net': self.policy_net.state_dict(),
           'target_net': self.target_net.state_dict(),
           'optimizer': self.optimizer.state_dict(),
           'config': self.config
       }, path)
       print(f"모델 저장: {path}")
  
   def load_model(self, path):
       """모델 로드"""
       checkpoint = torch.load(path, map_location=self.device)
       self.policy_net.load_state_dict(checkpoint['policy_net'])
       self.target_net.load_state_dict(checkpoint['target_net'])
       self.optimizer.load_state_dict(checkpoint['optimizer'])
       print(f"모델 로드: {path}")




# -----------------------------
# 평가 함수
# -----------------------------
class EvaluationMetrics:
   def __init__(self):
       self.reset()
  
   def reset(self):
       self.total_frames = 0
       self.collision_count = 0
       self.lane_deviation_count = 0
       self.safe_frames = 0
       self.total_reward = 0.0
       self.action_distribution = {0: 0, 1: 0, 2: 0}
  
   def update(self, reward, action, detection_info, config):
       self.total_frames += 1
       self.total_reward += reward
       self.action_distribution[action] += 1
      
       # 충돌 체크
       if detection_info['red_ratio'] > config.red_threshold_danger:
           self.collision_count += 1
      
       # 차선 이탈 체크
       if detection_info['over_line'] > config.over_line_threshold:
           self.lane_deviation_count += 1
      
       # 안전 프레임
       if (detection_info['red_ratio'] < config.red_threshold_warning and
           detection_info['over_line'] < config.over_line_threshold):
           self.safe_frames += 1
  
   def print_summary(self):
       print("\n" + "="*60)
       print("평가 결과 요약")
       print("="*60)
       print(f"총 프레임: {self.total_frames}")
       print(f"충돌 횟수: {self.collision_count} ({self.collision_count/max(1,self.total_frames)*100:.2f}%)")
       print(f"차선 이탈: {self.lane_deviation_count} ({self.lane_deviation_count/max(1,self.total_frames)*100:.2f}%)")
       print(f"안전 프레임: {self.safe_frames} ({self.safe_frames/max(1,self.total_frames)*100:.2f}%)")
       print(f"총 보상: {self.total_reward:.2f}")
       print(f"평균 보상: {self.total_reward/max(1,self.total_frames):.4f}")
       print(f"\n행동 분포:")
       for action, count in self.action_distribution.items():
           action_name = ['Left', 'Straight', 'Right'][action]
           percentage = count / max(1, self.total_frames) * 100
           print(f"  {action_name}: {count} ({percentage:.1f}%)")
       print("="*60)




def evaluate_agent(agent, eval_video_path, config, verbose=False):
   """에이전트 평가"""
   frames = load_video_frames(eval_video_path)
   if len(frames) == 0:
       print("평가할 프레임이 없습니다.")
       return None
  
   lane_detector = LaneDetector()
   frame_stack = FrameStack(config.frame_stack, config.img_height, config.img_width)
   frame_stack.reset(frames[0])
   metrics = EvaluationMetrics()
   prev_action = 1
  
   # Q값 통계 수집
   q_stats = {'left': [], 'straight': [], 'right': []}
   action_changes = []
  
   for idx, frame in enumerate(frames):
       detection_info = lane_detector.process_frame(frame)
       img_state = frame_stack.append(frame)
      
       # 원본 방식: left=0, right=1
       lane_to_num = {'left': 0, 'right': 1}
       lane_num = lane_to_num.get(detection_info['lane_state'], 1)
       red_x_norm = detection_info['red_center_x'] / frame.shape[1]
      
       vec_state = np.array([
           detection_info['red_ratio'],
           detection_info['over_line'],
           float(lane_num),  # 0 or 1
           float(prev_action) / 2.0,
           detection_info['lane_deviation'],
           detection_info['lane_balance'] / 2.0 + 0.5,
           float(detection_info['red_side']) if detection_info['red_side'] >= 0 else 0.5
       ], dtype=np.float32)
      
       action, q_values = agent.select_action(img_state, vec_state)
      
       # Q값 저장
       q_stats['left'].append(q_values[0])
       q_stats['straight'].append(q_values[1])
       q_stats['right'].append(q_values[2])
      
       # 행동 변화 추적
       if action != prev_action:
           action_changes.append(idx)
      
       # Verbose 모드: 첫 10프레임과 마지막 10프레임 출력
       if verbose and (idx < 10 or idx >= len(frames) - 10):
           action_name = ['Left', 'Straight', 'Right'][action]
           print(f"[Frame {idx:3d}] Action: {action_name:8s} | "
                 f"Q: L={q_values[0]:.3f} S={q_values[1]:.3f} R={q_values[2]:.3f} | "
                 f"Lane: {detection_info['lane_state']:6s} | Red: {detection_info['red_ratio']:.3f}")
      
       prev_action = action
      
       collector = ImprovedDataCollector(lane_detector, config)
       reward, _ = collector.calculate_reward(detection_info, action)
      
       metrics.update(reward, action, detection_info, config)
  
   metrics.print_summary()
  
   # Q값 통계 출력
   print("\n" + "="*60)
   print("Q값 통계")
   print("="*60)
   for action_name, q_list in q_stats.items():
       print(f"{action_name.capitalize():8s}: 평균={np.mean(q_list):.4f}, "
             f"표준편차={np.std(q_list):.4f}, "
             f"최소={np.min(q_list):.4f}, "
             f"최대={np.max(q_list):.4f}")
  
   print(f"\n행동 변화 횟수: {len(action_changes)}회")
   print("="*60)
  
   return metrics




# -----------------------------
# 실시간 카메라 추론
# -----------------------------
def run_realtime_inference(agent, config, camera_id=0, display=True):
   """실시간 카메라로 추론"""
   print(f"카메라 {camera_id} 시작...")
   cap = cv2.VideoCapture(camera_id)
  
   if not cap.isOpened():
       print(f"카메라 {camera_id}를 열 수 없습니다.")
       return
  
   lane_detector = LaneDetector()
   frame_stack = FrameStack(config.frame_stack, config.img_height, config.img_width)
  
   # 첫 프레임으로 초기화
   ret, frame = cap.read()
   if not ret:
       print("첫 프레임을 읽을 수 없습니다.")
       cap.release()
       return
  
   frame_stack.reset(frame)
   prev_action = 1
  
   print("실시간 추론 시작 (q를 눌러 종료)")
  
   try:
       while True:
           ret, frame = cap.read()
           if not ret:
               break
          
           # 상태 추출
           detection_info = lane_detector.process_frame(frame)
           img_state = frame_stack.append(frame)
          
           lane_to_num = {'left': 0, 'center': 1, 'right': 2}
           lane_num = lane_to_num[detection_info['lane_state']]
           red_x_norm = detection_info['red_center_x'] / frame.shape[1]
           red_y_norm = detection_info['red_center_y'] / frame.shape[0]
          
           vec_state = np.array([
               detection_info['red_ratio'],
               detection_info['over_line'],
               float(lane_num) / 2.0,
               float(prev_action) / 2.0,
               detection_info['lane_deviation'],
               red_x_norm,
               red_y_norm
           ], dtype=np.float32)
          
           # 행동 선택
           start_time = time.time()
           action, q_values = agent.select_action(img_state, vec_state)
           inference_time = (time.time() - start_time) * 1000  # ms
           prev_action = action
          
           # 결과 표시
           action_name = ['LEFT', 'STRAIGHT', 'RIGHT'][action]
          
           if display:
               # 프레임에 정보 오버레이
               display_frame = frame.copy()
               cv2.putText(display_frame, f"Action: {action_name}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
               cv2.putText(display_frame, f"Lane: {detection_info['lane_state']}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
               cv2.putText(display_frame, f"Red: {detection_info['red_ratio']:.3f}", (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               cv2.putText(display_frame, f"FPS: {1000/inference_time:.1f}", (10, 150),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
              
               # Q값 표시
               for i, q in enumerate(q_values):
                   q_text = ['Q[L]', 'Q[S]', 'Q[R]'][i]
                   cv2.putText(display_frame, f"{q_text}: {q:.2f}", (10, 190 + i*30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
              
               cv2.imshow('DQN Lane Keeping', display_frame)
          
           # 터미널 출력
           print(f"\rAction: {action_name:8s} | Lane: {detection_info['lane_state']:6s} | "
                 f"Red: {detection_info['red_ratio']:.3f} | FPS: {1000/inference_time:.1f}", end='')
          
           # 종료 체크
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
              
   except KeyboardInterrupt:
       print("\n\n추론 종료")
   finally:
       cap.release()
       cv2.destroyAllWindows()




# -----------------------------
# Main 함수
# -----------------------------
def main():
   parser = argparse.ArgumentParser(description='DQN Lane Keeping for Jetson Orin')
   parser.add_argument('--mode', type=str, choices=['train', 'eval', 'realtime'], required=True,
                      help='실행 모드: train(학습), eval(평가), realtime(실시간)')
   parser.add_argument('--video-dir', type=str, default='./videos',
                      help='학습용 비디오 디렉토리')
   parser.add_argument('--train-pattern', type=str, default='*',
                      help='학습에 포함할 파일 패턴 (예: "1_*.mov,4_*.mov")')
   parser.add_argument('--exclude-pattern', type=str, default='',
                      help='학습에서 제외할 파일 패턴 (예: "2_*.mov,3_*.mov")')
   parser.add_argument('--eval-video', type=str, default=None,
                      help='평가용 비디오 파일 경로 (쉼표로 여러 파일 구분 가능)')
   parser.add_argument('--model-path', type=str, default='./dqn_model.pth',
                      help='모델 저장/로드 경로')
   parser.add_argument('--camera-id', type=int, default=0,
                      help='카메라 디바이스 ID (실시간 모드)')
   parser.add_argument('--epochs', type=int, default=100,
                      help='학습 에포크 수')
   parser.add_argument('--batch-size', type=int, default=128,
                      help='배치 크기')
   parser.add_argument('--no-display', action='store_true',
                      help='실시간 모드에서 화면 표시 안함')
   parser.add_argument('--verbose', action='store_true',
                      help='평가 모드에서 상세 정보 출력')
  
   args = parser.parse_args()
  
   # Config 설정
   config = Config()
   config.epochs = args.epochs
   config.batch_size = args.batch_size
  
   # Device 설정
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Device: {device}")
   if torch.cuda.is_available():
       print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
  
   # Agent 생성
   agent = ImprovedDQNAgent(config, device)
  
   # 모드별 실행
   if args.mode == 'train':
       print("\n=== 학습 모드 ===")
      
       # 비디오 파일 로드
       all_video_files = glob.glob(os.path.join(args.video_dir, '*.mov'))
       all_video_files += glob.glob(os.path.join(args.video_dir, '*.mp4'))
      
       if len(all_video_files) == 0:
           print(f"비디오 파일이 없습니다: {args.video_dir}")
           return
      
       # 파일 필터링
       video_files = []
       excluded_files = []
      
       # exclude-pattern 처리
       exclude_patterns = [p.strip() for p in args.exclude_pattern.split(',') if p.strip()]
      
       for video_file in all_video_files:
           basename = os.path.basename(video_file)
          
           # 제외 패턴 체크
           should_exclude = False
           for pattern in exclude_patterns:
               if fnmatch.fnmatch(basename, pattern):
                   should_exclude = True
                   excluded_files.append(basename)
                   break
          
           if not should_exclude:
               video_files.append(video_file)
      
       print(f"총 비디오 파일: {len(all_video_files)}개")
       if excluded_files:
           print(f"\n제외된 파일: {len(excluded_files)}개")
           for ef in sorted(excluded_files):
               print(f"  - {ef}")
      
       print(f"\n학습에 사용할 파일: {len(video_files)}개")
       for vf in sorted(video_files):
           print(f"  + {os.path.basename(vf)}")
      
       if len(video_files) == 0:
           print("학습에 사용할 비디오 파일이 없습니다.")
           return
      
       # 데이터 수집
       lane_detector = LaneDetector()
       collector = ImprovedDataCollector(lane_detector, config)
      
       all_experiences = []
       action_counts = {0: 0, 1: 0, 2: 0}
      
       for vf in video_files:
           print(f"\n처리 중: {vf}")
           frames = load_video_frames(vf)
           experiences = collector.collect_from_frames(frames)
           all_experiences.extend(experiences)
          
           for exp in experiences:
               action_counts[exp['action']] += 1
      
       print(f"\n총 수집된 transitions: {len(all_experiences)}")
       print(f"\n행동 분포:")
       for action, count in action_counts.items():
           action_name = ['Left', 'Straight', 'Right'][action]
           print(f"  {action_name}: {count} ({count/len(all_experiences)*100:.1f}%)")
      
       # 버퍼에 추가
       print("\n데이터를 replay buffer에 추가 중...")
       agent.add_experiences(all_experiences)
       print(f"Replay buffer 크기: {len(agent.replay_buffer)}")
      
       # 학습
       print("\n학습 시작...")
       agent.train_offline(config.epochs, config.updates_per_epoch)
      
       # 모델 저장
       agent.save_model(args.model_path)
      
   elif args.mode == 'eval':
       print("\n=== 평가 모드 ===")
      
       # 모델 로드
       if not os.path.exists(args.model_path):
           print(f"모델 파일이 없습니다: {args.model_path}")
           return
      
       agent.load_model(args.model_path)
      
       # 평가 비디오
       if args.eval_video is None:
           print("--eval-video 인자가 필요합니다.")
           return
      
       # 여러 비디오 처리
       eval_videos = [v.strip() for v in args.eval_video.split(',')]
      
       all_metrics = []
       for video_path in eval_videos:
           if not os.path.exists(video_path):
               print(f"\n평가 비디오가 없습니다: {video_path}")
               continue
          
           print(f"\n{'='*60}")
           print(f"평가 비디오: {video_path}")
           print('='*60)
           metrics = evaluate_agent(agent, video_path, config, verbose=args.verbose)
           if metrics:
               all_metrics.append({
                   'video': video_path,
                   'metrics': metrics
               })
      
       # 여러 비디오 평가 시 요약
       if len(all_metrics) > 1:
           print("\n" + "="*60)
           print("전체 평가 요약")
           print("="*60)
          
           total_frames = sum(m['metrics'].total_frames for m in all_metrics)
           total_collisions = sum(m['metrics'].collision_count for m in all_metrics)
           total_lane_deviations = sum(m['metrics'].lane_deviation_count for m in all_metrics)
           total_safe_frames = sum(m['metrics'].safe_frames for m in all_metrics)
           total_reward = sum(m['metrics'].total_reward for m in all_metrics)
          
           combined_actions = {0: 0, 1: 0, 2: 0}
           for m in all_metrics:
               for action, count in m['metrics'].action_distribution.items():
                   combined_actions[action] += count
          
           print(f"총 비디오 수: {len(all_metrics)}개")
           print(f"총 프레임: {total_frames}")
           print(f"평균 충돌률: {total_collisions/total_frames*100:.2f}%")
           print(f"평균 차선 이탈률: {total_lane_deviations/total_frames*100:.2f}%")
           print(f"평균 안전률: {total_safe_frames/total_frames*100:.2f}%")
           print(f"평균 보상: {total_reward/total_frames:.4f}")
          
           print(f"\n전체 행동 분포:")
           for action, count in combined_actions.items():
               action_name = ['Left', 'Straight', 'Right'][action]
               percentage = count / total_frames * 100
               print(f"  {action_name}: {count} ({percentage:.1f}%)")
           print("="*60)
      
   elif args.mode == 'realtime':
       print("\n=== 실시간 추론 모드 ===")
      
       # 모델 로드
       if not os.path.exists(args.model_path):
           print(f"모델 파일이 없습니다: {args.model_path}")
           return
      
       agent.load_model(args.model_path)
      
       # 실시간 추론
       run_realtime_inference(agent, config, args.camera_id, display=not args.no_display)




if __name__ == "__main__":
   main()







