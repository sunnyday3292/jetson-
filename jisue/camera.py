#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson 실시간 카메라 테스트
- 학습된 DQN 모델로 실시간 판단
- 카메라 영상에서 액션 예측 (left/straight/right)
- 화면에 예측 결과 표시
"""


import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import time
from collections import deque




# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




# -----------------------------
# DQN Network (동일한 구조 필요)
# -----------------------------
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
           nn.Linear(64, action_dim)
       )


   def forward(self, x):
       return self.fc(x)




# -----------------------------
# LaneDetector (동일)
# -----------------------------
class LaneDetector:
   def __init__(self):
       self.prev_lane = "right"
       self.lower_red1 = np.array([0, 70, 50])
       self.upper_red1 = np.array([10, 255, 255])
       self.lower_red2 = np.array([170, 70, 50])
       self.upper_red2 = np.array([180, 255, 255])


   def process_frame(self, frame):
       height, width = frame.shape[:2]
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


       # 노란색 마스크
       mask_yellow = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
       left_half = mask_yellow[:, :width // 2]
       right_half = mask_yellow[:, width // 2:]
       left_count = cv2.countNonZero(left_half)
       right_count = cv2.countNonZero(right_half)


       if left_count > right_count and left_count > 0:
           lane_state = "left"
       elif right_count > left_count and right_count > 0:
           lane_state = "right"
       else:
           lane_state = self.prev_lane
       self.prev_lane = lane_state


       # 빨강 물체
       mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
       mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
       mask_red = cv2.bitwise_or(mask_red1, mask_red2)
       red_pixels = cv2.countNonZero(mask_red)
       red_ratio = red_pixels / float(height * width)


       return lane_state, red_ratio




# -----------------------------
# StateCalculator (상태 계산)
# -----------------------------
class StateCalculator:
   def __init__(self, lane_detector):
       self.lane_detector = lane_detector
       self.before_act = None
       self.prev_move_val = 0


   def get_state(self, frame):
       lane_state, red_ratio = self.lane_detector.process_frame(frame)
       act_str = 'left' if lane_state == 'left' else 'right'
       prev_act = self.before_act if self.before_act is not None else act_str


       # move 계산
       if prev_act == 'right' and act_str == 'left':
           move = 0
       elif prev_act == 'left' and act_str == 'right':
           move = 2
       else:
           move = 1


       self.before_act = act_str


       # 도로 이탈
       height, width, _ = frame.shape
       bottom_center_pixel = frame[height - 1, width // 2]
       over_line = float(np.all(bottom_center_pixel > 240))


       prev_lane_num = 0 if prev_act == 'left' else 1


       # lane_dev
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       yellow_mask = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
       left_half = yellow_mask[:, :width // 2]
       right_half = yellow_mask[:, width // 2:]
       left_count = cv2.countNonZero(left_half)
       right_count = cv2.countNonZero(right_half)
       total = left_count + right_count + 1e-6
       lane_balance = (right_count - left_count) / total
       lane_dev = float(abs(lane_balance))


       # steer_change
       steer_now = {0: -1, 1: 0, 2: +1}[move]
       steer_change = abs(steer_now - self.prev_move_val) / 2.0
       self.prev_move_val = steer_now


       state = np.array([
           red_ratio,
           over_line,
           float(prev_lane_num),
           float(move),
           lane_dev,
           steer_change
       ], dtype=np.float32)


       return state, red_ratio, lane_dev




# -----------------------------
# RealtimePredictor
# -----------------------------
class RealtimePredictor:
   def __init__(self, model_path, device='cpu'):
       self.device = device
      
       # 모델 로드
       checkpoint = torch.load(model_path, map_location=device)
       state_dim = checkpoint['state_dim']
       action_dim = checkpoint['action_dim']
      
       self.model = DQN(state_dim, action_dim).to(device)
       self.model.load_state_dict(checkpoint['policy_net_state_dict'])
       self.model.eval()
      
       print(f"✓ 모델 로드 완료: {model_path}")
       print(f"  State dim: {state_dim}, Action dim: {action_dim}")
      
       # 상태 계산기
       self.lane_detector = LaneDetector()
       self.state_calculator = StateCalculator(self.lane_detector)
      
       # 통계
       self.action_history = deque(maxlen=100)
       self.fps_history = deque(maxlen=30)
       self.prev_action = None
      
   def predict(self, frame):
       """프레임에서 액션 예측"""
       start_time = time.time()
      
       # 상태 계산
       state, red_ratio, lane_dev = self.state_calculator.get_state(frame)
      
       # 모델 예측
       with torch.no_grad():
           state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
           q_values = self.model(state_tensor)
           action = q_values.argmax(dim=1).item()
           q_vals = q_values.cpu().numpy()[0]
      
       # 액션 문자열 변환: 이전 액션과 비교
       if self.prev_action is None:
           action_str = "STRAIGHT"
       elif self.prev_action == action:
           action_str = "STRAIGHT"  # 같은 차선 유지
       elif self.prev_action == 1 and action == 0:
           action_str = "LEFT"  # 오른쪽 → 왼쪽
       elif self.prev_action == 0 and action == 1:
           action_str = "RIGHT"  # 왼쪽 → 오른쪽
       else:
           action_str = "STRAIGHT"
      
       self.prev_action = action
       self.action_history.append(action_str)
      
       # FPS 계산
       fps = 1.0 / (time.time() - start_time)
       self.fps_history.append(fps)
      
       return {
           'action': action,
           'action_str': action_str,
           'q_values': q_vals,
           'red_ratio': red_ratio,
           'lane_dev': lane_dev,
           'fps': np.mean(self.fps_history) if self.fps_history else 0
       }
  
   def get_stats(self):
       """최근 통계"""
       if not self.action_history:
           return {}
      
       actions = list(self.action_history)
       return {
           'left_ratio': actions.count('LEFT') / len(actions),
           'straight_ratio': actions.count('STRAIGHT') / len(actions),
           'right_ratio': actions.count('RIGHT') / len(actions),
       }




# -----------------------------
# 화면 표시 함수
# -----------------------------
def draw_info(frame, result, stats):
   """프레임에 정보 오버레이"""
   h, w = frame.shape[:2]
   overlay = frame.copy()
  
   # 반투명 배경
   cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
   frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
  
   # 액션 표시 (크게)
   action_str = result['action_str']
   action_color = {
       'LEFT': (0, 255, 255),      # 노란색 (왼쪽 회피)
       'STRAIGHT': (0, 255, 0),    # 초록색 (직진)
       'RIGHT': (255, 0, 255)      # 마젠타 (오른쪽 복귀)
   }[action_str]
  
   cv2.putText(frame, f"ACTION: {action_str}", (20, 50),
               cv2.FONT_HERSHEY_DUPLEX, 1.5, action_color, 3)
  
   # Q-values
   q_vals = result['q_values']
   cv2.putText(frame, f"Q[LEFT]:  {q_vals[0]:.3f}", (20, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
   cv2.putText(frame, f"Q[RIGHT]: {q_vals[1]:.3f}", (20, 115),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
   # 상태 정보
   cv2.putText(frame, f"Red Ratio: {result['red_ratio']:.4f}", (20, 145),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
   cv2.putText(frame, f"Lane Dev:  {result['lane_dev']:.4f}", (20, 165),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
  
   # FPS
   cv2.putText(frame, f"FPS: {result['fps']:.1f}", (w-150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
  
   # 통계 (우측 하단)
   if stats:
       y_offset = h - 100
       cv2.putText(frame, "Recent Actions:", (w-250, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
       cv2.putText(frame, f"  LEFT:     {stats['left_ratio']*100:.0f}%", (w-250, y_offset+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
       cv2.putText(frame, f"  STRAIGHT: {stats['straight_ratio']*100:.0f}%", (w-250, y_offset+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
       cv2.putText(frame, f"  RIGHT:    {stats['right_ratio']*100:.0f}%", (w-250, y_offset+65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
  
   # 종료 안내
   cv2.putText(frame, "Press 'q' to quit", (w-200, h-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
  
   return frame




# -----------------------------
# 카메라 열기 함수
# -----------------------------
def open_camera(camera_id=0, width=640, height=480, use_gstreamer=False, use_see3cam=False):
   """
   Jetson에서 카메라 열기
  
   Args:
       camera_id: 카메라 ID (0=기본, CSI 카메라는 특별 처리)
       width: 프레임 너비
       height: 프레임 높이
       use_gstreamer: GStreamer 파이프라인 사용 (CSI 카메라)
       use_see3cam: See3CAM_CU27 최적화 설정 사용
   """
   if use_gstreamer:
       # Jetson CSI 카메라용 GStreamer 파이프라인
       gst_str = (
           f"nvarguscamerasrc sensor-id={camera_id} ! "
           f"video/x-raw(memory:NVMM), width={width}, height={height}, "
           f"format=NV12, framerate=30/1 ! "
           f"nvvidconv flip-method=0 ! "
           f"video/x-raw, width={width}, height={height}, format=BGRx ! "
           f"videoconvert ! "
           f"video/x-raw, format=BGR ! appsink"
       )
       print(f"GStreamer 파이프라인 사용: CSI 카메라 {camera_id}")
       cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
   elif use_see3cam:
       # See3CAM_CU27 전용 설정 (V4L2 백엔드)
       print(f"See3CAM_CU27 카메라 {camera_id} 열기 (V4L2 백엔드)...")
       cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
      
       # See3CAM_CU27 최적 설정
       cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
       cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
       cap.set(cv2.CAP_PROP_FPS, 30)
       cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화 (지연 감소)
       cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPEG 코덱
      
       print(f"  해상도: {width}x{height}")
       print(f"  FPS: 30")
       print(f"  코덱: MJPEG")
       print(f"  버퍼: 최소화 (지연 감소)")
   else:
       # 일반 USB 카메라
       print(f"USB 카메라 {camera_id} 열기...")
       cap = cv2.VideoCapture(camera_id)
       cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
       cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
       cap.set(cv2.CAP_PROP_FPS, 30)
  
   if not cap.isOpened():
       raise RuntimeError(f"카메라를 열 수 없습니다 (ID: {camera_id})")
  
   print("✓ 카메라 연결 성공")
   return cap




# -----------------------------
# 메인 함수
# -----------------------------
def main():
   parser = argparse.ArgumentParser(description='Jetson 실시간 카메라 테스트')
   parser.add_argument('--model', type=str, default='./dqn_model.pth',
                       help='학습된 모델 경로')
   parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID (0=기본)')
   parser.add_argument('--width', type=int, default=640,
                       help='프레임 너비')
   parser.add_argument('--height', type=int, default=480,
                       help='프레임 높이')
   parser.add_argument('--gstreamer', action='store_true',
                       help='GStreamer 사용 (CSI 카메라)')
   parser.add_argument('--see3cam', action='store_true',
                       help='See3CAM_CU27 전용 설정 사용 (V4L2 + MJPEG)')
   parser.add_argument('--no-display', action='store_true',
                       help='화면 표시 안함 (터미널만)')
   parser.add_argument('--save-video', type=str, default=None,
                       help='비디오 저장 경로 (선택사항)')
  
   args = parser.parse_args()
  
   print("=" * 60)
   print("Jetson 실시간 카메라 테스트")
   print("=" * 60)
   print(f"모델: {args.model}")
   print(f"카메라: {args.camera}")
   print(f"해상도: {args.width}x{args.height}")
   print("=" * 60)
  
   # 모델 로드
   predictor = RealtimePredictor(args.model, device=device)
  
   # 카메라 열기
   try:
       cap = open_camera(
           camera_id=args.camera,
           width=args.width,
           height=args.height,
           use_gstreamer=args.gstreamer,
           use_see3cam=args.see3cam
       )
   except RuntimeError as e:
       print(f"오류: {e}")
       print("\n다른 방법으로 시도해보세요:")
       print("  1. USB 카메라: python3 jetson_camera_test.py --camera 0")
       print("  2. CSI 카메라: python3 jetson_camera_test.py --gstreamer --camera 0")
       print("  3. See3CAM_CU27: python3 jetson_camera_test.py --see3cam --width 1920 --height 1080")
       return
  
   # 비디오 저장 설정
   video_writer = None
   if args.save_video:
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       video_writer = cv2.VideoWriter(args.save_video, fourcc, 30.0,
                                      (args.width, args.height))
       print(f"✓ 비디오 저장: {args.save_video}")
  
   print("\n카메라 시작! ('q' 키로 종료)")
   print("-" * 60)
  
   frame_count = 0
   last_predict_time = time.time()
   last_result = None
   last_stats = None
  
   try:
       while True:
           ret, frame = cap.read()
           if not ret:
               print("프레임을 읽을 수 없습니다.")
               time.sleep(0.1)
               continue
          
           current_time = time.time()
          
           # 1초에 1번씩만 예측 (1Hz)
           if current_time - last_predict_time >= 1.0:
               # 예측
               result = predictor.predict(frame)
               stats = predictor.get_stats()
              
               last_result = result
               last_stats = stats
               last_predict_time = current_time
              
               # 터미널 출력 (1초마다)
               frame_count += 1
               print(f"[{frame_count:04d}] {result['action_str']:8s} | "
                     f"Q[L]={result['q_values'][0]:6.3f} Q[R]={result['q_values'][1]:6.3f} | "
                     f"Red={result['red_ratio']:.4f} Lane={result['lane_dev']:.4f}")
          
           # 화면 표시 (매 프레임마다, 마지막 예측 결과 사용)
           if not args.no_display and last_result is not None:
               display_frame = draw_info(frame.copy(), last_result, last_stats)
               cv2.imshow('Jetson Camera Test', display_frame)
              
               # 비디오 저장
               if video_writer:
                   video_writer.write(display_frame)
              
               # 키 입력
               key = cv2.waitKey(1) & 0xFF
               if key == ord('q'):
                   print("\n종료합니다...")
                   break
           elif not args.no_display:
               # 아직 예측 결과가 없으면 원본 프레임만 표시
               cv2.imshow('Jetson Camera Test', frame)
               key = cv2.waitKey(1) & 0xFF
               if key == ord('q'):
                   print("\n종료합니다...")
                   break
           else:
               # 화면 표시 없을 때는 Ctrl+C로 종료
               time.sleep(0.01)
  
   except KeyboardInterrupt:
       print("\n\nCtrl+C로 종료합니다...")
  
   finally:
       # 최종 통계
       print("\n" + "=" * 60)
       print("최종 통계")
       print("=" * 60)
       print(f"총 추론 횟수: {frame_count} (1초마다)")
       print(f"실행 시간: 약 {frame_count}초")
       if last_stats:
           print(f"최근 액션 분포 (최대 100회):")
           print(f"  LEFT:     {last_stats['left_ratio']*100:.1f}%")
           print(f"  STRAIGHT: {last_stats['straight_ratio']*100:.1f}%")
           print(f"  RIGHT:    {last_stats['right_ratio']*100:.1f}%")
      
       # 정리
       cap.release()
       if video_writer:
           video_writer.release()
       cv2.destroyAllWindows()
      
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
      
       print("\n완료!")




if __name__ == "__main__":
   main()







