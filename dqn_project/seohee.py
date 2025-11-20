#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson과 아두이노 RC카 연결 코드

L298N 모터 드라이버와 엔코더를 사용하는 RC카를 Jetson에서 제어합니다.
"""

import serial
import time
import threading
import queue
import cv2
import numpy as np
import torch
import json
import os

# OpenCV GUI 비활성화
os.environ["OPENCV_VIDEOIO_PRIORITY_V4L2"] = "1"
cv2.setNumThreads(1)

# DQN 모듈 import (jetson_orin_dqn_updated 대신 dqn.py에서 직접 import)
try:
    from dqn import LaneDetector, DQN
except ImportError:
    print("dqn.py에서 모듈을 찾을 수 없습니다.")


class ArduinoRCCarController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600):
        """
        아두이노 RC카 컨트롤러 초기화

        Args:
            port: 아두이노 시리얼 포트 (예: /dev/ttyUSB0, /dev/ttyACM0)
            baudrate: 시리얼 통신 속도
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False

        # RC카 파라미터 (아두이노 코드와 동일)
        self.WHEEL_DIAMETER_CM = 6.5
        self.COUNTS_PER_REV = 20
        self.DRIVE_PWM = 180
        self.STOP_BRAKE_TIME_MS = 120

        # 현재 상태
        self.current_speed = 0
        self.current_steering = 0
        self.left_count = 0
        self.right_count = 0

        # 명령 큐
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()

    def connect(self):
        """아두이노와 시리얼 연결"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=1
            )
            time.sleep(2)  # 아두이노 초기화 대기
            self.connected = True
            print(f"아두이노 연결 성공: {self.port}")

            # 통신 스레드 시작
            self.start_communication_thread()
            return True

        except Exception as e:
            print(f"아두이노 연결 실패: {e}")
            return False

    def start_communication_thread(self):
        """통신 스레드 시작"""
        self.comm_thread = threading.Thread(target=self.communication_loop)
        self.comm_thread.daemon = True
        self.comm_thread.start()

    def communication_loop(self):
        """시리얼 통신 루프"""
        while self.connected:
            try:
                # 명령 전송
                if not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    self.serial_conn.write(command.encode("utf-8"))
                    self.serial_conn.write(b"\n")

                # 응답 수신
                if self.serial_conn.in_waiting > 0:
                    response = self.serial_conn.readline().decode("utf-8").strip()
                    if response:
                        self.response_queue.put(response)

            except Exception as e:
                print(f"통신 오류: {e}")
                break

            time.sleep(0.01)

    def send_command(self, command_type, **kwargs):
        """아두이노에 명령 전송"""
        if not self.connected:
            return False

        command = {"type": command_type, **kwargs}

        try:
            self.command_queue.put(json.dumps(command))
            return True
        except Exception as e:
            print(f"명령 전송 실패: {e}")
            return False

    def move_forward(self, speed=None, distance_cm=None):
        """전진"""
        if speed is None:
            speed = self.DRIVE_PWM

        if distance_cm:
            # 특정 거리만큼 전진
            return self.send_command("move_forward", speed=speed, distance=distance_cm)
        else:
            # 계속 전진
            return self.send_command("move_forward", speed=speed)

    def move_backward(self, speed=None, distance_cm=None):
        """후진"""
        if speed is None:
            speed = self.DRIVE_PWM

        if distance_cm:
            return self.send_command("move_backward", speed=speed, distance=distance_cm)
        else:
            return self.send_command("move_backward", speed=speed)

    def turn_left(self, angle_degrees=None):
        """좌회전"""
        if angle_degrees:
            return self.send_command("turn_left", angle=angle_degrees)
        else:
            return self.send_command("turn_left")

    def turn_right(self, angle_degrees=None):
        """우회전"""
        if angle_degrees:
            return self.send_command("turn_right", angle=angle_degrees)
        else:
            return self.send_command("turn_right")

    def stop(self):
        """정지"""
        return self.send_command("stop")

    def get_encoder_data(self):
        """엔코더 데이터 요청"""
        return self.send_command("get_encoders")

    def set_speed(self, left_speed, right_speed):
        """좌우 모터 속도 개별 설정"""
        return self.send_command("set_speed", left=left_speed, right=right_speed)

    def get_status(self):
        """상태 정보 요청"""
        return self.send_command("get_status")


class JetsonAutonomousController:
    def __init__(self, arduino_port="/dev/ttyUSB0"):
        """Jetson 자율주행 컨트롤러"""
        # 아두이노 RC카 컨트롤러
        self.rc_car = ArduinoRCCarController(port=arduino_port)

        # 컴퓨터 비전 모듈
        self.lane_detector = LaneDetector()

        # DQN 모델
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(5, 2).to(self.device)
        self.policy_net.eval()

        # See3CAM_CU27 카메라 설정 (GUI 없이 실행)
        self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)  # V4L2 백엔드 사용
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # See3CAM_CU27 최대 해상도
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

        # GUI 없이 실행을 위한 설정
        self.headless_mode = True  # GUI 없이 실행

        # OpenCV 백엔드 설정 (GUI 완전 비활성화)
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)

        # 실제 설정값 확인
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"See3CAM_CU27 설정: {actual_width}x{actual_height}")

        # 실행 상태
        self.running = False
        self.autonomous_mode = False

        # 액션 매핑
        self.action_map = {
            0: self.action_left,  # 좌회전
            1: self.action_center,  # 직진
            2: self.action_right,  # 우회전
        }

    def action_left(self):
        """좌회전 액션"""
        self.rc_car.turn_left(30)
        time.sleep(0.5)  # 회전 시간
        self.rc_car.move_forward(150)  # 회전 후 전진

    def action_center(self):
        """직진 액션"""
        self.rc_car.move_forward(180)

    def action_right(self):
        """우회전 액션"""
        self.rc_car.turn_right(30)
        time.sleep(0.5)  # 회전 시간
        self.rc_car.move_forward(150)  # 회전 후 전진

    def get_state(self, frame):
        """
        프레임에서 상태(state)를 계산하고 move를 결정.
        state = [red_ratio, over_line, prev_lane_num, move]
        """
        # 현재 차선과 빨강 비율
        lane_state, red_ratio = self.lane_detector.process_frame(frame)
        act_str = "left" if lane_state == "left" else "right"

        # 이전 차선 기반 move 계산
        prev_act = self.before_act if self.before_act is not None else act_str

        if prev_act == "right" and act_str == "left":
            move = 0  # 좌회전
        elif prev_act == "left" and act_str == "right":
            move = 2  # 우회전
        else:
            move = 1  # 직진

        # 현재 차선을 저장
        self.before_act = act_str

        # 바닥 픽셀 over_line 판단
        height, width, _ = frame.shape
        bottom_center_pixel = frame[height - 1, width // 2]
        over_line = float(np.all(bottom_center_pixel > 240))

        # 이전 차선 숫자로 변환: left=0, right=1
        prev_lane_num = 0 if prev_act == "left" else 1

        state = np.array([red_ratio, over_line, prev_lane_num, move], dtype=np.float32)

        return state

    def autonomous_control_loop(self):
        """자율주행 제어 루프"""
        while self.running and self.autonomous_mode:
            move_list = []
            try:
                ret, frame = self.cap.read()
                if ret:
                    # 상태 추출
                    state = self.get_state(frame)

                    # DQN으로 액션 결정
                    with torch.no_grad():
                        state_tensor = torch.tensor(
                            state, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        q_values = self.policy_net(state_tensor)
                        move = q_values.argmax().item()
                        if move_list[-1] == 0 and action == 1:
                            action = 2
                        elif move_list[-1] == 1 and action == 0:
                            action = 0
                        else:
                            action = 1
                        move_list.append(move)

                    # 액션 실행
                    self.action_map[action]()

                    # 디버그 정보 출력
                    print(f"Action: {action}, Q-values: {q_values.cpu().numpy()}")

                    # GUI 모드가 아닌 경우 프레임 표시 생략
                    if not self.headless_mode:
                        # 프레임 표시
                        cv2.putText(
                            frame,
                            f"Action: {action}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Speed: {self.rc_car.current_speed}",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                        cv2.imshow("Autonomous Driving", frame)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    else:
                        # GUI 없이 실행 - 프레임 처리만 수행
                        pass

            except Exception as e:
                print(f"자율주행 제어 오류: {e}")
                self.rc_car.stop()

            time.sleep(0.1)

    def start_autonomous(self):
        """자율주행 시작"""
        if not self.rc_car.connected:
            print("아두이노가 연결되지 않았습니다.")
            return False

        print("자율주행 모드 시작...")
        self.autonomous_mode = True
        self.running = True

        # 자율주행 스레드 시작
        self.autonomous_thread = threading.Thread(target=self.autonomous_control_loop)
        self.autonomous_thread.daemon = True
        self.autonomous_thread.start()

        return True

    def stop_autonomous(self):
        """자율주행 중지"""
        print("자율주행 모드 중지...")
        self.autonomous_mode = False
        self.running = False
        self.rc_car.stop()

    def manual_control(self):
        """수동 제어 모드"""
        print("수동 제어 모드 (키보드 입력)")
        print("w: 전진, s: 후진, a: 좌회전, d: 우회전, x: 정지, q: 종료")

        while self.running:
            key = input("명령 입력: ").lower()

            if key == "w":
                self.rc_car.move_forward()
                print("전진")
            elif key == "s":
                self.rc_car.move_backward()
                print("후진")
            elif key == "a":
                self.rc_car.turn_left()
                print("좌회전")
            elif key == "d":
                self.rc_car.turn_right()
                print("우회전")
            elif key == "x":
                self.rc_car.stop()
                print("정지")
            elif key == "q":
                break
            elif key == "auto":
                self.start_autonomous()
            elif key == "status":
                self.rc_car.get_status()

    def start(self):
        """시스템 시작"""
        print("Jetson 아두이노 RC카 제어 시스템 시작...")

        # 아두이노 연결
        if not self.rc_car.connect():
            print("아두이노 연결 실패. 수동 모드로 진행합니다.")
            return False

        self.running = True

        # 모드 선택
        print("모드를 선택하세요:")
        print("1. 수동 제어")
        print("2. 자율주행")

        choice = input("선택 (1/2): ")

        if choice == "1":
            self.manual_control()
        elif choice == "2":
            self.start_autonomous()

            # 메인 스레드에서 키보드 입력 대기
            try:
                while self.running:
                    key = input("Press 'q' to quit, 's' to stop: ")
                    if key == "q":
                        break
                    elif key == "s":
                        self.stop_autonomous()

            except KeyboardInterrupt:
                print("중단됨")

        return True

    def cleanup(self):
        """정리"""
        self.running = False
        self.autonomous_mode = False
        self.rc_car.stop()
        if self.rc_car.serial_conn:
            self.rc_car.serial_conn.close()
        if self.cap:
            self.cap.release()
        # GUI 관련 함수 호출 제거
        # if not self.headless_mode:
        #     cv2.destroyAllWindows()


def main():
    """메인 함수"""
    print("Jetson 아두이노 RC카 제어 시스템")

    # 아두이노 포트 확인
    import glob

    ports = glob.glob("/dev/tty[A-Za-z]*")
    print("사용 가능한 포트:")
    for port in ports:
        print(f"  {port}")

    # 포트 선택
    arduino_port = input("아두이노 포트를 입력하세요 (예: /dev/ttyUSB0): ")
    if not arduino_port:
        arduino_port = "/dev/ttyACM0"

    # 컨트롤러 초기화
    controller = JetsonAutonomousController(arduino_port)

    try:
        # 시스템 시작
        controller.start()
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()
