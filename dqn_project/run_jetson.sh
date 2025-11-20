#!/bin/bash


# Jetson Orin DQN 자율주행 실행 스크립트


echo "=== Jetson Orin DQN 자율주행 시작 ==="


# Jetson 성능 모드 설정
echo "Jetson 성능 모드 설정 중..."
sudo nvpmodel -m 0
sudo jetson_clocks


# 현재 디렉토리 확인
echo "현재 디렉토리: $(pwd)"


# Python 버전 확인
echo "Python 버전: $(python3 --version)"


# CUDA 사용 가능 여부 확인
echo "CUDA 사용 가능: $(python3 -c 'import torch; print(torch.cuda.is_available())')"


# 비디오 파일 확인
echo "비디오 파일 확인 중..."
ls -la *.mp4 2>/dev/null || echo "현재 디렉토리에 mp4 파일이 없습니다."


# 메모리 상태 확인
echo "메모리 상태:"
free -h


# 실행
echo "DQN 학습 시작..."
python3 jetson_orin_dqn_updated.py


echo "=== 실행 완료 ==="




