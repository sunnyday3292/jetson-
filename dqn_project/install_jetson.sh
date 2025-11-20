#!/bin/bash

echo "Jetson Road Segmentation System Installation Script"
echo "=================================================="

# 시스템 업데이트
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# 필수 패키지 설치
echo "Installing required packages..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good-doc \
    gstreamer1.0-plugins-bad-doc \
    gstreamer1.0-plugins-ugly-doc \
    gstreamer1.0-libav-doc \
    gstreamer1.0-tools-doc \
    gstreamer1.0-x-doc \
    gstreamer1.0-alsa-doc \
    gstreamer1.0-gl-doc \
    gstreamer1.0-gtk3-doc \
    gstreamer1.0-qt5-doc \
    gstreamer1.0-pulseaudio-doc

# Python 패키지 설치
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    torch \
    torchvision \
    torchaudio \
    numpy \
    opencv-python \
    gymnasium \
    psutil \
    matplotlib \
    scipy \
    pillow

# Jetson 특화 패키지 설치
echo "Installing Jetson-specific packages..."
pip3 install \
    nvidia-ml-py3 \
    pycuda \
    tensorrt

# CUDA 설정 확인
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "CUDA is installed: $(nvcc --version)"
else
    echo "Warning: CUDA not found. Please install CUDA toolkit."
fi

# GPU 메모리 설정
echo "Setting GPU memory mode..."
sudo nvpmodel -m 0  # 최대 성능 모드
sudo jetson_clocks  # 최대 클럭 설정

# 환경 변수 설정
echo "Setting environment variables..."
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=1' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/dist-packages' >> ~/.bashrc

# 스크립트 실행 권한 설정
echo "Setting execution permissions..."
chmod +x road_segment_jetson.py

echo "Installation completed!"
echo "To run the system:"
echo "python3 road_segment_jetson.py --video your_video.mp4 --output result.mp4"
