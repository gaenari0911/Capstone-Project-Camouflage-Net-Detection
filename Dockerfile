# 1. NVIDIA 공식 CUDA 이미지 베이스 (Ubuntu 22.04 환경)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# 2. 필수 OS 패키지 및 Python 설치
# DEBIAN_FRONTEND 설정으로 tzdata 등의 설치 중 멈춤 현상 방지
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 'python3' 명령어를 'python'으로 매핑
RUN ln -s /usr/bin/python3 /usr/bin/python

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. PyTorch 설치 (CUDA 12.1 버전에 맞춤)
# requirements.txt 설치 전, GPU 가속이 완벽히 지원되는 PyTorch를 먼저 명시적으로 설치합니다.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# pip, setuptools, wheel 최신 버전으로 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 5. 의존성 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 전체 프로젝트 파일 복사
COPY . .

# 7. 수정한 로컬 Ultralytics 엔진을 시스템에 개발자 모드로 설치
RUN pip install -e ./ultralytics_lib

ENV PYTHONPATH="/app"

# docker run --rm --gpus all -it --ipc=host -v ${PWD}:/app cod_project:latest