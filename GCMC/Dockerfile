# NVIDIA CUDA 기반 이미지 사용 (PyTorch와 GPU 지원)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 3.8을 기본으로 설정
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch 및 관련 패키지 설치 (CUDA 지원)
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric 및 torch-scatter 설치
RUN pip install torch-geometric==2.3.0 torch-scatter==2.1.0

# 소스 코드 복사
COPY . .

# 기본 실행 명령 (필요 시 수정)
# CMD ["python", "main.py"]