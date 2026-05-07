# Optimized Dockerfile for RealSense + FoundationPose/DOPE (V2.1)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    software-properties-common \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    pkg-config \
    libusb-1.0-0-dev \
    libssl-dev \
    kmod \
    v4l-utils \
    usbutils \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.10 and Pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Build RealSense SDK from source with RSUSB backend for WSL2 compatibility
RUN git clone --depth 1 --branch v2.56.5 https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    mkdir build && cd build && \
    cmake .. \
    -DFORCE_RSUSB_BACKEND=ON \
    -DCMAKE_BUILD_TYPE=release \
    -DBUILD_EXAMPLES=true \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.10 && \
    make -j2 && \
    make install && \
    ldconfig && \
    cd ../.. && \
    rm -rf librealsense

# Install PyTorch and Vision (CUDA 11.8)
RUN pip3 install --no-cache-dir torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# RealSense Python bindings are included in the source build above

# Copy requirements and install
WORKDIR /workspace
COPY requirements.txt .
# Explicitly install setuptools/wheel and visdom (to fix pkg_resources errors)
RUN pip3 install --no-cache-dir setuptools==69.5.1 wheel
RUN pip3 install --no-cache-dir visdom==0.2.4 --no-deps --no-build-isolation
RUN pip3 install --no-cache-dir torchnet==0.0.4 --no-deps
# Handle blinker conflict separately (it's a distutils package in Ubuntu 22.04)
RUN pip3 install --no-cache-dir blinker --ignore-installed
# Remove version constraints for numpy if needed, but 3.10 should handle 1.26.4
RUN pip3 install --no-cache-dir -r requirements.txt

# Additional libraries for FoundationPose/DOPE
RUN pip3 install --no-cache-dir transforms3d roma scikit-image pyyaml imgaug pyrr kornia einops

# Set environment variables for GUI
ENV QT_X11_NO_MITSHM=1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

CMD ["bash"]
