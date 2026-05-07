# Stage 1: RealSense Base with RSUSB
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates software-properties-common \
    libusb-1.0-0-dev libssl-dev pkg-config libudev-dev python3.10 python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN git clone --depth 1 --branch v2.56.5 https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    mkdir build && cd build && \
    cmake .. \
    -DFORCE_RSUSB_BACKEND=ON \
    -DCMAKE_BUILD_TYPE=release \
    -DBUILD_EXAMPLES=false \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.10 && \
    make -j1 && \
    make install && \
    ldconfig && \
    cd ../.. && \
    rm -rf librealsense
