FROM nvcr.io/nvidia/tensorrt:24.01-py3 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    zip \
    unzip \
    tar \
    pkg-config \
    libapr1-dev \
    libaprutil1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
    /opt/vcpkg/bootstrap-vcpkg.sh

ENV CUDA_PATH=/usr/local/cuda
ENV TENSORRT_ROOT=/usr
ENV VCPKG_ROOT=/opt/vcpkg

WORKDIR /workspace

COPY vcpkg.json .

RUN VCPKG_COMMIT=$(git -C /opt/vcpkg rev-parse HEAD) && \
    sed -i "s/\"builtin-baseline\": \"[^\"]*\"/\"builtin-baseline\": \"${VCPKG_COMMIT}\"/" vcpkg.json && \
    cat vcpkg.json

RUN /opt/vcpkg/vcpkg install --triplet x64-linux

COPY . .

RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
          -DVCPKG_TARGET_TRIPLET=x64-linux \
          -DBUILD_TESTS=OFF \
          .. && \
    make -j$(nproc)

FROM nvcr.io/nvidia/tensorrt:24.01-py3 AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libapr1 \
    libaprutil1 \
    && rm -rf /var/lib/apt/lists/*

# Il binario è build/Inference/Inference (subdirectory/executable)
COPY --from=builder /workspace/build/Inference/Inference /app/Inference

# Gli Assets sono accanto al binario
COPY --from=builder /workspace/build/Inference/Assets /app/Assets

WORKDIR /app
ENTRYPOINT ["./Inference"]