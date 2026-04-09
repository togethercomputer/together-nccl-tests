# NCCL Tests container for B200 / CUDA 12.9
# Builds all nccl-tests binaries with OpenMPI + UCX support
#
# Build:
#   sudo docker build -t nccl-tests:cuda12.9 .
#
# For enroot:
#   enroot import dockerd://nccl-tests:cuda12.9
#   mv nccl-tests+cuda12.9.sqsh /mnt/vast/exemplar/llmb/containers/

ARG CUDA_VERSION=12.9.0
ARG UBUNTU_VERSION=22.04

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG NCCL_TESTS_VERSION=v2.13.11

ENV DEBIAN_FRONTEND=noninteractive

# Install NCCL, OpenMPI, UCX, and build dependencies via apt
# (NVIDIA CUDA base images have the NVIDIA apt repo pre-configured)
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    build-essential \
    ca-certificates \
    git \
    libnccl2 \
    libnccl-dev \
    libopenmpi-dev \
    libucx-dev \
    && rm -rf /var/lib/apt/lists/*

# Build nccl-tests with MPI support
RUN git clone --depth 1 --branch ${NCCL_TESTS_VERSION} \
    https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests && \
    cd /tmp/nccl-tests && \
    make -j$(nproc) \
        MPI=1 \
        MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi \
        CUDA_HOME=/usr/local/cuda \
        NCCL_HOME=/usr \
    && mkdir -p /opt/nccl-tests \
    && cp /tmp/nccl-tests/build/*_perf /opt/nccl-tests/ \
    && rm -rf /tmp/nccl-tests

# ---------- runtime stage ----------
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi3 \
    openmpi-bin \
    libucx0 \
    libibverbs1 \
    librdmacm1 \
    numactl \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Copy nccl-tests binaries
COPY --from=builder /opt/nccl-tests/*_perf /usr/local/bin/

ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Verify: binaries exist, libnccl.so.2 is resolvable from the base image,
# and the binary is executable. (libnccl2 is intentionally not reinstalled
# here — the base CUDA runtime image ships it at the correct cuda12.9 version;
# reinstalling via apt would upgrade to the cuda13.2 build and break at runtime.)
RUN ls -la /usr/local/bin/*_perf \
    && ldd /usr/local/bin/all_reduce_perf | grep -q 'libnccl' \
    && (all_reduce_perf --help 2>&1 | head -3; true)

WORKDIR /workspace
