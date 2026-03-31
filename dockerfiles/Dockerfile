FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y vim tmux zip unzip bzip2 wget git git-lfs build-essential libibverbs-dev ca-certificates \
    curl iproute2 libsm6 libxext6 kmod ccache libnuma-dev libssl-dev flex bison libgtk-3-dev libpango1.0-dev \
    libsoup2.4-dev libnice-dev libopus-dev libvpx-dev libx264-dev libsrtp2-dev libglib2.0-dev libdrm-dev libjpeg-dev libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && git lfs install

RUN conda install conda-forge::ffmpeg=8.0.0 -y && conda clean -all -y

RUN pip install --no-cache-dir packaging ninja cmake scikit-build-core uv meson ruff pre-commit fastapi uvicorn requests -U

RUN git clone https://github.com/vllm-project/vllm.git && cd vllm \
    && python use_existing_torch.py \
    && pip install --no-cache-dir -r requirements/build.txt \
    && MAX_JOBS=8 NVCC_THREADS=4 CMAKE_BUILD_PARALLEL_LEVEL=2 \
       pip install --no-cache-dir --no-build-isolation -v -e .

RUN git clone https://github.com/sgl-project/sglang.git && cd sglang/sgl-kernel \
    && make build MAX_JOBS=8 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1" && make clean

RUN pip install --no-cache-dir diffusers transformers tokenizers accelerate safetensors opencv-python numpy imageio \
    imageio-ffmpeg einops loguru qtorch ftfy av decord matplotlib debugpy soundfile jsonschema pymongo modelscope

RUN git clone https://github.com/Dao-AILab/flash-attention.git --recursive

RUN cd flash-attention && \
    MAX_JOBS=3 NVCC_THREADS=2 CMAKE_BUILD_PARALLEL_LEVEL=3 \
    python setup.py install && \
    rm -rf build

RUN cd flash-attention/hopper && \
    MAX_JOBS=3 NVCC_THREADS=2 CMAKE_BUILD_PARALLEL_LEVEL=3 \
    python setup.py install && \
    rm -rf build

RUN git clone https://github.com/ModelTC/SageAttention.git --depth 1

RUN cd SageAttention && CUDA_ARCHITECTURES="8.0,8.6,8.9,9.0,12.0" EXT_PARALLEL=1 NVCC_APPEND_FLAGS="--threads 2" MAX_JOBS=8 pip install --no-cache-dir -v -e .

RUN git clone https://github.com/ModelTC/SageAttention-1104.git --depth 1

RUN cd SageAttention-1104/sageattention3_blackwell && python setup.py install && rm -rf build

RUN git clone https://github.com/Tencent-Hunyuan/flex-block-attn.git --recursive

RUN cd flex-block-attn && python setup.py install && rm -rf build

RUN git clone https://github.com/ModelTC/FlashVSR.git --depth 1

RUN cd FlashVSR && pip install --no-cache-dir -v -e .

COPY lightx2v_kernel /app/lightx2v_kernel

RUN git clone https://github.com/NVIDIA/cutlass.git --depth 1 && cd /app/lightx2v_kernel && MAX_JOBS=32 && CMAKE_BUILD_PARALLEL_LEVEL=4 \
    uv build --wheel \
        -Cbuild-dir=build . \
        -Ccmake.define.CUTLASS_PATH=/app/cutlass \
        --verbose \
        --color=always \
        --no-build-isolation \
    && pip install dist/*whl --force-reinstall --no-deps \
    && rm -rf /app/lightx2v_kernel && rm -rf /app/cutlass

# cloud deploy
RUN ldconfig

# for base docker
RUN git clone https://github.com/KONAKONA666/q8_kernels.git --depth 1
RUN cd q8_kernels && git submodule init && git submodule update && python setup.py install && rm -rf build

RUN git clone https://github.com/thu-ml/SpargeAttn.git --depth 1
RUN cd SpargeAttn && TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" pip install --no-cache-dir --no-build-isolation -v -e .

RUN git clone https://github.com/facebookresearch/sam2.git 
RUN cd sam2 && SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e ".[notebooks]" --no-deps --no-build-isolation

RUN pip install gradio onnxruntime moviepy pydub pyannote.audio decord peft pandas matplotlib loguru sentencepiece

WORKDIR /workspace

RUN git clone https://github.com/gojasper/LBM.git
RUN cd LBM && pip install --no-deps -e .