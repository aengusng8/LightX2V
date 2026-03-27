#!/bin/bash
# Simple Gradio demo for Wan2.2 Animate (same workflow as run_wan22_animate.sh).

# Set paths first
lightx2v_path=${lightx2v_path:-/workspace/LightX2V}
model_path=${model_path:-/workspace/LightX2V/Wan2.2-Animate-14B}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Set environment variables (PYTHONPATH, LD_LIBRARY_PATH, etc.)
source "${lightx2v_path}/scripts/base/base.sh"

# Optional: port and share (e.g. --port 8030 --share)
port=7863
share=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) port="$2"; shift 2 ;;
        --share) share="1"; shift ;;
        *) shift ;;
    esac
done

cd "${lightx2v_path}/app"
python gradio_wan22_animate.py \
    --lightx2v_path "$lightx2v_path" \
    --model_path "$model_path" \
    --server_name "0.0.0.0" \
    --server_port "$port" \
    ${share:+--share}
