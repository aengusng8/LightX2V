#!/bin/bash

if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH

prepend_ld_library_path() {
    if [ -n "$1" ] && [ -d "$1" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$1:"*) ;;
            *) export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
}

# SAM2 _C.so needs PyTorch's libc10.so etc. on LD_LIBRARY_PATH.
# In Docker, these usually live under the Conda prefix and torch's site-packages directory.
prepend_ld_library_path "/usr/local/lib/x86_64-linux-gnu"
prepend_ld_library_path "/opt/conda/lib"
prepend_ld_library_path "${CONDA_PREFIX}/lib"

TORCH_LIB=""
for py_bin in python python3; do
    if command -v "$py_bin" >/dev/null 2>&1; then
        TORCH_LIB=$("$py_bin" -c "import torch, os; print(os.path.join(torch.__path__[0], 'lib'))" 2>/dev/null)
        if [ -n "$TORCH_LIB" ]; then
            break
        fi
    fi
done

if [ -n "$TORCH_LIB" ]; then
    prepend_ld_library_path "$TORCH_LIB"
fi

for torch_lib in /opt/conda/lib/python*/site-packages/torch/lib "${CONDA_PREFIX}"/lib/python*/site-packages/torch/lib; do
    prepend_ld_library_path "$torch_lib"
done

# always set false to avoid some warnings
export TOKENIZERS_PARALLELISM=false
# set expandable_segments to True to avoid OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# =====================================================================================
# ⚠️  IMPORTANT CONFIGURATION PARAMETERS - READ CAREFULLY AND MODIFY WITH CAUTION ⚠️
# =====================================================================================

# Model Inference Data Type Setting (IMPORTANT!)
# Key parameter affecting model accuracy and performance
# Available options: [BF16, FP16]
# If not set, default value: BF16
export DTYPE=BF16

# Sensitive Layer Data Type Setting (IMPORTANT!)
# Used for layers requiring higher precision
# Available options: [FP32, None]
# If not set, default value: None (follows DTYPE setting)
export SENSITIVE_LAYER_DTYPE=None

# Performance Profiling Debug Level (Debug Only)
# Enables detailed performance analysis output, such as time cost and memory usage
# Available options: [0, 1, 2]
# If not set, default value: 0
# Note: This option can be set to 0 for production.
export PROFILING_DEBUG_LEVEL=2


echo "==============================================================================="
echo "LightX2V Base Environment Variables Summary:"
echo "-------------------------------------------------------------------------------"
echo "lightx2v_path: ${lightx2v_path}"
echo "model_path: ${model_path}"
echo "-------------------------------------------------------------------------------"
echo "Model Inference Data Type: ${DTYPE}"
echo "Sensitive Layer Data Type: ${SENSITIVE_LAYER_DTYPE}"
echo "Performance Profiling Debug Level: ${PROFILING_DEBUG_LEVEL}"
echo "==============================================================================="
