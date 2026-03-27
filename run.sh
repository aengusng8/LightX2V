sudo docker run --rm -it --gpus all \
  -p 7863:7863 \
  -v /home/itsthanhtung/tungdo/code/wan_animate/LightX2V:/workspace/LightX2V \
  -v /home/itsthanhtung/tungdo/code/Wan2.2/Wan2.2-Animate-14B:/workspace/LightX2V/Wan2.2-Animate-14B \
  -v /home/itsthanhtung/tungdo/code/Wan2.2/Wan2.2-Animate-14B-GGUF:/workspace/LightX2V/Wan2.2-Animate-14B-GGUF \
  lightx2v_demo:latest bash


# lightx2v_path=/workspace/LightX2V model_path=/workspace/LightX2V/Wan2.2-Animate-14B bash scripts/wan22/run_wan22_animate_gradio.sh
