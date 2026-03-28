sudo docker run --rm -it --gpus all \
  -p 7863:7863 \
  -v /home/nmhung/wan_animate/LightX2V:/workspace/LightX2V \
  -v /home/nmhung/wan_animate/checkpoints/Wan2.2-Animate-14B:/workspace/LightX2V/Wan2.2-Animate-14B \
  lightx2v/lightx2v:wan_animate bash


# lightx2v_path=/workspace/LightX2V model_path=/workspace/LightX2V/Wan2.2-Animate-14B bash scripts/wan22/run_wan22_animate_gradio.sh
