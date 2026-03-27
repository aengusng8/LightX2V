model_path=/workspace/LightX2V/Wan2.2-Animate-14B
output_path=/workspace/LightX2V/Wan2.2-Animate-14B-FP8/wan_animate_encoder_fp8.safetensors
python quant_adapter.py \
    --model_path $model_path \
    --output_path $output_path \
    --mode wan_animate_encoders
