model_path=/workspace/LightX2V/Wan2.2-Animate-14B
lora_path=/workspace/LightX2V/lora_models/lightx2v_I2V_14B_720p_cfg_step_distill_rank_128_bf16.safetensors
output_merged_path=/workspace/LightX2V/Wan2.2-Animate-14B-Merged-720-r128

output_path=/workspace/LightX2V/Wan2.2-Animate-14B-FP8-720-r128

echo "Merging LoRA..."
python converter.py \
    --source $model_path \
    --output $output_merged_path \
    --output_name wan22_animate_merged_distill \
    --model_type wan_animate_dit \
    --lora_path $lora_path \
    --lora_strength 1.0 \
    --single_file

echo "Quantizing model..."
python converter.py \
    --source $output_merged_path \
    --output $output_path \
    --output_ext .safetensors \
    --output_name wan_animate_fp8 \
    --linear_type fp8 \
    --model_type wan_animate_dit \
    --quantized \
    --single_file