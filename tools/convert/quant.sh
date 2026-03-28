model_path=/workspace/LightX2V/Wan2.2-Animate-14B
lora_path1=/workspace/LightX2V/lora_models/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
lora_path2=/workspace/LightX2V/lora_models/FastWan_T2V_14B_480p_lora_rank_128_bf16.safetensors
output_merged_path=/workspace/LightX2V/Wan2.2-Animate-14B-Merged-720-r128

output_path=/workspace/LightX2V/Wan2.2-Animate-14B-FP8-720-r128

echo "Merging LoRA 1..."
python converter.py \
    --source $model_path \
    --output $output_merged_path \
    --output_name wan22_animate_merged_distill_lora1 \
    --model_type wan_animate_dit \
    --lora_path $lora_path1 \
    --lora_strength 1.0 \
    --single_file

echo "Merging LoRA 2..."
python converter.py \
    --source $output_merged_path/wan22_animate_merged_distill_lora1.safetensors \
    --output $output_merged_path \
    --output_name wan22_animate_merged_distill_both \
    --model_type wan_animate_dit \
    --lora_path $lora_path2 \
    --lora_strength 1.0 \
    --single_file

echo "Quantizing model..."
python converter.py \
    --source $output_merged_path/wan22_animate_merged_distill_both.safetensors \
    --output $output_path \
    --output_ext .safetensors \
    --output_name wan_animate_fp8 \
    --linear_type fp8 \
    --model_type wan_animate_dit \
    --quantized \
    --single_file