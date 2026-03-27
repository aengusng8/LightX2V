#!/bin/bash

# set path firstly
lightx2v_path=/workspace/LightX2V
model_path=/workspace/LightX2V/Wan2.2-Animate-14B
video_path=/workspace/LightX2V/examples_video/wan_animate/animate/dancing_4.mp4
refer_path=/workspace/LightX2V/examples_video/wan_animate/animate/sydney.jpeg
use_flux=false

process_dir=${lightx2v_path}/save_results/replace/process_results_${use_flux}
export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# process
preprocess_args=(
    --ckpt_path ${model_path}/process_checkpoint
    --video_path "$video_path"
    --refer_path "$refer_path"
    --save_path "$process_dir"
    --resolution_area 1280 720
    --iterations 3
    --k 7
    --w_len 1
    --h_len 1
    --replace_flag
)
# keep this option consistent with run_wan22_animate.sh
# preprocess_data requires --retarget_flag when --use_flux is set
if [[ "$use_flux" == true ]]; then
    preprocess_args+=(--retarget_flag --use_flux)
fi
python ${lightx2v_path}/tools/preprocess/preprocess_data.py "${preprocess_args[@]}"

# if [[ "$use_flux" == true ]]; then
#     src_ref_images=${process_dir}/refer_edit.png
# else
src_ref_images=${process_dir}/src_ref.png
# fi

python -m lightx2v.infer \
--model_cls wan2.2_animate \
--task animate \
--image_path $src_ref_images \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan22/wan_animate_replace_4090.json \
--src_pose_path ${process_dir}/src_pose.mp4 \
--src_face_path ${process_dir}/src_face.mp4 \
--src_ref_images $src_ref_images \
--src_bg_path ${process_dir}/src_bg.mp4 \
--src_mask_path ${process_dir}/src_mask.mp4 \
--prompt "视频中的人在做动作" \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_replace_${use_flux}.mp4
