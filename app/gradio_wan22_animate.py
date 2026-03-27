"""
Simple Gradio demo for Wan2.2 Animate: drive a reference face with a motion video.
Based on scripts/wan22/run_wan22_animate.sh
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import uuid

import gradio as gr


def _log(msg: str) -> None:
    """Print to terminal so server logs show run results."""
    print(f"[Wan22 Animate] {msg}", flush=True)

# Default paths (override via CLI)
DEFAULT_LIGHTX2V_PATH = os.environ.get("LIGHTX2V_PATH", "/workspace/LightX2V")
DEFAULT_MODEL_PATH = os.environ.get("WAN22_ANIMATE_MODEL_PATH", "/workspace/LightX2V/Wan2.2-Animate-14B")
DEFAULT_CONFIG_JSON = os.path.join(DEFAULT_LIGHTX2V_PATH, "configs/wan22/wan_animate.json")
DEFAULT_PROCESS_CKPT = os.path.join(DEFAULT_MODEL_PATH, "process_checkpoint")

DEFAULT_PROMPT = "视频中的人在做动作"
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def run_animate(
    video_file,
    refer_image,
    prompt,
    negative_prompt,
    use_flux,
    replace_mode,
    use_r128_lora,
    lightx2v_path,
    model_path,
    progress=gr.Progress(),
):
    if not video_file or not refer_image:
        _log("Skipped: upload both a driving video and a reference image.")
        return None, None, None, "Please upload both a driving video and a reference image."

    lightx2v_path = lightx2v_path or DEFAULT_LIGHTX2V_PATH
    model_path = model_path or DEFAULT_MODEL_PATH
    if replace_mode and use_r128_lora:
        config_json = os.path.join(lightx2v_path, "configs/wan22/wan_animate_replace_4090_r128.json")
    elif replace_mode:
        config_json = os.path.join(lightx2v_path, "configs/wan22/wan_animate_replace_4090.json")
    elif use_r128_lora:
        config_json = os.path.join(lightx2v_path, "configs/wan22/wan_animate_r128.json")
    else:
        config_json = os.path.join(lightx2v_path, "configs/wan22/wan_animate.json")
    process_ckpt = os.path.join(model_path, "process_checkpoint")

    if not os.path.isdir(lightx2v_path):
        _log(f"Error: LightX2V path not found: {lightx2v_path}")
        return None, None, None, f"LightX2V path not found: {lightx2v_path}"
    if not os.path.isdir(model_path):
        _log(f"Error: Model path not found: {model_path}")
        return None, None, None, f"Model path not found: {model_path}"
    if not os.path.isdir(process_ckpt):
        _log(f"Error: Process checkpoint not found: {process_ckpt}")
        return None, None, None, f"Process checkpoint not found: {process_ckpt}. Need process_checkpoint under model path."

    prompt = prompt or DEFAULT_PROMPT
    negative_prompt = negative_prompt or DEFAULT_NEGATIVE

    # Resolve paths from Gradio inputs (file path string or object with path/name)
    def _path(x):
        if x is None:
            return None
        if isinstance(x, str):
            return x
        if hasattr(x, "path") and x.path:
            return x.path
        if hasattr(x, "name") and x.name:
            return x.name
        return None
    video_path = _path(video_file)
    refer_path = _path(refer_image)
    if not video_path or not refer_path:
        _log("Error: Invalid video or image file.")
        return None, None, None, "Invalid video or image file."

    out_dir = os.path.join(lightx2v_path, "save_results", "animate", "gradio")
    os.makedirs(out_dir, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]
    process_dir = os.path.join(out_dir, f"process_{run_id}")
    os.makedirs(process_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"output_{run_id}.mp4")

    start_time = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{lightx2v_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"  # so child process output appears in terminal immediately

    # Step 1: Preprocess
    progress(0.2, desc="Preprocessing video and reference...")
    preprocess_cmd = [
        "python",
        os.path.join(lightx2v_path, "tools/preprocess/preprocess_data.py"),
        "--ckpt_path", process_ckpt,
        "--video_path", video_path,
        "--refer_path", refer_path,
        "--save_path", process_dir,
        "--resolution_area", "1280", "720",
    ]
    if replace_mode:
        preprocess_cmd.extend(["--replace_flag", "--iterations", "3", "--k", "7", "--w_len", "1", "--h_len", "1"])
    else:
        preprocess_cmd.append("--retarget_flag")
        if use_flux:
            preprocess_cmd.append("--use_flux")
    try:
        _log("Preprocess started (output below).")
        subprocess.run(
            preprocess_cmd,
            env=env,
            cwd=lightx2v_path,
            check=True,
            timeout=600,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        err = (getattr(e, "stderr") or getattr(e, "stdout") or str(e) or "see terminal above")[:500]
        _log(f"Preprocess failed: {err}")
        return None, None, None, f"Preprocess failed: {err}"
    except Exception as e:
        _log(f"Preprocess error: {e}")
        return None, None, None, f"Preprocess error: {e}"

    src_pose = os.path.join(process_dir, "src_pose.mp4")
    src_face = os.path.join(process_dir, "src_face.mp4")
    src_ref = os.path.join(process_dir, "refer_edit.png" if (not replace_mode and use_flux) else "src_ref.png")
    if not os.path.isfile(src_ref):
        _log(f"Error: Preprocess did not produce {os.path.basename(src_ref)}.")
        return None, None, None, f"Preprocess did not produce {os.path.basename(src_ref)}."
    if not all(os.path.isfile(p) for p in (src_pose, src_face)):
        _log("Error: Preprocess did not produce src_pose.mp4 or src_face.mp4.")
        return None, None, None, "Preprocess did not produce src_pose.mp4 or src_face.mp4."

    src_bg = os.path.join(process_dir, "src_bg.mp4")
    src_mask = os.path.join(process_dir, "src_mask.mp4")
    if replace_mode:
        if not os.path.isfile(src_bg) or not os.path.isfile(src_mask):
            _log("Error: Replace mode requires src_bg.mp4 and src_mask.mp4 from preprocess.")
            return None, None, None, "Replace mode: preprocess did not produce src_bg.mp4 or src_mask.mp4. Ensure process_checkpoint includes SAM2."

    # Step 2: Inference
    progress(0.5, desc="Running Wan2.2 Animate inference...")
    infer_cmd = [
        "python", "-m", "lightx2v.infer",
        "--model_cls", "wan2.2_animate",
        "--task", "animate",
        "--image_path", refer_path,
        "--model_path", model_path,
        "--config_json", config_json,
        "--src_pose_path", src_pose,
        "--src_face_path", src_face,
        "--src_ref_images", src_ref,
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--save_result_path", out_video,
    ]
    if replace_mode:
        infer_cmd.extend(["--src_bg_path", src_bg, "--src_mask_path", src_mask])
    try:
        _log("Inference started (output below).")
        subprocess.run(
            infer_cmd,
            env=env,
            cwd=lightx2v_path,
            check=True,
            timeout=1200,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        err = (getattr(e, "stderr") or getattr(e, "stdout") or str(e) or "see terminal above")[:500]
        _log(f"Inference failed: {err}")
        return None, None, None, f"Inference failed: {err}"
    except Exception as e:
        _log(f"Inference error: {e}")
        return None, None, None, f"Inference error: {e}"

    progress(1.0, desc="Done.")
    if not os.path.isfile(out_video):
        _log("Error: Output video was not created.")
        return None, None, None, "Output video was not created."
    elapsed = time.perf_counter() - start_time
    mins, secs = divmod(int(round(elapsed)), 60)
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    _log(f"Result saved to: {out_video}  (total time: {time_str})")
    return out_video, src_pose, src_ref, f"Success.\nTotal processing time: {time_str}"


def build_ui(default_lightx2v_path=None, default_model_path=None):
    lightx2v_val = default_lightx2v_path if default_lightx2v_path is not None else DEFAULT_LIGHTX2V_PATH
    model_val = default_model_path if default_model_path is not None else DEFAULT_MODEL_PATH
    with gr.Blocks(title="Wan2.2 Animate") as demo:
        gr.Markdown("# Wan2.2 Animate — Simple Demo")
        gr.Markdown("Upload a **driving video** (motion) and a **reference image** (face). The model animates the reference face with the motion from the video.")

        example_dir = os.path.join(lightx2v_val, "examples_video", "wan_animate", "animate")
        example_videos = [
            os.path.join(example_dir, "dancing_2.mp4"),
            os.path.join(example_dir, "dancing.mp4"),
        ]
        for ref_name in ("quang.jpg", "oanh.jpg"):
            example_ref = os.path.join(example_dir, ref_name)
            if os.path.isfile(example_ref):
                break
        else:
            example_ref = os.path.join(example_dir, "quang.jpg")
        # Only include examples that exist so the app works if examples are missing
        examples_list = []
        for v in example_videos:
            if os.path.isfile(v) and os.path.isfile(example_ref):
                examples_list.append([v, example_ref])

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Driving video", sources=["upload"])
                refer_in = gr.Image(label="Reference image", type="filepath")
                if examples_list:
                    gr.Examples(
                        examples=examples_list,
                        inputs=[video_in, refer_in],
                        label="Example inputs (click to load)",
                        run_on_click=False,
                    )
                prompt_in = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    lines=2,
                    placeholder="Describe the motion, e.g. 视频中的人在做动作",
                )
                neg_in = gr.Textbox(
                    label="Negative prompt",
                    value=DEFAULT_NEGATIVE,
                    lines=3,
                    placeholder="Optional",
                )
                use_flux_in = gr.Checkbox(
                    label="Use FLUX",
                    value=False,
                    info="Use FLUX for image editing in pose retargeting (recommended if reference or first frame is not front-facing). Ignored in Replace mode.",
                )
                replace_mode_in = gr.Checkbox(
                    label="Replace mode",
                    value=False,
                    info="Replacement mode: paste reference face onto video person (needs SAM2 in process_checkpoint). Uses src_bg and src_mask.",
                )
                use_r128_lora_in = gr.Checkbox(
                    label="Use rank 128 LoRA",
                    value=False,
                    info="Use FP8 checkpoint with rank-128 LoRA from Wan2.2-Animate-14B-FP8-720-r128. Applies to both animate and replace mode.",
                )
                with gr.Accordion("Paths (optional)", open=False):
                    lightx2v_path_in = gr.Textbox(
                        label="LightX2V root",
                        value=lightx2v_val,
                        placeholder="/path/to/LightX2V",
                    )
                    model_path_in = gr.Textbox(
                        label="Model path (Wan2.2-Animate-14B)",
                        value=model_val,
                        placeholder="/path/to/Wan2.2-Animate-14B",
                    )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=1):
                out_video = gr.Video(label="Output video")
                out_pose = gr.Video(label="Pose video (preprocess)")
                out_src_ref = gr.Image(label="Reference used (src_ref_images)", type="filepath")
                out_msg = gr.Textbox(label="Status", interactive=False, lines=3)

        run_btn.click(
            fn=run_animate,
            inputs=[video_in, refer_in, prompt_in, neg_in, use_flux_in, replace_mode_in, use_r128_lora_in, lightx2v_path_in, model_path_in],
            outputs=[out_video, out_pose, out_src_ref, out_msg],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 Animate — Simple Gradio demo")
    parser.add_argument("--lightx2v_path", type=str, default=DEFAULT_LIGHTX2V_PATH, help="LightX2V project root")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Wan2.2-Animate-14B model path")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7863)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    demo = build_ui(default_lightx2v_path=args.lightx2v_path, default_model_path=args.model_path)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[args.lightx2v_path],
        max_file_size="1gb",
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
