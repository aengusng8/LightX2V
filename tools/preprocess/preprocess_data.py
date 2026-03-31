# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os

from process_pipepline import ProcessPipeline


def get_preprocess_parser():
    parser = argparse.ArgumentParser(description="The preprocessing pipeline for Wan-animate.")

    parser.add_argument("--ckpt_path", type=str, default=None, help="The path to the preprocessing model's checkpoint directory. ")

    parser.add_argument("--video_path", type=str, default=None, help="The path to the driving video.")
    parser.add_argument("--refer_path", type=str, default=None, help="The path to the refererence image.")
    parser.add_argument("--save_path", type=str, default=None, help="The path to save the processed results.")

    parser.add_argument(
        "--resolution_area",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="The target resolution for processing, specified as [width, height]. To handle different aspect ratios, the video is resized to have a total area equivalent to width * height, while preserving the original aspect ratio.",
    )
    parser.add_argument("--fps", type=int, default=30, help="The target FPS for processing the driving video. Set to -1 to use the video's original FPS.")

    parser.add_argument("--replace_flag", action="store_true", default=False, help="Whether to use replacement mode.")
    parser.add_argument("--retarget_flag", action="store_true", default=False, help="Whether to use pose retargeting. Currently only supported in animation mode")
    parser.add_argument(
        "--use_flux",
        action="store_true",
        default=False,
        help="Whether to use image editing in pose retargeting. Recommended if the character in the reference image or the first frame of the driving video is not in a standard, front-facing pose",
    )
    parser.add_argument(
        "--compose_refer_on_first_frame",
        action="store_true",
        default=False,
        help="Compose reference on first frame (in memory for LBM path). With --lbm_relight and wavelet mode, only reference_wavelet_merged.png is written from that step.",
    )
    parser.add_argument(
        "--lbm_relight",
        action="store_true",
        default=False,
        help="After compose: LBM + wavelet merge; writes only reference_wavelet_merged.png (needs --lbm_fuse_mode wavelet, PyWavelets). Requires compose, CUDA, LBM.",
    )
    parser.add_argument(
        "--lbm_relight_ckpt",
        type=str,
        default=None,
        help="LBM relighting checkpoint directory or cache dir for HF download. Default: {ckpt_path}/lbm_relighting",
    )
    parser.add_argument("--lbm_relight_steps", type=int, default=1, help="LBM sampling steps (default 1 matches upstream).")
    parser.add_argument(
        "--lbm_fuse_mode",
        type=str,
        default="wavelet",
        choices=["wavelet", "alpha"],
        help="wavelet: write reference_wavelet_merged.png. alpha: LBM path writes no image from this step (use wavelet for replace ref output).",
    )
    parser.add_argument("--lbm_wavelet", type=str, default="haar", help="Wavelet name for --lbm_fuse_mode wavelet (PyWavelets); default Haar.")
    parser.add_argument("--lbm_wavelet_level", type=int, default=2, help="Decomposition depth for wavelet fusion.")

    # Parameters for the mask strategy in replacement mode. These control the mask's size and shape. Refer to https://arxiv.org/pdf/2502.06145
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for mask dilation.")
    parser.add_argument("--k", type=int, default=7, help="Number of kernel size for mask dilation.")
    parser.add_argument(
        "--w_len",
        type=int,
        default=1,
        help="The number of subdivisions for the grid along the 'w' dimension. A higher value results in a more detailed contour. A value of 1 means no subdivision is performed.",
    )
    parser.add_argument(
        "--h_len",
        type=int,
        default=1,
        help="The number of subdivisions for the grid along the 'h' dimension. A higher value results in a more detailed contour. A value of 1 means no subdivision is performed.",
    )
    return parser


def process_input_video(args):
    args_dict = vars(args)
    print(args_dict)

    assert len(args.resolution_area) == 2, "resolution_area should be a list of two integers [width, height]"
    assert not args.use_flux or args.retarget_flag, "Image editing with FLUX can only be used when pose retargeting is enabled."
    if args.lbm_relight and not args.compose_refer_on_first_frame:
        print("Warning: --lbm_relight has no effect without --compose_refer_on_first_frame")

    lbm_ckpt_dir = None
    if args.lbm_relight:
        if args.lbm_relight_ckpt:
            lbm_ckpt_dir = args.lbm_relight_ckpt
        elif args.ckpt_path:
            lbm_ckpt_dir = os.path.join(args.ckpt_path, "lbm_relighting")
        else:
            lbm_ckpt_dir = os.path.abspath("lbm_relighting")

    pose2d_checkpoint_path = os.path.join(args.ckpt_path, "pose2d/vitpose_h_wholebody.onnx")
    det_checkpoint_path = os.path.join(args.ckpt_path, "det/yolov10m.onnx")

    sam2_checkpoint_path = os.path.join(args.ckpt_path, "sam2/sam2_hiera_large.pt") if args.replace_flag else None
    # Use FLUX.2 Klein (9B) for image editing during pose retargeting.
    # Distilled FLUX.2 Klein weights location under the checkpoint folder.
    # The FLUX pipeline expects the model files inside `models/flux_klein`.
    flux_kontext_path = os.path.join(args.ckpt_path, "flux_klein") if args.use_flux else None
    process_pipeline = ProcessPipeline(
        det_checkpoint_path=det_checkpoint_path,
        pose2d_checkpoint_path=pose2d_checkpoint_path,
        sam_checkpoint_path=sam2_checkpoint_path,
        flux_kontext_path=flux_kontext_path,
        lbm_relight_ckpt_dir=lbm_ckpt_dir,
        lbm_relight_steps=args.lbm_relight_steps,
        lbm_relight=args.lbm_relight,
        lbm_fuse_mode=args.lbm_fuse_mode,
        lbm_wavelet=args.lbm_wavelet,
        lbm_wavelet_level=args.lbm_wavelet_level,
    )
    os.makedirs(args.save_path, exist_ok=True)
    process_pipeline(
        video_path=args.video_path,
        refer_image_path=args.refer_path,
        output_path=args.save_path,
        resolution_area=args.resolution_area,
        fps=args.fps,
        iterations=args.iterations,
        k=args.k,
        w_len=args.w_len,
        h_len=args.h_len,
        retarget_flag=args.retarget_flag,
        use_flux=args.use_flux,
        replace_flag=args.replace_flag,
        compose_refer_on_first_frame=args.compose_refer_on_first_frame,
    )


if __name__ == "__main__":
    parser = get_preprocess_parser()
    args = parser.parse_args()
    process_input_video(args)
