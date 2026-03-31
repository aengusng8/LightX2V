# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import shutil

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from loguru import logger

try:
    import moviepy.editor as mpy
except:  # noqa
    import moviepy as mpy

import sam2.modeling.sam.transformer as transformer
from decord import VideoReader
from human_visualization import draw_aapose_by_meta_new
from pose2d import Pose2d
from pose2d_utils import AAPoseMeta
from retarget_pose import get_retarget_pose
from utils import (
    compose_refer_on_first_frame as compose_refer_by_pose,
    fuse_relight_wavelet_dwt2,
    get_aug_mask,
    get_face_bboxes,
    get_frame_indices,
    get_mask_body_img,
    padding_resize,
    paste_relit_onto_refer_canvas,
    resize_by_area,
)

transformer.USE_FLASH_ATTN = False
transformer.MATH_KERNEL_ON = True
transformer.OLD_GPU = True
from sam_utils import build_sam2_image_predictor, build_sam2_video_predictor  # noqa


class ProcessPipeline:
    def __init__(
        self,
        det_checkpoint_path,
        pose2d_checkpoint_path,
        sam_checkpoint_path,
        flux_kontext_path,
        lbm_relight_ckpt_dir=None,
        lbm_relight_steps=1,
        lbm_relight=False,
        lbm_fuse_mode="wavelet",
        lbm_wavelet="haar",
        lbm_wavelet_level=2,
    ):
        self.pose2d = Pose2d(checkpoint=pose2d_checkpoint_path, detector_checkpoint=det_checkpoint_path)

        model_cfg = "sam2_hiera_l.yaml"
        if sam_checkpoint_path is not None:
            self.predictor = build_sam2_video_predictor(model_cfg, sam_checkpoint_path)
            self.image_predictor = build_sam2_image_predictor(model_cfg, sam_checkpoint_path)
        if flux_kontext_path is not None:
            # Load FLUX.2 Klein (9B) pipeline for image editing during pose retargeting.
            self.flux_kontext = Flux2KleinPipeline.from_pretrained(flux_kontext_path, torch_dtype=torch.bfloat16).to("cuda")

        self.lbm_relight = bool(lbm_relight)
        self.lbm_relight_ckpt_dir = lbm_relight_ckpt_dir
        self.lbm_relight_steps = int(lbm_relight_steps)
        self.lbm_fuse_mode = (lbm_fuse_mode or "wavelet").lower()
        self.lbm_wavelet = str(lbm_wavelet)
        self.lbm_wavelet_level = int(lbm_wavelet_level)
        self._lbm_relight_model = None

    def __call__(
        self,
        video_path,
        refer_image_path,
        output_path,
        resolution_area=[1280, 720],
        fps=30,
        iterations=3,
        k=7,
        w_len=1,
        h_len=1,
        retarget_flag=False,
        use_flux=False,
        replace_flag=False,
        compose_refer_on_first_frame=False,
    ):
        if replace_flag:
            video_reader = VideoReader(video_path)
            frame_num = len(video_reader)
            print("frame_num: {}".format(frame_num))

            video_fps = video_reader.get_avg_fps()
            print("video_fps: {}".format(video_fps))
            print("fps: {}".format(fps))

            # TODO: Maybe we can switch to PyAV later, which can get accurate frame num
            duration = video_reader.get_frame_timestamp(-1)[-1]
            expected_frame_num = int(duration * video_fps + 0.5)
            ratio = abs((frame_num - expected_frame_num) / frame_num)
            if ratio > 0.1:
                print("Warning: The difference between the actual number of frames and the expected number of frames is two large")
                frame_num = expected_frame_num

            if fps == -1:
                fps = video_fps

            target_num = int(frame_num / video_fps * fps)
            print("target_num: {}".format(target_num))
            idxs = get_frame_indices(frame_num, video_fps, target_num, fps)
            frames = video_reader.get_batch(idxs).asnumpy()

            frames = [resize_by_area(frame, resolution_area[0] * resolution_area[1], divisor=16) for frame in frames]
            height, width = frames[0].shape[:2]
            logger.info(f"Processing pose meta")

            tpl_pose_metas = self.pose2d(frames)

            face_images = []
            for idx, meta in enumerate(tpl_pose_metas):
                face_bbox_for_image = get_face_bboxes(meta["keypoints_face"][:, :2], scale=1.3, image_shape=(frames[0].shape[0], frames[0].shape[1]))

                x1, x2, y1, y2 = face_bbox_for_image
                face_image = frames[idx][y1:y2, x1:x2]
                face_image = cv2.resize(face_image, (512, 512))
                face_images.append(face_image)

            logger.info(f"Processing reference image: {refer_image_path}")
            refer_img = cv2.imread(refer_image_path)
            src_ref_path = os.path.join(output_path, "src_ref.png")
            shutil.copy(refer_image_path, src_ref_path)
            refer_img = refer_img[..., ::-1]

            refer_img = padding_resize(refer_img, height, width)
            if compose_refer_on_first_frame:
                refer_pose_for_compose = self.pose2d([refer_img])[0]
                refer_fg = None
                refer_fg = self.get_refer_person_mask_sam2(refer_img, refer_pose_for_compose)
                if refer_fg is not None:
                    try:
                        imageio.imwrite(
                            os.path.join(output_path, "refer_sam_mask.png"),
                            (np.clip(refer_fg, 0.0, 1.0) * 255.0).astype(np.uint8),
                        )
                    except Exception:
                        pass

                # Extract SAM2 mask for the first frame (canvas) so the paste mask covers it.
                canvas_cover_mask = None
                try:
                    canvas_mask = self.get_refer_person_mask_sam2(frames[0], tpl_pose_metas[0])
                    if canvas_mask is not None:
                        cm = canvas_mask.astype(np.float32)
                        if cm.max() > 1.5:
                            cm = cm / 255.0
                        cm = np.clip(cm, 0.0, 1.0)
                        canvas_cover_mask = cm
                        imageio.imwrite(
                            os.path.join(output_path, "canvas_sam_mask.png"),
                            (canvas_cover_mask * 255.0).astype(np.uint8),
                        )
                except Exception as e:
                    logger.warning(f"SAM2 canvas-mask extraction failed; continuing without canvas cover. Error: {e}")

                composed, warped_a, aff_m, alpha_ref = compose_refer_by_pose(
                    refer_img,
                    frames[0],
                    refer_pose_for_compose,
                    tpl_pose_metas[0],
                    refer_fg_mask=refer_fg,
                    canvas_cover_mask=canvas_cover_mask,
                    use_masks_for_placement=True,
                )
                # Save compose outputs for debugging/inspection.
                try:
                    imageio.imwrite(os.path.join(output_path, "composited_on_canvas_raw.png"), composed)
                    imageio.imwrite(os.path.join(output_path, "warped_a.png"), (np.clip(warped_a, 0.0, 1.0) * 255.0).astype(np.uint8))
                except Exception:
                    pass
                self._save_composed_with_optional_lbm(
                    composed,
                    output_path,
                    refer_rgb=refer_img,
                    compose_extra={"warped_a": warped_a, "M": aff_m, "alpha_ref": alpha_ref},
                )
            logger.info(f"Processing template video: {video_path}")
            tpl_retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in tpl_pose_metas]
            cond_images = []

            for idx, meta in enumerate(tpl_retarget_pose_metas):
                canvas = np.zeros_like(refer_img)
                conditioning_image = draw_aapose_by_meta_new(canvas, meta)
                cond_images.append(conditioning_image)
            masks = self.get_mask(frames, 400, tpl_pose_metas)

            bg_images = []
            aug_masks = []

            for frame, mask in zip(frames, masks):
                if iterations > 0:
                    _, each_mask = get_mask_body_img(frame, mask, iterations=iterations, k=k)
                    each_aug_mask = get_aug_mask(each_mask, w_len=w_len, h_len=h_len)
                else:
                    each_aug_mask = mask

                each_bg_image = frame * (1 - each_aug_mask[:, :, None])
                bg_images.append(each_bg_image)
                aug_masks.append(each_aug_mask)

            src_face_path = os.path.join(output_path, "src_face.mp4")
            mpy.ImageSequenceClip(face_images, fps=fps).write_videofile(src_face_path)

            src_pose_path = os.path.join(output_path, "src_pose.mp4")
            mpy.ImageSequenceClip(cond_images, fps=fps).write_videofile(src_pose_path)

            src_bg_path = os.path.join(output_path, "src_bg.mp4")
            mpy.ImageSequenceClip(bg_images, fps=fps).write_videofile(src_bg_path)

            aug_masks_new = [np.stack([mask * 255, mask * 255, mask * 255], axis=2) for mask in aug_masks]
            src_mask_path = os.path.join(output_path, "src_mask.mp4")
            mpy.ImageSequenceClip(aug_masks_new, fps=fps).write_videofile(src_mask_path)
            return True
        else:
            logger.info(f"Processing reference image: {refer_image_path}")
            refer_img = cv2.imread(refer_image_path)
            src_ref_path = os.path.join(output_path, "src_ref.png")
            shutil.copy(refer_image_path, src_ref_path)
            refer_img = refer_img[..., ::-1]

            refer_img = resize_by_area(refer_img, resolution_area[0] * resolution_area[1], divisor=16)

            refer_pose_meta = self.pose2d([refer_img])[0]

            # Visualize reference pose right after pose extraction
            refer_pose_meta_np = dict(refer_pose_meta)
            for k in ("keypoints_body", "keypoints_left_hand", "keypoints_right_hand", "keypoints_face"):
                if k in refer_pose_meta_np and refer_pose_meta_np[k] is not None and not isinstance(refer_pose_meta_np[k], np.ndarray):
                    refer_pose_meta_np[k] = np.asarray(refer_pose_meta_np[k])
            refer_pose_canvas = np.zeros_like(refer_img)
            refer_pose_visual = draw_aapose_by_meta_new(refer_pose_canvas, AAPoseMeta.from_humanapi_meta(refer_pose_meta_np))
            refer_pose_path = os.path.join(output_path, "refer_pose.png")
            refer_img_path = os.path.join(output_path, "refer_img.png")

            imageio.imwrite(refer_pose_path, refer_pose_visual)
            imageio.imwrite(refer_img_path, refer_img)

            logger.info(f"Processing template video: {video_path}")
            video_reader = VideoReader(video_path)
            frame_num = len(video_reader)
            print("frame_num: {}".format(frame_num))

            video_fps = video_reader.get_avg_fps()
            print("video_fps: {}".format(video_fps))
            print("fps: {}".format(fps))

            # TODO: Maybe we can switch to PyAV later, which can get accurate frame num
            duration = video_reader.get_frame_timestamp(-1)[-1]
            expected_frame_num = int(duration * video_fps + 0.5)
            ratio = abs((frame_num - expected_frame_num) / frame_num)
            if ratio > 0.1:
                print("Warning: The difference between the actual number of frames and the expected number of frames is two large")
                frame_num = expected_frame_num

            if fps == -1:
                fps = video_fps

            target_num = int(frame_num / video_fps * fps)
            print("target_num: {}".format(target_num))
            idxs = get_frame_indices(frame_num, video_fps, target_num, fps)
            frames = video_reader.get_batch(idxs).asnumpy()

            logger.info(f"Processing pose meta")

            tpl_pose_meta0 = self.pose2d(frames[:1])[0]
            tpl_pose_metas = self.pose2d(frames)

            if compose_refer_on_first_frame:
                composed, warped_a, aff_m, alpha_ref = compose_refer_by_pose(refer_img, frames[0], refer_pose_meta, tpl_pose_meta0)
                self._save_composed_with_optional_lbm(
                    composed,
                    output_path,
                    first_frame_rgb=frames[0],
                    refer_rgb=refer_img,
                    compose_extra={"warped_a": warped_a, "M": aff_m, "alpha_ref": alpha_ref},
                )

            face_images = []
            for idx, meta in enumerate(tpl_pose_metas):
                face_bbox_for_image = get_face_bboxes(meta["keypoints_face"][:, :2], scale=1.3, image_shape=(frames[0].shape[0], frames[0].shape[1]))

                x1, x2, y1, y2 = face_bbox_for_image
                face_image = frames[idx][y1:y2, x1:x2]
                face_image = cv2.resize(face_image, (512, 512))
                face_images.append(face_image)

            if retarget_flag:
                if use_flux:
                    tpl_prompt, refer_prompt = self.get_editing_prompts(tpl_pose_metas, refer_pose_meta)
                    refer_input = Image.fromarray(refer_img)
                    refer_input.save("refer_input.png")
                    print("refer_prompt: ", refer_prompt)
                    # Distilled FLUX.2 Klein typically uses ~4 steps.
                    refer_edit = self.flux_kontext(
                        image=[refer_input],
                        height=refer_img.shape[0],
                        width=refer_img.shape[1],
                        prompt=refer_prompt,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                    ).images[0]

                    refer_edit = Image.fromarray(padding_resize(np.array(refer_edit), refer_img.shape[0], refer_img.shape[1]))
                    refer_edit_path = os.path.join(output_path, "refer_edit.png")
                    refer_edit.save(refer_edit_path)
                    refer_edit_pose_meta = self.pose2d([np.array(refer_edit)])[0]

                    tpl_img = frames[1]
                    tpl_input = Image.fromarray(tpl_img)
                    tpl_input.save("tpl_input.png")
                    print("tpl_prompt: ", tpl_prompt)
                    tpl_edit = self.flux_kontext(
                        image=[tpl_input],
                        height=tpl_img.shape[0],
                        width=tpl_img.shape[1],
                        prompt=tpl_prompt,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                    ).images[0]

                    tpl_edit = Image.fromarray(padding_resize(np.array(tpl_edit), tpl_img.shape[0], tpl_img.shape[1]))
                    tpl_edit_path = os.path.join(output_path, "tpl_edit.png")
                    tpl_edit.save(tpl_edit_path)
                    tpl_edit_pose_meta0 = self.pose2d([np.array(tpl_edit)])[0]
                    tpl_retarget_pose_metas = get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, tpl_edit_pose_meta0, refer_edit_pose_meta)
                else:
                    tpl_retarget_pose_metas = get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, None, None)
            else:
                tpl_retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in tpl_pose_metas]

            cond_images = []
            for idx, meta in enumerate(tpl_retarget_pose_metas):
                if retarget_flag:
                    canvas = np.zeros_like(refer_img)
                    conditioning_image = draw_aapose_by_meta_new(canvas, meta)
                else:
                    canvas = np.zeros_like(frames[0])
                    conditioning_image = draw_aapose_by_meta_new(canvas, meta)
                    conditioning_image = padding_resize(conditioning_image, refer_img.shape[0], refer_img.shape[1])

                cond_images.append(conditioning_image)

            src_face_path = os.path.join(output_path, "src_face.mp4")
            mpy.ImageSequenceClip(face_images, fps=fps).write_videofile(src_face_path)

            src_pose_path = os.path.join(output_path, "src_pose.mp4")
            mpy.ImageSequenceClip(cond_images, fps=fps).write_videofile(src_pose_path)
            return True

    def _ensure_lbm_relight_model(self):
        if self._lbm_relight_model is not None:
            return self._lbm_relight_model
        if not self.lbm_relight_ckpt_dir:
            return None
        from lbm_relight_utils import load_lbm_relighting_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lbm_relight_model = load_lbm_relighting_model(self.lbm_relight_ckpt_dir, device=device)
        return self._lbm_relight_model

    def _save_composed_with_optional_lbm(
        self,
        composed_rgb,
        output_path,
        refer_rgb=None,
        compose_extra=None,
    ):
        """Save composed image, relit composed image, and optional reference wavelet merge."""
        p = output_path
        composed_out = os.path.join(p, "composited_on_canvas.png")
        relit_composed_out = os.path.join(p, "relit_composited_on_canvas.png")
        wave_out = os.path.join(p, "reference_wavelet_merged.png")

        # Always save the composed image.
        imageio.imwrite(composed_out, composed_rgb)

        if not self.lbm_relight or not self.lbm_relight_ckpt_dir:
            return

        try:
            from lbm_relight_utils import lbm_available, relight_numpy_rgb

            if not lbm_available():
                logger.warning("LBM package not installed; skipping reference_wavelet_merged.png")
                return
            if not torch.cuda.is_available():
                logger.warning("LBM expects CUDA; skipping reference_wavelet_merged.png")
                return

            model = self._ensure_lbm_relight_model()
            relit = relight_numpy_rgb(model, composed_rgb, num_sampling_steps=self.lbm_relight_steps)
            imageio.imwrite(relit_composed_out, relit)

            if self.lbm_fuse_mode != "wavelet":
                logger.info("Saved composed + relit composed; skipping reference_wavelet_merged.png (set --lbm_fuse_mode wavelet).")
                return
            if compose_extra is None or refer_rgb is None:
                logger.warning("Cannot build reference_wavelet_merged.png without compose_extra and refer_rgb.")
                return

            relit_on_ref = paste_relit_onto_refer_canvas(
                relit,
                refer_rgb,
                compose_extra["M"],
                compose_extra["alpha_ref"],
                frame_alpha=compose_extra.get("warped_a"),
            )
            imageio.imwrite(os.path.join(p, "relit_on_ref.png"), relit_on_ref)
            try:
                refer_final = fuse_relight_wavelet_dwt2(
                    relit_on_ref,
                    refer_rgb,
                    wavelet=self.lbm_wavelet,
                    level=self.lbm_wavelet_level,
                )
            except ImportError:
                logger.warning("PyWavelets not installed; reference_wavelet_merged.png not written")
                return
            except Exception as e:
                logger.warning(f"Wavelet merge failed; reference_wavelet_merged.png not written: {e}")
                return

            imageio.imwrite(wave_out, refer_final)
            logger.info(f"Saved {wave_out}")
        except Exception as e:
            logger.warning(f"LBM / reference_wavelet_merged.png failed: {e}")

    def get_editing_prompts(self, tpl_pose_metas, refer_pose_meta):
        arm_visible = False
        leg_visible = False
        for tpl_pose_meta in tpl_pose_metas:
            tpl_keypoints = tpl_pose_meta["keypoints_body"]
            if tpl_keypoints[3].all() != 0 or tpl_keypoints[4].all() != 0 or tpl_keypoints[6].all() != 0 or tpl_keypoints[7].all() != 0:
                if (
                    (tpl_keypoints[3][0] <= 1 and tpl_keypoints[3][1] <= 1 and tpl_keypoints[3][2] >= 0.75)
                    or (tpl_keypoints[4][0] <= 1 and tpl_keypoints[4][1] <= 1 and tpl_keypoints[4][2] >= 0.75)
                    or (tpl_keypoints[6][0] <= 1 and tpl_keypoints[6][1] <= 1 and tpl_keypoints[6][2] >= 0.75)
                    or (tpl_keypoints[7][0] <= 1 and tpl_keypoints[7][1] <= 1 and tpl_keypoints[7][2] >= 0.75)
                ):
                    arm_visible = True
            if tpl_keypoints[9].all() != 0 or tpl_keypoints[12].all() != 0 or tpl_keypoints[10].all() != 0 or tpl_keypoints[13].all() != 0:
                if (
                    (tpl_keypoints[9][0] <= 1 and tpl_keypoints[9][1] <= 1 and tpl_keypoints[9][2] >= 0.75)
                    or (tpl_keypoints[12][0] <= 1 and tpl_keypoints[12][1] <= 1 and tpl_keypoints[12][2] >= 0.75)
                    or (tpl_keypoints[10][0] <= 1 and tpl_keypoints[10][1] <= 1 and tpl_keypoints[10][2] >= 0.75)
                    or (tpl_keypoints[13][0] <= 1 and tpl_keypoints[13][1] <= 1 and tpl_keypoints[13][2] >= 0.75)
                ):
                    leg_visible = True
            if arm_visible and leg_visible:
                break

        if leg_visible:
            if tpl_pose_meta["width"] > tpl_pose_meta["height"]:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."

            if refer_pose_meta["width"] > refer_pose_meta["height"]:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."
        elif arm_visible:
            if tpl_pose_meta["width"] > tpl_pose_meta["height"]:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."

            if refer_pose_meta["width"] > refer_pose_meta["height"]:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."
        else:
            tpl_prompt = "Change the person to face forward."
            refer_prompt = "Change the person to face forward."

        tpl_prompt += " KEEP THE PERSON'S IDENTITY. KEEP THE ORIGINAL FACIAL EXPRESSION, AND FACE ANGLE CONSISTENT. MAINTAIN THE EXISTING BACKGROUND. PRESERVE THE LIGHTING AND OVERALL COMPOSITION OF THE IMAGE."
        refer_prompt += " KEEP THE PERSON'S IDENTITY. KEEP THE ORIGINAL FACIAL EXPRESSION, AND FACE ANGLE CONSISTENT. MAINTAIN THE EXISTING BACKGROUND. PRESERVE THE LIGHTING AND OVERALL COMPOSITION OF THE IMAGE."
        return tpl_prompt, refer_prompt

    def get_refer_person_mask_sam2(self, refer_rgb, pose_meta):
        """Segment the person on the reference image (same SAM2 path as video).

        Returns:
            float32 (H,W) in [0,1] or None.
        """
        pred = getattr(self, "predictor", None)
        if pred is None:
            return None
        meta = self.convert_list_to_array([dict(pose_meta)])[0]
        m = self.get_mask_one_frame(refer_rgb, meta)
        if m is None:
            return None
        m = m.astype(np.float32)
        if m.max() > 1.5:
            m = m / 255.0
        m = np.clip(m, 0.0, 1.0)
        h, w = refer_rgb.shape[:2]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        if float(m.sum()) < 64.0:
            return None
        return m

    def get_mask(self, frames, th_step, kp2ds_all):
        frame_num = len(frames)
        if frame_num < th_step:
            num_step = 1
        else:
            num_step = (frame_num + th_step) // th_step

        all_mask = []
        for index in range(num_step):
            each_frames = frames[index * th_step : (index + 1) * th_step]

            kp2ds = kp2ds_all[index * th_step : (index + 1) * th_step]
            if len(each_frames) > 4:
                key_frame_num = 4
            elif 4 >= len(each_frames) > 0:
                key_frame_num = 1
            else:
                continue

            key_frame_step = len(kp2ds) // key_frame_num
            key_frame_index_list = list(range(0, len(kp2ds), key_frame_step))

            key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]
            key_frame_body_points_list = []
            for key_frame_index in key_frame_index_list:
                keypoints_body_list = []
                body_key_points = kp2ds[key_frame_index]["keypoints_body"]
                for each_index in key_points_index:
                    each_keypoint = body_key_points[each_index]
                    if None is each_keypoint:
                        continue
                    keypoints_body_list.append(each_keypoint)

                keypoints_body = np.array(keypoints_body_list)[:, :2]
                wh = np.array([[kp2ds[0]["width"], kp2ds[0]["height"]]])
                points = (keypoints_body * wh).astype(np.int32)
                key_frame_body_points_list.append(points)

            inference_state = self.predictor.init_state_v2(frames=each_frames)
            self.predictor.reset_state(inference_state)
            ann_obj_id = 1
            for ann_frame_idx, points in zip(key_frame_index_list, key_frame_body_points_list):
                labels = np.array([1] * points.shape[0], np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}

            for out_frame_idx in range(len(video_segments)):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    out_mask = out_mask[0].astype(np.uint8)
                    all_mask.append(out_mask)

        return all_mask

    def get_mask_one_frame(self, frame, kp2d):
        """
        SAM2 foreground mask for a single frame using pose body keypoints as positive prompts.

        Args:
            frame: np.ndarray [H, W, 3]
            kp2d: dict containing keys:
                - "keypoints_body" (normalized keypoints [N, 3] or equivalent)
                - "width", "height"

        Returns:
            np.ndarray [H, W] uint8 mask, or None if unavailable.
        """
        pred = getattr(self, "predictor", None)
        if pred is None:
            return None
        if frame is None or kp2d is None:
            return None

        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]
        keypoints_body_list = []
        body_key_points = kp2d.get("keypoints_body", None)
        if body_key_points is None:
            return None

        for each_index in key_points_index:
            if each_index >= len(body_key_points):
                continue
            each_keypoint = body_key_points[each_index]
            if each_keypoint is None:
                continue
            keypoints_body_list.append(each_keypoint)

        if len(keypoints_body_list) == 0:
            return None

        keypoints_body = np.array(keypoints_body_list)[:, :2]
        wh = np.array([[kp2d["width"], kp2d["height"]]])
        keypoints_px = keypoints_body * wh
        center_point = keypoints_px.mean(axis=0, keepdims=True)
        points = center_point.astype(np.int32)

        labels = np.array([1], np.int32)
        out_mask = None
        image_predictor = getattr(self, "image_predictor", None)
        if image_predictor is not None:
            image_predictor.set_image(frame)
            masks, scores, _ = image_predictor.predict(
                point_coords=points.astype(np.float32),
                point_labels=labels,
                multimask_output=True,
            )
            if masks is not None and len(masks) > 0:
                masks_bin = (masks > 0).astype(np.uint8)
                areas = masks_bin.reshape(masks_bin.shape[0], -1).sum(axis=1).astype(np.float32)
                if scores is None:
                    scores = np.zeros((masks_bin.shape[0],), dtype=np.float32)
                else:
                    scores = np.asarray(scores, dtype=np.float32)
                    if scores.shape[0] != masks_bin.shape[0]:
                        scores = np.zeros((masks_bin.shape[0],), dtype=np.float32)
                # Choose whole-person candidate: largest area first, score as tie-break.
                best_idx = int(np.argmax(areas * 1e3 + scores))
                out_mask = masks_bin[best_idx]
        else:
            # Fallback to video predictor on a single-frame sequence.
            inference_state = self.predictor.init_state_v2(frames=[frame])
            self.predictor.reset_state(inference_state)
            self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                if out_frame_idx != 0:
                    continue
                for i, _ in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0].astype(np.uint8)
                    break
                if out_mask is not None:
                    break

        return out_mask

    def convert_list_to_array(self, metas):
        metas_list = []
        for meta in metas:
            for key, value in meta.items():
                if type(value) is list:
                    value = np.array(value)
                meta[key] = value
            metas_list.append(meta)
        return metas_list
