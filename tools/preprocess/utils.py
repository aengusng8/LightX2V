# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import random

import cv2
import numpy as np

from pose2d_utils import AAPoseMeta


def get_mask_boxes(mask):
    """

    Args:
        mask: [h, w]
    Returns:

    """
    y_coords, x_coords = np.nonzero(mask)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(np.int32)
    return bbox


def get_aug_mask(body_mask, w_len=10, h_len=20):
    body_bbox = get_mask_boxes(body_mask)

    bbox_wh = body_bbox[2:4] - body_bbox[0:2]
    w_slice = np.int32(bbox_wh[0] / w_len)
    h_slice = np.int32(bbox_wh[1] / h_len)

    for each_w in range(body_bbox[0], body_bbox[2], w_slice):
        w_start = min(each_w, body_bbox[2])
        w_end = min((each_w + w_slice), body_bbox[2])
        # print(w_start, w_end)
        for each_h in range(body_bbox[1], body_bbox[3], h_slice):
            h_start = min(each_h, body_bbox[3])
            h_end = min((each_h + h_slice), body_bbox[3])
            if body_mask[h_start:h_end, w_start:w_end].sum() > 0:
                body_mask[h_start:h_end, w_start:w_end] = 1

    return body_mask


def get_mask_body_img(img_copy, hand_mask, k=7, iterations=1):
    kernel = np.ones((k, k), np.uint8)
    dilation = cv2.dilate(hand_mask, kernel, iterations=iterations)
    mask_hand_img = img_copy * (1 - dilation[:, :, None])

    return mask_hand_img, dilation


def get_face_bboxes(kp2ds, scale, image_shape, ratio_aug):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[23:91, :2]

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    if ratio_aug:
        if random.random() > 0.5:
            delta_width += random.uniform(0, initial_width // 10)
        else:
            delta_height += random.uniform(0, initial_height // 10)

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]


def calculate_new_size(orig_w, orig_h, target_area, divisor=64):
    target_ratio = orig_w / orig_h

    def check_valid(w, h):
        if w <= 0 or h <= 0:
            return False
        return w * h <= target_area and w % divisor == 0 and h % divisor == 0

    def get_ratio_diff(w, h):
        return abs(w / h - target_ratio)

    def round_to_64(value, round_up=False, divisor=64):
        if round_up:
            return divisor * ((value + (divisor - 1)) // divisor)
        return divisor * (value // divisor)

    possible_sizes = []

    max_area_h = int(np.sqrt(target_area / target_ratio))
    max_area_w = int(max_area_h * target_ratio)

    max_h = round_to_64(max_area_h, round_up=True, divisor=divisor)
    max_w = round_to_64(max_area_w, round_up=True, divisor=divisor)

    for h in range(divisor, max_h + divisor, divisor):
        ideal_w = h * target_ratio

        w_down = round_to_64(ideal_w)
        w_up = round_to_64(ideal_w, round_up=True)

        for w in [w_down, w_up]:
            if check_valid(w, h, divisor):
                possible_sizes.append((w, h, get_ratio_diff(w, h)))

    if not possible_sizes:
        raise ValueError("Can not find suitable size")

    possible_sizes.sort(key=lambda x: (-x[0] * x[1], x[2]))

    best_w, best_h, _ = possible_sizes[0]
    return int(best_w), int(best_h)


def resize_by_area(image, target_area, keep_aspect_ratio=True, divisor=64, padding_color=(0, 0, 0)):
    h, w = image.shape[:2]
    try:
        new_w, new_h = calculate_new_size(w, h, target_area, divisor)
    except:  # noqa
        aspect_ratio = w / h

        if keep_aspect_ratio:
            new_h = math.sqrt(target_area / aspect_ratio)
            new_w = target_area / new_h
        else:
            new_w = new_h = math.sqrt(target_area)

        new_w, new_h = int((new_w // divisor) * divisor), int((new_h // divisor) * divisor)

    interpolation = cv2.INTER_AREA if (new_w * new_h < w * h) else cv2.INTER_LINEAR

    resized_image = padding_resize(image, height=new_h, width=new_w, padding_color=padding_color, interpolation=interpolation)
    return resized_image


def padding_resize(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    ori_height = img_ori.shape[0]
    ori_width = img_ori.shape[1]
    channel = img_ori.shape[2]

    img_pad = np.zeros((height, width, channel))
    if channel == 1:
        img_pad[:, :, 0] = padding_color[0]
    else:
        img_pad[:, :, 0] = padding_color[0]
        img_pad[:, :, 1] = padding_color[1]
        img_pad[:, :, 2] = padding_color[2]

    if (ori_height / ori_width) > (height / width):
        new_width = int(height / ori_height * ori_width)
        img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
        padding = int((width - new_width) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img_pad[:, padding : padding + new_width, :] = img
    else:
        new_height = int(width / ori_width * ori_height)
        img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
        padding = int((height - new_height) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img_pad[padding : padding + new_height, :, :] = img

    img_pad = np.uint8(img_pad)

    return img_pad


def get_frame_indices(frame_num, video_fps, clip_length, train_fps):
    start_frame = 0
    times = np.arange(0, clip_length) / train_fps
    frame_indices = start_frame + np.round(times * video_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, frame_num - 1)

    return frame_indices.tolist()


def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[1:] * (w, h)

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]


def _meta_dict_to_aa(meta):
    m = dict(meta)
    for k in ("keypoints_body", "keypoints_left_hand", "keypoints_right_hand", "keypoints_face"):
        if k in m and m[k] is not None and not isinstance(m[k], np.ndarray):
            m[k] = np.asarray(m[k])
    return AAPoseMeta.from_humanapi_meta(m)


def skeleton_reference_length(aa, p_thresh=0.3):
    """Characteristic body scale in pixels: shoulder width, else neck–mid-hip, else bbox diagonal."""
    kb, kb_p = aa.kps_body, aa.kps_body_p
    # Body layout: 0 nose, 1 neck, 2 RShoulder, …, 5 LShoulder, 8 RHip, 11 LHip
    if kb_p[2] >= p_thresh and kb_p[5] >= p_thresh:
        return float(np.linalg.norm(kb[5] - kb[2]))
    if kb_p[1] >= p_thresh and kb_p[8] >= p_thresh and kb_p[11] >= p_thresh:
        mid_hip = (kb[8] + kb[11]) / 2.0
        return float(np.linalg.norm(kb[1] - mid_hip))
    mask = kb_p >= p_thresh
    if np.count_nonzero(mask) >= 2:
        pts = kb[mask]
        span = pts.max(axis=0) - pts.min(axis=0)
        return float(np.linalg.norm(span))
    return None


def skeleton_anchor_point(aa, p_thresh=0.3):
    """Pixel anchor for alignment: neck, else nose, else centroid of confident body keypoints."""
    kb, kb_p = aa.kps_body, aa.kps_body_p
    if kb_p[1] >= p_thresh:
        return kb[1].copy()
    if kb_p[0] >= p_thresh:
        return kb[0].copy()
    mask = kb_p >= p_thresh
    if np.count_nonzero(mask) > 0:
        return kb[mask].mean(axis=0)
    return np.array([aa.width / 2.0, aa.height / 2.0], dtype=np.float64)


def eye_center_from_meta(meta, p_thresh=0.3):
    """
    Eye center from face landmarks in AAPoseMeta coordinate system (pixels).

    human_visualization.py maps:
      - left_eye:  indexs [36..41]
      - right_eye: indexs [42..47]
    """
    aa = _meta_dict_to_aa(meta)
    if aa.kps_face is None or aa.kps_face_p is None:
        return None
    if aa.kps_face.shape[0] < 48:
        return None

    left_idx = np.arange(36, 42)
    right_idx = np.arange(42, 48)
    left_p = aa.kps_face_p[left_idx]
    right_p = aa.kps_face_p[right_idx]

    left_pts = aa.kps_face[left_idx][left_p >= p_thresh]
    right_pts = aa.kps_face[right_idx][right_p >= p_thresh]

    left_mean = None if left_pts.shape[0] == 0 else left_pts.mean(axis=0)
    right_mean = None if right_pts.shape[0] == 0 else right_pts.mean(axis=0)

    if left_mean is not None and right_mean is not None:
        return (left_mean + right_mean) / 2.0
    if left_mean is not None:
        return left_mean
    if right_mean is not None:
        return right_mean
    return None


def build_person_mask_from_pose(
    refer_rgb_shape,
    meta,
    p_thresh=0.3,
    dilate_kernel=35,
    dilate_iters=5,
):
    """
    Binary person mask from 2D pose (body + hands + face): convex hull of confident keypoints,
    then hole-filling so the region is solid (no morphological dilation).
    """
    h, w = int(refer_rgb_shape[0]), int(refer_rgb_shape[1])
    aa = _meta_dict_to_aa(meta)
    sx = w / float(max(aa.width, 1))
    sy = h / float(max(aa.height, 1))
    pts = []
    for kp, kp_p in (
        (aa.kps_body, aa.kps_body_p),
        (aa.kps_lhand, aa.kps_lhand_p),
        (aa.kps_rhand, aa.kps_rhand_p),
        (aa.kps_face, aa.kps_face_p),
    ):
        if kp is None or kp_p is None or len(kp) == 0:
            continue
        for i in range(len(kp)):
            if kp_p[i] >= p_thresh:
                pts.append([kp[i, 0] * sx, kp[i, 1] * sy])
    pts = np.asarray(pts, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    dk = max(5, int(dilate_kernel))
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    elif len(pts) == 2:
        p0 = tuple(np.clip(pts[0].astype(int), [0, 0], [w - 1, h - 1]))
        p1 = tuple(np.clip(pts[1].astype(int), [0, 0], [w - 1, h - 1]))
        cv2.line(mask, p0, p1, 255, thickness=dk)
    elif len(pts) == 1:
        c = tuple(np.clip(pts[0].astype(int), [0, 0], [w - 1, h - 1]))
        cv2.circle(mask, c, dk * 2, 255, -1)
    else:
        mask[:, :] = 255

    # Fill holes inside the drawn region.
    # Method: flood-fill the background from the image border, then keep the
    # pixels that were *not* reachable (holes) and union them back.
    mask_bin = (mask > 0).astype(np.uint8) * 255
    flood = mask_bin.copy()
    h, w = mask_bin.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    # If (0,0) is inside the object (rare), pick another border pixel.
    seed = (0, 0)
    if mask_bin[seed[1], seed[0]] > 0:
        # find any border background pixel
        found = False
        for x in range(w):
            if mask_bin[0, x] == 0:
                seed = (x, 0)
                found = True
                break
        if not found:
            for x in range(w):
                if mask_bin[h - 1, x] == 0:
                    seed = (x, h - 1)
                    found = True
                    break
        if not found:
            for y in range(h):
                if mask_bin[y, 0] == 0:
                    seed = (0, y)
                    found = True
                    break
        if not found:
            # fallback: no holes to fill
            return (mask_bin.astype(np.float32) / 255.0).clip(0.0, 1.0)

    cv2.floodFill(flood, ff_mask, seedPoint=seed, newVal=255)
    holes = ((flood == 0) & (mask_bin == 0)).astype(np.uint8) * 255
    filled = cv2.bitwise_or(mask_bin, holes)
    return (filled.astype(np.float32) / 255.0).clip(0.0, 1.0)


def compose_refer_on_first_frame(
    refer_rgb,
    first_frame_rgb,
    refer_meta,
    tpl_meta,
    p_thresh=0.3,
    scale_min=0.12,
    scale_max=8.0,
    mask_value_thresh=24,
    refer_fg_mask=None,
    mask_feather_sigma=1.5,
    ensure_pose_cover=True,
    pose_cover_thresh=0.5,
    pose_cover_dilate_ratio=0.02,
    canvas_cover_mask=None,
    use_masks_for_placement=False,
    masks_bin_thresh=0.5,
):
    """
    Scale the reference RGB image using pose skeleton scale (shoulder span / torso length),
    align anchor (neck) to the first-frame pose, and composite the foreground onto the first frame.

    refer_fg_mask: optional float32 (H, W) in [0, 1], same size as refer_rgb — person segmentation.
    If None, a mask is built from pose keypoints (convex hull + dilation).
    """
    if refer_rgb.dtype != np.uint8:
        refer_rgb = np.clip(refer_rgb, 0, 255).astype(np.uint8)
    if first_frame_rgb.dtype != np.uint8:
        first_frame_rgb = np.clip(first_frame_rgb, 0, 255).astype(np.uint8)

    h_r, w_r = refer_rgb.shape[:2]
    if refer_fg_mask is None:
        alpha_ref = build_person_mask_from_pose(refer_rgb.shape, refer_meta, p_thresh=0)
    else:
        alpha_ref = np.clip(refer_fg_mask.astype(np.float32), 0.0, 1.0)
        if alpha_ref.shape[0] != h_r or alpha_ref.shape[1] != w_r:
            alpha_ref = cv2.resize(alpha_ref, (w_r, h_r), interpolation=cv2.INTER_LINEAR)
        # Fill holes inside the SAM2 foreground mask without dilating/expanding.
        bin_mask = (alpha_ref > 0.5).astype(np.uint8) * 255
        flood = bin_mask.copy()
        ff_mask = np.zeros((h_r + 2, w_r + 2), dtype=np.uint8)
        seed = None
        for x in range(w_r):
            if bin_mask[0, x] == 0:
                seed = (x, 0)
                break
        if seed is None:
            for x in range(w_r):
                if bin_mask[h_r - 1, x] == 0:
                    seed = (x, h_r - 1)
                    break
        if seed is None:
            for y in range(h_r):
                if bin_mask[y, 0] == 0:
                    seed = (0, y)
                    break
        if seed is None:
            for y in range(h_r):
                if bin_mask[y, w_r - 1] == 0:
                    seed = (w_r - 1, y)
                    break
        if seed is not None:
            cv2.floodFill(flood, ff_mask, seedPoint=seed, newVal=255)
            holes = ((flood == 0) & (bin_mask == 0)).astype(np.uint8) * 255
            bin_filled = cv2.bitwise_or(bin_mask, holes)
            alpha_ref = (bin_filled.astype(np.float32) / 255.0).clip(0.0, 1.0)

    # Compute scale and translation either from pose, or from masks.
    if use_masks_for_placement and canvas_cover_mask is not None:
        cover = np.clip(canvas_cover_mask.astype(np.float32), 0.0, 1.0)
        if cover.shape[0] != first_frame_rgb.shape[0] or cover.shape[1] != first_frame_rgb.shape[1]:
            cover = cv2.resize(
                cover,
                (first_frame_rgb.shape[1], first_frame_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        ref_bin = (alpha_ref > float(masks_bin_thresh))
        cover_bin = (cover > float(masks_bin_thresh))
        h_t, w_t = first_frame_rgb.shape[:2]

        ref_y, ref_x = np.nonzero(ref_bin)
        cov_y, cov_x = np.nonzero(cover_bin)
        if ref_x.size > 0 and cov_x.size > 0:
            ref_x0, ref_x1 = ref_x.min(), ref_x.max()
            ref_y0, ref_y1 = ref_y.min(), ref_y.max()
            cov_x0, cov_x1 = cov_x.min(), cov_x.max()
            cov_y0, cov_y1 = cov_y.min(), cov_y.max()

            ref_h = max(1, int(ref_y1 - ref_y0 + 1))
            ref_w = max(1, int(ref_x1 - ref_x0 + 1))
            cov_h = max(1, int(cov_y1 - cov_y0 + 1))
            cov_w = max(1, int(cov_x1 - cov_x0 + 1))

            # Scale from mask sizes (prefer max ratio to ensure coverage).
            s_h = cov_h / float(ref_h)
            s_w = cov_w / float(ref_w)
            s = float(max(s_h, s_w) * 1.03)
            s = float(np.clip(s, scale_min, scale_max))

            # Translate using eye-center alignment from pose.
            eye_ref = eye_center_from_meta(refer_meta, p_thresh=p_thresh)
            eye_cov = eye_center_from_meta(tpl_meta, p_thresh=p_thresh)

            if eye_ref is not None and eye_cov is not None:
                tx = float(eye_cov[0] - s * eye_ref[0])
                ty = float(eye_cov[1] - s * eye_ref[1])

                # Coverage refinement: ensure pasted reference mask covers the
                # frame (canvas) mask as much as possible.
                cover_count = float(cover_bin.sum())
                if cover_count > 0:
                    ref_u8 = ref_bin.astype(np.uint8)
                    cover_u8 = cover_bin.astype(np.uint8)
                    target_frac = 0.995
                    check_iters = 5
                    scale_step = 1.03

                    for _ in range(check_iters):
                        new_w = max(1, int(round(w_r * float(s))))
                        new_h = max(1, int(round(h_r * float(s))))
                        resized_ref = cv2.resize(ref_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                        pasted = np.zeros((h_t, w_t), dtype=np.uint8)
                        x0 = int(round(tx))
                        y0 = int(round(ty))
                        x1 = max(0, x0)
                        y1 = max(0, y0)
                        x2 = min(w_t, x0 + new_w)
                        y2 = min(h_t, y0 + new_h)
                        if x2 > x1 and y2 > y1:
                            src_x1 = x1 - x0
                            src_y1 = y1 - y0
                            src_x2 = src_x1 + (x2 - x1)
                            src_y2 = src_y1 + (y2 - y1)
                            pasted[y1:y2, x1:x2] = resized_ref[src_y1:src_y2, src_x1:src_x2]

                        frac = float((pasted * cover_u8).sum()) / cover_count
                        if frac >= target_frac:
                            break
                        s = float(np.clip(s * scale_step, scale_min, scale_max))
                        # Keep eye-center aligned as scale changes.
                        tx = float(eye_cov[0] - s * eye_ref[0])
                        ty = float(eye_cov[1] - s * eye_ref[1])
            else:
                # Fallback translation if face landmarks are missing.
                ref_cx = 0.5 * (ref_x0 + ref_x1)
                ref_cy = 0.5 * (ref_y0 + ref_y1)
                cov_cx = 0.5 * (cov_x0 + cov_x1)
                cov_cy = 0.5 * (cov_y0 + cov_y1)
                tx = float(cov_cx - s * ref_cx)
                ty = float(cov_cy - s * ref_cy)
        else:
            use_masks_for_placement = False  # fall back to pose

    if not use_masks_for_placement:
        aa_r = _meta_dict_to_aa(refer_meta)
        aa_t = _meta_dict_to_aa(tpl_meta)

        len_r = skeleton_reference_length(aa_r, p_thresh=p_thresh)
        len_t = skeleton_reference_length(aa_t, p_thresh=p_thresh)
        if len_r is not None and len_t is not None and len_r > 1e-3:
            s = len_t / len_r
        else:
            s = 1.0

        s = float(np.clip(s, scale_min, scale_max))

        anchor_r = skeleton_anchor_point(aa_r, p_thresh=p_thresh)
        anchor_t = skeleton_anchor_point(aa_t, p_thresh=p_thresh)

        tx = float(anchor_t[0] - s * anchor_r[0])
        ty = float(anchor_t[1] - s * anchor_r[1])

    h_t, w_t = first_frame_rgb.shape[:2]
    M = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float64)

    # Paste-only mode: resize + translate paste (no warpAffine for RGB).
    new_w = max(1, int(round(w_r * float(s))))
    new_h = max(1, int(round(h_r * float(s))))
    resized = cv2.resize(refer_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = int(round(tx))
    y0 = int(round(ty))

    ov = np.zeros((h_t, w_t, 3), dtype=np.uint8)
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(w_t, x0 + new_w)
    y2 = min(h_t, y0 + new_h)

    if x2 > x1 and y2 > y1:
        src_x1 = x1 - x0
        src_y1 = y1 - y0
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        ov[y1:y2, x1:x2] = resized[src_y1:src_y2, src_x1:src_x2]
    warped = ov

    # Build compositing mask in first-frame space via resize+translate paste.
    resized_a = cv2.resize(alpha_ref, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    warped_a = np.zeros((h_t, w_t), dtype=np.float32)
    if x2 > x1 and y2 > y1:
        src_x1 = x1 - x0
        src_y1 = y1 - y0
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        warped_a[y1:y2, x1:x2] = resized_a[src_y1:src_y2, src_x1:src_x2]
    warped_a = np.clip(warped_a, 0.0, 1.0)

    # No extra post-processing: use pasted reference mask directly.

    base = first_frame_rgb.astype(np.float32)
    ov = warped.astype(np.float32)
    alpha = warped_a[:, :, None]
    if float(alpha.sum()) < 1e-3:
        alpha = (np.max(ov, axis=2) > mask_value_thresh).astype(np.float32)[:, :, None]
    out = base * (1.0 - alpha) + ov * alpha
    composed = np.clip(out, 0, 255).astype(np.uint8)
    alpha2d = alpha.squeeze(-1) if alpha.ndim == 3 else alpha
    # alpha2d: person alpha in first-frame space (matches compose blend); M: 2x3 affine refer -> frame; alpha_ref: on refer
    return composed, alpha2d.astype(np.float32), M, alpha_ref.astype(np.float32)


def blend_relit_foreground_on_original(original_rgb, relit_rgb, fg_alpha):
    """Composite relit_rgb using fg_alpha onto original_rgb (same HxW)."""
    if original_rgb.dtype != np.uint8:
        original_rgb = np.clip(original_rgb, 0, 255).astype(np.uint8)
    if relit_rgb.dtype != np.uint8:
        relit_rgb = np.clip(relit_rgb, 0, 255).astype(np.uint8)
    a = np.clip(fg_alpha.astype(np.float32), 0.0, 1.0)
    a = a[:, :, None]
    out = original_rgb.astype(np.float32) * (1.0 - a) + relit_rgb.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_relit_affine_to_refer(relit_on_frame_rgb, M, refer_hw):
    """Inverse affine: map relit (same space as first frame) onto reference canvas shape refer_hw (H, W)."""
    h_r, w_r = int(refer_hw[0]), int(refer_hw[1])
    Minv = cv2.invertAffineTransform(M.astype(np.float64))
    img = np.clip(relit_on_frame_rgb, 0, 255).astype(np.uint8)
    return cv2.warpAffine(img, Minv, (w_r, h_r), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def fuse_relight_wavelet_dwt2(relit_rgb, original_rgb, wavelet="haar", level=2, mode="symmetric"):
    """
    Discrete 2D wavelet decomposition: use LL from relit and LH/HL/HH from original,
    then inverse DWT. Operates on all RGB channels together (no channel loop).
    Requires WaveDiff-style DWT layers:
      from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
    """
    try:
        import torch
        from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
    except Exception as e:
        raise ImportError(
            "WaveDiff DWT layers not available. Install/provide "
            "`DWT_IDWT.DWT_IDWT_layer` with `DWT_2D` and `IDWT_2D`."
        ) from e

    h, w = original_rgb.shape[:2]
    relit = np.asarray(relit_rgb, dtype=np.float32)
    if relit.shape[0] != h or relit.shape[1] != w:
        relit = cv2.resize(relit, (w, h), interpolation=cv2.INTER_LINEAR)
    orig = np.asarray(original_rgb, dtype=np.float32)

    level = int(max(1, level))
    if wavelet != "haar":
        raise ValueError("DWT_2D/IDWT_2D path currently supports only wavelet='haar'.")
    _ = mode  # kept for signature compatibility

    # [H, W, 3] -> [1, 3, H, W], all channels together.
    xr = torch.from_numpy(relit).permute(2, 0, 1).unsqueeze(0).float()
    xo = torch.from_numpy(orig).permute(2, 0, 1).unsqueeze(0).float()

    dwt = DWT_2D("haar")
    idwt = IDWT_2D("haar")

    ll_r = xr
    ll_o = xo
    hi_o_levels = []
    for _ in range(level):
        ll_r, _, _, _ = dwt(ll_r)
        ll_o, xlh, xhl, xhh = dwt(ll_o)
        hi_o_levels.append((xlh, xhl, xhh))

    rec = ll_r
    for xlh, xhl, xhh in reversed(hi_o_levels):
        rec = idwt(rec, xlh, xhl, xhh)

    out = rec.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    return np.clip(out, 0, 255).astype(np.uint8)


def paste_relit_onto_refer_canvas(
    relit_on_frame_rgb, refer_rgb, M, alpha_ref, mask_feather_sigma=1.5, frame_alpha=None
):
    """
    Map relit full-frame result back onto the reference image canvas using inverse affine,
    then blend with alpha_ref (person region on reference).
    """
    if refer_rgb.dtype != np.uint8:
        refer_rgb = np.clip(refer_rgb, 0, 255).astype(np.uint8)
    h_r, w_r = refer_rgb.shape[:2]
    relit_on_ref = warp_relit_affine_to_refer(relit_on_frame_rgb, M, (h_r, w_r))
    a = np.clip(alpha_ref.astype(np.float32), 0.0, 1.0)
    if frame_alpha is not None:
        frame_alpha = np.clip(frame_alpha.astype(np.float32), 0.0, 1.0)
        if frame_alpha.shape[:2] != relit_on_frame_rgb.shape[:2]:
            frame_alpha = cv2.resize(
                frame_alpha,
                (relit_on_frame_rgb.shape[1], relit_on_frame_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        Minv = cv2.invertAffineTransform(np.asarray(M, dtype=np.float32))
        valid_on_ref = cv2.warpAffine(
            frame_alpha,
            Minv,
            (w_r, h_r),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        a = a * np.clip(valid_on_ref, 0.0, 1.0)
    if mask_feather_sigma and mask_feather_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=float(mask_feather_sigma), sigmaY=float(mask_feather_sigma))
    a = a[:, :, None]
    out = refer_rgb.astype(np.float32) * (1.0 - a) + relit_on_ref.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)
