# Copyright note: inference logic adapted from LBM (Latent Bridge Matching), ICCV 2025.
# https://github.com/gojasper/LBM — licensed under CC BY-NC 4.0.
# Weights: https://huggingface.co/jasperai/LBM_relighting
"""Optional relighting via Jasper LBM. Install the package first, e.g.:
    pip install git+https://github.com/gojasper/LBM.git
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

# Same supported aspect ratios as upstream LBM inference (see src/lbm/inference/inference.py).
ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}


def lbm_available() -> bool:
    try:
        import lbm.inference.utils  # noqa: F401

        return True
    except ImportError:
        return False


def load_lbm_relighting_model(ckpt_dir: str, device: str | None = None, torch_dtype=torch.bfloat16):
    """
    Load LBM relighting weights from a local directory (yaml + safetensors/ckpt), or download
    jasperai/LBM_relighting into ckpt_dir if the folder has no config yet.
    """
    from lbm.inference.utils import get_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_repo = "jasperai/LBM_relighting"
    has_yaml = False
    if os.path.isdir(ckpt_dir):
        try:
            has_yaml = any(f.endswith(".yaml") for f in os.listdir(ckpt_dir))
        except OSError:
            has_yaml = False

    if has_yaml:
        return get_model(ckpt_dir, save_dir=None, torch_dtype=torch_dtype, device=device)

    os.makedirs(ckpt_dir, exist_ok=True)
    return get_model(hf_repo, save_dir=ckpt_dir, torch_dtype=torch_dtype, device=device)


@torch.no_grad()
def relight_pil(model, source_image: Image.Image, num_sampling_steps: int = 1) -> Image.Image:
    """
    Relight a pasted composite (foreground on background) to match target lighting.
    Upstream LBM returns a PIL image that omitted assigning resize(); we resize back explicitly.
    """
    ori_w, ori_h = source_image.size
    ar = ori_h / max(ori_w, 1)
    closest_ar = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar))
    source_dimensions = ASPECT_RATIOS[closest_ar]

    resized = source_image.resize(source_dimensions, Image.Resampling.LANCZOS)
    img_tensor = ToTensor()(resized).unsqueeze(0) * 2 - 1
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch = {"source_image": img_tensor.to(device=dev, dtype=dtype)}

    z_source = model.vae.encode(batch[model.source_key])
    output = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    output = (output[0].float().cpu() + 1) / 2
    out_pil = ToPILImage()(output)
    out_pil = out_pil.resize((ori_w, ori_h), Image.Resampling.LANCZOS)
    return out_pil


def relight_numpy_rgb(model, rgb_uint8: np.ndarray, num_sampling_steps: int = 1) -> np.ndarray:
    pil = Image.fromarray(np.asarray(rgb_uint8, dtype=np.uint8))
    out = relight_pil(model, pil, num_sampling_steps=num_sampling_steps)
    return np.asarray(out.convert("RGB"), dtype=np.uint8)
