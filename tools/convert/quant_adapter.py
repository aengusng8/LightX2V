import argparse
import re
import sys
from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file

sys.path.append(str(Path(__file__).parent.parent.parent))

quant_path = str(Path(__file__).parent / "quant")
if quant_path not in sys.path:
    sys.path.insert(0, quant_path)

from quant import *  # noqa: E402

from lightx2v.utils.quant_utils import FloatQuantizer  # noqa: E402
from lightx2v.utils.utils import load_pt_safetensors  # noqa: E402


WAN_ANIMATE_ADAPTER_PATTERN = re.compile(r"^face_adapter\.fuser_blocks\.\d+\.(linear1_kv|linear1_q|linear2)\.weight$")
WAN_ANIMATE_ENCODER_PATTERN = re.compile(r"^(face_encoder|motion_encoder)\..*\.weight$")


def should_quantize_key(key: str, mode: str) -> bool:
    if mode == "audio":
        return key.startswith("ca") and ".to" in key and key.endswith("weight")
    if mode == "wan_animate_adapter":
        return bool(WAN_ANIMATE_ADAPTER_PATTERN.match(key))
    if mode == "wan_animate_encoders":
        return bool(WAN_ANIMATE_ENCODER_PATTERN.match(key))
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Quantize selected checkpoint weights to FP8")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(project_root / "models" / "SekoTalk-Distill" / "audio_adapter_model.safetensors"),
        help="Path to input model file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(project_root / "models" / "SekoTalk-Distill-fp8" / "audio_adapter_model_fp8.safetensors"),
        help="Path to output quantized model file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="audio",
        choices=["audio", "wan_animate_adapter", "wan_animate_encoders"],
        help="Which key pattern to quantize",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = load_pt_safetensors(str(model_path))

    new_state_dict = {}
    converted_count = 0
    kept_existing_count = 0

    for key, value in state_dict.items():
        if should_quantize_key(key, args.mode):
            scale_key = key + "_scale"
            if scale_key in state_dict:
                print(f"Keeping existing quantized tensor {key}")
                new_state_dict[key] = value
                kept_existing_count += 1
                continue

            print(f"Converting {key} to FP8, dtype: {value.dtype}")
            weight = value.to(torch.float32).cuda()
            w_quantizer = FloatQuantizer("e4m3", True, "per_channel")
            weight, weight_scale, _ = w_quantizer.real_quant_tensor(weight)
            weight = weight.to(torch.float8_e4m3fn)
            weight_scale = weight_scale.to(torch.float32)

            new_state_dict[key] = weight.cpu()
            new_state_dict[scale_key] = weight_scale.cpu()
            converted_count += 1
        else:
            new_state_dict[key] = value

    save_file(new_state_dict, str(output_path))
    print(f"Converted {converted_count} tensors and kept {kept_existing_count} pre-quantized tensors")
    print(f"Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
