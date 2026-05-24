#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:  # pragma: no cover
    safe_load_file = None

try:
    from diffusers.models import AutoencoderKL
except Exception:  # pragma: no cover
    AutoencoderKL = None

from diffusers.models.transformers import ProMoETransformer2DModel
from diffusers.pipelines.promoe import ProMoEPipeline
from diffusers.schedulers import ProMoEFlowMatchScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images from a converted ProMoE Diffusers pipeline.")
    parser.add_argument("--model", required=True, help="Path to converted pipeline directory.")
    parser.add_argument("--class-label", type=int, action="append", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="samples")
    parser.add_argument("--output-type", choices=["pil", "np", "pt", "latent"], default="pil")
    return parser.parse_args()


def _load_transformer(model_dir: Path) -> ProMoETransformer2DModel:
    config_path = model_dir / "transformer" / "config.json"
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    config.pop("_class_name", None)
    model = ProMoETransformer2DModel(**config)
    safetensors_path = model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    bin_path = model_dir / "transformer" / "diffusion_pytorch_model.bin"
    if safetensors_path.exists():
        if safe_load_file is None:
            raise ImportError("Install safetensors to load safetensors checkpoints.")
        state_dict = safe_load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError("No transformer weight file found.")
    model.load_state_dict(state_dict, strict=True)
    return model


def _load_scheduler(model_dir: Path) -> ProMoEFlowMatchScheduler:
    config_path = model_dir / "scheduler" / "config.json"
    if not config_path.exists():
        return ProMoEFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    config.pop("_class_name", None)
    return ProMoEFlowMatchScheduler(**config)


def _load_vae_if_available(model_dir: Path):
    if AutoencoderKL is None:
        return None
    vae_dir = model_dir / "vae"
    if vae_dir.exists():
        return AutoencoderKL.from_pretrained(str(vae_dir))
    return None


def main():
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model)
    transformer = _load_transformer(model_dir).to(device=device, dtype=dtype)
    scheduler = _load_scheduler(model_dir)
    vae = _load_vae_if_available(model_dir)
    if vae is not None:
        vae = vae.to(device=device, dtype=dtype)
    pipe = ProMoEPipeline(transformer=transformer, scheduler=scheduler, vae=vae).to(device)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    output = pipe(
        class_labels=args.class_label,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type=args.output_type,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_type == "pil":
        for index, image in enumerate(output.images):
            image.save(output_dir / f"{index:06d}.png")
    elif args.output_type == "np":
        import numpy as np

        np.save(output_dir / "samples.npy", output.images)
    else:
        torch.save(output.images, output_dir / "samples.pt")


if __name__ == "__main__":
    main()
