#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers import ProMoETransformer2DModel


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "DiT_B": {
        "architecture": "dit",
        "model_config": {"input_size": 32, "patch_size": 2, "hidden_size": 768, "depth": 12, "num_heads": 12},
    },
    "DiT_L": {
        "architecture": "dit",
        "model_config": {"input_size": 32, "patch_size": 2, "hidden_size": 1024, "depth": 24, "num_heads": 16},
    },
    "DiT_XL": {
        "architecture": "dit",
        "model_config": {"input_size": 32, "patch_size": 2, "hidden_size": 1152, "depth": 28, "num_heads": 16},
    },
    "TCDiT_L_E8": {
        "architecture": "tcdit",
        "model_config": {
            "input_size": 32,
            "patch_size": 2,
            "hidden_size": 1024,
            "depth": 24,
            "num_heads": 16,
            "MoE_config": {
                "n_shared_experts": 0,
                "num_experts": 8,
                "capacity": 1,
                "init_MoeMLP": False,
                "interleave": True,
            },
        },
    },
    "ECDiT_L_E8": {
        "architecture": "ecdit",
        "model_config": {
            "input_size": 32,
            "patch_size": 2,
            "hidden_size": 1024,
            "depth": 24,
            "num_heads": 16,
            "MoE_config": {
                "n_shared_experts": 0,
                "num_experts": 8,
                "capacity": 1,
                "init_MoeMLP": False,
                "interleave": True,
            },
        },
    },
    "DiffMoE_B_E8": {
        "architecture": "diffmoe",
        "model_config": {
            "input_size": 32,
            "patch_size": 2,
            "hidden_size": 768,
            "depth": 12,
            "num_heads": 12,
            "MoE_config": {
                "n_shared_experts": 0,
                "num_experts": 8,
                "capacity": 1,
                "init_MoeMLP": False,
                "interleave": True,
            },
        },
    },
    "DiffMoE_L_E8": {
        "architecture": "diffmoe",
        "model_config": {
            "input_size": 32,
            "patch_size": 2,
            "hidden_size": 1024,
            "depth": 24,
            "num_heads": 16,
            "MoE_config": {
                "n_shared_experts": 0,
                "num_experts": 8,
                "capacity": 1,
                "init_MoeMLP": False,
                "interleave": True,
            },
        },
    },
    "DiffMoE_XL_E8": {
        "architecture": "diffmoe",
        "model_config": {
            "input_size": 32,
            "patch_size": 2,
            "hidden_size": 1152,
            "depth": 28,
            "num_heads": 16,
            "MoE_config": {
                "n_shared_experts": 0,
                "num_experts": 8,
                "capacity": 1,
                "init_MoeMLP": False,
                "interleave": True,
            },
        },
    },
    "ProMoE_TC_S": {
        "architecture": "promoe_tc",
        "model_config": {"input_size": 32, "patch_size": 2, "hidden_size": 384, "depth": 12, "num_heads": 6},
    },
    "ProMoE_TC_B": {
        "architecture": "promoe_tc",
        "model_config": {"input_size": 32, "patch_size": 2, "hidden_size": 768, "depth": 12, "num_heads": 12},
    },
    "ProMoE_TC_L": {
        "architecture": "promoe_tc",
        "model_config": {"input_size": 32, "patch_size": 2, "hidden_size": 1024, "depth": 24, "num_heads": 16},
    },
    "ProMoE_TC_XL": {
        "architecture": "promoe_tc",
        "model_config": {"input_size": 32, "patch_size": 2, "hidden_size": 1152, "depth": 28, "num_heads": 16},
    },
    "ProMoE_EC_L": {
        "architecture": "promoe_ec",
        "model_config": {"input_size": 32, "patch_size": 2, "hidden_size": 1024, "depth": 24, "num_heads": 16},
    },
}


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        return safe_load_file(checkpoint_path, device="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("ema_model_state_dict", "model_state_dict", "state_dict", "model", "module"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _add_backbone_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("backbone.") for key in state_dict):
        return state_dict
    return {f"backbone.{key}": value for key, value in state_dict.items()}


def _save_config(path: Path, config: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(state_dict, str(output_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ProMoE-style checkpoints into a Diffusers-native layout.")
    parser.add_argument("--checkpoint", required=True, help="Path to legacy checkpoint (.pth/.bin/.safetensors).")
    parser.add_argument("--output", required=True, help="Output model directory.")
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse", help="VAE reference stored in output metadata.")
    parser.add_argument("--check-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"
    preset = json.loads(json.dumps(MODEL_PRESETS[args.model_name]))
    preset["model_config"]["num_classes"] = args.num_classes

    state_dict = _add_backbone_prefix(_load_state_dict(args.checkpoint))
    model = ProMoETransformer2DModel(**preset)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if args.check_load and (missing or unexpected):
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        raise SystemExit(1)

    transformer_config = {"_class_name": "ProMoETransformer2DModel", **preset}
    _save_config(transformer_dir / "config.json", transformer_config)
    _save_weights(transformer_dir, model.state_dict(), args.safe_serialization)

    scheduler_config = {"_class_name": "ProMoEFlowMatchScheduler", "num_train_timesteps": 1000, "shift": 1.0}
    _save_config(scheduler_dir / "config.json", scheduler_config)
    if args.vae:
        with open(output_dir / "vae_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as file:
            file.write(args.vae + os.linesep)

    model_index = {
        "_class_name": "ProMoEPipeline",
        "_diffusers_version": "0.32.2",
        "scheduler": ["diffusers", "ProMoEFlowMatchScheduler"],
        "transformer": ["diffusers", "ProMoETransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    _save_config(output_dir / "model_index.json", model_index)
    print(f"Saved Diffusers-native ProMoE model to {output_dir}")


if __name__ == "__main__":
    main()
