# ProMoE Diffusers Native Refactor

This repository has been fully refactored to a native Diffusers-style layout under `src/diffusers`.

Legacy standalone training/sampling/model files from the source repository were removed. The codebase now provides:

- `src/diffusers/models/transformers/transformer_promoe.py`
  - `ProMoETransformer2DModel` (`ModelMixin` + `ConfigMixin`)
  - Supports `dit`, `tcdit`, `ecdit`, `diffmoe`, `promoe_tc`, `promoe_ec`
- `src/diffusers/pipelines/promoe/pipeline_promoe.py`
  - `ProMoEPipeline` for class-conditional latent sampling and optional VAE decode
- `src/diffusers/schedulers/scheduling_flow_match_promoe.py`
  - `ProMoEFlowMatchScheduler` (FlowMatch-compatible scheduler wrapper)
- `scripts/convert_promoe_to_diffusers.py`
  - Converts legacy checkpoints into Diffusers-native artifact layout
- `scripts/sample_promoe.py`
  - Samples from a converted model directory

## Package layout

```text
src/diffusers/
  __init__.py
  models/
    __init__.py
    transformers/
      __init__.py
      modeling_promoe_common.py
      backbone_*.py
      transformer_promoe.py
  pipelines/
    __init__.py
    promoe/
      __init__.py
      pipeline_promoe.py
  schedulers/
    __init__.py
    scheduling_flow_match_promoe.py
scripts/
  convert_promoe_to_diffusers.py
  sample_promoe.py
```

## Convert legacy checkpoint

```bash
python scripts/convert_promoe_to_diffusers.py \
  --checkpoint /path/to/ckpt_step_500000.pth \
  --output ./promoe-l-diffusers \
  --model-name ProMoE_TC_L
```

Optional flags:

- `--num-classes 1000`
- `--safe-serialization` / `--no-safe-serialization`
- `--check-load`

## Sample

```bash
python scripts/sample_promoe.py \
  --model ./promoe-l-diffusers \
  --class-label 207 \
  --height 256 \
  --width 256 \
  --num-inference-steps 250 \
  --guidance-scale 1.5 \
  --output-dir ./samples
```

## Notes

- This refactor targets Diffusers-native module boundaries and serialization patterns.
- To upstream into `huggingface/diffusers`, copy the `src/diffusers` files into matching package locations.
