"""CPU check of the GPU-profile config, short of training: the trainer's own config staging accepts it,
the teacher model builds at 128^3 and runs one forward pass, and the dataset finds training and
validation patches on the full label set.

usage: python -I check_gpu_config.py VILLA_SRC CONFIG_JSON

Copied from the October prep; two changes for labelshift-w01. The model returns a dict, so the output
shapes are read from its values (the October copy ran the forward pass but then failed to print the
shapes). len(InkDataset) counts training and validation patches together, so the training count is
now read from training_patches (October's "train_patches" 3154 was 2478 training plus 676 validation).
"""
import json
import sys
import time

src, cfg = sys.argv[1:3]
sys.path.insert(0, src)
import torch  # noqa: E402
from vesuvius.ink_detection.data.dataset import InkDataset  # noqa: E402
from vesuvius.ink_detection.models.model import make_model  # noqa: E402
from vesuvius.ink_detection.training.train import stage_training_request, training_dataset_config  # noqa: E402

t0 = time.time()
config = stage_training_request(cfg).config
print("staged:", {k: getattr(config, k, None) for k in ("num_iterations", "batch_size", "loader_patch_size", "model_crop_size", "mixed_precision",
                                                        "val_every", "seed")}, "benchmark", config.benchmark, flush=True)
model = make_model(config.ink)
n = sum(p.numel() for p in model.parameters())
try:
    with torch.no_grad():
        y = model(torch.zeros((1, 1, *config.model_crop_size)))
    outs = y.values() if isinstance(y, dict) else (y if isinstance(y, (list, tuple)) else [y])
    shapes = [tuple(t.shape) for t in outs]
except Exception as e:  # input layout differs from (1, 1, D, H, W); the param count is still valid
    shapes = f"dummy forward not run: {type(e).__name__}: {e}"
print(f"model params {n / 1e6:.1f} M, forward out {shapes if isinstance(shapes, str) else shapes[:2]}", flush=True)
ds = InkDataset(training_dataset_config(config), do_augmentations=False)
segs = {}
for pt in ds.validation_patches:
    segs[pt.segment.segment_name] = segs.get(pt.segment.segment_name, 0) + 1
print(json.dumps({"training_patches": len(ds.training_patches), "validation_patches": len(ds.validation_patches),
                  "all_patches": len(ds),
                  "validation_patches_per_segment": segs, "seconds": round(time.time() - t0, 1)}), flush=True)
