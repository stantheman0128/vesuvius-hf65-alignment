"""Count the held-out patches villa's InkDataset finds for eval configs (eval offset scan pre-check, CPU only).

usage: python -I count_val_patches.py VILLA_SRC CONFIG_JSON [CONFIG_JSON ...]

Builds the dataset the way eval_heldout.py does and prints, per config, the number of validation patches in
total and per segment. Reads label zarrs and tifxyz only; no model and no GPU.
"""
import sys

sys.path.insert(0, sys.argv[1])
from vesuvius.ink_detection.data.dataset import InkDataset  # noqa: E402
from vesuvius.ink_detection.training.train import stage_training_request, training_dataset_config  # noqa: E402

for cfg in sys.argv[2:]:
    config = stage_training_request(cfg).config
    ds = InkDataset(training_dataset_config(config), do_augmentations=False)
    per = {}
    for pt in ds.validation_patches:
        per[pt.segment.segment_name] = per.get(pt.segment.segment_name, 0) + 1
    print(cfg, sum(per.values()), per, flush=True)
