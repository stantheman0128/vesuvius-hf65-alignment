"""Write one label-shift run's training config, derived from villa's dino_guided_teacher.json.

usage: python -I mkconfig.py PROFILE TEACHER_JSON ARM_ROOT OUT_DIR SEED OUT_JSON [--steps N]
       [--cache-dir DIR] [--workers N] [--batch N]

PROFILE gpu   : the teacher config with only the changes listed in PREREG.md section 4 (two segments,
                volume_scale 2, 128^3 crops, STEPS iterations with every step-based schedule value
                scaled by STEPS / 60001, validation on the held-out mask).
PROFILE smoke : the same data path on CPU with a 2-stage, 2/4-feature U-Net, 21 iterations, batch 1,
                fp32, validation at step 20 over 2 batches. Its numbers are not results.
ARM_ROOT holds <ARM>/phercparis4/w00_20231016151002 and <ARM>/pherc1667/w018_20240304144031_2um
(scripts/build_labels.py); pass ARM_ROOT/<ARM>.
"""
import argparse
import copy
import json
import os

TEACHER_ITERS = 60001
VOLUMES = {
    "phercparis4": ("w00_20231016151002",
                    "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"),
    "pherc1667": ("w018_20240304144031_2um",
                  "s3://vesuvius-challenge-open-data/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/"),
}
TINY_MODEL = {
    "autoconfigure": True, "z_projection_mode": "none",
    "basic_encoder_block": "BasicBlockD", "basic_decoder_block": "ConvBlock",
    "features_per_stage": [2, 4], "n_blocks_per_stage": [1, 1], "n_conv_per_stage_decoder": [1],
    "kernel_sizes": [[3, 3, 3], [3, 3, 3]], "strides": [[1, 1, 1], [2, 2, 2]],
    "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2]], "spacing": [1, 1, 1],
}


def scaled(value, steps):
    return max(1, round(value * steps / TEACHER_ITERS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("profile", choices=("gpu", "smoke"))
    p.add_argument("teacher")
    p.add_argument("arm_root")
    p.add_argument("out_dir")
    p.add_argument("seed", type=int)
    p.add_argument("out_json")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--benchmark-warmup", type=int, default=None,
                   help="enable the trainer's inline benchmark after this many steps (pod throughput probe)")
    p.add_argument("--max-steps-mode", choices=("teacher", "anneal"), default="teacher",
                   help="teacher: scheduler horizon scaled from the teacher's max_steps (PREREG); "
                        "anneal: horizon = --steps, so the cosine schedule ends near zero (villa's default, Amendment E)")
    a = p.parse_args()
    teacher = json.load(open(a.teacher))
    c = copy.deepcopy(teacher)
    c["description"] = (f"label-shift {a.profile} run derived from dino_guided_teacher.json; arm {os.path.basename(a.arm_root.rstrip('/'))}, "
                        f"seed {a.seed}; changes from the teacher are listed in PREREG.md section 4")
    c["out_dir"] = a.out_dir
    c["seed"] = a.seed
    c["patch_size"] = [128, 128, 128]
    c["datasets"] = [{"segments_path": os.path.join(a.arm_root, scroll), "segments": [seg],
                      "volume_path": vol, "volume_scale": 2}
                     for scroll, (seg, vol) in VOLUMES.items()]
    c["volume_cache_dir"] = a.cache_dir
    c["volume_cache_max_gb"] = 200
    if a.cache_dir is None:
        del c["volume_cache_dir"], c["volume_cache_max_gb"]
    if a.profile == "gpu":
        steps = a.steps
        c["num_iterations"] = steps
        c["max_steps"] = steps if a.max_steps_mode == "anneal" else scaled(teacher["max_steps"], steps)
        c["warmup_steps"] = scaled(teacher["warmup_steps"], steps)
        c["ema"]["start_step"] = scaled(teacher["ema"]["start_step"], steps)
        c["save_iterations"] = [steps]
        c["save_every"] = steps
        c["val_every"] = max(1, steps // 16)
        c["dataloader_workers"] = a.workers or 8
        c["batch_size"] = a.batch or teacher["batch_size"]
    else:
        c["model_config"] = TINY_MODEL
        c["num_iterations"] = 21
        c["max_steps"] = 21
        c["warmup_steps"] = 5
        c["ema"]["start_step"] = 5
        c["save_iterations"] = [21]
        c["save_every"] = 21
        c["val_every"] = 20
        c["val_steps"] = 2
        c["mixed_precision"] = "no"
        c["batch_size"] = a.batch or 1
        c["dataloader_workers"] = a.workers or 2
        c["prefetch_factor"] = 2
        c["pin_memory"] = False
    if a.benchmark_warmup is not None:
        c["benchmark"] = {"enabled": True, "warmup_steps": a.benchmark_warmup}
        # the benchmark timer runs from warmup to the end of the loop, validation included; validate
        # once (last step) so the GPU path is exercised without STEPS/16 passes inflating s/step
        c["val_every"] = max(1, a.steps - 1)
    with open(a.out_json, "w") as f:
        json.dump(c, f, indent=2)


if __name__ == "__main__":
    main()
