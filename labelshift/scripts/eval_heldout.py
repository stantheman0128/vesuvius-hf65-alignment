"""Score one checkpoint on every held-out patch against the C0 (corrected) labels.

usage: python -I eval_heldout.py VILLA_SRC CKPT EVAL_CONFIG_JSON OUT_JSON [--batch N] [--workers N]
       [--max-patches N] [--weights ema_model|model]

EVAL_CONFIG_JSON is a config written by mkconfig.py for the C0 arm (same profile and seed as the
run being scored), so the held-out patches, their 3D-projected C0 labels and the validation mask
come out of villa's own InkDataset exactly as during training. Unlike the trainer's validation (a
random val_steps-batch subset), this runs every held-out patch once, in order, without augmentation,
with the weights inference would use (EMA when present).

Per voxel where the validation mask projects (ignore <= 0): sigmoid probability and C0 target.
Probabilities go into 2000-bin histograms per class, per segment and pooled, from which:
  auc        ROC AUC (trapezoid over the binned ROC; binning error is far below 1e-3 here)
  f05_best   best F0.5 over the bin thresholds, and that threshold
  f05_at_0p5, balanced_accuracy_at_0p5   at the trainer's 0.5 threshold
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BINS = 2000


def summarize(hpos, hneg):
    hpos, hneg = hpos.astype(np.float64), hneg.astype(np.float64)
    P, N = hpos.sum(), hneg.sum()
    out = {"n_pos": int(P), "n_neg": int(N)}
    if P == 0 or N == 0:
        return out
    # threshold at bin edge e: predicted positive when prob >= e (bins from high to low)
    tp = np.concatenate([[0.0], np.cumsum(hpos[::-1])])
    fp = np.concatenate([[0.0], np.cumsum(hneg[::-1])])
    tpr, fpr = tp / P, fp / N
    out["auc"] = float(np.trapezoid(tpr, fpr))
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = tpr
    f05 = np.divide(1.25 * prec * rec, 0.25 * prec + rec, out=np.zeros_like(tp), where=(0.25 * prec + rec) > 0)
    i = int(np.argmax(f05))
    edges = np.linspace(1.0, 0.0, BINS + 1)  # edge for tp[i] is the lower edge of the i-th bin from the top
    out["f05_best"] = float(f05[i])
    out["f05_best_threshold"] = float(edges[i])
    j = BINS // 2  # prob >= 0.5
    out["f05_at_0p5"] = float(f05[j])
    tnr = 1 - fpr[j]
    out["balanced_accuracy_at_0p5"] = float((tpr[j] + tnr) / 2)
    out["positive_fraction"] = float(P / (P + N))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("villa_src")
    p.add_argument("ckpt")
    p.add_argument("eval_config")
    p.add_argument("out_json")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-patches", type=int, default=None)
    p.add_argument("--weights", default=None, help="checkpoint key; default: ema_model if present")
    a = p.parse_args()
    sys.path.insert(0, a.villa_src)
    import torch
    from torch.utils.data import DataLoader
    from vesuvius.ink_detection.data.dataset import InkDataset
    from vesuvius.ink_detection.models.checkpoint import (
        config_from_checkpoint, load_checkpoint, load_model_state, select_inference_weights)
    from vesuvius.ink_detection.models.model import make_model
    from vesuvius.ink_detection.training.stitching import run_model_forward
    from vesuvius.ink_detection.training.train import (
        prepare_model_input, prepare_validation_loss_inputs, stage_training_request, training_dataset_config)

    t0 = time.time()
    config = stage_training_request(a.eval_config).config
    data_config = training_dataset_config(config)
    shared = InkDataset(data_config, do_augmentations=False)
    patches = list(shared.validation_patches)
    if a.max_patches:  # smoke only: take patches round-robin over segments so each one is exercised
        by_seg = {}
        for pt in patches:
            by_seg.setdefault(pt.segment.segment_name, []).append(pt)
        rr = [pt for group in zip(*by_seg.values()) for pt in group]
        taken = {id(pt) for pt in rr}
        patches = (rr + [pt for pt in patches if id(pt) not in taken])[: a.max_patches]
    val = InkDataset(data_config, do_augmentations=False, patches=patches, segments=shared.segments)
    seg_of = [pt.segment.segment_name for pt in patches]

    ckpt = load_checkpoint(a.ckpt)
    model = make_model(config_from_checkpoint(ckpt, source=a.ckpt))
    if a.weights:
        name, state = a.weights, ckpt[a.weights]
    else:
        name, state = select_inference_weights(ckpt, source=a.ckpt)
    load_model_state(model, state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    amp = config.mixed_precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

    loader = DataLoader(val, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                        **({"multiprocessing_context": "spawn", "persistent_workers": False} if a.workers else {}))
    hist = {}
    edges = np.linspace(0.0, 1.0, BINS + 1)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                pred = run_model_forward(model, prepare_model_input(batch, config), config.model_crop_size,
                                         stitched=config.use_stitched_forward,
                                         use_gradient_checkpointing=config.stitched_gradient_checkpointing)
            pred, targets, ignore = prepare_validation_loss_inputs(pred, batch, mode=config.ink.data.mode)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            prob = torch.sigmoid(pred.float())
            for b in range(prob.shape[0]):
                m = ignore[b] <= 0
                pr = prob[b][m].cpu().numpy()
                tg = targets[b][m].cpu().numpy() >= 0.5
                seg = seg_of[seen + b]
                for k in (seg, "pooled"):
                    hp, hn = hist.setdefault(k, (np.zeros(BINS, np.int64), np.zeros(BINS, np.int64)))
                    hp += np.histogram(pr[tg], edges)[0]
                    hn += np.histogram(pr[~tg], edges)[0]
            seen += prob.shape[0]
    res = {"ckpt": os.path.abspath(a.ckpt), "weights": name, "eval_config": os.path.abspath(a.eval_config),
           "n_patches": len(patches), "n_patches_scored": seen, "device": device.type,
           "seconds": round(time.time() - t0, 1),
           "metrics": {k: summarize(*v) for k, v in hist.items()},
           "histograms": {k: {"pos": v[0].tolist(), "neg": v[1].tolist()} for k, v in hist.items()}}
    with open(a.out_json, "w") as f:
        json.dump(res, f)
    print(json.dumps({k: v for k, v in res.items() if k != "histograms"}))


if __name__ == "__main__":
    main()
