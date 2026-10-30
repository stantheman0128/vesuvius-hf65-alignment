"""Build results/segments.csv and results/summary.json from results/phase1.json and results/phase2/*.

Flags (training segments; unused/ segments get the same measurements but the loader never reads them):
  a  any label-kind .zarr in the segment dir whose (H, W) differs from the canvas (H, W)
     (label zarrs that match the canvas but not the tifxyz full-res grid are counted in a separate column;
      that is a consequence of flag b)
     (a_picked: one of them is what villa's loader picks)
  b  tifxyz full-res (H, W) != canvas (H, W); predicted max offset per axis = |full - canvas| px
     (the linear part of the #2370 Paris4 fit, which is 0 at canvas px 0 and full - canvas at the far edge)
  c  a file the loader or the check needs is missing or unreadable
Phase-2 verdict per region (high-pass NCC peak, best other local peak > 4 level-2 px away):
  inconclusive  peak NCC < 0.4, or best other peak > peak / 2, or peak on the search-window edge
  aligned       |dy - 0.35| <= 1.5 and |dx - 0.35| <= 1.5 level-0 px (0.35 = renderer half-pixel, README control)
  offset        otherwise
"""
import os, sys, json, csv, glob

AUDIT = os.environ.get("AUDIT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P1 = json.load(open(os.path.join(AUDIT, "results", "phase1.json")))
TG = json.load(open(os.path.join(AUDIT, "code", "targets.json")))
P2 = os.path.join(AUDIT, "results", "phase2")
LC = {o["path"]: o for o in json.load(open(os.path.join(AUDIT, "results", "labels_check.json")))}

# villa origin/main 37d1897fb, vesuvius/src/vesuvius/ink_detection/configs/dino_guided_*.json, all mode full_3d,
# no segment allowlist; local dataset dir name -> bucket scroll dir (name mapping is ours, see REPORT)
CONFIG_DIRS = {
    "dino_guided_teacher": ["phercparis4", "0139", "0500p2", "814", "841", "1667", "man5", "0009b"],
    "dino_guided_v1": ["phercparis4"], "dino_guided_v2": ["phercparis4"],
    "dino_guided_v3": ["phercparis4"], "dino_guided_v3_fullsup": ["phercparis4"],
}
COVERED = {
    "ink/phercparis4/w00_20231016151002": "#2370", "ink/phercparis4/w01_20230702185753_r15": "#2370",
    "ink/phercparis4/w05_4424": "#2370", "ink/1667/w018_20240304144031_2um": "#2370",
    "ink/man5/MAN5_outer_3": "#2369",
}


def hw(z):
    s = z.get("shape")
    if not s:
        return None
    d = [int(v) for v in s if int(v) != 1]
    return (d[-2], d[-1]) if len(d) >= 2 else None


def verdict(r):
    hp = r["hp"]
    pk, sec = hp["ncc"], hp["second"]
    if pk is None or pk < 0.4 or hp["on_edge"] or (sec is not None and sec > pk / 2):
        why = []
        if pk is None or pk < 0.4:
            why.append(f"peak NCC {pk:.2f} < 0.4" if pk is not None else "no peak")
        if hp["on_edge"]:
            why.append("peak on window edge")
        if sec is not None and pk is not None and sec > pk / 2:
            why.append(f"second peak {sec:.2f} > peak/2")
        return "inconclusive", "; ".join(why)
    if abs(r["level0_dy"] - 0.35) <= 1.5 and abs(r["level0_dx"] - 0.35) <= 1.5:
        return "aligned", ""
    return "offset", ""


def main():
    # one key per segment; t040b (same segment, other CT export) is a method check reported separately
    key_of = {v["seg"]: k for k, v in TG.items() if k != "t040b"}
    rows, summ = [], {"segments_total": 0, "training": 0, "unused": 0}
    for r in P1:
        seg = r["path"]
        canvas = tuple(r["canvas_hw"]) if r.get("canvas_hw") else None
        full = tuple(r["full_hw"]) if r.get("full_hw") else None
        labz = []
        a_any, a_picked, lab_vs_full = [], [], []
        picked = set(v for v in r["loader_pick"].values() if v)
        for n, z in sorted(r["zarrs"].items()):
            if n == f"{r['segment']}.zarr":
                continue
            isl = ("_inklabels" in n or "_supervision_mask" in n or "_validation_mask" in n or "inklabels" in n.lower())
            h = hw(z)
            labz.append(f"{n}:{'x'.join(map(str, h)) if h else z.get('error', '?')}" + (" [loader]" if n in picked else ""))
            if isl and h and canvas and h != canvas:
                a_any.append(n)
                if n in picked:
                    a_picked.append(n)
            if isl and h and full and h != full:
                lab_vs_full.append(n)
        flag_b = bool(canvas and full and canvas != full)
        c_items = list(r["errors"])
        if not r["unused"]:
            for kind in ("inklabels", "supervision_mask"):
                if not r["loader_pick"].get(kind):
                    c_items.append(f"loader finds no {r['segment']}_{kind}*.zarr")
        cfgs = [c for c, dirs in CONFIG_DIRS.items() if (not r["unused"]) and r["scroll"] in dirs]
        # versions the loader cannot see (label assets whose prefix != segment dir name)
        newer = sorted({a["name"] for a in r["label_assets"] if a["ext"] == ".zarr" and a["prefix"] != r["segment"] and a["version"] > 1})
        # phase 2
        key = key_of.get(seg)
        regs = []
        if key:
            p = os.path.join(P2, f"reg_{key}.jsonl")
            if os.path.exists(p):
                seen = {}
                for l in open(p):
                    x = json.loads(l)
                    seen[x["region"]] = x  # last run of a region wins
                regs = [seen[k] for k in sorted(seen)]
        vlist = [verdict(x) for x in regs]
        meas = "; ".join(f"r{x['region']}@({x['y0'] + 512},{x['x0'] + 512}): {x['level0_dy']:+.1f},{x['level0_dx']:+.1f}" for x in regs)
        ncc = "; ".join(f"{x['hp']['ncc']:.3f}/{x['hp']['second']:.3f}" if x['hp']['second'] is not None else f"{x['hp']['ncc']:.3f}/-" for x in regs)
        vtxt = "; ".join(v + (f" ({why})" if why else "") for v, why in vlist)
        if not regs:
            overall = "not registered"
        elif all(v == "inconclusive" for v, _ in vlist):
            overall = "inconclusive"
        elif any(v == "offset" for v, _ in vlist):
            overall = "offset"
        else:
            overall = "aligned" if all(v == "aligned" for v, _ in vlist) else "aligned (some regions inconclusive)"
        row = {
            "scroll": r["scroll"], "segment": r["segment"], "unused_dir": "yes" if r["unused"] else "no",
            "in_full3d_config": "yes" if cfgs else "no", "configs": " ".join(cfgs),
            "tifxyz_scale_xy": " ".join(f"{float(s):.6f}" for s in r["meta_scale"]) if r.get("meta_scale") else "",
            "tifxyz_stored_hw": "x".join(map(str, r["stored_hw"])) if r.get("stored_hw") else "",
            "tifxyz_full_hw": "x".join(map(str, full)) if full else "",
            "canvas_hw": "x".join(map(str, canvas)) if canvas else "",
            "full_minus_canvas_hw": f"{full[0] - canvas[0]:+d} {full[1] - canvas[1]:+d}" if (full and canvas) else "",
            "label_zarrs": " | ".join(labz),
            "loader_inklabels": r["loader_pick"].get("inklabels") or "",
            "loader_supervision": r["loader_pick"].get("supervision_mask") or "",
            "loader_validation": r["loader_pick"].get("validation_mask") or "",
            "loader_ignores_versioned_labels": " ".join(newer),
            "villa_validator_metadata": "pass" if not r["validator_issues"] else "FAIL: " + " | ".join(r["validator_issues"]),
            "flag_a_label_shape": " ".join(a_any) if a_any else "no",
            "flag_a_loader_picked": "yes" if a_picked else "no",
            "label_zarrs_not_on_tifxyz_grid": len(lab_vs_full),
            "flag_b_tifxyz_vs_canvas": "yes" if flag_b else ("no" if (full and canvas) else "n/a"),
            "flag_b_pred_max_offset_px": f"{abs(full[0] - canvas[0])} {abs(full[1] - canvas[1])}" if flag_b else "",
            "flag_c_missing": " | ".join(c_items) if c_items else "no",
            "flag_d_ignored_prefix": " ".join(LC[seg]["d_ignored_prefix"]) or "no",
            "flag_e_rejected_name": " ".join(LC[seg]["e_rejected"]) or "no",
            "flag_f_ignored_newer_or_only_validation": " | ".join(LC[seg]["f_newer_or_only"]) or "no",
            "raises_if_label_version_v2": "yes" if LC[seg]["raises_if_label_version"]["v2"] else "no",
            "already_filed": COVERED.get(seg, ""),
            "phase2_key": key or "", "phase2_volume": TG[key]["vol"] if key else "",
            "registered_regions": len(regs),
            "measured_dy_dx_level0": meas, "ncc_peak/second": ncc, "region_verdicts": vtxt, "verdict": overall,
        }
        rows.append(row)
    rows.sort(key=lambda x: (x["unused_dir"] == "yes", x["segment"] != "w018_20240304144031_2um", x["scroll"], x["segment"]))
    with open(os.path.join(AUDIT, "results", "segments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tr = [x for x in rows if x["unused_dir"] == "no"]
    un = [x for x in rows if x["unused_dir"] == "yes"]
    summ = {
        "segments_total": len(rows), "training": len(tr), "unused": len(un),
        "training_in_full3d_config": sum(x["in_full3d_config"] == "yes" for x in tr),
        "flag_a_training": sum(x["flag_a_label_shape"] != "no" for x in tr),
        "flag_a_loader_picked_training": sum(x["flag_a_loader_picked"] == "yes" for x in tr),
        "flag_b_training": sum(x["flag_b_tifxyz_vs_canvas"] == "yes" for x in tr),
        "flag_c_training": sum(x["flag_c_missing"] != "no" for x in tr),
        "validator_fail_training": sum(x["villa_validator_metadata"] != "pass" for x in tr),
        "flag_d_training": sum(x["flag_d_ignored_prefix"] != "no" for x in tr),
        "flag_e_training": sum(x["flag_e_rejected_name"] != "no" for x in tr),
        "flag_f_training": sum(x["flag_f_ignored_newer_or_only_validation"] != "no" for x in tr),
        "raises_if_label_version_v2_training": sum(x["raises_if_label_version_v2"] == "yes" for x in tr),
        "flag_d_unused": sum(x["flag_d_ignored_prefix"] != "no" for x in un),
        "flag_e_unused": sum(x["flag_e_rejected_name"] != "no" for x in un),
        "loader_ignores_versioned_training": sum(bool(x["loader_ignores_versioned_labels"]) for x in tr),
        "flag_a_unused": sum(x["flag_a_label_shape"] != "no" for x in un),
        "flag_b_unused": sum(x["flag_b_tifxyz_vs_canvas"] == "yes" for x in un),
        "flag_c_unused": sum(x["flag_c_missing"] != "no" for x in un),
        "validator_fail_unused": sum(x["villa_validator_metadata"] != "pass" for x in un),
        "registered_segments": sum(x["registered_regions"] > 0 for x in rows),
        "registered_regions": sum(x["registered_regions"] for x in rows),
        "verdicts": {},
    }
    for x in rows:
        if x["registered_regions"]:
            summ["verdicts"][x["verdict"]] = summ["verdicts"].get(x["verdict"], 0) + 1
    json.dump(summ, open(os.path.join(AUDIT, "results", "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
