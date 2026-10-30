"""Print the markdown tables used in REPORT.md from results/segments.csv, results/labels_check.json,
results/phase2/*.jsonl and the ledgers, so every count in the report comes from an output file."""
import os, csv, json, glob

AUDIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(AUDIT, "results", "segments.csv"), encoding="utf-8")))
tr = [r for r in rows if r["unused_dir"] == "no"]

print("## phase 1, training segments")
print("| scroll | segment | tifxyz full | canvas | full - canvas | a | b | d | f | validator | registered |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in tr:
    print(f"| {r['scroll']} | {r['segment']} | {r['tifxyz_full_hw']} | {r['canvas_hw']} | {r['full_minus_canvas_hw']} | "
          f"{'yes' if r['flag_a_label_shape'] != 'no' else 'no'} | {r['flag_b_tifxyz_vs_canvas']} | "
          f"{'yes' if r['flag_d_ignored_prefix'] != 'no' else 'no'} | {'yes' if r['flag_f_ignored_newer_or_only_validation'] != 'no' else 'no'} | "
          f"{'pass' if r['villa_validator_metadata'] == 'pass' else 'FAIL'} | {r['registered_regions']} |")

print("\n## phase 2, every registered region")
print("| scroll | segment | key | region centre (y, x) | dy, dx (level-0 px) | peak NCC | best other | verdict | CT volume |")
print("|---|---|---|---|---|---|---|---|---|")
TG = json.load(open(os.path.join(AUDIT, "code", "targets.json")))
for p in sorted(glob.glob(os.path.join(AUDIT, "results", "phase2", "reg_*.jsonl"))):
    seen = {}
    for l in open(p):
        x = json.loads(l)
        seen[x["region"]] = x
    for ri in sorted(seen):
        x = seen[ri]
        hp = x["hp"]
        ok = hp["ncc"] >= 0.4 and not hp["on_edge"] and (hp["second"] is None or hp["second"] <= hp["ncc"] / 2)
        al = abs(x["level0_dy"] - 0.35) <= 1.5 and abs(x["level0_dx"] - 0.35) <= 1.5
        v = "inconclusive" if not ok else ("aligned" if al else "offset")
        sc, sg = x["seg"].split("/")[1:3]
        print(f"| {sc} | {sg} | {x['key']} | {x['y0'] + 512}, {x['x0'] + 512} | {x['level0_dy']:+.1f}, {x['level0_dx']:+.1f} | "
              f"{hp['ncc']:.3f} | {hp['second']:.3f} | {v} | {TG[x['key']]['vol'].split('/')[-1]} |")

print("\n## bytes")
DATA = os.environ.get("HF65_DATA", os.path.abspath("hf65_data"))
for name in ("ledger.tsv", "ledger_listing.tsv"):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        print(f"{name}: not found under $HF65_DATA")
        continue
    n = sum(int(l.split("\t", 1)[0]) for l in open(p) if l.strip())
    print(f"{name}: {n} bytes ({n / 1e6:.1f} MB), {sum(1 for _ in open(p))} responses")
