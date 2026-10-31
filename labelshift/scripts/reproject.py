"""Recompute the smoke projection for another STEPS / run count / cap from the same benchmark_summary.json.

Same arithmetic and output lines as the projection block in pod_run.sh smoke(); no new measurement.
usage: python -I reproject.py BENCHMARK_SUMMARY_JSON STEPS NRUNS CAP_HOURS
"""
import json
import sys

b = json.load(open(sys.argv[1]))
steps, n, cap = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
sps = b["steps_per_second"]
per_run = steps / sps / 3600 * 1.10  # +10%: validation passes and the held-out eval
print(f"measured {sps:.3f} steps/s ({1 / sps:.2f} s/step, data wait {b.get('data_wait_seconds_per_step', float('nan')):.2f} s/step)")
print(f"projected {per_run:.2f} GPU h per run, {2 * per_run:.2f} per arm (2 seeds), {n * per_run:.1f} for {n} runs; cap {cap}")
print("PROJECTION_OK" if n * per_run <= cap else "PROJECTION_OVER_CAP")
