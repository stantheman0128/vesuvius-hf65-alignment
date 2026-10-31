# Amendment J (2026-11-01): a fourth seed for C0, C16y and C32y on a second pod

Written on 2026-11-01 from about 04:35 +0800. On the first pod (e105f54d-966b-42da-8eb0-4cb27acf7d39) Amendment I
starts only after Amendment H ends, so no Amendment I result can exist yet. The freeze time, the hashes, the state of
both pods and what had been seen at that moment are in the freeze record at the end and in
`logs/amendment_j_timestamp.txt`. Exploratory like F, G, H and I, since G0 failed in all three attempts.

## Why

At about 04:20 +0800 Stan asked for one more seed and offered to open a second pod so that it runs next to I. With
seed 30, C0, C16y and C32y have four seeds each and the pooled seed noise has df 11 instead of 8. The w00 interval for
C16y or C32y becomes d +/- 2.201 s sqrt(2/4) = d +/- 1.556 s, against I's d +/- 1.883 s, about 17% narrower. These
runs are decided now, before any Amendment I result, so whether they run does not depend on what I shows.

## Runs

- A second pod trains seed 30 of C0, C32y and C16y, in that order, with every setting of Amendment I's seed 29 runs:
  STEPS=12000, MAXSTEPS_MODE=anneal, WORKERS=16, G0_WAIVED=1,
  BUILD_ARMS=AS,C0,C2y,C4y,C8y,C2x,C4x,C8x,C16y,C32y, villa 37d1897fb, `scripts/pod_requirements.txt`, and the same
  teacher config, regions, held-out boxes and scoring.
- The code is the same: `pod_bundle.tar.gz` (sha256 b60e7917...) and `h_extra.tar.gz` (1c354589...) that the first
  pod runs, plus `j_extra.tar.gz`, which only adds this file, `AMENDMENT-I-2026-10-31.md`, `scripts/decide_i.py`,
  `scripts/decide_j.py`, `go_j.sh`, `vault_sync_j.sh`, `final_sync_j.sh` and a SHA256SUMS file. `pod_run.sh` is
  unchanged.
- `go_j.sh` is the Amendment I part of `go9c.sh` with SEEDS=30. The second pod is fresh, so it waits for nothing and
  its `pod_run.sh prepare` builds the venv, downloads the public data and builds the label arms itself. The label
  check against `labels/manifest_full.json` is the same as on the first pod. If it fails, nothing trains.
- The first pod and Amendment I are not changed in any way.
- The second pod should be the same GPU model and image as the first (NVIDIA GeForce RTX 5090, driver 580 or newer).
  `prepare_J.log` records the GPU name, driver and torch build.
- Stop: a local watcher stops the second pod when `go_j.sh` writes ALL_DONE_J, after 60 minutes without a prepare,
  training or evaluation process, or at 2026-11-01 12:00:00 +0800, whichever comes first.
- Expected time: about 40 minutes of setup on a fresh pod (H's fresh-tree prepare on the first pod took about 37
  minutes) and about 83 minutes per run (C16y seed 27 took 82.7 minutes), so about 4.8 hours. Started by 05:45 +0800,
  it ends at about 10:30 +0800. At the first pod's rate of NT$30.29/h this is about NT$150, and at most about NT$195
  if the pod runs until the 12:00 stop.

## Readout (`scripts/decide_j.py`)

1. `decide_j.py` is `decide_i.py`'s reading with seeds 27, 28, 29 and 30. It takes a third runs folder for the
   second pod's seed 30 runs: `decide_j.py F_RUNS_DIR H_RUNS_DIR J_RUNS_DIR OUT_MD`. Seed 30 of C0, C16y and C32y is
   read only from that folder. Everything else, including where seeds 27 to 29 come from and H's session check,
   is as in `decide_i.py`. The t table gains df 11 (2.201).
2. Seed noise, the C0 lower bound, d, se, the interval, the verdicts and the dose response slope are computed as in
   Amendment I with these n, s and t. With four seeds for C0, C16y and C32y and two for AS and C8y, df is 11.
3. With no seed 30 run present, `decide_j.py` gives `decide_i.py`'s output byte for byte.
4. It runs locally once both pods' results are pulled, as
   `python -I scripts/decide_j.py pod_results_F/runs pod_results_H/runs pod_results_J/runs results_J.md`. It is
   not run on the second pod, which does not hold the first pod's runs.
5. Amendment I's three-seed reading stays in `results_I.md` as frozen and is not replaced.

## Which reading is primary

- If all three seed 30 runs (C0_s30, C32y_s30 and C16y_s30) have an `eval.json` written on the second pod by
  2026-11-01 11:30:00 +0800, the four-seed reading in `results_J.md` is primary. The file time on the pod counts;
  the copies in /vault/labelshift_J and `pod_results_J` keep it. `results_I.md` is then reported next to it as the
  three-seed reading.
- Otherwise `results_I.md` is primary and J is reported as incomplete, with the seed 30 runs that did finish named.
  `results_J.md` is kept but is not used for any conclusion.
- A seed 29 run that is missing on the first pod is handled as Amendment I says. It does not change which reading is
  primary.

## Limits

- The limits of H and I hold.
- AS and C8y keep two seeds.
- Seed 30 is trained on a second pod and in a new session, while seed 29 and C16y and C32y seeds 27 and 28 come
  from the first pod and C0 seeds 27 and 28 from F's sessions. H's session check (C0 seed 28 retrained on the first
  pod) shows how far a session moves C0. J adds no check of its own for the second pod.
- The second pod rebuilds the label arms. Its check against `labels/manifest_full.json` covers the eight earlier
  arms. C16y and C32y are not in that manifest, as on the first pod.

## Freeze record

- Frozen at: 2026-11-01 04:50:34 +0800
- Seen at that moment: of Amendment H, only the C16y_s27 eval line that a watcher printed at about 00:26 +0800,
  and the check line of scan2.sh's first pair (CHECK_OK). No other eval value, no scan readout and no Amendment I
  result had been looked at, and no Amendment I result existed.
- First pod at that moment (checked at 2026-10-31 20:50:24 UTC): Amendment H's session check, C0 seed 28, was training,
  at step 5221 of 12000. eval.json files under runs/: 4, those of C16y and C32y seeds 27 and 28. Seed 29 runs: 0.
  No ALL_DONE_H and no prepare_I.log. scan2.sh had written 12 of its 18 pair files.
- Second pod at that moment: not created.
- sha256 of the J files:
  - scripts/decide_j.py: ea4519a989c0985abf794aa6bd1719f7097b838e3e69551ab5460d7df379b4e8
  - go_j.sh: c435f19e46e5e8e569697d4ff921c3db1be4925e091edbdacb9e013525a9f35b
  - vault_sync_j.sh: fedd985c2fe8f7dbf4cd6d441f3fcffd47bee21b1a05cf279a800aef5f393676
  - final_sync_j.sh: 55f157d88588e19b81b4a5c31cbbc62e2d69d7e9b8d1059c99f9a084be5b0984
- sha256 of the files J reuses unchanged:
  - AMENDMENT-I-2026-10-31.md: 77dd833476cda56ef00d2f9267f16dc403205c2850a4f7415e5975c1720fe991
  - scripts/decide_i.py: 25ef0820ce3cbb2a7d4eafc48c11111afb85f5c89192b0eb31ad6443480d69d2
  - pod_bundle.tar.gz: b60e7917f213f6ed264e7fd074e52c527bd996b520349a50c3b8cf4ab3d3fde2
  - h_extra.tar.gz: 1c354589cfbc5e71a965400c220d5b5149ac05f30f190b53836536a81cf5a13b
- The hashes of this file, of `j_extra.tar.gz` (which holds this file) and of `pod_setup_j.sh` (which holds
  `j_extra.tar.gz`'s hash) cannot be written here. They go into `logs/amendment_j_timestamp.txt`.
