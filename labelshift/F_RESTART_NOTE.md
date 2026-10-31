# Amendment F restart note

Written 2026-10-31 14:26 +0800 (clock read with `date`), before anything of Amendment F ran again.

- The pod stopped between 05:25:50 UTC (the last /vault write) and about 05:29 UTC, 13:25:50 to 13:29 +0800.
  Stan says he did not stop it. The session's autostop did not trigger (no output), and the session made no
  stop call. The cause is unknown: get-instance-status gives no reason, get-instance-events is empty
  (podExists false), no GPUtw email arrived in the last two days, and the balance was NT$341.
- When it stopped, AS s27 had trained for about 5 minutes. Nothing else of Amendment F had run.
- The session restarted it at 14:24:37 +0800 on the same node (restart-instance, mode same) under Stan's
  instruction to keep the GPU work going, after he confirmed the stop was not his.
- /root was partly wiped again (`pod_run.sh`, `pilots.txt` and `~/.local` were gone). The old tree was moved to
  ~/labelshift_wiped_1031b and is not used. The new tree is pod_bundle.tar.gz (sha256 4d41974c62f6afd6...) with
  Amendment F's files on top. The venv, data, label arms and volume cache are rebuilt from scratch.
- `f_choice.txt` (written 05:22:36 UTC) is reused unchanged: L2 setting, C0 seed 27 from L2. Its `eval.json`,
  `train.done`, `config.json` and logs are restored into `runs/C0_s27` from the local copy pulled at 05:22 UTC,
  which is byte-identical to the L2 copy and to `/vault/labelshift_F/runs/C0_s27`. The choice is not made again.
- AS s27 starts again from scratch. The run order, rules and readout are unchanged. `go7.sh` replaces `go6.sh`
  only to skip the wait for go5 and the choice step.

## Correction (2026-10-31 14:45 +0800)

The cause given above as unknown is known, and it was the session's own autostop, not the platform. The
earlier autostop for Amendment E (bzkhfiqn1) had been stopped with TaskStop at 04:50:48 UTC, which reported
success, but its Git Bash script kept running. It saw go5's ALL_DONE (written 05:22:12 UTC), ran
`final_sync.sh` (05:26:14 UTC "autostop final sync" in /vault/labelshift_E/last_sync.txt) and called
stop-instance at 05:26:34.904 UTC (headless transcript 5e152fcc-f490-496e-b500-6ed48e6bbfa6). The sentences
"The session's autostop did not trigger" and "the session made no stop call" are withdrawn. Nothing about the
run order, rules or readout changes.
