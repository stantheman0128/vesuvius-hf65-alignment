# Changes after the freeze

PREREG.md and the scripts in this folder were frozen at commit e8b832b on 2026-11-04, before any training
run of A. This file lists what changed after that, in order.

## 2026-11-05: AS seed 27 restarted after a network outage

The first AS seed 27 training stopped at step 2314 of 12000 at 02:25 +0800. A DataLoader worker could not
connect to S3 while reading the Paris4 volume (botocore ConnectTimeoutError), and the trainer exited. A
remote check of the machine at about 02:30 failed as well, and S3 answered normally again by 02:41. The
attempt wrote no checkpoint and no held-out evaluation. Its training log has the trainer's own validation
metrics up to step 2250.

C0 seed 27 had finished at 01:16, so its result was known before this change: primary AUC 0.8380, which
passes the section 5 gate. No AS result existed yet.

The volume cache does not make later runs independent of S3. zarr's CacheStore keeps only chunks that S3
returns. Some chunks of the masked volume do not exist on S3 (the one in the error returns 404), and every
run asks S3 for those again. With s3fs's 5 s connect timeout and botocore's default of 5 attempts, a read
gives up after about 30 s without a connection.

What changed:

- `desk_supervise.sh` (new) starts `desk_launch.sh` with `AWS_RETRY_MODE=legacy` and `AWS_MAX_ATTEMPTS=14`.
  Legacy mode waits rand * 2^(n-1) s after attempt n, so one S3 read keeps trying for about an hour on
  average. If a training stops with a network error in its stderr, the script moves that training to
  `failed/` in the work folder and starts `desk_launch.sh` again after 15 minutes, at most twice. An
  evaluation that stops on a network error starts again the same way, without moving anything.
- AS seed 27 runs again from step 0 with the same config and seed. The first attempt's logs stay in
  `failed/`.
- C0 seed 27 ran with the default retry settings and every later run uses 14 attempts. Retries only change
  how long a read waits. A read either returns the same bytes as before or stops the run.

Sections 3 to 7 of PREREG.md are unchanged.
