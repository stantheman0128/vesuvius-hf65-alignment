#!/usr/bin/env bash
# Eval offset scan (SCAN-2026-11-01.md, revision of 2026-11-01 02:40 +0800): run scan2.sh again once C32y_s28 has
# trained, so pair 11 does not wait for ALL_DONE_H. scan2.sh's lock queues it behind a call that is still running.
cd ~/labelshift
until [ -f runs/C32y_s28/train.done ] || [ -f ALL_DONE_H ]; do sleep 60; done
nice -n 10 bash scan2.sh >> scan.log 2>&1 < /dev/null
