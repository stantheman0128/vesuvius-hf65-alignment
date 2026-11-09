#!/bin/bash
cd ~/labelshift-w01 || { echo "### no work dir" >> ~/figA_out/run_s29.status; exit 1; }
export AWS_RETRY_MODE=legacy AWS_MAX_ATTEMPTS=14
PY="$HOME/labelshift/venv/bin/python"
ST=~/figA_out/run_s29.status
ts() { TZ=Asia/Taipei date '+%F %T'; }
echo "### smoke start $(ts)" >> "$ST"
"$PY" -I ~/figA/render_beforeafter.py ~/labelshift/villa/vesuvius/src ~/figA_out/smoke_s29 --seed 29 --work . --boxes 5 > ~/figA_out/smoke_s29.log 2>&1
rc=$?
echo "### smoke exit $rc at $(ts)" >> "$ST"
if [ $rc -ne 0 ]; then echo "### stop: smoke failed" >> "$ST"; touch ~/figA_out/DONE_s29; exit $rc; fi
echo "### full start $(ts)" >> "$ST"
"$PY" -I ~/figA/render_beforeafter.py ~/labelshift/villa/vesuvius/src ~/figA_out/s29 --seed 29 --work . > ~/figA_out/s29.log 2>&1
rc=$?
echo "### full exit $rc at $(ts)" >> "$ST"
touch ~/figA_out/DONE_s29
