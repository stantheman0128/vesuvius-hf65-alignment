# Amendment J setup, run once on the second (fresh) pod with pod_bundle.tar.gz, h_extra.tar.gz and j_extra.tar.gz
# already copied to ~. Unlike pod_setup_h.sh it does not move an old tree aside: an existing ~/labelshift means this
# is not the fresh second pod (the first pod runs H and I there), so it stops without touching anything.
set -eu
cd ~
[ -e labelshift ] && { echo "~/labelshift exists: not the fresh second pod, nothing done"; exit 1; }
echo "b60e7917f213f6ed264e7fd074e52c527bd996b520349a50c3b8cf4ab3d3fde2  pod_bundle.tar.gz" | sha256sum -c
echo "1c354589cfbc5e71a965400c220d5b5149ac05f30f190b53836536a81cf5a13b  h_extra.tar.gz" | sha256sum -c
echo "6f5f779a2c535263d3f0e0236d571cc59353680a579ced8585978aabc632a0dd  j_extra.tar.gz" | sha256sum -c
tar xzf pod_bundle.tar.gz
tar xzf h_extra.tar.gz
tar xzf j_extra.tar.gz
cd ~/labelshift
sha256sum -c SHA256SUMS
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h ~ | tail -1
ls -d /vault/labelshift_* 2>/dev/null || true
setsid nohup bash go_j.sh > go_j.log 2>&1 < /dev/null &
sleep 5
echo "go_j processes: $(ps -eo args | grep -c '^bash go_j[.]sh')"
tail -3 go_j.log
