"""Run villa's unmodified ink trainer (train.main) from one source tree.

usage: python -I launch.py VILLA_SRC CONFIG_JSON
VILLA_SRC (a tree's vesuvius/src) goes first on sys.path; spawned DataLoader workers inherit that
sys.path, so every process imports vesuvius from it. Each process logs which vesuvius it imported.
If LAUNCH_EXTRA_SITE is set, that directory goes on sys.path behind VILLA_SRC (the local smoke test
uses it for imagecodecs, which the fix2277 venv lacks; the pod installs it into its own venv).
"""
import os
import sys

if os.environ.get("LAUNCH_EXTRA_SITE"):
    sys.path.insert(0, os.environ["LAUNCH_EXTRA_SITE"])


def _log(message):
    print(f"[launch pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    src, config = sys.argv[1], sys.argv[2]
    sys.path.insert(0, src)
    import vesuvius
    from vesuvius.ink_detection.training import train

    _log(f"main vesuvius={vesuvius.__file__}")
    raise SystemExit(train.main([config]))
else:
    import vesuvius

    _log(f"worker vesuvius={vesuvius.__file__}")
