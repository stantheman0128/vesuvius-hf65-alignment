import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
from vio import fetch_counted, s3, hf, mine
from vlist import ls
for segp in ["PHerc0139/segments/20250831000000-w040_2025083102", "PHerc0139/segments/20260115000000-w044_2026011522"]:
    for sub in ["surface-volumes/", "ink-labels/", "mesh/"]:
        for e in ls(f"{segp}/{sub}"):
            print("  ", e)
            if e.endswith(".tifxyz/"):
                m = fetch_counted(s3(f"{e}meta.json"))
                if m:
                    j = json.loads(m); print("      meta", {k: j.get(k) for k in ("bbox", "scale", "volume", "uuid")})
                x = fetch_counted(s3(f"{e}x.tif"), (0, 15))
            if e.endswith(".zarr/"):
                for L in (0,):
                    z = fetch_counted(s3(f"{e}{L}/.zarray"))
                    print("      zarray L0", json.loads(z)["shape"] if z else None, json.loads(z)["chunks"] if z else None)
                za = fetch_counted(s3(f"{e}.zattrs"))
                print("      zattrs", za.decode()[:300].replace("\n", " ") if za else None)
for s in ["ink/0139/w040_2025083102", "ink/0139/w044_2026011522"]:
    j = json.loads(fetch_counted(hf(f"{s}/meta.json")))
    print(s, "bucket meta", {k: j.get(k) for k in ("bbox", "scale", "volume", "uuid")})
print("mine", mine())
