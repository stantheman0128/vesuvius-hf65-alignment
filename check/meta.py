import sys, json, struct
sys.path.insert(0, sys.argv[1])
import store
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
from urllib.parse import quote
def hf(p): return HF + quote(p, safe="")
segs = {"w018": ("ink/1667/w018_20240304144031_2um", "w018_20240304144031_2um.zarr"),
        "w00": ("ink/phercparis4/w00_20231016151002", "w00_20231016151002.zarr"),
        "w01": ("ink/phercparis4/w01_20230702185753_r15", "w01_20230702185753_r15.zarr"),
        "w05": ("ink/phercparis4/w05_4424", "w05_4424.zarr")}
import numpy as np, math
def tiff_hw(url):
    h = store.get_range(url, 0, 15)
    bo = "<" if h[:2] == b"II" else ">"
    big = struct.unpack(bo+"H", h[2:4])[0] == 43
    off = struct.unpack(bo+("Q" if big else "I"), h[8:16] if big else h[4:8])[0]
    ifd = store.get_range(url, off, off+4095, short_ok=True)
    n = struct.unpack(bo+("Q" if big else "H"), ifd[:8] if big else ifd[:2])[0]
    p, esz = (8, 20) if big else (2, 12)
    t = {}
    for i in range(n):
        e = ifd[p+i*esz:p+(i+1)*esz]
        tag, typ = struct.unpack(bo+"HH", e[:4])
        if tag in (256, 257, 258, 259, 278, 322, 339):
            fmt = {3: "H", 4: "I", 16: "Q"}.get(typ)
            if fmt: t[tag] = struct.unpack(bo+fmt, e[(12 if big else 8):(12 if big else 8)+struct.calcsize(fmt)])[0]
    return t
for k, (d, z) in segs.items():
    m = json.loads(store.get(hf(d + "/meta.json")))
    sx, sy = m["scale"]
    t = tiff_hw(hf(d + "/x.tif"))
    W, H = t[256], t[257]
    shapes = []
    for lv in range(6):
        try:
            za = json.loads(store.get(hf(f"{d}/{z}/{lv}/.zarray")))
            shapes.append((lv, za["shape"], za["chunks"], za.get("compressor", {}) and za["compressor"].get("id")))
        except Exception as e:
            shapes.append((lv, "ERR " + str(e)[:60]))
    s32x, s32y = float(np.float32(sx)), float(np.float32(sy))
    fh = max(1, int(math.floor(H * (1.0 / s32y) + 0.5))); fw = max(1, int(math.floor(W * (1.0 / s32x) + 0.5)))
    print(k, "scale[sx,sy]", m["scale"], "stored HxW", H, W, "tiftags", t, "fullres HxW", fh, fw, "volume", m.get("volume"))
    for s in shapes: print("   ", s)
    L0 = shapes[0][1]
    if isinstance(L0, list):
        print("    ratio-1 y %.6f x %.6f" % (fh / L0[1] - 1, fw / L0[2] - 1), " diff rows", fh - L0[1], "cols", fw - L0[2])
print("new bytes", store.new_bytes())
