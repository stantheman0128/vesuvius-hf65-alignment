"""PNG strip: HF65 slice | tifxyz-sampled CT at identity | at the NCC peak (high-passed, level-2 px).
argv: key region out.png"""
import os, sys, zlib, struct, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import DATA
from lib import highpass

key, ri, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
z = np.load(os.path.join(DATA, f"reg_{key}_{ri}.npz"))
I, S, Im, Sm, Mh = z["I"], z["S"], z["Im"], z["Sm"], z["Mh"]
R = Mh.shape[0] // 2
h, w = I.shape
i, j = np.unravel_index(np.nanargmax(Mh), Mh.shape)
dy, dx = i - R, j - R
def norm(a, m):
    b = highpass(a, m, 4)
    lo, hi = np.percentile(b[m], [1, 99])
    return (np.clip((b - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
A = norm(I, Im)
B = norm(S, Sm)
pan = [A, B[R:R + h, R:R + w], B[R + dy:R + dy + h, R + dx:R + dx + w]]
img = np.concatenate([np.pad(p, ((0, 0), (0, 6)), constant_values=255) for p in pan], 1)
def png(a, fn):
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(a.shape[0]))
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(fn, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", a.shape[1], a.shape[0], 8, 0, 0, 0, 0))
                + ch(b"IDAT", zlib.compress(raw, 6)) + ch(b"IEND", b""))
png(img, out)
print("peak", dy, dx, "->", out)
