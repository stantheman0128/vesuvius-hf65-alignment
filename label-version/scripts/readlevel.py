"""Read one pyramid level of a bucket label zarr (zarr v2, '/' separator, chunks (65,128,128))
chunk by chunk, keep slice z=SURF (default shape[0]//2) plus the max over z, save as .npz.

usage: readlevel.py <zarr path in bucket> <level> <out.npz>"""
import os, sys, json
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import numcodecs
import hfio

numcodecs.blosc.set_nthreads(1)

zp, lv, out = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.loads(hfio.get(f"{zp}/{lv}/.zarray").decode())
shape, chunks = meta["shape"], meta["chunks"]
assert meta["dimension_separator"] == "/" and meta["order"] == "C", meta
assert chunks[0] == shape[0], (chunks, shape)
codec = numcodecs.get_codec(meta["compressor"])
dtype = np.dtype(meta["dtype"])
fill = meta["fill_value"] or 0
Z, H, W = shape
surf = Z // 2
ny, nx = -(-H // chunks[1]), -(-W // chunks[2])
mid = np.full((H, W), fill, dtype=dtype)
zmax = np.full((H, W), fill, dtype=dtype)
zmin = np.full((H, W), fill, dtype=dtype)
missing = []
present_allzero = []
present = []


def one(ij):
    i, j = ij
    d = hfio.get(f"{zp}/{lv}/0/{i}/{j}")
    if d is None:
        return ij, None
    a = np.frombuffer(codec.decode(d), dtype=dtype).reshape(chunks)
    return ij, a


with ThreadPoolExecutor(max_workers=2) as ex:
    for (i, j), a in ex.map(one, [(i, j) for i in range(ny) for j in range(nx)]):
        y0, x0 = i * chunks[1], j * chunks[2]
        y1, x1 = min(H, y0 + chunks[1]), min(W, x0 + chunks[2])
        if a is None:
            missing.append((i, j))
            continue
        present.append((i, j))
        if not a.any():
            present_allzero.append((i, j))
        a = a[:, : y1 - y0, : x1 - x0]
        mid[y0:y1, x0:x1] = a[surf]
        zmax[y0:y1, x0:x1] = a.max(axis=0)
        zmin[y0:y1, x0:x1] = a.min(axis=0)

np.savez_compressed(out, mid=mid, zmax=zmax, zmin=zmin, shape=np.array(shape), surf=surf,
                    missing=np.array(missing, dtype=np.int64).reshape(-1, 2),
                    present=np.array(present, dtype=np.int64).reshape(-1, 2))
nz = int(np.count_nonzero(mid))
print(json.dumps({"zarr": zp, "level": lv, "shape": shape, "surf": surf, "chunks_total": ny * nx,
                  "chunks_missing": len(missing), "chunks_present_allzero": len(present_allzero), "mid_nonzero": nz, "mid_frac": nz / mid.size,
                  "zmax_nonzero": int(np.count_nonzero(zmax)),
                  "mid_eq_zmax": bool((mid == zmax).all()), "mid_eq_zmin": bool((mid == zmin).all()),
                  "values": [int(v) for v in np.unique(mid)[:10]], "spent": hfio.spent()}))
