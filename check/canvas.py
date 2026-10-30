"""My own HF65 canvas reader (zarr v2, blosc) through store.py."""
import json
from urllib.parse import quote
import numpy as np
import numcodecs
import store
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
SEGS = {"w018": ("ink/1667/w018_20240304144031_2um", "w018_20240304144031_2um.zarr"),
        "w00": ("ink/phercparis4/w00_20231016151002", "w00_20231016151002.zarr"),
        "w01": ("ink/phercparis4/w01_20230702185753_r15", "w01_20230702185753_r15.zarr"),
        "w05": ("ink/phercparis4/w05_4424", "w05_4424.zarr")}


def hf(p):
    return HF + quote(p, safe="")


class Canvas:
    def __init__(self, key, level):
        d, z = SEGS[key]
        self.base = f"{d}/{z}/{level}"
        za = json.loads(store.get(hf(self.base + "/.zarray")))
        self.shape, self.ch = za["shape"], za["chunks"]
        self.sep = za.get("dimension_separator", ".")
        self.codec = numcodecs.get_codec(za["compressor"])
        assert za["dtype"] == "|u1" and za["order"] == "C", za
        self.fill = za.get("fill_value") or 0

    def chunk(self, cy, cx):
        raw = store.get(hf(f"{self.base}/0{self.sep}{cy}{self.sep}{cx}"))
        if raw is None:
            return np.full(self.ch, self.fill, np.uint8)
        return np.frombuffer(self.codec.decode(raw), np.uint8).reshape(self.ch)

    def read(self, zs, y0, y1, x0, x1):
        """slices zs (list), rows [y0,y1), cols [x0,x1) at this level; outside -> 0."""
        out = np.zeros((len(zs), y1 - y0, x1 - x0), np.uint8)
        cy0, cy1 = max(0, y0) // self.ch[1], (min(y1, self.shape[1]) - 1) // self.ch[1]
        cx0, cx1 = max(0, x0) // self.ch[2], (min(x1, self.shape[2]) - 1) // self.ch[2]
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                c = self.chunk(cy, cx)
                gy, gx = cy * self.ch[1], cx * self.ch[2]
                a0, a1 = max(y0, gy), min(y1, gy + self.ch[1], self.shape[1])
                b0, b1 = max(x0, gx), min(x1, gx + self.ch[2], self.shape[2])
                out[:, a0 - y0:a1 - y0, b0 - x0:b1 - x0] = c[zs][:, a0 - gy:a1 - gy, b0 - gx:b1 - gx]
        return out
