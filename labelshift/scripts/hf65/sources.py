"""Per-key data sources: tifxyz surface, image reader at a level, volume, mid slice index."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import s3, MIRROR
from lib import Volume, Image, S3Image, Surf, surf_from_url

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = s3("PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr")
C1667 = s3("PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr")
HFSEG = {
    "c018": ("ink/1667/w018_20240304144031_2um", "w018_20240304144031_2um.zarr", 1, C1667, "w018_20240304144031_2um_supervision_mask.zarr"),
    "w00": ("ink/phercparis4/w00_20231016151002", "w00_20231016151002.zarr", 2, P4, "w00_20231016151002_supervision_mask.zarr"),
    "w01": ("ink/phercparis4/w01_20230702185753_r15", "w01_20230702185753_r15.zarr", 2, P4, "w01_20230702185753_r15_supervision_mask.zarr"),
    "w05": ("ink/phercparis4/w05_4424", "w05_4424.zarr", 2, P4, "4424_supervision_mask.zarr"),
}
S3SEG = {
    "c018s3": ("PHerc1667/segments/20240304144031-w018_20240304144031_flatboi",
               "mesh/20240304144031-on-20251217075048-2.399um.tifxyz",
               "surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr", C1667),
    "w00s3": ("PHercParis4/segments/20231016151002",
              "mesh/20231016151002-on-20260411134726-2.4um.tifxyz",
              "surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr", P4),
}
S3SLICES = list(range(50, 59))  # 109-slice renders: middle is 54


def pyr_shift(key):
    p = os.path.join(HERE, f"pyrshift_{key}.json")
    if os.path.exists(p):
        return tuple(json.load(open(p))["shift"])
    return {"c018": (0, 0), "w00": (3, 0)}.get(key)


def get(key):
    """-> dict(surf, img, klev, vol, mid, shift)"""
    if key in HFSEG:
        segdir, zname, klev, vol, _ = HFSEG[key]
        local = os.path.join(MIRROR, "r4", *segdir.split("/")[1:])
        surf = Surf(segdir, local if os.path.isdir(local) else None)
        img = Image(f"{segdir}/{zname}", klev)
        img_l0 = Image(f"{segdir}/{zname}", 0).shape[1:]
        mid = 32
    else:
        seg, mesh, sv, vol = S3SEG[key]
        surf = surf_from_url(s3(f"{seg}/{mesh}"))
        klev = 2
        img = S3Image(s3(f"{seg}/{sv}"), klev, S3SLICES)
        mid = S3SLICES.index(54)
        img_l0 = S3Image(s3(f"{seg}/{sv}"), 0, [54]).full_shape[1:]
    return {"surf": surf, "img": img, "klev": klev, "vol": Volume(vol, 2), "mid": mid, "shift": pyr_shift(key), "img_l0": img_l0}
