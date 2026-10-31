"""Run villa main's find_segment_patches (patch_finding_default.py, extracted verbatim with ast)
with dino_guided_teacher settings on level-4 middle slices saved by readlevel.py.

Settings from configs/dino_guided_teacher.json: patch_size [256,256,256], patch_overlap 0.5,
patch_min_labeled_coverage 0.02, patch_finding_scale 4, volume_scale 0. CT level shapes from
PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr: L0 [37076,15229,15229],
L4 [2318,952,952] (only .shape is used, for scale_y and the bbox z).

usage: patches.py <sup.npz> <ink.npz> [<val.npz>|-]
Prints counts and the supervised level-4 pixel total under the training patches."""
import ast, json, os, sys
from dataclasses import dataclass
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "villa_patch_finding_default_9a92af33e.py"), encoding="utf-8").read()
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (
    "surface_patch_bbox", "labeled_patch_coverage", "combined_patch_discovery_support",
    "_scan_geometry", "find_segment_patches")]


@dataclass(frozen=True)
class Patch:
    segment: object
    bbox: tuple
    is_validation: bool = False
    supervision_mask_override: object = None


ns = {"np": np, "Patch": Patch, "Segment": object, "Callable": object, "__name__": "pf"}
exec(compile(ast.Module(body=keep, type_ignores=[]), "patch_finding_default.py", "exec"), ns)


class Mid:
    """(65, H, W) array view that only serves slice 32, the one the patch finder reads."""
    def __init__(self, mid, z):
        self.mid, self.z = mid, int(z)
        self.shape = (65, mid.shape[0], mid.shape[1])
    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert key[0] == self.z, key
            return self.mid[key[1:]]
        assert key == self.z, key
        return self.mid


class Shape:
    def __init__(self, shape):
        self.shape = tuple(shape)


@dataclass(frozen=True)
class PF:
    overlap: float = 0.5
    min_labeled_coverage: float = 0.02
    scan_scale: int = 4


@dataclass(frozen=True)
class DC:
    patch_finding: PF = PF()


@dataclass(frozen=True)
class Seg:
    supervision_mask: object
    inklabels: object
    validation_mask: object
    image_volume: str = "CT"
    scale: int = 0
    patch_size: tuple = (256, 256, 256)
    data_config: DC = DC()


def load(p):
    d = np.load(p)
    return Mid(d["mid"], d["surf"]), d["mid"]


sup_p, ink_p = sys.argv[1], sys.argv[2]
val_p = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
vols = {"CT@0": Shape((37076, 15229, 15229)), "CT@4": Shape((2318, 952, 952))}
sup, sup_mid = load(sup_p)
ink, ink_mid = load(ink_p)
vols[f"sup@4"], vols[f"ink@4"] = sup, ink
val_mid = None
if val_p:
    val, val_mid = load(val_p)
    vols["val@4"] = val


def open_volume(path, scale):
    return vols[f"{path}@{scale}"]


seg = Seg(supervision_mask="sup", inklabels="ink", validation_mask="val" if val_p else None)
try:
    training, held = ns["find_segment_patches"](seg, open_volume)
    err = None
except ValueError as e:
    training, held, err = [], [], str(e)

# supervised area actually covered by training patches (union, level-4 px of the middle slice,
# validation pixels excluded as dataset.py exclude_validation_voxels does)
cover = np.zeros(sup_mid.shape, bool)
for p in training:
    _, y0, x0, _, y1, x1 = p.bbox
    ys, xs = int(round(y0 / (15229 / 952))), int(round(x0 / (15229 / 952)))
    cover[ys:ys + 16, xs:xs + 16] = True
supervised = sup_mid > 0
if val_mid is not None:
    supervised &= ~(val_mid > 0)
print(json.dumps({
    "sup": os.path.basename(sup_p), "ink": os.path.basename(ink_p),
    "val": None if not val_p else os.path.basename(val_p), "error": err,
    "training_patches": len(training), "validation_patches": len(held),
    "sup_px_L4": int((sup_mid > 0).sum()), "sup_frac_L4": float((sup_mid > 0).mean()),
    "train_supervised_px_L4": int((supervised & cover).sum()),
    "ink_px_L4": int((ink_mid > 0).sum()),
    "ink_in_sup_px_L4": int(((ink_mid > 0) & (sup_mid > 0)).sum()),
    "ink_in_train_supervised_px_L4": int(((ink_mid > 0) & supervised & cover).sum()),
}))
