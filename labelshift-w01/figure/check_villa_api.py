"""Static check: every villa name render_beforeafter.py imports or calls exists at A's villa commit and
accepts the arguments the script passes. Reads source with `git show COMMIT:PATH` (no checkout, no
fetch) and inspects it with ast; nothing is imported or executed.

usage: python -B check_villa_api.py VILLA_REPO [COMMIT]   (COMMIT default 37d1897fb)
Works locally and on the remote box (VILLA_REPO = the villa checkout whose vesuvius/src is VILLA_SRC).
"""
import ast
import subprocess
import sys

REPO = sys.argv[1]
COMMIT = sys.argv[2] if len(sys.argv) > 2 else "37d1897fb"
BASE = "vesuvius/src/vesuvius/"

# (module path, qualified name, positional arg count passed, keyword names passed)
CALLS = [
    ("ink_detection/data/dataset.py", "InkDataset", 1, ["do_augmentations"]),
    ("ink_detection/data/dataset.py", "InkDataset._native_sample", 1, []),
    ("ink_detection/data/dataset.py", "InkDataset._tifxyz", 1, []),
    ("ink_detection/data/dataset.py", "InkDataset._open", 2, []),
    ("ink_detection/data/dataset.py", "InkDataset._normal_thicknesses", 1, []),
    ("ink_detection/data/dataset.py", "_read_flat_surface", 1, ["y0", "y1", "x0", "x1"]),
    ("ink_detection/data/geometry.py", "native_tifxyz_pyramid_params", 1, []),
    ("ink_detection/data/geometry.py", "read_tifxyz_on_flat_grid", 1,
     ["y0", "y1", "x0", "x1", "flat_grid_stride", "native_coordinate_scale"]),
    ("ink_detection/data/geometry.py", "compute_native_crop_bbox", 3, []),
    ("ink_detection/data/patch_finding_default.py", "_scan_geometry", 2, []),
    ("ink_detection/data/patch_finding_default.py", "surface_patch_bbox", 4, []),
    ("ink_detection/training/train.py", "stage_training_request", 1, []),
    ("ink_detection/training/train.py", "training_dataset_config", 1, []),
    ("ink_detection/training/train.py", "prepare_model_input", 2, []),
    ("ink_detection/training/train.py", "prepare_validation_loss_inputs", 2, ["mode"]),
    ("ink_detection/training/stitching.py", "run_model_forward", 3, ["stitched", "use_gradient_checkpointing"]),
    ("ink_detection/models/checkpoint.py", "load_checkpoint", 1, []),
    ("ink_detection/models/checkpoint.py", "config_from_checkpoint", 1, ["source"]),
    ("ink_detection/models/checkpoint.py", "select_inference_weights", 1, ["source"]),
    ("ink_detection/models/checkpoint.py", "load_model_state", 2, []),
    ("ink_detection/models/model.py", "make_model", 1, []),
    ("ink_detection/types.py", "Patch", 0, ["segment", "bbox", "is_validation", "supervision_mask_override"]),
    ("tifxyz/types.py", "Tifxyz.get_normals", 4, []),
]
# attributes/fields read by the script, checked as text in the named file
TEXT = [
    ("ink_detection/data/dataset.py", "self.patch_size = config.patch_size"),
    ("ink_detection/data/dataset.py", "self.validation_patches = "),
    ("ink_detection/data/patch_finding_default.py", "segment.data_config.patch_finding.overlap"),
    ("ink_detection/data/patch_finding_default.py", "bbox = surface_patch_bbox(surface, y0, x0, patch_size)"),
    ("ink_detection/data/patch_finding_default.py", "supervision_mask_override=segment.validation_mask"),
    ("ink_detection/training/train.py", "config.ink.model.input_pad_depth_to"),
    ("ink_detection/data/dataset.py", "crop_bbox = compute_native_crop_bbox("),
    ("ink_detection/data/dataset.py", "patch_positions, patch_valid, self.patch_size"),
    ("ink_detection/data/dataset.py", "if self.do_augmentations:\n            crop_bbox = maybe_translate_crop_bbox("),
    ("ink_detection/data/geometry.py", "local_z = int(z) - z0"),
    ("ink_detection/data/geometry.py", "if abs(step) > half_thickness + 1e-6:"),
]


def show(path):
    return subprocess.run(["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false", "-C", REPO, "show",
                           f"{COMMIT}:{BASE}{path}"], capture_output=True, text=True, check=True).stdout


def find(tree, qual):
    parts = qual.split(".")
    nodes = tree.body
    found = None
    for part in parts:
        found = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part), None)
        if found is None:
            return None
        nodes = found.body
    return found


def params(node):
    """(positional names, keyword-acceptable names, has **kwargs, has *args) of a def, or of a class's
    __init__ / dataclass fields."""
    if isinstance(node, ast.ClassDef):
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is None:  # dataclass: annotated fields in order
            fields = [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]
            return fields, set(fields), False, False
        node, drop_self = init, True
    else:
        drop_self = bool(node.args.args) and node.args.args[0].arg in ("self", "cls")
    a = node.args
    pos = [x.arg for x in a.posonlyargs + a.args]
    if drop_self:
        pos = pos[1:]
    kw = set(x.arg for x in a.args) | set(x.arg for x in a.kwonlyargs)
    return pos, kw, a.kwarg is not None, a.vararg is not None


bad = 0
cache = {}
for path, qual, npos, kws in CALLS:
    src = cache.setdefault(path, show(path))
    node = find(ast.parse(src), qual)
    if node is None:
        print(f"FAIL {path}:{qual} not found at {COMMIT}")
        bad += 1
        continue
    pos, kw, has_kwargs, has_varargs = params(node)
    problems = []
    if npos > len(pos) and not has_varargs:
        problems.append(f"{npos} positional args passed, accepts {len(pos)} ({pos})")
    missing = [k for k in kws if k not in kw and not has_kwargs]
    if missing:
        problems.append(f"keywords not accepted: {missing}")
    print(("FAIL " if problems else "ok   ") + f"{path}:{node.lineno} {qual}({', '.join(pos)}; kw {sorted(kw)})"
          + (f" -> {problems}" if problems else ""))
    bad += bool(problems)
for path, snippet in TEXT:
    src = cache.setdefault(path, show(path))
    ok = snippet in src
    print(("ok   " if ok else "FAIL ") + f"{path}: {snippet!r}")
    bad += not ok
print(f"{bad} problems at {COMMIT}")
sys.exit(1 if bad else 0)
