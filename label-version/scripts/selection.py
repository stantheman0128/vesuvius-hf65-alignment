"""Run villa main's own discover_segment_labels (extracted verbatim with ast from
villa_segment_9a92af33e.py) on the fresh bucket listing of the six 1667 training folders,
with dino_guided_teacher settings: no label_version, discovery 'labeled' (required=True)."""
import ast, json, os, sys
from dataclasses import dataclass, replace, field
from pathlib import Path, PurePosixPath

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "villa_segment_9a92af33e.py"), encoding="utf-8").read()
tree = ast.parse(src)
keep = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("_LABEL_SUFFIXES", "_LABEL_EXTENSIONS") for t in node.targets):
        keep.append(node)
    if isinstance(node, ast.FunctionDef) and node.name in ("parse_label_asset_path", "discover_segment_labels"):
        keep.append(node)
mod = ast.Module(body=keep, type_ignores=[])


class FakeDir:
    def __init__(self, path, names):
        self.path, self.names = path, names
    def iterdir(self):
        return [PurePosixPath(self.path) / n for n in self.names]
    def __str__(self):
        return self.path


@dataclass(frozen=True)
class DataCfg:
    label_version: object = None


@dataclass(frozen=True)
class Seg:
    data_config: DataCfg
    segment_dir: object
    segment_name: str
    inklabels: object = None
    supervision_mask: object = None
    validation_mask: object = None


ns = {"Path": PurePosixPath, "replace": replace, "Segment": Seg, "InkDataConfig": DataCfg,
      "__name__": "villa_segment_extract"}
exec(compile("from __future__ import annotations\n", "x", "exec"), ns)
exec(compile(mod, "villa_segment_9a92af33e.py", "exec"), ns)

lst = json.load(open(sys.argv[1]))
out = {}
for s, ents in lst["folders"].items():
    names = [e["path"].rsplit("/", 1)[-1] for e in ents]
    seg = Seg(DataCfg(None), FakeDir(f"ink/1667/{s}", names), s)
    r = ns["discover_segment_labels"](seg, extension=".zarr", required=True)
    out[s] = {k: (None if getattr(r, k) is None else str(getattr(r, k)).rsplit("/", 1)[-1])
              for k in ("inklabels", "supervision_mask", "validation_mask")}
    print(s, json.dumps(out[s]))
json.dump(out, open(sys.argv[2], "w"), indent=1)
