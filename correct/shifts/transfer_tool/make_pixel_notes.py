"""Write pixel_notes.json (the checked_on_pixels column) from the w01 and w00 compare outputs."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W01 = os.path.join(HERE, "w01", "compare_agreement.json")
W00 = os.path.join(HERE, "w00", "compare_agreement.json")


def pct(p):
    return f"{100 * p['fraction_equal']:.3f}% equal, ink IoU {p['ink_iou']:.4f}"


notes = {}
a = json.load(open(W01))["pairs"]
notes["w01"] = ("partly: the w01 run used the affine3 refit constant 11.3181 11.1463, not this row's model; "
                f"tool output vs that model on the tool grid {pct(a['offset~AF_tool'])}; "
                f"vs the model on the loader grid {pct(a['offset~AF'])}")
if os.path.exists(W00):
    b = json.load(open(W00))["pairs"]
    notes["w00"] = (f"yes: DY DX run vs this row on the tool grid {pct(b['offset~AF_tool'])}, "
                    f"on the loader grid {pct(b['offset~AF'])}; loader_DY loader_DX run vs this row "
                    f"on the loader grid {pct(b['loader~AF'])}; no-offset control on the loader grid {pct(b['control~AF'])}")
with open(os.path.join(HERE, "pixel_notes.json"), "w", newline="\n") as f:
    json.dump(notes, f, indent=1)
print(json.dumps(notes, indent=1))
