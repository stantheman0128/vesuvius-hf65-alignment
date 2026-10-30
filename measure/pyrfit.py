import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyrcheck import img_fit
from common import ledger_total
w = sys.argv[1]
if w == "p4":
    z = "ink/phercparis4/w00_20231016151002/w00_20231016151002.zarr"
    img_fit(z, 1, 125, 200, 32, [32], kref=0)
    img_fit(z, 2, 125, 200, 32, [32], kref=0)
    img_fit(z, 2, 62, 100, 32, [32], kref=1)
elif w == "c":
    z = "ink/1667/w018_20240304144031_2um/w018_20240304144031_2um.zarr"
    img_fit(z, 1, 166, 384, 32, [32], kref=0)
    img_fit(z, 1, 166, 384, 32, [64, 65], kref=0)
    img_fit(z, 1, 166, 384, 16, [32, 33], kref=0)
print("ledger total bytes", ledger_total())
