import sys
sys.path.insert(0, sys.argv[1])
import store
from geom import CT, S3
for u in [CT["p1667"] + "/2/.zarray", CT["p4"] + "/2/.zarray", CT["p4"] + "/0/.zarray",
          S3 + "PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/2/.zarray",
          S3 + "PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/0/.zarray",
          S3 + "PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/mesh/20240304144031-on-20251217075048-2.399um.tifxyz/meta.json"]:
    print(u.split("/")[-4:], store.get(u).decode()[:400].replace("\n", " "))
