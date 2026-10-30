import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
from vio import fetch_counted, s3, mine
for v in ["20250820105138-2.403um-0.2m-77keV-masked.zarr", "20250822062710-2.403um-0.2m-77keV-masked.zarr",
          "20260102150214-2.399um-0.2m-78keV-masked.zarr", "20260319133050-2.403um-0.2m-77keV-masked.zarr",
          "20260319133554-2.403um-0.2m-77keV-masked.zarr", "20260930100252-2.399um-0.2m-78keV-masked.zarr"]:
    b = f"PHerc0139/volumes/{v}"
    for L in (0, 2):
        z = fetch_counted(s3(f"{b}/{L}/.zarray"))
        print(v, L, json.loads(z)["shape"] if z else None, json.loads(z)["chunks"] if z else "", json.loads(z).get("compressor") if z else "", json.loads(z).get("dimension_separator") if z else "")
    m = fetch_counted(s3(f"{b}/metadata.json"))
    if m:
        try:
            j = json.loads(m); a = j.get("metadata", j)
            s = json.dumps(a)
            import re
            print("   meta:", re.findall(r'"(scanRadix|date|samplePixelSize|energy)": ("[^"]*"|[0-9.]+)', s)[:6])
        except Exception as e: print("   meta err", e)
print("mine", mine())
