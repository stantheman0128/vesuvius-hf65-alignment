import os, sys, re, json
HERE = os.path.dirname(os.path.abspath(__file__))  # audit/independent/scripts
AUDIT = os.path.dirname(os.path.dirname(HERE))  # audit/
sys.path.insert(0, HERE)
from vio import fetch_counted, s3, mine
def ls(prefix, delim="/"):
    out, tok = [], None
    while True:
        u = s3(f"?list-type=2&prefix={prefix}&delimiter={delim}") + (f"&continuation-token={tok}" if tok else "")
        x = fetch_counted(u).decode()
        out += re.findall(r"<Prefix>([^<]*)</Prefix>", x)[1:] + re.findall(r"<Key>([^<]*)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", x)
        if not m: break
        from urllib.parse import quote
        tok = quote(m.group(1), safe="")
    return out
for p in sys.argv[1:]:
    r = ls(p)
    print(p, len(r))
    for e in r: print("  ", e)
print("mine", mine())
