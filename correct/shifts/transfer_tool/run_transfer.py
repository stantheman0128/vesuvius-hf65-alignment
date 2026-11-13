"""Run vesuvius.tifxyz_label_transfer.transfer from a villa source tree with a prebuilt vc_reader.

usage: VILLA_SRC=... [VC_READER=...] [VC_READER_BUILD=...] python -I run_transfer.py single ...

VILLA_SRC, VC_READER and VC_READER_BUILD are as in villa_boot.py; without VC_READER an installed
vc_reader is used. All arguments go to the transfer CLI unchanged. The vesuvius and vesuvius.utils
package __init__ files are skipped (they pull in unrelated extras); every module the transfer CLI
uses is loaded unchanged.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import villa_boot  # noqa: E402

villa_boot.boot(os.environ["VILLA_SRC"], os.environ.get("VC_READER"), os.environ.get("VC_READER_BUILD"))
from vesuvius.tifxyz_label_transfer import transfer  # noqa: E402

print("transfer module:", transfer.__file__, flush=True)
raise SystemExit(transfer.main(sys.argv[1:]))
