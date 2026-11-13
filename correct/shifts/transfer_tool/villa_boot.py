"""Make villa's tifxyz_label_transfer importable from a source tree, without installing villa.

villa_src is villa's `vesuvius/src`. The `vesuvius` and `vesuvius.utils` package __init__ files are
skipped (they pull in unrelated extras); every module that is used is loaded unchanged.

vc_reader is villa's `vc_reader` folder with a scikit-build build in it: the Python sources in
`<vc_reader>/src/vc_reader` and the compiled extensions in `<vc_reader>/build/<tag>`, where <tag> is
for example cp314-cp314-linux_x86_64. Pass vc_reader_build to name that build folder directly. With
vc_reader None, an installed vc_reader package is imported instead.
"""
import glob
import importlib.util
import os
import sys
import types

sys.dont_write_bytecode = True


def _load_vc_reader(root, build=None):
    src = os.path.join(root, "src", "vc_reader")
    if build is None:
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        found = sorted(glob.glob(os.path.join(root, "build", f"{tag}-{tag}-*")))
        if len(found) != 1:
            raise SystemExit(f"expected one build/{tag}-{tag}-* folder under {root}, found {found}; "
                             "pass the build folder explicitly")
        build = found[0]
    spec = importlib.util.spec_from_file_location(
        "vc_reader", os.path.join(src, "__init__.py"), submodule_search_locations=[src, build])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vc_reader"] = mod
    spec.loader.exec_module(mod)


def boot(villa_src, vc_reader=None, vc_reader_build=None):
    if vc_reader is not None:
        _load_vc_reader(vc_reader, vc_reader_build)
    for name, path in (("vesuvius", os.path.join(villa_src, "vesuvius")),
                       ("vesuvius.utils", os.path.join(villa_src, "vesuvius", "utils"))):
        module = types.ModuleType(name)
        module.__path__ = [path]
        sys.modules[name] = module
