r"""mock_paths.py -- where the mock ensemble's json files live.

`results_mock/` held 20,319 json files alongside the 15 figures the analysis writes there,
which made the figures impossible to find.  The json now lives in `results_mock/json/`
and everything else -- pdf, png, txt -- stays where it was.

Every script that touches the ensemble routes its paths through here, so the split is one
decision in one file rather than nine.  Reads FALL BACK to the flat location: a fresh
`mock_idl.py --task lsweep` run, or a copy from a cluster, still writes json into
`results_mock/`, and those files keep working until

    python mock_paths.py            # sweep stray json into results_mock/json/

is run again.  Writes always go to the json directory.

The subdirectories that were already separate -- `addcell/`, `cluster/`, `local_ref/` --
are left exactly as they are; they are their own collections and nothing globs across them.
"""
import glob
import os
import sys

JSON_SUBDIR = 'json'


def _is_pattern(name):
    return any(c in name for c in '*?[')


def rpath(res_dir, name, write=False):
    """The path for one file: json/<name> for .json, <name> otherwise.

    A .json that exists ONLY in the flat directory resolves there, so a file written by a
    tool that does not know about the split is still found.  `write=True` skips that and
    always returns the json directory.
    """
    if not name.endswith('.json'):
        return os.path.join(res_dir, name)
    p = os.path.join(res_dir, JSON_SUBDIR, name)
    if write or _is_pattern(name):
        return p
    flat = os.path.join(res_dir, name)
    return flat if (not os.path.exists(p) and os.path.exists(flat)) else p


def rglob(res_dir, pattern):
    """Matches from json/ first, then any the flat directory adds -- by BASENAME, so a file
    present in both is counted once and the json/ copy is the one returned."""
    out = glob.glob(os.path.join(res_dir, JSON_SUBDIR, pattern))
    seen = {os.path.basename(p) for p in out}
    out += [p for p in glob.glob(os.path.join(res_dir, pattern))
            if os.path.basename(p) not in seen]
    return sorted(out)


def tidy(res_dir, quiet=False):
    """Move stray top-level json into json/.  Never overwrites: a name present in both is
    reported and left alone, because the two are not necessarily the same fit."""
    dest = os.path.join(res_dir, JSON_SUBDIR)
    os.makedirs(dest, exist_ok=True)
    moved = clash = 0
    for p in glob.glob(os.path.join(res_dir, '*.json')):
        q = os.path.join(dest, os.path.basename(p))
        if os.path.exists(q):
            clash += 1
            print('[tidy] BOTH: %s' % os.path.basename(p))
            continue
        os.replace(p, q)
        moved += 1
    if not quiet:
        print('[tidy] %s: moved %d, left %d in place' % (res_dir, moved, clash))
    return moved, clash


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(1 if tidy(os.path.join(here, 'results_mock'))[1] else 0)
