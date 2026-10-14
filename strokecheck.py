#!/usr/bin/env python3
"""Stroke-count guard for matplotlib figures. Standard library only.

A page with thousands of *separate* stroked paths can defeat a printer's RIP
even when it is small and looks fine on screen. Total data volume is not the
predictor -- stroke count is. Measured on DES_cosmography.pdf:

    page 23  648,326 points in 1,059 strokes  -> printed fine
    page 26   88,156 points in 4,568 strokes  -> would not print

Drop this next to your plotting scripts. One line, and every PDF save reports
itself -- no savefig call needs changing:

    import strokecheck; strokecheck.enable()

    ok  figD1.pdf:  517 / 2000 strokes ( 26%)
    !!  figD2.pdf: 2518 / 2000 strokes (126%) -- THIS FIGURE MAY NOT PRINT.

Or sweep a folder of finished figures (no arguments means the current
directory), exiting 1 if any figure is over the limit:

    python3 strokecheck.py figures/
    python3 strokecheck.py

Multi-page PDFs are skipped automatically -- a document is not a figure, and
its total means nothing against a per-figure limit. Use check_strokes.py there.

Everything else is included by default -- you never list the figures. To leave a
few out, pass -x/--exclude (repeatable, glob), or put the patterns one per line
in a .strokecheckignore file in the folder. Anything left out is named in the
summary rather than dropped silently:

    python3 strokecheck.py figures/ -x 'draft_*' -x scratch.pdf

The limit is THRESHOLD below; it governs on-screen reporting only. For an
assembled multi-page document use check_strokes.py instead, which gates per
page -- the worst single page is what matters, not the document total.

Needs no third-party packages, so it travels with the scripts.
"""
import fnmatch
import functools
import glob
import os
import re
import sys
import zlib

# Warn above this many strokes in one figure. Not a measured limit: 1,059
# strokes on a page printed fine and 4,568 did not, with nothing between tested,
# so this sits in that gap erring safe. Affects on-screen reporting only --
# edit it freely, or set strokecheck.THRESHOLD at runtime.
THRESHOLD = 2000

# Stroke-painting operators as whole tokens: S s B B* b b*
STROKE_OP = re.compile(rb'(?<![A-Za-z0-9*])(S|s|B\*|B|b\*|b)(?![A-Za-z0-9*])')
# Text is written as kerned arrays like [(X)-250(s)15(ty)30]TJ, so isolated
# letters inside string literals would otherwise read as stroke operators.
STR_LIT = re.compile(rb'\((?:\\.|[^\\()])*\)', re.S)
HEX_LIT = re.compile(rb'<[0-9A-Fa-f\s]*>')
PRINTABLE = bytes(range(32, 127)) + b'\n\r\t'


def _is_content(chunk):
    """True if a decompressed stream looks like a content stream, not a font
    program or image. Content streams from matplotlib are ASCII operators;
    binary payloads would generate false operator matches."""
    if not chunk:
        return False
    sample = chunk[:4096]
    printable = sum(1 for b in sample if b in PRINTABLE)
    return printable / len(sample) > 0.90


PAGE_OBJ = re.compile(rb'/Type\s*/Page(?![sA-Za-z])')


def count_pages(path):
    """Number of page objects, counting inside compressed object streams."""
    data = open(path, 'rb').read()
    n = len(PAGE_OBJ.findall(data))
    for m in re.finditer(rb'stream\r?\n', data):
        s = m.end()
        e = data.find(b'endstream', s)
        if e < 0:
            continue
        try:
            chunk = zlib.decompress(data[s:e])
        except zlib.error:
            continue
        n += len(PAGE_OBJ.findall(chunk))
    return n


def count_strokes(path):
    """Approximate number of stroked paths in a PDF.

    Counts stroke operators across all Flate-compressed content streams,
    including matplotlib's embedded Form XObjects. Approximate: the regex does
    not skip operators inside string literals or inline image data. Good enough
    to tell 200 from 2,000, which is the decision that matters.
    """
    data = open(path, 'rb').read()
    total = 0
    for m in re.finditer(rb'stream\r?\n', data):
        start = m.end()
        end = data.find(b'endstream', start)
        if end < 0:
            continue
        try:
            chunk = zlib.decompress(data[start:end])
        except zlib.error:
            continue  # uncompressed, or an image/font codec -- skip
        if _is_content(chunk):
            chunk = HEX_LIT.sub(b'<>', STR_LIT.sub(b'()', chunk))
            total += len(STROKE_OP.findall(chunk))
    return total


def _report(path, n, limit, quiet=False):
    """Print one line showing the count against the limit, so the margin is
    visible at a glance. Over-threshold goes to stderr with what to do."""
    pages = count_pages(path)
    if pages > 1:
        # Summing a whole document against a per-figure limit is meaningless:
        # what matters is the worst single page, which needs check_strokes.py.
        print(f"  -- {path}: {n} strokes across {pages} pages. This is a "
              f"document, not a figure;\n     the per-figure limit does not "
              f"apply. Use check_strokes.py for the per-page gate.")
        return n
    pct = 100.0 * n / limit if limit else float('inf')
    head = f"{path}: {n} / {limit} strokes ({pct:.0f}%)"
    if n > limit:
        print(f"  !! {head} -- THIS FIGURE MAY NOT PRINT.\n"
              f"     Set rasterized=True on the dense line collections (not the "
              f"axes), and\n     remember any insets: they redraw every trace, "
              f"which doubles the count.", file=sys.stderr)
    elif not quiet:
        print(f"  ok {head}")
    return n


def savefig_checked(fig, path, threshold=None, quiet=False, **kwargs):
    """fig.savefig(...) then report the stroke count. Returns the count."""
    threshold = THRESHOLD if threshold is None else threshold
    fig.savefig(path, **kwargs)
    n = count_strokes(path)
    return _report(path, n, threshold, quiet)


def enable(threshold=None, quiet=False):
    """Patch Figure.savefig so *every* PDF save reports its stroke count.

    One line at the top of a plotting script, and no savefig call needs
    changing:

        import strokecheck; strokecheck.enable()

    Safe to call more than once. Never raises: a failed count must not break
    someone's plotting run, so problems are reported and swallowed.
    """
    from matplotlib.figure import Figure
    if getattr(Figure.savefig, '_strokecheck', False):
        return
    original = Figure.savefig

    # wraps() carries over __qualname__; without it, importing pyplot after
    # enable() dies in its docstring machinery ("Wrapped method from
    # unexpected class"), so import order would silently matter.
    @functools.wraps(original)
    def savefig(self, fname, *args, **kwargs):
        result = original(self, fname, *args, **kwargs)
        try:
            path = str(fname)
            if path.lower().endswith('.pdf'):
                limit = THRESHOLD if threshold is None else threshold
                _report(path, count_strokes(path), limit, quiet)
        except Exception as exc:                      # never break a plot run
            print(f"  ?? strokecheck skipped {fname}: {exc}", file=sys.stderr)
        return result

    savefig._strokecheck = True
    Figure.savefig = savefig


IGNORE_FILE = '.strokecheckignore'


def load_ignore(directory):
    """Glob patterns from `.strokecheckignore` in `directory`, if present.

    One pattern per line, '#' comments. Saves retyping the same few exclusions
    on every run, which complex projects always seem to accumulate.
    """
    path = os.path.join(directory, IGNORE_FILE)
    patterns = []
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                line = line.split('#', 1)[0].strip()
                if line:
                    patterns.append(line)
    return patterns


def is_excluded(path, patterns):
    """Match a glob against the basename or the path as given."""
    base = os.path.basename(path)
    return any(fnmatch.fnmatch(base, p) or fnmatch.fnmatch(path, p)
               for p in patterns)


def collect(args, excludes=()):
    """Expand CLI arguments to (kept, excluded, missing) sorted lists.

    An argument is a folder to sweep if the filesystem says it is a directory,
    and a specific PDF otherwise; no arguments at all means the current
    directory. Everything found is included by default -- listing dozens of
    figures by hand is what this exists to avoid -- so exclusion is the only
    thing ever spelled out.

    `.strokecheckignore` is read from each directory swept. It does not apply
    to files named explicitly on the command line: naming a file is taken to
    mean you want that one looked at.
    """
    paths, named, patterns, missing = [], [], list(excludes), []
    for a in (args or ['.']):
        if os.path.isdir(a):
            paths += glob.glob(os.path.join(a, '*.pdf'))
            paths += glob.glob(os.path.join(a, '*.PDF'))
            patterns += load_ignore(a)
        elif os.path.isfile(a):
            named.append(a)
        else:
            missing.append(a)
    swept = sorted(set(paths))
    kept = [p for p in swept if not is_excluded(p, patterns)]
    excluded = [p for p in swept if is_excluded(p, patterns)]
    # Explicitly named files still honour -x, but not .strokecheckignore.
    for p in sorted(set(named)):
        (excluded if is_excluded(p, excludes) else kept).append(p)
    return kept, excluded, sorted(set(missing))


if __name__ == '__main__':
    argv, args, excludes = sys.argv[1:], [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-h', '--help'):
            sys.exit(__doc__)
        elif a in ('-x', '--exclude'):
            i += 1
            if i >= len(argv):
                sys.exit("--exclude needs a pattern")
            excludes.append(argv[i])
        elif a.startswith('--exclude='):
            excludes.append(a.split('=', 1)[1])
        else:
            args.append(a)
        i += 1

    files, skipped, missing = collect(args, excludes)
    for m in missing:
        print(f"  ?? {m}: no such file or directory", file=sys.stderr)
    if not files:
        sys.exit("no PDFs found (looked in: %s)" % ', '.join(args or ['.']))

    # A multi-page PDF is a document, not a figure. The per-figure limit is
    # meaningless against a document total -- what matters there is the worst
    # single page, which is check_strokes.py's job.
    figures, documents, unreadable = [], [], []
    for p in files:
        pages = count_pages(p)
        if pages == 0:
            unreadable.append(p)      # not a PDF, or truncated
        elif pages > 1:
            documents.append(p)
        else:
            figures.append(p)

    over = 0
    for p in figures:
        n = count_strokes(p)
        if n > THRESHOLD:
            over += 1
        _report(p, n, THRESHOLD)

    print(f"\n  {len(figures)} figure(s) checked, {over} over {THRESHOLD} "
          f"strokes")
    # Name anything left out: a silently skipped figure is how a bad one slips
    # through unnoticed.
    if skipped:
        print(f"  {len(skipped)} excluded: "
              + ', '.join(os.path.basename(p) for p in skipped))
    if unreadable:
        print(f"  {len(unreadable)} unreadable, no pages found: "
              + ', '.join(os.path.basename(p) for p in unreadable))
    if documents:
        print(f"  {len(documents)} skipped as multi-page documents: "
              + ', '.join(f"{os.path.basename(p)} ({count_pages(p)} pp)"
                          for p in documents)
              + "\n     -- check these with check_strokes.py, which gates per "
                "page")
    sys.exit(1 if (over or missing or unreadable) else 0)
