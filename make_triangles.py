r"""
make_triangles.py -- the appendix triangle plots, one per analysis.

READS ONLY.  It never samples.

    python make_triangles.py                        # all five
    python make_triangles.py --figure cosmo_A lcdm

EVERY figure here carries DES and DES+legacy together, and the set covers every table
in the paper:

    cosmo_A       Table 1, top     cosmography, configuration A (Planck r_d)
    cosmo_B       Table 1, bottom  cosmography, configuration B (CC-derived r_d)
    lcdm_A        Table 1          flat LCDM, configuration A
    lcdm_B        Table 1          flat LCDM, configuration B
    legacy        Table 3          the legacy zero-point, Delta fixed vs sampled
    evidence_exp  Table 2          the exp late-time model, no anchor

    lcdm          Table 1          flat LCDM, BOTH configurations in one 4-contour panel.
                                   SUPERSEDED by lcdm_A / lcdm_B, which the manuscript
                                   tiles with two \includegraphics commands; kept only so
                                   an un-updated \includegraphics still builds.

TITLES are OFF.  Each figure names itself in its file name and is described in its
manuscript caption, so a suptitle would only repeat the caption inside the float.  Pass
--titles to draw them when browsing these outside the paper.

That is five, not four: the fifth is the one where `z_1` runs to the edge of its prior,
which is far easier to see in a triangle than in a
`mean +- std`.  Drop it from FIGURES if four is wanted.

WEIGHTS.  The evidence cells come from dynesty and are NOT equally weighted -- their early
samples carry weights of order 1e-300.  Those chains are loaded with their stored
`weights` array; the NUTS chains are loaded with unit weights, which is correct for them.
Plotting a nested-sampling chain unweighted would draw the prior, not the posterior.

COLOURS are imported from make_figures.SERIES, so the triangles use the same black/red
DES / DES+legacy pair as the rest of the paper.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Stroke-count guard (strokecheck.py).  A page carrying
# thousands of SEPARATE stroked paths can defeat a printer's RIP even though it
# renders instantly and the file is small, and nothing in normal use surfaces
# that.  This patches Figure.savefig, so every savefig call below is left exactly
# as written and each PDF reports its stroke count against the limit as it is
# saved.  Warnings go to stderr.
import strokecheck
strokecheck.enable()

from getdist import MCSamples, plots

import dovekie_cosmography as dc
import make_figures as mf

DPI = 200
P = dc.BASELINE['P']

# LaTeX names getdist will typeset.  getdist wants them WITHOUT the surrounding $.
#
# UNITS.  They are given on the dimensional parameters -- r_d in cMpc, and H0
# and the excess amplitude A in km/s/Mpc (its prior is U(-10, +10) km/s/Mpc)
# -- so that a reader meeting the appendix first can judge the scale without going back
# to the tables.  Delta M_legacy is deliberately NOT here: mag is carried by the label
# itself everywhere else in the paper.  They were once skipped on the evidence triangles, which were
# then six parameters wide with a three-entry key; those panels are five wide since
# H0_LCDM came out of them, so they carry the units too.  UNIT holds the suffixes and each figure opts in with `units=True`; LABEL
# itself stays bare, so a figure that does not opt in is untouched.
UNIT = {
    # STACKED, not inline.  `H_0 [km s^-1 Mpc^-1]` set inline is about 1.9 in of type at
    # the size these axes use, against a 1.29 in panel, so it runs straight into the
    # neighbouring parameter's label.  \genfrac with empty delimiters and a zero-thickness
    # rule is a two-line label in mathtext (\atop is not implemented in matplotlib's
    # parser), which fits in the panel and keeps the paper's own unit notation rather than
    # compressing it to km/s/Mpc.  r_d's unit is short enough to sit inline.
    'H0': ('stack', r'\mathrm{km\,s^{-1}\,Mpc^{-1}}'),
    'H0_LCDM': ('stack', r'\mathrm{km\,s^{-1}\,Mpc^{-1}}'),
    'H_at_0': ('stack', r'\mathrm{km\,s^{-1}\,Mpc^{-1}}'),
    # A shares H0's unit and H0's panel width, so it is stacked for the same reason.
    'A': ('stack', r'\mathrm{km\,s^{-1}\,Mpc^{-1}}'),
    'rd': ('inline', r'\mathrm{cMpc}'),
    # H_1..H_k (see _nuts) are deliberately ABSENT: four more stacked unit labels in an
    # eight-wide triangle overlap one another (2026-10-12).  They share H_0's unit, which
    # the H_0 axis carries and the caption states.
}
LABEL = {
    # M_DES as the manuscript's \Mdes writes it, M_{\mathrm{DES}} -- NOT \mathcal{M}, which
    # the article never uses for it.
    'H0': r'H_0', 'q0': r'q_0', 'M': r'M_{\mathrm{DES}}', 'rd': r'r_d',
    # H_at_0 is the model's PRESENT expansion rate, H0_LCDM + A, and it is the quantity
    # Table 3's H0 column quotes, so it is labelled H_0.  The background normalisation keeps its
    # own subscripted label, so the two are still distinct on the same panel.
    'H0_LCDM': r'H_{0,\Lambda{\rm CDM}}', 'H_at_0': r'H_0',
    'Om': r'\Omega_{\rm m}', 'Delta': r'\Delta M_\mathrm{legacy}', 'A': r'A', 'z1': r'z_1',
    'w0': r'w_0', 'wa': r'w_a',
    # the DIMENSIONLESS shape coefficients of E(y) = H/H0 (since 2026-09-28)
    'C1': r'C_1', 'C2': r'C_2', 'C3': r'C_3', 'C4': r'C_4', 'C5': r'C_5', 'C6': r'C_6',
    # the expansion rate at k_max equally spaced points, H_k = H(k y_max / k_max)
    **{'H%d' % _k: r'H_%d' % _k for _k in range(1, 7)},
}


# =============================================================================
# Loading
# =============================================================================
def _nuts(group, name, params):
    """A NUTS chain as an MCSamples, unit weights.

    `C1`..`CP` are unpacked from the (N, P) `C` block: the dimensionless shape
    coefficients C_1..C_P of E(y) = H/H0.  H0 is stored under its own key.

    `H1`..`HP` are DERIVED: H_k = H(y_k) = H0 E(y_k) at y_k = k y_max / P, the expansion
    rate at P equally spaced points out to the envelope's ceiling (y_max = 0.70, z = 2.33).
    Why: the C_k of a polynomial are hard to read, and
    together with H_0 the H_k carry the same information, one-to-one.  At fixed H0 the map
    C -> (H_1..H_P) is linear, so the shape prior stays uniform in them -- see
    check_Hk_prior.py for what that leaves non-flat."""
    c = dc.cache_load(group, name)
    if c is None:
        raise SystemExit('missing chain %s/%s -- run the corresponding --task'
                         % (group, name))
    dc.require_quality('chains/%s/%s.npz' % (group, name), dc._unpack(c))
    cols = []
    for p in params:
        if p.startswith('C') and p[1:].isdigit():
            k = int(p[1:])
            if 'C' not in c:
                raise SystemExit('%s/%s has no C block' % (group, name))
            cols.append(np.asarray(c['C'])[:, k - 1])
        elif p.startswith('H') and p[1:].isdigit() and int(p[1:]) >= 1:
            if 'C' not in c:
                raise SystemExit('%s/%s has no C block' % (group, name))
            C = np.asarray(c['C'])
            K = C.shape[1]
            yk = dc.shape_y_max() * int(p[1:]) / K
            cols.append(np.asarray(c['H0']) * (1.0 + C @ yk ** np.arange(1, K + 1)))
        else:
            cols.append(np.asarray(c[p]))
    return np.column_stack(cols), None


def _dynesty(name, params):
    """An evidence cell, WITH its dynesty weights.

    `H_at_0` is DERIVED, not stored: in the exp model the sampled normalisation is
    H0_LCDM and the present expansion rate is H(0) = H0_LCDM + A, which is what Table 2
    quotes.  Ask for whichever you mean -- there is no longer a plain `H0` here that
    silently resolves to one of them.  Same construction as print_cls_sigma.read_chain."""
    import dovekie_evidence as de
    # same manifest check the evidence module itself applies -- the nested-sampling
    # controls and the prior bounds live in dovekie_evidence, not in BASELINE, so
    # cache_load cannot see them unless they are handed to it
    _smp, _anc, _mod = name.split('__')
    if EVIDENCE_TASK == 'evidence_cmb':
        return _dynesty_cmb(name, params, _anc, _mod)
    c = dc.cache_load('evidence', name, extra=de.cache_extra(_anc, _mod))
    if c is None:
        raise SystemExit('missing or stale evidence chain %s.\n'
                         'Run: python dovekie_evidence.py --task grid' % name)
    import json
    summ = json.loads(str(c['summary_json']))
    order = summ['params']
    s, w = np.asarray(c['samples']), np.asarray(c['weights'])
    cols = []
    for pr in params:
        if pr == 'H_at_0':
            if summ.get('model') != 'exp':
                raise SystemExit('H_at_0 is defined for the exp model only, not %r'
                                 % summ.get('model'))
            cols.append(s[:, order.index('H0_LCDM')] + s[:, order.index('A')])
        else:
            cols.append(s[:, order.index(pr)])
    return np.column_stack(cols), w


# WHICH evidence chains the Table 3 triangles read.  Since 2026-10-08 Table 3 quotes the
# comparison WITH the CMB (dovekie_evidence_cmb.py, chains/evidence_cmb/), so that is the
# default; EVIDENCE_TASK=evidence redraws the figures from the Planck-r_d chains.
EVIDENCE_TASK = os.environ.get('EVIDENCE_TASK', 'evidence_cmb')
if EVIDENCE_TASK not in ('evidence', 'evidence_cmb'):
    raise SystemExit('EVIDENCE_TASK must be evidence or evidence_cmb, not %r' % EVIDENCE_TASK)


def _dynesty_cmb(name, params, anchor, model):
    """The same columns as `_dynesty`, from a WITH-CMB cell.  The layout differs: the
    sampled block is (omega_b, Omega_bc, H0_LCDM, A, z1), and Om (= Omega_bc + Omega_nu)
    and r_d (derived from omega_b, omega_bc) live in the `derived` array."""
    import dovekie_evidence_cmb as ec
    c = dc.cache_load(ec.TASK, name, extra=ec.cache_extra(anchor, model))
    if c is None:
        raise SystemExit('missing or stale CMB evidence chain %s.\n'
                         'Run: python dovekie_evidence_cmb.py --task grid' % name)
    import json
    summ = json.loads(str(c['summary_json']))
    order, dnames = summ['params'], summ['derived']
    s, der, w = (np.asarray(c['samples']), np.asarray(c['derived']),
                 np.asarray(c['weights']))
    cols = []
    for pr in params:
        if pr == 'H_at_0':
            cols.append(s[:, order.index('H0_LCDM')] + s[:, order.index('A')])
        elif pr in dnames:
            cols.append(der[:, dnames.index(pr)])
        else:
            cols.append(s[:, order.index(pr)])
    return np.column_stack(cols), w


# Hard prior edges, so getdist does not smooth a density out past a boundary the sampler
# could never cross.  This matters for real: z1 piles up against BOTH ends of its box, and
# without the range the KDE draws it running to z1 < 0, which is not a posterior feature.
# Read from the code that imposes the priors -- see make_tables.table5.
def _ranges(params):
    import dovekie_evidence as de
    B = dc.BASELINE
    full = {
        # the LADDER box.  The evidence module samples H0 from its OWN, wider box, so a
        # dynesty panel drawn against this one would be clipped by 10 km/s/Mpc at each
        # end -- H0_LCDM below carries the evidence range instead.
        'H0': (B['H0_LO'], B['H0_HI']),
        'H0_LCDM': (de.H0_LO, de.H0_HI),
        # H(0) = H0_LCDM + A is a SUM of two independent uniform draws, so it has no hard
        # edge of its own; getdist is left to find the support.
        'Om': (B['OM_LO'], B['OM_HI']),
        'M': (B['M_LO'], B['M_HI']),
        'Delta': (-B['DELTA_HALFWIDTH'], B['DELTA_HALFWIDTH']),
        'A': (-de.A_MAX, de.A_MAX),
        'z1': (de.Z1_MIN, de.Z1_MAX),
        'w0': (de.W0_LO, de.W0_HI),
        'wa': (de.WA_LO, de.WA_HI),
    }
    # The C_k have no hard edge of their own: their prior is uniform over a tilted
    # parallelepiped, so a marginal's support is far outside the posterior and giving it
    # to getdist would only stretch the KDE.
    return {p: full[p] for p in params if p in full}


def _label(p, units, bare=()):
    """The axis name, with its unit attached only where the figure asked for units, and
    never on a parameter the figure lists as `bare`."""
    u = UNIT.get(p) if units and p not in bare else None
    if u is None:
        return LABEL[p]
    how, txt = u
    if how == 'stack':
        return r'\genfrac{}{}{0}{}{%s}{[%s]}' % (LABEL[p], txt)
    return r'%s\,[%s]' % (LABEL[p], txt)


def _mcsamples(loader, args, params, label, ranges=True, units=False, bare=()):
    x, w = loader(*args, params)
    return MCSamples(samples=x, weights=w, names=list(params),
                     labels=[_label(p, units, bare) for p in params], label=label,
                     ranges=_ranges(params) if ranges else None,
                     settings=dict(smooth_scale_2D=0.4, smooth_scale_1D=0.4))


def _two_ticks(g, margin=0.16, min_scale=0.80):
    """Every axis carries at least two labelled ticks -- where two will legibly fit.

    getdist asks matplotlib for at most `axis_tick_max_labels`, and matplotlib's locator
    is free to return ONE when the range is narrow.  On the bottom row of the cosmography
    triangle that happens three times over -- a single tick at 147 for r_d, one at -19.4
    for M_DES, one at -3000 for c_3 -- and one tick gives the reader no sense of scale,
    which is what this fixes.

    Three stages, applied only to axes that show fewer than two labels:

    1.  Ticks are chosen from a 1/2/2.5/5 x 10^n ladder inside the MIDDLE of the range,
        `margin` of the span clear of each end.  The panels of a triangle share their
        edges, so a tick against a limit collides with the NEXT parameter's first label,
        not with anything in its own panel.  The largest step that still yields two ticks
        wins, so an axis gets two labels and not five.
    2.  If the two labels then overlap each other, or reach outside their own panel, that
        axis alone is set smaller -- but no smaller than `min_scale` of the size PAGE_PT
        asked for.
    3.  If they still do not fit, the axis is PUT BACK exactly as getdist had it.  Two
        labels printed on top of each other are worse than one, and this is
        worth trying and fine to fail: "it is fine if not as the
        range is clear from the y-axis" -- c_3 and its neighbours are 5-character labels
        in a 1.3 in panel, and they also appear as the y axis of the left column, where
        there is room for three.

    Axes that already showed two or more labels are never touched.
    """
    fig = g.fig
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def labels(axis):
        return [t for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]

    def misfit(ax, axis):
        """Pixels by which this axis's labels overlap each other or leave the panel."""
        bb = [t.get_window_extent(renderer=rend) for t in labels(axis)]
        if len(bb) < 2:
            return 0.0, 1.0
        if axis is ax.xaxis:
            lo_, hi_ = [b.x0 for b in bb], [b.x1 for b in bb]
            edge, room = (ax.bbox.x0, ax.bbox.x1), ax.bbox.width
        else:
            lo_, hi_ = [b.y0 for b in bb], [b.y1 for b in bb]
            edge, room = (ax.bbox.y0, ax.bbox.y1), ax.bbox.height
        spill = max(edge[0] - min(lo_), max(hi_) - edge[1], 0.0)
        touch = max([hi_[i] - lo_[i + 1] for i in range(len(bb) - 1)] + [0.0])
        return max(spill, touch), room

    for ax in fig.get_axes():
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            was = [t for t in axis.get_majorticklocs() if lo - 1e-12 <= t <= hi + 1e-12]
            if len(was) >= 2:
                continue
            a, b = lo + margin * (hi - lo), hi - margin * (hi - lo)
            want = None
            for e in range(int(np.floor(np.log10(max(hi - lo, 1e-30)))) + 1, -12, -1):
                for m in (5.0, 2.5, 2.0, 1.0):
                    step = m * 10.0 ** e
                    t = np.arange(np.ceil(a / step), np.floor(b / step) + 1) * step
                    if len(t) >= 2:
                        want = t
                        break
                if want is not None:
                    break
            if want is None:
                continue
            axis.set_ticks(want)
            fig.canvas.draw()

            # A shared or hidden axis draws no labels at all -- the interior panels of a
            # triangle only show ticks on the bottom row and the left column.  Their tick
            # POSITIONS still matter (the marks line up across a column), but there is
            # nothing to measure or shrink, so the fitting stages are skipped.
            if not labels(axis):
                continue
            size0 = labels(axis)[0].get_fontsize()
            over, room = misfit(ax, axis)
            while over > 0.01 * room:
                size = labels(axis)[0].get_fontsize() * 0.94
                if size < min_scale * size0:
                    break
                for t in labels(axis):
                    t.set_fontsize(size)
                fig.canvas.draw()
                over, room = misfit(ax, axis)
            if over > 0.01 * room:
                # stage 3: it does not fit.  Put the axis back the way getdist had it.
                for t in labels(axis):
                    t.set_fontsize(size0)
                axis.set_ticks(was)
                fig.canvas.draw()



def _single_x_tick(g, params, names):
    """Leave ONE labelled tick on the bottom-row x axis of each named parameter.

    For the cosmography triangles: M_DES's x axis was
    drawing -19.425, -19.400 and -19.375 into a 1.0 in panel, where they print as one
    unreadable run of digits.  getdist chose three because `axis_tick_max_labels` is a
    cap, not a fit test, and `_two_ticks` never looked at the axis -- it only ever ADDS a
    second tick to an axis showing fewer than two.

    The tick kept is the one nearest the middle of the range, so the mark sits inside the
    contour rather than out at a limit.  The reader takes the SCALE off the same
    parameter's y axis in the left column, which has room for three -- the arrangement
    r_d already had, and which is accepted for this parameter: "this
    would end up similar to r_d, where readers have to get the range from the y-axis but
    not the x-axis".

    `names` maps a parameter to the requested tick value, or to None to let one
    be chosen.  A requested value is used only if it falls inside the middle of the axis --
    a tick against a limit collides with the next parameter's first label, and one outside
    the range would leave the panel with NO tick at all, which is the failure this pass
    exists to avoid.  Otherwise the value comes off a 1/2/2.5/5 x 10^n ladder, coarsest
    step first, nearest the centre of the range: -19.40 for M_DES, and the same rule would
    have put c_3 at -2500 had -3000 not been requested.

    Run AFTER `_two_ticks`, which would otherwise put a second tick straight back.
    """
    for p, want in dict(names).items():
        if p not in params:
            continue
        j = list(params).index(p)
        ax = g.subplots[-1][j]
        lo, hi = sorted(ax.get_xlim())
        have = [t for t in ax.xaxis.get_majorticklocs() if lo <= t <= hi]
        if len(have) < 2:
            continue
        mid = 0.5 * (lo + hi)
        a, b = lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)
        if want is not None and a <= want <= b:
            ax.set_xticks([want])
            continue
        pick = None
        for e in range(int(np.floor(np.log10(max(hi - lo, 1e-30)))) + 1, -12, -1):
            for m in (5.0, 2.5, 2.0, 1.0):
                step = m * 10.0 ** e
                cand = np.arange(np.ceil(a / step), np.floor(b / step) + 1) * step
                if len(cand):
                    pick = min(cand, key=lambda v: abs(v - mid))
                    break
            if pick is not None:
                break
        ax.set_xticks([pick if pick is not None
                       else min(have, key=lambda v: abs(v - mid))])


def _fixed_x_labels(g, params, names):
    """Put author-chosen ticks AND tick text on a bottom-row x axis.

    `names` maps a parameter to [(value, text), ...].  For the legacy
    triangle: getdist drew M_DES at -19.425, -19.400, -19.375, which run into
    one another at panel width.  Two ticks, -19.43 and -19.37, both written in full --
    there is no room for a third between them.  Run AFTER `_two_ticks` and `_single_x_tick`; `_label_overlaps`
    then checks the result like any other axis.
    """
    for p, ticks in dict(names).items():
        if p not in params:
            continue
        ax = g.subplots[-1][list(params).index(p)]
        # widen only if a requested tick would fall outside; a no-op otherwise
        lo, hi = ax.get_xlim()
        vals = [v for v, _ in ticks]
        if min(vals) < lo or max(vals) > hi:
            ax.set_xlim(min(lo, min(vals)), max(hi, max(vals)))
        ax.set_xticks([v for v, _ in ticks])
        ax.set_xticklabels([t.replace('-', '−') for _, t in ticks])


def _fixed_y_labels(g, params, names):
    """The y-axis twin of `_fixed_x_labels`: author-chosen ticks on a parameter's ROW.

    Every off-diagonal panel in the row gets the same tick positions, so the tick marks line
    up across the row, and the left-most panel carries the text.  The axis is widened if
    needed so that every requested tick is inside it.  Added for r_d on the evidence
    triangles (ticks at 147 and 148 cMpc in all three).
    """
    for p, ticks in dict(names).items():
        if p not in params:
            continue
        i = list(params).index(p)
        vals = [v for v, _ in ticks]
        for j in range(i):
            ax = g.subplots[i][j]
            if ax is None:
                continue
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, min(vals)), max(hi, max(vals)))
            ax.set_yticks(vals)
            if j == 0:
                ax.set_yticklabels([t.replace('-', '−') for _, t in ticks])


def _clear_axis_names(g, gap_em=0.3, tries=4):
    """Move each axis NAME away from its own tick labels until `gap_em` separates them.

    The stacked H_0 / A names (name over unit, then boosted 15 per cent) are taller than
    the line getdist reserves for them, so on the legacy triangle "H_0" sat ON the 70 and
    80 below the panel.  matplotlib's labelpad is in points, so the shortfall is added to
    it directly and the canvas is measured again.
    """
    fig = g.fig
    px = fig.dpi / 72.0
    for _ in range(tries):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        moved = False
        for ax in fig.get_axes():
            for axis in (ax.xaxis, ax.yaxis):
                name = axis.label
                ticks = [t for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
                if not (name.get_visible() and name.get_text() and ticks):
                    continue
                nb = name.get_window_extent(renderer=rend)
                tb = [t.get_window_extent(renderer=rend) for t in ticks]
                if axis is ax.xaxis:
                    gap = min(b.y0 for b in tb) - nb.y1
                else:
                    gap = min(b.x0 for b in tb) - nb.x1
                want = gap_em * ticks[0].get_fontsize() * px
                if gap < want:
                    axis.labelpad += (want - gap) / px + 0.5
                    moved = True
        if not moved:
            return


def _label_overlaps(fig, gap_em=0.0):
    """Every pair of visible text boxes on the canvas closer than `gap_em`, as strings.

    Tick labels of NEIGHBOURING panels are compared too: a triangle's panels share their
    edges, so the last label of one column and the first of the next are the commonest
    collision, and a per-axis test never sees it.  Tick labels, axis names and the legend
    are all included.  A pair is reported when it is closer than `gap_em` of the smaller
    of its two font sizes, horizontally and vertically; the default 0 reports text that
    actually touches.  Raise it to see near-misses as well.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = []
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_ticklabels():
                if t.get_visible() and t.get_text():
                    items.append(('tick %r' % t.get_text(), t))
            if axis.label.get_visible() and axis.label.get_text():
                items.append(('axis name %r' % axis.label.get_text(), axis.label))
    for leg in [fig.legends] + [[a.get_legend()] for a in fig.get_axes()]:
        for L in leg:
            if L is not None:
                items += [('legend %r' % t.get_text(), t) for t in L.get_texts()]
    px = fig.dpi / 72.0
    boxes = [(n, t.get_window_extent(renderer=rend), t.get_fontsize() * px)
             for n, t in items]
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (na, a, fa), (nb, b, fb) = boxes[i], boxes[j]
            g = gap_em * min(fa, fb)
            dx = max(b.x0 - a.x1, a.x0 - b.x1)
            dy = max(b.y0 - a.y1, a.y0 - b.y1)
            if dx < g and dy < g:
                bad.append('%s  within %.2f em of  %s' % (na, max(dx, dy) / min(fa, fb), nb))
    return bad


def _boost_stacked_labels(g, params, factor):
    """Set the two-line (name over unit) axis names `factor` times larger.

    The author asked for H_0 and A to be typed 15 per cent up on the evidence and legacy
    triangles (2026-09-11).  Those are exactly the parameters whose label is STACKED: the
    \\genfrac puts the name on the upper line of a two-line label, where it prints smaller
    to the eye than the single-line names beside it even though matplotlib is setting both
    at the same point size.  Keying the boost off the stack -- rather than off a list of
    parameter names -- is what makes it mean the same thing in a figure that carries H_0
    under another name (`H_at_0`) or that gains a third dimensional parameter later.

    Both copies of each name move: the bottom-row x label and the left-column y label.
    """
    if not factor or factor == 1.0:
        return
    for j, p in enumerate(params):
        if UNIT.get(p, (None,))[0] != 'stack':
            continue
        for lbl in ([g.subplots[-1][j].xaxis.label] +
                    ([g.subplots[j][0].yaxis.label] if j else [])):
            lbl.set_fontsize(lbl.get_fontsize() * factor)

# =============================================================================
# The five figures.  (key, params, [(label, colour, loader, loader-args), ...])
# =============================================================================
BLACK, RED = mf.SERIES['des_only']['c'], mf.SERIES['full']['c']
# The blue is the Delta-varied series of the one palette, not a local choice -- the h0_q0
# figure gives the joint-Delta contour the same colour, so the front figure and the
# appendix agree on what blue means.
BLUE, GREY = mf.SERIES['full_dleg']['c'], '0.45'

# SPA flat-LCDM reference lines on every appendix triangle: a faint
# dotted orange line at the SPA central value of H_0 and of Omega_m or q_0, no band, keyed
# as SPA (LCDM) like the early figures.  H_at_0 is the exp model's present expansion rate,
# labelled H_0, so it gets the SPA H_0 too; the background normalisation H0_LCDM is not
# plotted.  Same orange as every other SPA mark in the paper.
# The cosmography triangles' H_1..H_P get the SPA flat-LCDM H(z_k) as well, at the same y_k = k y_max / P that _nuts uses, with the same
# matter + Lambda H(z) as every other SPA curve in the paper (dc.H_lcdm, no radiation).
SPA_MARKERS = {'H0': mf.SPA_H0[0], 'H_at_0': mf.SPA_H0[0], 'q0': mf.SPA_Q0, 'Om': mf.SPA_OM}
for _k in range(1, P + 1):
    _y = dc.shape_y_max() * _k / P
    SPA_MARKERS['H%d' % _k] = float(dc.H_lcdm(_y / (1.0 - _y), mf.SPA_H0[0], mf.SPA_OM))
del _k, _y
SPA_COLOUR = mf.SPA_MARKER['c']
SPA_ALPHA = 0.85

FIGS = {
    'cosmo_A': dict(
        units=True,
        # M_DES's three x labels collide at panel width -- see _single_x_tick.
        single_x_tick={'M': None},
        # H_0, q_0, then H_1..H_4 in place of C_2..C_4.  q_0 is
        # derivable from the H_k but is kept for its importance; the caption says so.
        params=('H0', 'q0') + tuple('H%d' % k for k in range(1, P + 1)) + ('M', 'rd'),
        # H_0 keeps its stacked unit; H_1..H_4 carry none -- the
        # caption defines H_k and gives km/s/Mpc.  See UNIT.  Add a name to `bare` to strip
        # the unit from any other parameter.
        title=r'Cosmography, configuration A (SNIa + BAO + $\it{Planck}$ $r_d$)',
        series=[('DES', BLACK, _nuts, ('ladder', 'A_des_only_cosmo_P%d' % P)),
                ('DES+legacy', RED, _nuts, ('ladder', 'A_full_cosmo_P%d' % P))]),
    'cosmo_B': dict(
        units=True,
        # M_DES's three x labels collide at panel width -- see _single_x_tick.
        single_x_tick={'M': None},
        # H_0, q_0, then H_1..H_4 in place of C_2..C_4.  q_0 is
        # derivable from the H_k but is kept for its importance; the caption says so.
        params=('H0', 'q0') + tuple('H%d' % k for k in range(1, P + 1)) + ('M', 'rd'),
        # H_0 keeps its stacked unit; H_1..H_4 carry none -- the
        # caption defines H_k and gives km/s/Mpc.  See UNIT.  Add a name to `bare` to strip
        # the unit from any other parameter.
        title=r'Cosmography, configuration B (SNIa + BAO + CC)',
        series=[('DES', BLACK, _nuts, ('ladder', 'B_des_only_cosmo_P%d' % P)),
                ('DES+legacy', RED, _nuts, ('ladder', 'B_full_cosmo_P%d' % P))]),
    # The two configurations are SEPARATE figures, one per configuration, so they can be
    # tiled with two \includegraphics commands rather than shipped as one four-contour
    # panel.  Splitting them also frees the palette: with only one configuration per
    # figure, DES is black and DES+legacy is red in both, exactly as everywhere else in
    # the paper, instead of config B being pushed onto grey and blue to stay distinct.
    # `lcdm` below is the combined four-contour version, kept so an un-updated
    # \includegraphics still builds.
    'lcdm_A': dict(
        units=True,
        params=('Om', 'H0', 'M', 'rd'),
        title=r'Flat $\Lambda$CDM, configuration A (SNIa + BAO + $\it{Planck}$ $r_d$)',
        series=[('DES', BLACK, _nuts, ('ladder', 'A_des_only_lcdm_P%d' % P)),
                ('DES+legacy', RED, _nuts, ('ladder', 'A_full_lcdm_P%d' % P))]),
    'lcdm_B': dict(
        units=True,
        params=('Om', 'H0', 'M', 'rd'),
        title=r'Flat $\Lambda$CDM, configuration B (SNIa + BAO + CC)',
        series=[('DES', BLACK, _nuts, ('ladder', 'B_des_only_lcdm_P%d' % P)),
                ('DES+legacy', RED, _nuts, ('ladder', 'B_full_lcdm_P%d' % P))]),
    'lcdm': dict(
        units=True,
        params=('Om', 'H0', 'M', 'rd'),
        title=r'Flat $\Lambda$CDM, both configurations',
        series=[('DES (A)', BLACK, _nuts, ('ladder', 'A_des_only_lcdm_P%d' % P)),
                ('DES+legacy (A)', RED, _nuts, ('ladder', 'A_full_lcdm_P%d' % P)),
                ('DES (B)', GREY, _nuts, ('ladder', 'B_des_only_lcdm_P%d' % P)),
                ('DES+legacy (B)', BLUE, _nuts, ('ladder', 'B_full_lcdm_P%d' % P))]),
    'legacy': dict(
        units=True,
        label_boost=1.15,
        params=('H0', 'q0', 'M', 'rd', 'Delta'),
        # getdist's -19.425 / -19.400 / -19.375 overlap -- see _fixed_x_labels.
        fixed_x_labels={'M': [(-19.43, '-19.43'), (-19.37, '-19.37')]},
        # The boosted, stacked H_0 name touched the 70 and 80 above it -- see _clear_axis_names.
        clear_axis_names=True,
        title=r'The legacy zero-point $\Delta M_\mathrm{legacy}$, sampled jointly',
        # ORDER IS THE AUTHOR'S: DES, then DES + legacy, then the case with the legacy
        # offset freed -- the same order every other figure and table in the paper reads
        # in, so the key does not have to be re-sorted in the reader's head.  Delta exists
        # in the joint cell alone, so the two Delta-fixed cells carry the reduced
        # parameter list and getdist leaves them out of that row and column.
        series=[('DES', BLACK, _nuts, ('legacy', 'cosmo_des_only_fixed_P%d' % P),
                 ('H0', 'q0', 'M', 'rd')),
                (r'DES+legacy, $\Delta M_\mathrm{legacy} = 0$', RED, _nuts,
                 ('legacy', 'cosmo_full_fixed_P%d' % P), ('H0', 'q0', 'M', 'rd')),
                (r'DES+legacy, $\Delta M_\mathrm{legacy}$ sampled', BLUE, _nuts,
                 ('legacy', 'cosmo_full_joint_P%d' % P))]),
    'evidence_exp': dict(
        units=True,
        label_boost=1.15,
        # H_at_0 = H0_LCDM + A is the model's present expansion rate, the column Table 2
        # quotes, and it is the one plotted -- labelled H_0.  The background
        # normalisation H0_LCDM is NOT plotted: of H0_LCDM, A and H_at_0 only two are
        # independent, so drawing all three put a row and a column in the triangle that
        # carry no information the other two do not already fix.  It is still what `_dynesty` reads to build H_at_0, and it keeps
        # its LABEL and UNIT entries for anything else that asks for it.
        #
        # ORDER: H_0 comes SECOND, straight after Omega_m, so these panels open on the same two cosmological parameters as
        # the lcdm_A / lcdm_B triangles rather than leading with the excess amplitude.
        # A and z_1 -- the parameters of the excess itself -- follow, then r_d.
        params=('Om', 'H_at_0', 'A', 'z1', 'rd'),
        title=r'Late-time excess $\Delta H = A\,e^{-z/z_1}$, no $H_0$ anchor',
        series=[('DES', BLACK, _dynesty, ('des_only__none__exp',)),
                ('DES+legacy', RED, _dynesty, ('full__none__exp',)),
                (r'DES+legacy, $\Delta M_\mathrm{legacy}$ free', BLUE, _dynesty,
                 ('full_dleg__none__exp',))],
        # Three series, one of them a long label: at the common size the key runs off the
        # canvas and the row labels climb into the row above.  A point off the labels and
        # three off the key, page size.  Kept as it was when this panel was six
        # parameters wide, so the anchored twins below match it exactly.
        pt=dict(label=5.5, tick=4.25, legend=4.0),
        # r_d ticks at 147 and 148 cMpc on both of its axes, in all three evidence
        # triangles, so the panels read alike.
        fixed_x_labels={'rd': [(147.0, '147'), (148.0, '148')]},
        fixed_y_labels={'rd': [(147.0, '147'), (148.0, '148')]}),
}

# The same figure as `evidence_exp`, once per H0 anchor.  The anchor-free version leaves A
# consistent with zero and z1 at its prior, which is the least interesting of the three
# cases; adding an anchor is what makes the late-time solution testable, so these two are a
# separate appendix figure, drawn as TWO files and stacked vertically in the float (panels
# are combined in LaTeX, never before).  Identical parameters, series,
# colours and type sizes as `evidence_exp`, so the three figures can be read against each
# other -- only the anchor changes.
for _key, _anchor, _title in (
        ('evidence_exp_lfree3', 'lfree3', r'three $H_0$ anchors that do not use SNIa'),
        ('evidence_exp_breuval', 'breuval', r'the SH0ES $H_0$ anchor')):
    FIGS[_key] = dict(
        units=FIGS['evidence_exp']['units'],
        label_boost=FIGS['evidence_exp']['label_boost'],
        params=FIGS['evidence_exp']['params'],
        title=r'Late-time excess $\Delta H = A\,e^{-z/z_1}$, ' + _title,
        series=[(lab, col, ldr, (args[0].replace('__none__', '__%s__' % _anchor),))
                for lab, col, ldr, args in FIGS['evidence_exp']['series']],
        pt=dict(FIGS['evidence_exp']['pt']),
        fixed_x_labels=FIGS['evidence_exp']['fixed_x_labels'],
        fixed_y_labels=FIGS['evidence_exp']['fixed_y_labels'])
del _key, _anchor, _title


# What the type should measure ON THE PAGE, in points.  See build().
# HALVED on 2026-08-27: at 12 pt page the keys and the tick labels
# were covering the contours.  These are still PAGE points -- build() multiplies them by
# each figure's own reduction factor.
PAGE_PT = dict(label=6.0, tick=4.5, legend=5.5, title=5.0)
DISPLAY_IN = 3.35          # \columnwidth in MNRAS, and 0.49\linewidth of a figure* too


def _add_spa_key(g, lw):
    """Append `SPA (LCDM)` to getdist's figure legend.

    getdist builds the key from the contours alone, so it is rebuilt here with the same
    anchor triangle_plot gave it (the top-right corner of the grid) plus one dotted line.
    The handle is drawn opaque and a little heavier than the panel lines, so the dots
    stay legible at key size; the lines themselves stay faint."""
    from matplotlib.lines import Line2D
    L = g.legend
    hs = list(L.legend_handles)
    labs = [t.get_text() for t in L.get_texts()]
    size = L.get_texts()[0].get_fontsize()
    L.remove()
    hs.append(Line2D([], [], color=SPA_COLOUR, ls=':', lw=1.5 * lw))
    labs.append(mf.SPA_LABEL_LCDM)
    g.legend = g.fig.legend(hs, labs, loc='upper right', prop={'size': size},
                            bbox_to_anchor=(g.plot_col, 1),
                            bbox_transform=g.subplots[0, 0].transAxes, borderaxespad=0,
                            frameon=g.settings.figure_legend_frame)
    if not g.settings.legend_rect_border:
        for rect in g.legend.get_patches():
            rect.set_edgecolor(rect.get_facecolor())


def build(key, titles=False):
    C = FIGS[key]
    sets, colours = [], []
    units = C.get('units', False)
    # A series may carry its OWN parameter list.  `legacy` needs that: Delta exists in the
    # jointly-sampled cell alone, so the two Delta-fixed cells go in on the reduced set and
    # getdist simply leaves them out of that row and column.  Keeping one ordered list
    # rather than a `series` block followed by an `extra` block is what lets the LEGEND
    # order be chosen freely -- DES, then DES+legacy, then the
    # zero-point-sampled case, which is not the order the ragged sets have to be built in.
    for entry in C['series']:
        lab, col, loader, args = entry[:4]
        prm = entry[4] if len(entry) > 4 else C['params']
        sets.append(_mcsamples(loader, args, prm, lab, units=units,
                               bare=C.get('bare', ())))
        colours.append(col)

    # TYPE SIZED FOR THE PAGE, NOT FOR THE CANVAS.  Every one of these triangles is
    # reduced to about \columnwidth in the manuscript -- 1.0\linewidth in a one-column
    # float, or 0.49\linewidth in a two-column one, both ~3.35 in -- from a canvas of 7.3
    # to 8.9 in.  That is a factor 2.2 to 2.7, so the 12 pt axis labels this used to draw
    # arrived at 4.5 to 5.5 pt.  Sizes are therefore given in PAGE points and multiplied by
    # the figure's own reduction factor: page_pt = python_pt x displayed_in / canvas_in.
    # getdist's own `scaling` would rescale them again by canvas size, so it is turned off.
    width = min(2.0 + 1.35 * len(C['params']), 9.0)
    k = width / DISPLAY_IN
    pt = dict(PAGE_PT, **C.get('pt', {}))
    g = plots.get_subplot_plotter(width_inch=width)
    g.settings.scaling = False
    g.settings.alpha_filled_add = 0.55
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = pt['legend'] * k
    g.settings.axes_fontsize = pt['tick'] * k
    g.settings.axes_labelsize = pt['label'] * k
    g.settings.title_limit_fontsize = pt['title'] * k
    # Lines follow the same reduction, or a 0.6 pt contour prints at 0.22 pt.
    g.settings.linewidth = 1.0 * k
    g.settings.linewidth_contour = 0.6 * k
    # Three ticks per axis: at page size the labels are 2.5x wider than they were and seven
    # of them collide.
    # Four ticks and horizontal again: the rotation existed only because the labels were
    # twice this size, and rotated ticks are harder to read for no remaining reason.
    g.settings.axis_tick_max_labels = 4
    g.settings.axis_tick_x_rotation = 0
    spa_lw = 0.75 * k
    markers = {p: SPA_MARKERS[p] for p in C['params'] if p in SPA_MARKERS}
    g.triangle_plot(sets, C['params'], filled=True, contour_colors=colours,
                    legend_loc='upper right', markers=markers,
                    marker_args=dict(color=SPA_COLOUR, ls=':', lw=spa_lw, alpha=SPA_ALPHA))
    if markers:
        _add_spa_key(g, spa_lw)
    # An axis showing one tick shows no scale.  See _two_ticks.
    _two_ticks(g)
    # ... except where two cannot be read at all: _single_x_tick runs after it and wins.
    _single_x_tick(g, C['params'], C.get('single_x_tick') or {})
    _fixed_x_labels(g, C['params'], C.get('fixed_x_labels') or {})
    _fixed_y_labels(g, C['params'], C.get('fixed_y_labels') or {})
    _boost_stacked_labels(g, C['params'], C.get('label_boost'))
    # Text that touches other text is a defect that should be caught here, not by a
    # reader.  Checked on the finished canvas, after every tick pass above.
    # Opt-in: only where a name actually sat on its ticks.  The author judged the other
    # triangles fine as drawn (2026-10-06), so their names stay where getdist put them.
    if C.get('clear_axis_names'):
        _clear_axis_names(g)
    clash = _label_overlaps(g.fig)
    for c in clash:
        print('[%s] LABEL OVERLAP: %s' % (key, c))
    # NO title by default.  The file name says which analysis this is (triangle_lcdm_A,
    # triangle_cosmo_B, ...), and the manuscript caption says it again in a sentence, so
    # a suptitle only repeats the caption in a smaller font inside the float.  `C['title']`
    # is kept as the one-line description of the figure and is still drawn with --titles,
    # which is useful when these are flicked through outside the paper.
    if titles:
        g.fig.suptitle(C['title'], fontsize=pt['label'] * k, y=1.005)

    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    p = os.path.join(dc.RESULT_DIR, 'triangle_%s' % key)
    g.fig.savefig(p + '.pdf', dpi=DPI, bbox_inches='tight')
    g.fig.savefig(p + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(g.fig)
    print('wrote %s.pdf and .png' % p)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    ap.add_argument('--figure', choices=sorted(FIGS), nargs='*')
    ap.add_argument('--titles', action='store_true',
                    help='draw the suptitle (off by default: the caption carries it)')
    a = ap.parse_args(argv)
    print(dc.banner())
    for k in (a.figure or sorted(FIGS)):
        try:
            build(k, titles=a.titles)
        except SystemExit as e:
            print('[%s] %s' % (k, e))


if __name__ == '__main__':
    main()
