r"""make_mock_traces_figure.py -- the floor-sweep mock ensemble, as traces.

Main figure:  (H0, q0) x (DES+legacy, DES-only).  Companion figure: the joint-Delta arm
(H0, q0, Delta_legacy), one column.

Each main panel carries every realisation's best-fit curve (thin, grey), unfilled
15.87-84.13 and 2.275-97.725 percentile outlines so the observed curve's local percentile
can be read straight off, the ensemble median, the NOISELESS null (the pure truncation
bias -- what this estimator returns when the data ARE the fiducial), the fiducial itself,
and the observed catalogue through the identical estimator.

Redundant cuts are drawn ONCE.  Where several z_floor values select an identical supernova
sample (empty redshift interval between them) only the lowest is plotted, and the observed
curve carries a thin horizontal whisker spanning the redundant range: one fit, invariant
across those cuts, rather than a flat segment that looks like a stable measurement.

Companion strip below each panel:  at each distinct z_floor, a solid bar of half-length
median(sigma) over the ensemble, a light rectangle spanning its 15.87-84.13 percentile
range, and a dashed bar, x-dodged, of half-length 0.5 x (P84.13 - P15.87) of the best fits.
Solid ~ dashed is the statement that the quoted curvature errors are honest.  The bars are
centred on zero: only their LENGTH and their z_floor carry meaning.

DES+legacy stops at z_floor = 0.085: no legacy SN survives a higher cut, so above z = 0.087
the arm IS des_only, to every digit, in the mocks and in the data alike.
"""
import glob
import json
import os
import re

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
from matplotlib.gridspec import GridSpec
# (HandlerTuple is no longer needed: the strip key is four single-colour entries, not
# two paired ones -- see combined().)
from matplotlib.ticker import MaxNLocator
from mock_paths import rglob, rpath

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results_mock')

SPA_H0 = 67.19
SPA_OM = 0.3175
SPA_Q0 = 1.5 * SPA_OM - 1.0            # -0.52625

S1 = (15.87, 84.13)                    # 1 sigma, Gaussian equivalent
S2 = (2.275, 97.725)                   # 2 sigma, Gaussian equivalent

# HOW MANY INDIVIDUAL TRACES TO DRAW.  The envelopes, the median and every number come
# from the WHOLE ensemble; this caps only the grey spaghetti behind them.  At 10,000
# realisations the traces saturate to a solid block that hides the envelopes it exists to
# motivate, and the pdf grows to hundreds of MB.  The subsample is a fixed STRIDE and not a
# random draw, so the figure is reproducible from the file list alone.
N_TRACE = 300

C_TRACE = '0.55'
C_ENV = '0.25'
C_MED = '#1f77b4'
C_NULL = '#2e7d32'
C_ARM = {'full': '#c62828', 'des_only': 'black'}
C_TRUTH = '#e8a33d'

XL_FULL = (0.027, 0.0885)
XL_DES = (0.068, 0.312)

# The strip below each panel plots +/- sigma of the SAME quantity on the SAME scale as
# the panel above it, so it carries the same units.  It had none, which was
# fixed on the H0 figure; q0 is dimensionless and so keeps a bare label.
STRIP_UNIT = {'H0': r'  [km s$^{-1}$ Mpc$^{-1}$]', 'q0': '', 'Delta': r'  [mag]'}

P_H0 = ('H0', 'sig_H0', r'$H_0$  [km s$^{-1}$ Mpc$^{-1}$]', SPA_H0, 5.0)
P_Q0 = ('q0', 'sig_q0', r'$q_0$', SPA_Q0, 0.5)
P_DL = ('Delta', 'sig_Delta', r'$\Delta M_{\rm legacy}$  [mag]', 0.0, None)


def load(path):
    return {(r['sample'], bool(r['joint_delta']), round(r['z_floor'], 4)): r
            for r in json.load(open(path))}


def floor_groups(row0, samp, joint, xlim):
    """Distinct cuts within xlim, each as (lowest floor, highest redundant floor)."""
    fl = [f for f in sorted(k[2] for k in row0 if k[0] == samp and bool(k[1]) == joint)
          if xlim[0] <= f <= xlim[1]]
    groups, seen = [], None
    for f in fl:
        n = row0[(samp, joint, f)]['n_sn']
        if n != seen:
            groups.append([f, f])
            seen = n
        else:
            groups[-1][1] = f
    return groups


def panel(fig, gs, r, c, E, NL, RD, samp, joint, par, xlim, arm, show_leg,
          hide_x, shade_box, leg_ncol=4):
    key, sig, ylab, truth, tick = par
    ax = fig.add_subplot(gs[2 * r, c])
    bx = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)

    groups = floor_groups(E[0], samp, joint, xlim)
    fl = np.array([g[0] for g in groups])
    V = np.array([[e[(samp, joint, f)][key] for f in fl] for e in E])
    S = np.array([[e[(samp, joint, f)][sig] for f in fl] for e in E])
    med = np.median(V, axis=0)
    nul = np.array([NL[(samp, joint, f)][key] for f in fl])
    dat = np.array([RD[(samp, joint, f)][key] for f in fl])
    lo1, hi1 = np.percentile(V, S1, axis=0)
    lo2, hi2 = np.percentile(V, S2, axis=0)

    step = max(1, len(V) // N_TRACE)
    for row in V[::step]:
        ax.plot(fl, row, color=C_TRACE, lw=0.4, alpha=0.09, zorder=1)
    for y in (lo2, hi2):
        ax.plot(fl, y, color=C_ENV, ls=':', lw=1.1, zorder=4)
    for y in (lo1, hi1):
        ax.plot(fl, y, color=C_ENV, ls='-', lw=1.0, zorder=4)
    ax.axhline(truth, color=C_TRUTH, ls=':', lw=1.8, zorder=3)
    ax.plot(fl, nul, color=C_NULL, ls='--', lw=2.2, zorder=5,
            label='null, no noise (truncation bias)')
    ax.plot(fl, med, color=C_MED, ls='-', lw=2.4, zorder=6,
            label=('ensemble median (%d mocks%s)'
                   % (len(E), '; %d traces drawn' % len(V[::step])
                      if step > 1 else '')))
    ax.plot(fl, dat, color=C_ARM[samp], ls='-', lw=2.2, marker='o', ms=3.6,
            zorder=7, label='observed catalogue')

    # whiskers on the observed curve are off here too, for the same reason as in _arm:
    # they are left out of these plots.
    lo = min(lo2.min(), dat.min(), truth, nul.min())
    hi = max(hi2.max(), dat.max(), truth, nul.max())
    pad = 0.05 * (hi - lo)
    ax.set_ylim(lo - 1.6 * pad, hi + 3.0 * pad)
    ax.set_ylabel(ylab, fontsize=11.5)
    ax.axvline(0.0733, color='0.55', ls='--', lw=0.9, zorder=2)
    if shade_box:
        ax.axvspan(0.24, xlim[1], color='0.85', alpha=0.55, lw=0, zorder=0)
        ax.text(0.905, 0.02, 'coefficient box binds', rotation=90,
                transform=ax.transAxes, fontsize=7.6, color='0.35',
                ha='center', va='bottom', style='italic')
    ax.text(0.015, 0.965, arm, transform=ax.transAxes, ha='left', va='top',
            fontsize=11.5, color=C_ARM[samp], fontweight='bold')
    if samp == 'full':
        ax.text(0.985, 0.035, 'no legacy SN survives $z_{\\rm floor} > 0.087$',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
                color='0.35', style='italic')
    plt.setp(ax.get_xticklabels(), visible=False)
    if show_leg:
        h, l = ax.get_legend_handles_labels()
        h += [plt.Line2D([], [], color=C_ENV, ls='-', lw=1.0),
              plt.Line2D([], [], color=C_ENV, ls=':', lw=1.1),
              plt.Line2D([], [], color=C_TRUTH, ls=':', lw=1.8),
              plt.Line2D([], [], color=C_MED, ls='-', lw=1.5),
              plt.Line2D([], [], color='#444444', ls='--', lw=1.5)]
        l += [u'mock 1\u03c3 (15.87\u201384.13%)', u'mock 2\u03c3 (2.275\u201397.725%)',
              'fiducial (truth)', r'strip: median mock-specific $\sigma$',
              'strip: mock-to-mock scatter of best fits']
        fig.legend(h, l, loc='upper center', ncol=leg_ncol, fontsize=8.6,
                   frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4,
                   handlelength=2.3)

    # ---- error strip ----------------------------------------------------------
    dx = 0.006 * (xlim[1] - xlim[0])
    top = 0.0
    for i, f in enumerate(fl):
        s16, s50, s84 = np.percentile(S[:, i], [S1[0], 50.0, S1[1]])
        sd = 0.5 * (np.percentile(V[:, i], S1[1]) - np.percentile(V[:, i], S1[0]))
        for y0 in (s16, -s84):
            bx.add_patch(plt.Rectangle((f - 1.8 * dx, y0), 1.6 * dx, s84 - s16,
                                       color=C_MED, alpha=0.50, lw=0, zorder=2))
        bx.errorbar(f - dx, 0.0, yerr=s50, color=C_MED, ls='-', lw=1.5,
                    capsize=2.2, capthick=1.2, zorder=3)
        bx.errorbar(f + dx, 0.0, yerr=sd, color='#444444', ls='--', lw=1.5,
                    capsize=2.2, capthick=1.2, zorder=3)
        top = max(top, s84, sd)
    bx.axhline(0.0, color='0.75', lw=0.7, zorder=1)
    bx.set_ylim(-1.16 * top, 1.16 * top)
    if tick is not None:
        # The strip's own scale is the point of it: pin a tick at a round sigma so the bar
        # lengths can be READ, not just compared with each other.  Then thin by minimum
        # separation, requested ticks first -- the strip is an inch tall and three labels
        # inside half a sigma of zero collide.  Zero survives only when it fits; the
        # baseline is drawn as a line regardless, so losing its label costs nothing.
        lo, hi = bx.get_ylim()
        rank = [t for t in (tick, -tick) if lo <= t <= hi] + [0.0] + \
               sorted((t for t in bx.get_yticks() if lo <= t <= hi), key=abs)
        keep = []
        for t in rank:
            if all(abs(t - k) >= 0.11 * (hi - lo) for k in keep):
                keep.append(t)
        bx.set_yticks(sorted(keep))
    bx.set_xlim(*xlim)
    bx.set_ylabel(r'$\pm\,\sigma$', fontsize=10.5)
    bx.tick_params(labelsize=9)
    if hide_x:
        plt.setp(bx.get_xticklabels(), visible=False)
    else:
        bx.set_xlabel(r'$z_{\rm floor}$', fontsize=12)


def one(E, NL, RD, samp, joint, par, xlim, arm, shade_box, out):
    fig = plt.figure(figsize=(6.9, 6.1))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.05], hspace=0.07,
                  left=0.145, right=0.975, top=0.845, bottom=0.095)
    panel(fig, gs, 0, 0, E, NL, RD, samp, joint, par, xlim, arm, True, False,
          shade_box, leg_ncol=2)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RES, '%s.%s' % (out, ext)), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print('  %s.{pdf,png}' % out)



# =============================================================================
# The COMBINED figures (appendix D): one per parameter, H0 and q0, so they can be placed
# independently in the article.
#
# Both samples on one pair of axes: the ensemble (traces, 1 and 2 sigma outlines, median)
# and the observed catalogue on top, the error strip below, drawn in the palette of
# figure 4 -- DES BLACK, DES + legacy RED -- so the two figures can be read against each
# other without a translation step.  What the two-panel version carried and this one does
# not: the noiseless null, the coefficient-box shading, the "no legacy SN survives"
# annotation.
#
# THE TWO PANELS SHARE A SCALE.  The strip's height is set so that one unit of the
# parameter covers the same number of centimetres in the strip as in the main panel; a bar
# in the strip can therefore be laid against the spread of the curves above it directly.
# Everything else follows from that: the figure's height is whatever the two panels need.
#
# In the strip, each sample's pair of bars straddles its z_floor SYMMETRICALLY -- the bars
# are dodged so they can be told apart, and the cut they belong to is still read off the
# midpoint between them.  DES + legacy takes the tighter dodge because its cuts crowd
# together (0.005 apart below z_floor = 0.07); DES takes the wider one, which also keeps
# the two samples apart at the two floors, 0.073 and 0.080, they have in common.
#
# The vertical dotted lines are each sample's OWN untruncated redshift floor -- the minimum
# z actually IN that catalogue, 0.02509 and 0.07334 -- the same lines figure 4 carries.
# They are NOT the first swept cut: the sweep grid starts at z_floor = 0.030, so the red
# line sits to the LEFT of the first red point, while the black line lands on the first
# black point only because 0.073 happens to be the DES minimum to three decimals.
#
# DES + legacy stops at z_floor = 0.085.  Above 0.087 no legacy supernova survives the cut,
# so the red curve would be the black one repainted; it is truncated rather than annotated.
# =============================================================================

N_TRACE_COMBINED = 500                 # realisations drawn, both arms of each

# RASTERISATION FLOOR.  The 500 traces per arm are drawn at zorder 1 and nothing else in
# the panel is below 2, so `set_rasterization_zorder(Z_RASTER)` flattens the traces -- and
# ONLY the traces -- into one image layer, leaving the envelopes, the medians, the observed
# curves, the axes, the ticks and the key as vector.
#
# WHY.  Each of these figures carried ~2,235 separate stroked paths (500 traces x 2 arms,
# drawn a second time inside the inset, which is exactly what doubles it), against a
# working limit of 2,000 -- see strokecheck.py.  A raster image processor computes a
# stroke outline and a graphics-state change per path, and a page of this kind is what took
# an institute printer out of service.  After this the two figures are 18 strokes each.
#
# `set_rasterization_zorder` rather than a per-artist `rasterized=True`: it fixes the
# compositing order, so the raster layer is unambiguously the BACKGROUND and no vector
# artist can end up underneath it.
#
# THE COST, accepted: the trace cloud is now a 600 dpi image (savefig.dpi, from
# matplotlibrc -- the explicit dpi=200 argument had to go, or the layer would have been
# baked at 200), the file is LARGER rather than smaller, and the traces are no longer
# separable in Illustrator.  Everything a reader reads off the figure stays vector.
Z_RASTER = 2

ZMIN = {'des_only': 0.07334, 'full': 0.02509}
ZMAX_ARM = {'des_only': None, 'full': 0.0855}
ARM_LAB = {'des_only': 'DES', 'full': 'DES + legacy'}
XL_BOTH = (0.0, 0.312)

# TYPE AND LINE SCALE FOR THE PAGE.  Both figures go into a two-column float at
# 0.49\linewidth, i.e. 3.40 in of an 8.53 in canvas -- a factor 0.40, so 10 pt type drawn
# here arrives at 4 pt printed.  FS scales every font so the page gets 9-11 pt; LS thickens
# the curves by less, because a hairline that stays a hairline is what keeps 500 traces
# readable as a cloud rather than a smear.  page_pt = python_pt x displayed_in / canvas_in.
FS = 1.55                  # HALVED 2026-08-27 (2.15 -> 1.075: the keys covered the data);
                           # then +20% on 2026-09-11, and +20% AGAIN later the same day,
                           # (1.075 -> 1.29 -> 1.55). Only the
                           # TYPE scales -- the axis limits and the locators are untouched,
                           # so every tick stays exactly where it was.
LS = 1.55

H_MAIN_IN = 4.6                        # inches of the main panel; the strip follows it
W_FIG_IN = 8.6
MARG = dict(left=0.092, right=0.985, top_in=0.10, bot_in=0.62, gap_in=0.10)

# Half-width of the dodge, in units of the LOCAL floor spacing, with a floor in absolute
# z for DES so that its pair stays outside the DES + legacy pair where the two samples
# share a cut.
DODGE = {'full': (0.135, 0.0), 'des_only': (0.135, 0.0018)}


# Inset: the region where the two samples separate, at its own scale.  `rect` is in axes
# fractions of the main panel.
# `x` is the inset's horizontal span in axes fractions and `top` its upper edge; the
# BOTTOM is computed so the inset -- and the tick labels that hang below it -- clear every
# observed point underneath.  H0's curves rise, so the upper left is free; q0's fall, and
# there the DES + legacy points run along the top of the panel, so its inset moves right.
INSET = {
    # `x` is the inset's horizontal span in axes fractions.  The inset must clear every
    # TRACE that is drawn, not merely the observed points or the envelopes: a curve behind
    # a white box is a curve the reader cannot check.
    'H0': dict(x=(0.055, 0.455), xlim=(0.0195, 0.0925), corner='top', edge=0.985,
               min_in=0.0),
    # q0's traces climb into the top right and its cloud fills the middle, so there is no
    # box up there that clears them without compressing the panel by half.  Below the cloud
    # there is: nothing any sample draws goes under -4.6 before z_floor = 0.16.  The inset
    # therefore sits at the BOTTOM left, and `min_in` buys its height by dropping the
    # panel's floor -- ~20 per cent of extra range, against the ~60 per cent the same inset
    # would have cost at the top.
    'q0': dict(x=(0.075, 0.520), xlim=(0.0195, 0.0925), corner='bottom', edge=0.015,
               min_in=1.5),
}

# The strip's two line styles, shared by the bars and by the key that names them, so the
# two cannot drift apart: SOLID = median mock sigma, DOTTED = mock-to-mock scatter.  The
# dotted pattern is given explicitly rather than as ':' because matplotlib scales ':' with
# the line width, and at lw = 1.3 x LS the dots close up into a dash.
SOLID, DOT = '-', (0, (1.1, 1.7))

LABELS_IN = 0.30                       # tick labels hang outside the inset, on the panel
GAP_IN = 0.10


def _edge(cfg, SR, step):
    """The nearest drawn thing on the side the inset comes from: the highest trace under a
    top inset, the lowest under a bottom one.  Computed from the traces ACTUALLY DRAWN
    (the same fixed stride), so the box clears what the reader can see."""
    x0, x1 = cfg['x']
    z0 = XL_BOTH[0] + x0 * (XL_BOTH[1] - XL_BOTH[0])
    z1 = XL_BOTH[0] + x1 * (XL_BOTH[1] - XL_BOTH[0])
    up = cfg['corner'] == 'top'
    e = -np.inf if up else np.inf
    for sr in SR.values():
        fl, m = sr['fl'], (sr['fl'] >= z0) & (sr['fl'] <= z1)
        if not m.any() or len(fl) < 2:
            continue
        # The curves are piecewise linear, so a segment running from a floor INSIDE the box's
        # x-span to one outside it enters the box before its endpoint does.  Take the value
        # AT the two vertical edges as well, or the box clips the very lines it exists to
        # leave alone.
        rows = np.vstack([sr['V'][::step], sr['dat'][None, :]])
        edges = np.array([[np.interp(z, fl, r) for z in (z0, z1)] for r in rows])
        v = np.hstack([rows[:, m], edges])
        e = max(e, v.max()) if up else min(e, v.min())
    return e


def _headroom(cfg, SR, ylim, step):
    """Extend the panel AWAY from the inset's corner, if it must, so an inset of
    cfg['min_in'] inches fits clear of everything drawn.  One step, not iterated: the box's
    height in data units is itself proportional to the span being solved for."""
    if not cfg.get('min_in'):
        return ylim
    need = (cfg['min_in'] + LABELS_IN + GAP_IN) / H_MAIN_IN
    if need >= 1.0:
        return ylim
    e = _edge(cfg, SR, step)
    if cfg['corner'] == 'top':
        return (ylim[0], max(ylim[1], ylim[0] + (e - ylim[0]) / (1.0 - need)))
    return (min(ylim[0], ylim[1] - (ylim[1] - e) / (1.0 - need)), ylim[1])


PARITY_MIN = 1.0                       # an inset magnifies BOTH axes


def _parity_headroom(cfg, SR, ylim, step, truth):
    """Extend the panel AWAY from the inset's corner until one unit of the parameter takes
    at least as many cm inside the inset as outside it: parity = h * span / span_inset >=
    PARITY_MIN, with h the height _inset_rect will give the box.  h itself depends on the
    span, so this is solved in closed form rather than iterated.  Never bought by clipping
    the inset's range.  The cubic sweep's narrower DES band (2026-10-05) left the inset at
    its 0.22 minimum height, i.e. parity 0.5, which this exists to catch."""
    lab, gap = LABELS_IN / H_MAIN_IN, GAP_IN / H_MAIN_IN
    lo_in, hi_in = _inset_ylim(SR, cfg['xlim'], truth)
    need, e = PARITY_MIN * (hi_in - lo_in), _edge(cfg, SR, step)
    if cfg['corner'] == 'top':
        span = (need + e - ylim[0]) / (cfg['edge'] - lab - gap)
        return (ylim[0], max(ylim[1], ylim[0] + span))
    span = (need + ylim[1] - e) / (1.0 - cfg['edge'] - lab - gap)
    return (min(ylim[0], ylim[1] - span), ylim[1])


def _check_parity(cfg, SR, ylim, step, truth, tag):
    """Printed on every run: the inset's magnification in y and in z_floor."""
    x0, y0, w, h = _inset_rect(cfg, SR, ylim, step)
    lo_in, hi_in = _inset_ylim(SR, cfg['xlim'], truth)
    py = h * (ylim[1] - ylim[0]) / (hi_in - lo_in)
    px = w * (XL_BOTH[1] - XL_BOTH[0]) / (cfg['xlim'][1] - cfg['xlim'][0])
    print('[%s] inset parity: %.2f in y, %.2f in z_floor %s' % (
        tag, py, px, 'OK' if min(py, px) >= PARITY_MIN - 1e-9 else '-- TOO FLAT'))


def _inset_rect(cfg, SR, ylim, step):
    """[x0, y0, w, h] in axes fractions, clear of every drawn trace by GAP_IN, with
    LABELS_IN left between the box and the panel for the tick labels."""
    f = (_edge(cfg, SR, step) - ylim[0]) / (ylim[1] - ylim[0])
    lab, gap = LABELS_IN / H_MAIN_IN, GAP_IN / H_MAIN_IN
    if cfg['corner'] == 'top':
        y0 = float(np.clip(f + lab + gap, 0.05, cfg['edge'] - 0.22))
        h = cfg['edge'] - y0
    else:
        y0 = cfg['edge'] + lab
        h = max(0.22, f - gap - y0)
    return [cfg['x'][0], y0, cfg['x'][1] - cfg['x'][0], h]


def _inset_ylim(SR, xlim, truth):
    """Limits from what actually falls inside the inset's x-range."""
    lo = hi = truth
    for sr in SR.values():
        m = (sr['fl'] >= xlim[0]) & (sr['fl'] <= xlim[1])
        if m.any():
            lo = min(lo, sr['lo2'][m].min(), sr['dat'][m].min())
            hi = max(hi, sr['hi2'][m].max(), sr['dat'][m].max())
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def _round_tick(x):
    """The largest 1/2/5 x 10^n at or below x -- a tick the reader can do arithmetic on."""
    e = np.floor(np.log10(x))
    return max(m for m in (1.0, 2.0, 5.0) if m * 10 ** e <= x + 1e-12) * 10 ** e


def _series(E, RD, samp, par, xlim):
    """Everything both panels need for one sample, computed once."""
    key, sig, _ylab, _truth, _tick = par
    zmax = ZMAX_ARM[samp]
    groups = floor_groups(E[0], samp, False, (xlim[0], zmax if zmax else xlim[1]))
    fl = np.array([g[0] for g in groups])
    V = np.array([[e[(samp, False, f)][key] for f in fl] for e in E])
    S = np.array([[e[(samp, False, f)][sig] for f in fl] for e in E])
    d = np.diff(fl)
    frac, floor = DODGE[samp]
    return dict(
        groups=groups, fl=fl, V=V,
        med=np.median(V, axis=0),
        dat=np.array([RD[(samp, False, f)][key] for f in fl]),
        lo1=np.percentile(V, S1[0], axis=0), hi1=np.percentile(V, S1[1], axis=0),
        lo2=np.percentile(V, S2[0], axis=0), hi2=np.percentile(V, S2[1], axis=0),
        s16=np.percentile(S, S1[0], axis=0), s50=np.median(S, axis=0),
        s84=np.percentile(S, S1[1], axis=0),
        sd=0.5 * (np.percentile(V, S1[1], axis=0) - np.percentile(V, S1[0], axis=0)),
        dx=np.maximum(frac * np.minimum(np.r_[d[0], d], np.r_[d, d[-1]]).clip(max=0.013),
                      floor))


def _arm(ax, sr, samp, step):
    """Draw one sample's ensemble and observed curve in the main panel."""
    cl = C_ARM[samp]
    fl, dat = sr['fl'], sr['dat']

    for row in sr['V'][::step]:
        ax.plot(fl, row, color=cl, lw=0.35 * LS, alpha=0.045, zorder=1)
    for y in (sr['lo2'], sr['hi2']):
        ax.plot(fl, y, color=cl, ls=':', lw=1.2 * LS, alpha=0.9, zorder=4)
    for y in (sr['lo1'], sr['hi1']):
        ax.plot(fl, y, color=cl, ls='-', lw=1.0 * LS, alpha=0.9, zorder=4)
    ax.plot(fl, sr['med'], color=cl, ls='-', lw=2.6 * LS, zorder=6)
    ax.plot(fl, dat, color=cl, ls='-', lw=1.8 * LS, marker='o', ms=4.5 * LS, zorder=7)

    # WHISKERS ARE OFF, on the real-data curve.  The redundant
    # cuts are still COLLAPSED to one plotted point each (floor_groups), which is the part
    # that matters: several z_floor values can select an identical supernova sample, and
    # plotting them all draws a flat stretch that no cut produced.  What is no longer drawn
    # is the thin horizontal marker showing how far such a point reaches.
    ax.axvline(ZMIN[samp], color=cl, ls=':', lw=1.3 * LS, alpha=0.75, zorder=2)


def _strip(bx, sr, samp):
    """Draw one sample's error strip: the bars straddle their z_floor symmetrically.

    Solid bar, with a light rectangle over its 15.87-84.13 percentile range: the MEDIAN
    mock-specific sigma at that floor, and how much that sigma itself varies from mock to
    mock.  Dotted bar: half the 15.87-84.13 range of the best fits themselves, i.e. the
    scatter the ensemble actually produces.  Solid ~ dotted is the statement that the
    quoted errors are honest.

    DOTTED, not dashed: colour names the sample and
    the line style names the quantity, which is the convention the main panel above
    already reads in -- where its envelopes are dotted too.  Bars are centred on zero: only their LENGTH, and the
    z_floor they straddle, carry meaning.
    """
    cl = C_ARM[samp]
    for f, dx, s16, s50, s84, sd in zip(sr['fl'], sr['dx'], sr['s16'], sr['s50'],
                                        sr['s84'], sr['sd']):
        # The rectangle is centred on the solid bar and never reaches the dashed one; it
        # is widened where the dodge is tight, or DES + legacy's would be half a point
        # wide and invisible.
        w = max(0.9 * dx, min(0.0012, 1.6 * dx))
        for y0 in (s16, -s84):
            bx.add_patch(plt.Rectangle((f - dx - 0.5 * w, y0), w, s84 - s16,
                                       color=cl, alpha=0.32, lw=0, zorder=2))
        # Line2D, NOT errorbar: matplotlib drops `linestyle` when it builds the error-bar
        # LineCollection, so an errorbar asked for ':' comes out SOLID.  Here the two
        # bars of a sample share its colour, so the DOTS are the discriminator.
        bx.plot([f - dx, f - dx], [-s50, s50], color=cl, ls=SOLID, lw=1.3 * LS,
                marker='_', ms=4.2 * LS, mew=1.1 * LS, zorder=3)
        bx.plot([f + dx, f + dx], [-sd, sd], color=cl, ls=DOT,
                lw=1.3 * LS, marker='_', ms=4.2 * LS, mew=1.1 * LS, zorder=3)
    bx.axvline(ZMIN[samp], color=cl, ls=':', lw=1.1 * LS, alpha=0.6, zorder=1)


def _strip_key():
    """The strip's key: four entries, and the kwargs that place it.

    Returned rather than drawn so it can be MEASURED before the panel's limits are fixed
    (_strip_headroom) and then drawn into the real strip unchanged -- one definition, so
    the box that is measured is the box that appears.
    """
    def bar(samp, ls):
        return plt.Line2D([], [], color=C_ARM[samp], ls=ls, lw=1.5 * LS)

    handles = [bar('des_only', SOLID), bar('full', SOLID),
               bar('des_only', DOT), bar('full', DOT)]
    labels = [r'Median mock $\sigma$ (%s)' % ARM_LAB['des_only'],
              r'Median mock $\sigma$ (%s)' % ARM_LAB['full'],
              'Mock-to-mock scatter (%s)' % ARM_LAB['des_only'],
              'Mock-to-mock scatter (%s)' % ARM_LAB['full']]
    return handles, labels, dict(
        loc='upper left', ncol=2, fontsize=7.6 * FS, frameon=True, framealpha=0.92,
        handlelength=2.0, columnspacing=1.2, borderpad=0.4, labelspacing=0.32)


def _key_box_in(handles, labels, legkw):
    """The key's height and width IN INCHES, measured off a throwaway canvas.

    A legend's size depends on its type size, its rows and its padding -- not on the axes
    it ends up in -- so it can be measured before the real figure exists, which is what
    lets the strip's limits be chosen with the key already accounted for."""
    f = plt.figure(figsize=(W_FIG_IN, 2.0))
    a = f.add_axes([0.0, 0.0, 1.0, 1.0])
    lg = a.legend(handles, labels, **legkw)
    f.canvas.draw()
    bb = lg.get_window_extent()
    h, w = bb.height / f.dpi, bb.width / f.dpi
    plt.close(f)
    return h, w


def _strip_headroom(SR, blim, span, handles, labels, legkw, pad_in=0.10):
    """Open the top of the strip until the key clears the bars drawn underneath it.

    The key sits at the strip's upper left and the bars grow to the right, so what it can
    collide with is whatever falls inside its own WIDTH.  Both panels are drawn at the same
    data units per inch -- that is the point of deriving the strip's height from its range
    -- and that scale is fixed by the MAIN panel, so the key's measured height converts to
    data units without the strip's own limits entering, and one solve is exact.

    Needed because the type was enlarged twice (FS above): the
    symmetric +-1.16 sigma that fitted the key at half this size puts it on top of the
    DES bars at z_floor >= 0.15 in the q0 panel.  Only ever raises the top.
    """
    kh_in, kw_in = _key_box_in(handles, labels, legkw)
    upi = span / H_MAIN_IN                       # data units per inch, both panels
    panel_w_in = W_FIG_IN * (MARG['right'] - MARG['left'])
    z_key = XL_BOTH[0] + (kw_in + 0.12) * (XL_BOTH[1] - XL_BOTH[0]) / panel_w_in
    tallest = 0.0
    for sr in SR.values():
        m = sr['fl'] <= z_key
        if m.any():
            tallest = max(tallest, float(np.max(np.maximum(sr['s84'][m], sr['sd'][m]))))
    return (blim[0], max(blim[1], tallest + (kh_in + pad_in) * upi))


def combined(E, RD, par=None, out='floorsweep_mock_H0', inset=None):
    par = par or P_H0
    key, sig, ylab, truth, tick = par
    step = max(1, len(E) // N_TRACE_COMBINED)
    SR = {s: _series(E, RD, s, par, XL_BOTH) for s in ('full', 'des_only')}

    # ---- limits FIRST: the panel heights are derived from them --------------------
    lo = min([truth] + [min(sr['lo2'].min(), sr['dat'].min()) for sr in SR.values()])
    hi = max([truth] + [max(sr['hi2'].max(), sr['dat'].max()) for sr in SR.values()])
    pad = 0.05 * (hi - lo)
    ylim = (lo - 1.6 * pad, hi + 3.2 * pad)
    ylim = _headroom(inset or INSET[key], SR, ylim, step)
    ylim = _parity_headroom(inset or INSET[key], SR, ylim, step, truth)
    _check_parity(inset or INSET[key], SR, ylim, step, truth, out)
    top = max(max(sr['s84'].max(), sr['sd'].max()) for sr in SR.values())
    # The strip's key is built HERE, before the limits, because the limits depend on it:
    # see _strip_headroom.  Symmetric +-1.16 sigma is the floor; the top opens further if
    # the key would otherwise land on a bar.
    KEY_H, KEY_L, LEGKW = _strip_key()
    blim = _strip_headroom(SR, (-1.16 * top, 1.16 * top), ylim[1] - ylim[0],
                           KEY_H, KEY_L, LEGKW)

    span, bspan = ylim[1] - ylim[0], blim[1] - blim[0]
    h_bot = H_MAIN_IN * bspan / span            # SAME cm per unit in both panels
    fig_h = MARG['top_in'] + H_MAIN_IN + MARG['gap_in'] + h_bot + MARG['bot_in']

    fig = plt.figure(figsize=(W_FIG_IN, fig_h))
    gs = GridSpec(2, 1, height_ratios=[span, bspan],
                  hspace=MARG['gap_in'] / (0.5 * (H_MAIN_IN + h_bot)),
                  left=MARG['left'], right=MARG['right'],
                  top=1.0 - MARG['top_in'] / fig_h, bottom=MARG['bot_in'] / fig_h)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1], sharex=ax)

    ax.set_rasterization_zorder(Z_RASTER)    # the trace cloud only -- see Z_RASTER
    ax.axhline(truth, color=C_TRUTH, ls=':', lw=1.8 * LS, zorder=3)
    for samp in ('full', 'des_only'):           # DES last, so it lands on top
        _arm(ax, SR[samp], samp, step)
    ax.set_ylim(*ylim)
    ax.set_xlim(*XL_BOTH)
    ax.set_ylabel(ylab, fontsize=13 * FS)
    ax.tick_params(labelsize=11 * FS)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---- inset: the crowded end of the sweep -------------------------------------
    # The main panel has to run to z_floor = 0.30, where sigma is 11 km/s/Mpc, and that
    # scale flattens the 0.025-0.09 region where the two samples separate and where every
    # p-value in the appendix is computed.  The inset is that region at its own scale; it
    # sits where the legend used to, which is the one large area of the panel no curve
    # and no envelope reaches.
    ins = inset or INSET[key]
    axin = ax.inset_axes(_inset_rect(ins, SR, ylim, step))
    # The inset redraws all 500 traces of both arms, which is the half of the cost that is
    # easiest to forget, so it gets the same treatment as the main panel.
    axin.set_rasterization_zorder(Z_RASTER)
    axin.set_facecolor('white')
    axin.patch.set_alpha(0.97)
    axin.axhline(truth, color=C_TRUTH, ls=':', lw=1.5 * LS, zorder=3)
    for samp in ('full', 'des_only'):
        _arm(axin, SR[samp], samp, step)
    axin.set_xlim(*ins['xlim'])
    axin.set_ylim(*_inset_ylim(SR, ins['xlim'], truth))
    axin.xaxis.set_major_locator(MaxNLocator(4))
    # 1/2/5 steps only: a 1.5 step gives '67.5'-wide labels, which reach back across the
    # panel's own spine and interleave with its tick labels (cubic sweep, 2026-10-05).
    axin.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    # A y tick in the bottom 6 per cent puts its label on the first x tick label, in the
    # corner they share.
    ylo_in, yhi_in = axin.get_ylim()
    axin.set_yticks([t for t in axin.get_yticks()
                     if ylo_in + 0.06 * (yhi_in - ylo_in) <= t <= yhi_in])
    axin.tick_params(labelsize=8.4 * FS, length=3 * LS, pad=2, direction='in')
    # BLACK, not grey.  The frame stays -- an inset laid over a cloud of traces
    # with no edge has no boundary at all -- but it is now the same black hairline as the
    # panel's own spines rather than a grey box that reads as an annotation of the region
    # underneath.  The zoom rectangle and its connector lines are a separate thing and
    # have been gone since the request below.
    for sp in axin.spines.values():
        sp.set_edgecolor('k')
        sp.set_linewidth(0.8)
    # The tick labels sit OUTSIDE the inset, i.e. on top of the main panel's traces; a
    # white patch behind each keeps both readable.
    for t in axin.get_xticklabels() + axin.get_yticklabels():
        t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.8))
    # NO indicate_inset_zoom: the inset does not point at
    # anything.  The rectangle and its two connector lines were drawn across the main
    # panel's traces, and the inset's own x tick labels already say which z_floor range it
    # covers.  Its caption sentence says so as well.

    # ---- error strip ---------------------------------------------------------------
    for samp in ('full', 'des_only'):
        _strip(bx, SR[samp], samp)
    bx.axhline(0.0, color='0.75', lw=0.7, zorder=1)
    bx.set_ylim(*blim)
    if tick is not None:
        # Pin a tick at a round sigma so the bar lengths can be READ, not just compared
        # with each other; then thin by minimum separation, requested ticks first.  The
        # pinned value is chosen from the strip's OWN height (the parameter's default one
        # is set for the two-panel figures, whose strips are a different size).
        tick = _round_tick(0.45 * blim[1])
        rank = [t for t in (tick, -tick) if blim[0] <= t <= blim[1]] + [0.0] + \
            sorted((t for t in bx.get_yticks() if blim[0] <= t <= blim[1]), key=abs)
        keep = []
        for t in rank:
            if all(abs(t - k) >= 0.09 * bspan for k in keep):
                keep.append(t)
        bx.set_yticks(sorted(keep))
    # FOUR ENTRIES, not two: each sample's two bars are named
    # separately, in that sample's own colour, so the key can be read without first
    # working out that a black/red pair of handles stands for both samples at once.
    #
    # Laid out 2 x 2 with DES on the TOP row.
    # matplotlib fills a multi-column legend COLUMN-MAJOR, so the handles are ordered
    # down-then-across -- DES sigma, DES+legacy sigma, DES scatter, DES+legacy scatter --
    # to come out as
    #
    #     Median mock sigma (DES)            Mock-to-mock scatter (DES)
    #     Median mock sigma (DES + legacy)   Mock-to-mock scatter (DES + legacy)
    #
    # The QUANTITY leads and the sample follows in brackets, so the two columns are two
    # quantities and the two rows are two samples; naming the sample first made all four
    # entries start with the same word.  Colour still says which sample and the line style
    # which quantity, so neither has to be read off the text.
    #
    # The key stays IN the strip at its top left: that is where the bars are shortest
    # (both samples' smallest sigma is at the lowest floors) and nothing else is drawn.
    bx.legend(KEY_H, KEY_L, **LEGKW)
    bx.set_xlim(*XL_BOTH)
    bx.set_xlabel(r'$z_{\rm floor}$', fontsize=13 * FS)
    bx.set_ylabel(r'$\pm\,\sigma$' + STRIP_UNIT.get(key, ''), fontsize=12 * FS)
    bx.tick_params(labelsize=11 * FS)

    # NO explicit dpi on the PDF: it would override savefig.dpi from matplotlibrc and bake
    # the rasterised trace layer at 200 dpi.  The PNG keeps 200, where dpi sets the whole
    # image and 600 would only make a large file of a screen preview.
    fig.savefig(os.path.join(RES, '%s.pdf' % out), bbox_inches='tight')
    fig.savefig(os.path.join(RES, '%s.png' % out), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  %s.{pdf,png}  (%d of %d traced; panels %.2f + %.2f in, %.4g units/in)'
          % (out, len(E[::step]), len(E), H_MAIN_IN, h_bot, span / H_MAIN_IN))


def main():
    files = [f for f in rglob(RES, 'lsweep_real*.json')
             if re.search(r'real\d+\.json$', f)]
    E = [load(f) for f in files]
    NL = load(rpath(RES, 'lsweep_noiseless.json'))
    RD = load(rpath(RES, 'lsweep_realdata.json'))
    print('realisations: %d' % len(E))

    DL = r'DES + legacy, $\Delta M_{\rm legacy}$ free'
    JOBS = (
        ('full', False, P_H0, XL_FULL, 'DES + legacy', False, 'deslegacy_H0'),
        ('full', False, P_Q0, XL_FULL, 'DES + legacy', False, 'deslegacy_q0'),
        ('des_only', False, P_H0, XL_DES, 'DES only', True, 'desonly_H0'),
        ('des_only', False, P_Q0, XL_DES, 'DES only', True, 'desonly_q0'),
        ('full', True, P_H0, XL_FULL, DL, False, 'deltaarm_H0'),
        ('full', True, P_Q0, XL_FULL, DL, False, 'deltaarm_q0'),
        ('full', True, P_DL, XL_FULL, DL, False, 'deltaarm_Delta'),
    )
    for samp, joint, par, xlim, arm, box, tag in JOBS:
        one(E, NL, RD, samp, joint, par, xlim, arm, box, 'floorsweep_mock_' + tag)

    # the combined figures that replace the desonly/deslegacy pair in appendix D: one per
    # parameter, so H0 and q0 can be placed independently in the article
    combined(E, RD, P_H0, 'floorsweep_mock_H0')
    combined(E, RD, P_Q0, 'floorsweep_mock_q0')


if __name__ == '__main__':
    main()
