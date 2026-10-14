r"""
make_figures.py -- render the paper's figures from the cached chains.

READS ONLY.  It never samples.

    python make_figures.py                    # all five
    python make_figures.py --figure floorsweep ladder

Output: results/<article basename>.pdf and .png, where <article basename> is the name
the manuscript \includegraphics, so a build can be dropped straight into the manuscript
directory:

    h0_q0_P4_spa          fig:h0_q0          (H0-q0 joint posterior vs SPA)
    floorsweep_real_H0    fig:floorsweep-real-new  (upper tile, one column)
    floorsweep_real_q0    fig:floorsweep-real-new  (lower tile, one column)
    floorsweep_real_paper  -- the same sweep as ONE two-panel float, superseded by the
                              two tiles above but kept so an old \includegraphics builds
    ladder                fig:offset_result / fig:offset_split   (both panels)
    h0                    fig:legacy-h0
    FAP_ratio_LCDM_paper  fig:AP-new

STYLE -- READ THIS BEFORE CHANGING A COLOUR.
Every number in the CONFIG blocks below (colour, marker, font size, limit, alpha, z-order,
label string, legend placement) was transcribed from the scripts that produced the figures
now in the manuscript:

    h0_q0_P4_spa          <- dovekie_h0_q0_spt_figure.py           (CONFIG block)
    floorsweep_real_paper <- paper_figures.py                      (FLOORSWEEP_REAL + STYLE)
    ladder, h0            <- legacy_taylor_figures.py              (STYLE + COL + AXES + ...)
    FAP_ratio_LCDM_paper  <- dovekie_cosmo_FAP_nuts.py::plot_ratio

so a rebuild here is drop-in replaceable with the published float.

COLOURS -- changed 2026-08-17.  Each figure used to carry
its OWN DES / DES+legacy pair (h0_q0 red/blue, floorsweep orange/blue, F_AP blue/red), so
the captions contradicted each other and the baseline was drawn in the paler colour of the
two.  There is now ONE pair, in SERIES, used by every figure:

    DES (the baseline)  = black       DES+legacy = red

which also fixes the
caption contradiction.  The CMB marker is a light orange star so it cannot be mistaken for
either series.  Change SERIES, not the individual figure configs.

A THIRD entry joined SERIES on 2026-08-18: `full_dleg`, "DES+legacy ($\Delta M_\mathrm{
legacy}$)", in blue -- the same supernovae as `full` but with the legacy zero-point
sampled rather than asserted.  It is a TREATMENT of the second series, not a third
dataset.  The blue is the one make_triangles uses for its Delta-varied contour, so the
appendix and the front figure agree.  It is carried by h0_q0 and, since 2026-08-20, by
the floor sweep as well.

THE SYMBOL for that zero-point is `\Delta M_\mathrm{legacy}` -- settled 2026-08-20, for
consistency.  A bare
`\Delta` is NOT it: the paper also carries `\Delta H(z)`, `\Delta\chi^2` and `\Delta BIC`,
and the bare symbol was being read as those.  Every axis label, legend entry and contour
label that means the legacy offset spells it out in full.

CMB REFERENCE -- changed 2026-08-17.  Every "CMB expectation" drawn by this file is now
SPA (SPT-3G + ACT DR6 + Planck 2018; Camphuis et al. 2026), not Planck 2018 alone, which
brings the figures into line with the tensions already quoted in the text and tables.  See
the SPA_* constants below.  The r_d prior inside the likelihood is a separate thing and is
NOT changed -- see the note there.
"""
import argparse
import json
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

import dovekie_cosmography as dc

DPI = 200                      # every published float was written at dpi=200

# --------------------------------------------------------------------------- #
# The CMB reference EVERY comparison in this paper is made against.
#
# SPA = SPT-3G D1 + ACT DR6 + Planck 2018, the combination of Camphuis et al. (2026),
# their table 1.  It replaces Planck 2018 alone (67.36 +/- 0.54, Om = 0.3153) everywhere
# a figure draws a "CMB expectation" line, point or normalisation, so that the figures
# agree with the tensions quoted in the text and in Tables 1 and 3, which already use it.
#
# This is a COMPARISON reference only.  The r_d = 147.05 +/- 0.30 cMpc prior that
# configuration A imports (dc.BASELINE['RD_PLANCK']) is a Planck 2018 measurement used as
# an ASSUMPTION inside the likelihood; it is deliberately untouched here, because changing
# it would invalidate every cached chain.
# --------------------------------------------------------------------------- #
SPA_H0 = (67.19, 0.38)
SPA_OM = 0.3175
SPA_Q0 = 1.5 * SPA_OM - 1.0    # flat LCDM: q0 = 3/2 Om - 1  (= -0.524)
SPA_LABEL_PLAIN = 'SPA'
SPA_LABEL_TEX = r'SPA'
# The form used for an in-panel ANNOTATION: a horizontal line at
# the SPA H0 or q0 is where a flat-LCDM fit to that CMB combination sits, not a
# model-independent measurement, and the annotation should say so where the reader meets
# it rather than only in the caption.
SPA_LABEL_LCDM = r'SPA ($\Lambda$CDM)'
# The local anchor, named for the programme rather than for the paper.  The VALUE is
# Breuval et al. (2024), 73.17 +/- 0.86, and is unchanged; only the label moved.  Both are
# correct; this is the clearer one.
SHOES_LABEL = 'SH0ES'
SPA_MARKER = dict(c='#e8a33d', marker='*', ms=16, mec='0.25', mew=0.8)   # light orange

# --------------------------------------------------------------------------- #
# The ONE series palette.  DES is the baseline and is therefore the black one; DES+legacy
# is the extension under suspicion and is red.  Every figure reads from here.
# --------------------------------------------------------------------------- #
# Figure 7 (fig:legacy-h0) colours by MODEL, not by sample: every cosmography point is
# black and every flat-LCDM point is this dark blue, on BOTH panels, so the row's treatment
# of the legacy zero-point is carried by the marker SHAPE alone (circle vs square), so
# the legend stays clear -- before this the cosmography
# circles wore three different colours (black / C3 red / dark blue) while the legend showed
# only one, and the LCDM squares were grey.
# 🚨 MODEL_BLUE is the SAME dark blue as SERIES['full_dleg'], which in the h0_q0 figure and
# the appendix triangles means the Delta-sampled COSMOGRAPHY treatment.  Blue therefore
# carries one meaning in Figure 7 and another elsewhere.
MODEL_BLACK = 'k'
MODEL_BLUE = '#1f4e79'

SERIES = dict(
    des_only=dict(c='k',  lab='DES'),
    full=dict(c='C3', lab='DES+legacy'),
    # The SAME supernovae as `full`, but with the legacy zero-point Delta sampled jointly
    # instead of held at its face value -- Table 3's joint cell.  It is a third TREATMENT,
    # not a third dataset, and it is drawn in the blue the appendix triangles already use
    # for the Delta-varied contour (make_triangles.BLUE).
    full_dleg=dict(c='#1f4e79', lab=r'DES+legacy ($\Delta M_\mathrm{legacy}$)'),
)


# =============================================================================
# Figure 1 -- the (H0, q0) joint posterior against SPT-3G D1
#   source: dovekie_h0_q0_spt_figure.py
# =============================================================================
H0Q0 = dict(
    out='h0_q0_P4_spa',
    figsize=(7.2, 5.8),
    fontsize_label=14, fontsize_tick=12, fontsize_legend=11.5,
    # Drawn in this order, so `full` (the compact blob) goes on TOP of the des_only ridge
    # it sits inside.  Config A, cosmography, P=4 -- the same cells as Table 1.
    #
    # The third entry is the SAME fit with the legacy zero-point Delta sampled jointly
    # (Table 3's joint cell, chains/legacy/cosmo_full_joint_P4).  Its scalars are
    # bit-identical to A_full_cosmo_P4 apart from `joint_delta`, so it belongs in this
    # plane: it is what DES+legacy becomes once the legacy offset is not asserted.  It is
    # broader (H0 71.17 +- 1.97 against 68.11 +- 0.71) and so is drawn LAST but with the
    # other two still visible through it -- hence unfilled edges are kept for all three.
    series=[('des_only', dict(cell=('ladder', 'A_des_only_cosmo_P%d'),
                              c=SERIES['des_only']['c'], lab=SERIES['des_only']['lab'])),
            ('full', dict(cell=('ladder', 'A_full_cosmo_P%d'),
                          c=SERIES['full']['c'], lab=SERIES['full']['lab'])),
            ('full_dleg', dict(cell=('legacy', 'cosmo_full_joint_P%d'),
                               c=SERIES['full_dleg']['c'],
                               lab=SERIES['full_dleg']['lab']))],
    levels=(0.68, 0.95),                 # enclosed posterior MASS, not sigma of a Gaussian
    alpha_1sig=0.55, alpha_2sig=0.22,    # darker inner, lighter outer
    lw=1.6,
    xlim=(60.0, 88.0), ylim=(-5.6, 1.0),
    xlabel=r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', ylabel=r'$q_0$',
    # A POINT, not an ellipse: q0 = 1.5*Om - 1 holds only inside flat LCDM, so the marker
    # is "where a flat-LCDM CMB fit sits in this plane", not a measurement in these
    # coordinates with an error bar.  This figure was ALREADY on SPA.
    ref=dict(H0=SPA_H0[0], Om=SPA_OM, lab=SPA_LABEL_TEX + r' ($\Lambda$CDM)', **SPA_MARKER),
    # The local anchor as a VERTICAL band.  It constrains H0 and
    # says nothing about q0, so it is a band across the panel and not a point with an
    # error bar -- the same distinction the SPA marker makes in the other direction.
    # Drawn BEFORE the contours (see fig_h0_q0) so it sits under them rather than washing
    # them out, and keyed simply as SH0ES.
    shoes=dict(val=73.17, sig=0.86, lab=SHOES_LABEL, c='green', alpha=0.15, lw=1.3),
    legend_loc='lower left', legend_framealpha=0.92,
)


# =============================================================================
# Figure 2 -- the redshift-floor sweep
#   source: paper_figures.py FLOORSWEEP_REAL, with STYLE fontsizes
# =============================================================================
FLOORSWEEP = dict(
    out='floorsweep_real_paper',
    figsize=(13.2, 5.2),                 # the side-by-side (H0, q0) pair
    fontsize_label=15, fontsize_legend=12, fontsize_annot=12,
    # (tag, legend label, colour, marker, that sample's own survey floor).  DES leads the
    # list so that the legend order follows the H0 ordering at
    # the right of the panel, where DES is the upper curve.
    # (tag, legend label, colour, marker, own survey floor, highest z_floor worth showing)
    #
    # The last field matters and was missing.  The legacy supernovae stop at z = 0.087, so
    # at any z_floor above that the "DES+legacy" fit contains NO legacy supernovae -- it is
    # the DES fit, re-plotted in a second colour on top of itself.  That is what used to
    # make the two curves lie exactly on each other from 0.09 to 0.15, and it is the point
    # the caption makes.  Truncating the series says it in the figure
    # instead.  None -> no truncation.
    #
    # The THIRD arm is the same supernovae as `full` with the
    # legacy zero-point SAMPLED rather than held at face value -- cells `full_joint_*`, the
    # floor-sweep version of Table 3's joint cell.  It carries the palette's blue and the
    # `\Delta M_\mathrm{legacy}` label, so it reads as the same treatment the h0_q0 figure and
    # the appendix triangles already draw in blue.  It stops at 0.085 rather than 0.087:
    # above that NO legacy supernova survives the cut, the offset samples its prior, and
    # the band would read like a measured null.  See dc.FLOORS_DELTA_MAX.
    samples=[('des_only', SERIES['des_only']['lab'], SERIES['des_only']['c'], 'o',
              0.073, None),
             ('full', SERIES['full']['lab'], SERIES['full']['c'], 'o',
              0.025, 0.087),
             ('full_joint', SERIES['full_dleg']['lab'], SERIES['full_dleg']['c'], 's',
              0.025, dc.FLOORS_DELTA_MAX)],
    # (2026-08-24) Each arm's dashed floor line sat to the LEFT of that arm's first data
    # point: FLOORS runs the sweep only from the first grid value at or above the sample's
    # own z_min, so DES started at z_floor = 0.080 under a line at 0.073, and both
    # DES+legacy arms started at 0.030 under a line at 0.025.  The missing point is not
    # missing physics -- at z_floor = z_min NO supernova is cut, so that cell simply IS the
    # untruncated fit, already cached as the Table 3 legacy cell.  Anchoring each curve on
    # it closes the gap with no new sampling and puts the marker exactly on the line.  It
    # is also the point the text already quotes: the DES value at z_floor = 0.073 is the
    # untruncated 76.83, and 76.83 - 72.04 is the 4.78 that the 10 legacy SNIa buy.
    # tag -> (sample, joint_delta) for _legacy_cell; the sample also fixes z_min.
    baselines={'des_only': ('des_only', False),
               'full': ('full', False),
               'full_joint': ('full', True)},
    ms=5, lw=1.7, capsize=2.5,
    xlabel=r'$z_{\rm floor}$',
    ylabel=r'recovered $H_0$  [km s$^{-1}$ Mpc$^{-1}$]',
    q0_ylabel=r'recovered $q_0$',
    # (value, sigma, name, colour, side, x).  x=None anchors the name just to the RIGHT of
    # the DES floor line, left-aligned; a number centres it on that z_floor instead.
    # 'above' for SPA.  Breuval stays below: above, it collides with the rising DES points.
    refs=[(SPA_H0[0], SPA_H0[1], SPA_LABEL_LCDM, '#e8a33d', 'above', None),
          (73.17, 0.86, SHOES_LABEL, 'green', 'below', 0.150)],
    ref_label_floor=0.073,
    # The right panel carries a CMB comparison too: the left panel's band is SPA, and
    # this is the matching q0.  Set to None to drop it.
    # The q0 panel's SPA annotation takes an extra downward nudge, in points:
    # with va='top' alone the text sat hard against its own line.
    q0_ref=(SPA_Q0, SPA_LABEL_LCDM, '#e8a33d'),
    q0_ref_dy_pt=-3.0,
    # a tick at 0 on both panels, since there are no gridlines
    extra_xtick=0.0,
    q0_legend_loc=None,
    # PINNED to the grid.  FLOORS ends at z_floor = 0.30
    # and every tail cell has now run, so the axis stops there rather than being fitted by
    # matplotlib -- which padded it out to 0.35 and spent the last seventh of the panel on
    # blank space.  0.305 rather than 0.300: a hair of margin, ~1.6 per cent of the span,
    # so the marker and the error bar of the last point are not cut in half by the spine.
    # FS_XLIM/FS_YLIM still override.
    xlim=tuple(float(v) for v in os.environ['FS_XLIM'].split(',')) if os.environ.get('FS_XLIM') else (0.0, 0.305),
    ylim=tuple(float(v) for v in os.environ['FS_YLIM'].split(',')) if os.environ.get('FS_YLIM') else None,
    q0_ylim=tuple(float(v) for v in os.environ['FS_Q0_YLIM'].split(',')) if os.environ.get('FS_Q0_YLIM') else None,
    # The single data legend lives on the LEFT (H0) panel; the right panel carries none,
    # so the two panels are not labelled twice.  q0_legend_loc was 'upper right' between
    # 2026-08-18 and 2026-08-20 and is back to None: the three
    # series are the same in both panels, so the second copy only repeated the first.
    legend_loc='lower right', legend_framealpha=0.95,
    # ------------------------------------------------------------------ #
    # The SPLIT.  fig:floorsweep-real-new is now two \includegraphics commands tiled
    # vertically inside a ONE-column `figure` (not a `figure*`), which is what the shared
    # x axis buys: z_floor is the same independent variable over the same range in both,
    # so stacking them puts the H0 point and the q0 point of a given cut on the same
    # vertical.  Each file therefore has to stand alone -- hence its OWN legend, where the
    # combined float could put one legend on the left panel and let the right borrow it.
    out_split=dict(H0='floorsweep_real_H0', q0='floorsweep_real_q0'),
    # Half the combined width, so the single-column reduction is the same as the
    # two-panel float's was at \textwidth and the fontsizes above still land at the same
    # point size on the page.
    figsize_split=(6.6, 4.4),
    legend_loc_split=dict(H0='lower right', q0='upper right'),
)


# =============================================================================
# Figures 3 and 4 -- the legacy offset ladder and the legacy H0
#   source: legacy_taylor_figures.py (STYLE, COL, RUNGS, SPLITS, AXES, H0_ROWS, ...)
# =============================================================================
LEG = dict(
    fontsize_label=12, fontsize_tick=10, fontsize_annot=9, fontsize_legend=8.5,
    ms=4.5, capsize=2.5, lw=1.4,
    # ladder_rungs / ladder_splits are the two halves as STANDALONE files (the manuscript
    # uses two \includegraphics rather than an externally stacked float); `ladder` is the
    # legacy combined version, kept so an un-updated \includegraphics still builds.
    figsize=dict(ladder=(6.4, 4.6), ladder_rungs=(6.4, 2.6), ladder_splits=(6.4, 2.6),
                 h0=(6.4, 3.75), h0_rows=(6.4, 2.75), h0_delta=(6.4, 1.8)),
    height_ratio=dict(ladder=(1.0, 1.05), h0=(1.0, 0.34)),
    top_pad=0.8, top_pad_jdelta=0.55,
    # one colour per rung / per treatment, not per sample.
    # `cosmo`/`fap`/`lcdm`/`lcdm_fap` are the LADDER panels' rung colours and are still
    # live.  `desonly`/`uncorr`/`joint` are the Figure 7 ROW colours and are now DEAD for
    # the markers -- both entries of `h0_models` carry an explicit override, so the
    # `mcol or C['col'][ckey]` fallback in draw_h0 never fires.  They are kept because the
    # fallback is the safety net if a third model is ever added to that figure.
    # `shoes` moved off C3 because C3 is the DES+legacy series colour and the two appear
    # on the same axes.
    col=dict(cosmo='C0', fap='C4', lcdm='C3', lcdm_fap='C5', split='0.35', zero='0.55',
             efst='0.85', uncorr=SERIES['full']['c'], joint=SERIES['full_dleg']['c'],
             desonly=SERIES['des_only']['c'], shoes='green', planck='#e8a33d'),
    # (rung key, label, colour key) -- the four rigid references for the DES relation
    rungs=[('cosmo', r'Cosmography', 'cosmo'),
           ('cosmo_fap', r'Cosmography $+$ AP', 'fap'),
           ('lcdm', r'$\Lambda$CDM', 'lcdm'),
           ('lcdm_fap', r'$\Lambda$CDM $+$ AP', 'lcdm_fap')],
    # whose subsample contrasts the lower panel shows
    headline_rung='lcdm_fap',
    splits=[('Foundation', r'Foundation'), ('CfA/CSP', r'CfA/CSP'),
            ('z > 0.05', r'$z>0.05$'), ('z >= DES floor', r'$z\geq z_{\rm floor}$')],
    delta_xlim=(-0.085, 0.17), delta_xlabel=r'$\Delta M_\mathrm{legacy}$  [mag]',
    # A different quantity centred on zero: sharing delta_xlim would render every contrast
    # as a dot on the zero line.
    contrast_xlim=(-0.075, 0.075),
    contrast_xlabel=r'$\Delta M_\mathrm{legacy} - \Delta M_\mathrm{legacy}^\mathrm{rest}$  [mag]',
    jdelta_xlim=(-0.02, 0.155),
    h0_xlim=(65.0, 80.5), h0_xlabel=r'$H_0$  [km s$^{-1}$ Mpc$^{-1}$]',
    # (row key, label, colour key).  None is a SPACER -- a blank slot with no marker and
    # no tick label.  Order: the two rows that need NO estimate of Delta
    # at the top, then the row that measures it.  The spacer that used to sit before the
    # last row is gone (the gap looked odd); h0_rows figsize
    # shrank 3.2 -> 2.75 in with it, so the row pitch is unchanged.
    # Labels are the SERIES labels every other figure uses.  The bracketed
    # Delta goes on its own line: on one line it was the
    # widest tick label and squeezed this panel's axes narrower than h0_delta's.
    h0_rows=[(('des_only', False), SERIES['des_only']['lab'], 'desonly'),
             (('full', False), SERIES['full']['lab'], 'uncorr'),
             (('full', True), SERIES['full_dleg']['lab'].replace(' (', '\n('), 'joint')],
    # (background key, legend label, marker, colour override or None -> the row's colour)
    # k_max, not P: the manuscript calls the truncation order k_max everywhere, and a
    # legend that calls it P makes the reader translate.
    h0_models=[('cosmo', r'Cosmography $k_{\max}{=}%d$' % dc.BASELINE['P'], 'o',
                MODEL_BLACK),
               ('lcdm', r'$\Lambda$CDM', 's', MODEL_BLUE)],
    # Figure 7 alone takes a slightly larger marker: with both series now drawn in two
    # fixed colours the circle/square distinction does the work the colours used to, and
    # at ms=4.5 the two shapes are hard to tell apart.  A local override rather than a
    # bump to LEG['ms'], which the two ladder panels also read.
    ms_legacy=5.5,
    h0_dy=0.20,                          # vertical split between the two backgrounds
    jdelta_row=('full', True), jdelta_label=r'$\Delta M_\mathrm{legacy}$ sampled',
    # The ~0.04 mag low-redshift offset of Efstathiou (2025), for comparison only -- never
    # a constraint.  It is an ABSOLUTE offset, so it goes on the two absolute-Delta panels
    # and NOT on the contrast panel, where zero is the null.
    efst=dict(val=0.04, half=0.01),
    # SH0ES, not 'Breuval et al.'.  The author asked for this on the order sweep and the
    # floor sweep ("both are correct but the former is clearer"); the reason is not
    # figure-specific, and one band wearing two names across the paper is worse than
    # either name, so the legacy ladder follows.  The VALUE is unchanged and is still
    # Breuval et al. (2024) -- the manuscript's \citet stays as it is.
    shoes=dict(val=73.17, sig=0.86, label=SHOES_LABEL),
    # kept under the key `planck` so the row/colour wiring below is unchanged; the VALUE
    # is now SPA, matching every other CMB comparison in the paper.
    planck=dict(val=SPA_H0[0], sig=SPA_H0[1], label=SPA_LABEL_TEX),
)


# =============================================================================
# Figure 5 -- the Alcock-Paczynski ratio
#   source: dovekie_cosmo_FAP_nuts.py::plot_ratio
# =============================================================================
AP = dict(
    out='FAP_ratio_LCDM_paper',
    figsize=(7.2, 4.8),
    fontsize_label=15, fontsize_tick=12, fontsize_legend=12,
    colors={k: v['c'] for k, v in SERIES.items()},
    labels={k: v['lab'] for k, v in SERIES.items()},
    order=['des_only', 'full'],
    band_alpha=0.18, lw=1.6,
    # DESI was black squares, which now collides with the DES series.  Dark blue keeps the
    # measurement visually separate from both SN curves.
    desi=dict(marker='s', color='#1f4e79', ms=6, capsize=3, label='DESI DR2'),
    ref_line_color='0.5',
    zgrid=(0.40, 1.02, 200), xlim=(0.40, 1.02),
    xlabel=r'redshift $z$',
    # F_AP^LCDM depends on Omega_m ONLY (H0 cancels), so the reference moves with SPA's
    # Om = 0.3175, not Planck 2018's 0.3153.  This renormalises the panel; it does NOT
    # touch the AP chi2 or p-values, which never reference a fiducial cosmology.
    ref_om=SPA_OM,
    ylabel=r'$F_{\rm AP}\,/\,F_{\rm AP}^{\Lambda{\rm CDM,\ SPA}}$',
    legend_loc='best',
)


# --------------------------------------------------------------------------- #
# The published floats carry no grid and no rcParams overrides beyond the sizes set
# per axis, so the defaults are left alone on purpose.
# --------------------------------------------------------------------------- #
def _save(fig, name, tight_bbox=False):
    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    p = os.path.join(dc.RESULT_DIR, name)
    kw = dict(bbox_inches='tight') if tight_bbox else {}
    fig.savefig(p + '.pdf', dpi=DPI, **kw)
    fig.savefig(p + '.png', dpi=DPI, **kw)
    plt.close(fig)
    print('wrote %s.pdf and .png' % p)
    return p


# =============================================================================
def fig_h0_q0():
    """The (H0, q0) joint posterior of the two samples, with the SPA point.

    Contours enclose 68% and 95% of the posterior MASS -- the standard cosmology
    convention, so a broad ridge and a compact blob are directly comparable."""
    from scipy.stats import gaussian_kde
    C = H0Q0
    P = dc.BASELINE['P']
    fig, ax = plt.subplots(figsize=C['figsize'])

    # FIRST, before any contour: the band is context, so it goes underneath.  Drawing it
    # afterwards would put a translucent green wash over the DES ridge it overlaps.
    sh = C.get('shoes')
    if sh:
        ax.axvspan(sh['val'] - sh['sig'], sh['val'] + sh['sig'], color=sh['c'],
                   alpha=sh['alpha'], lw=0, zorder=0, label=sh['lab'])
        ax.axvline(sh['val'], color=sh['c'], ls=':', lw=sh['lw'], zorder=0)

    for name, style in C['series']:
        group, cell = style['cell'][0], style['cell'][1] % P
        c = dc.cache_load(group, cell)
        if c is not None:
            dc.require_quality('chains/%s/%s.npz' % (group, cell), dc._unpack(c))
        if c is None:
            raise SystemExit('missing chain chains/%s/%s.npz.\n'
                             'Run: python dovekie_cosmography.py --task %s'
                             % (group, cell, group))
        x, y = np.asarray(c['H0']), np.asarray(c['q0'])
        pad_x, pad_y = 0.15 * np.ptp(x), 0.15 * np.ptp(y)
        xs = np.linspace(x.min() - pad_x, x.max() + pad_x, 260)
        ys = np.linspace(y.min() - pad_y, y.max() + pad_y, 260)
        X, Y = np.meshgrid(xs, ys)
        Z = gaussian_kde(np.vstack([x, y]))(
            np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        flat = np.sort(Z.ravel())[::-1]
        csum = np.cumsum(flat) / flat.sum()
        c68, c95 = (flat[np.searchsorted(csum, lv)] for lv in C['levels'])
        ax.contourf(X, Y, Z, levels=[c95, c68], colors=[style['c']], alpha=C['alpha_2sig'])
        ax.contourf(X, Y, Z, levels=[c68, Z.max()], colors=[style['c']],
                    alpha=C['alpha_1sig'])
        ax.contour(X, Y, Z, levels=[c95, c68], colors=style['c'], linewidths=C['lw'])
        # a fat line stands in for the filled patch in the legend
        ax.plot([], [], color=style['c'], lw=6, alpha=0.6, label=style['lab'])

    r = C['ref']
    ax.plot(r['H0'], 1.5 * r['Om'] - 1.0, marker=r['marker'], ms=r['ms'], color=r['c'],
            ls='none', mec=r['mec'], mew=r['mew'], zorder=6, label=r['lab'])
    ax.set_xlim(*C['xlim']); ax.set_ylim(*C['ylim'])
    ax.set_xlabel(C['xlabel'], fontsize=C['fontsize_label'])
    ax.set_ylabel(C['ylabel'], fontsize=C['fontsize_label'])
    ax.tick_params(labelsize=C['fontsize_tick'])
    ax.legend(loc=C['legend_loc'], fontsize=C['fontsize_legend'],
              framealpha=C['legend_framealpha'])
    fig.tight_layout()
    return _save(fig, C['out'])


# =============================================================================
def _floorsweep_cell(tag, fl, P):
    """One floor-sweep cell, gated.  Returns None when absent, raises when unpublishable."""
    cell = '%s_%.3f_P%d' % (tag, fl, P)
    c = dc.cache_load('floorsweep', cell)
    if c is not None:
        dc.require_quality('chains/floorsweep/%s.npz' % cell, dc._unpack(c))
    return c


def _floorsweep_rows(C, P):
    """The swept cells, anchored on each arm's untruncated fit and collapsed over
    identical-sample cuts.  Returns (rows, floor_x, spans) -- see fig_floorsweep."""
    rows = {}
    for tag, _lab, _cl, _mk, _fl, zmax in C['samples']:
        pts = [dc._unpack(c) for c in
               (_floorsweep_cell(tag, fl, P)
                for fl in dc.FLOORS if zmax is None or fl <= zmax) if c is not None]
        if pts:
            rows[tag] = sorted(pts, key=lambda r: r['z_floor'])
    if not rows:
        raise SystemExit('no cached floor-sweep cells.\n'
                         'Run: python dovekie_cosmography.py --task floorsweep')

    # Anchor each curve on its own untruncated fit -- see FLOORSWEEP['baselines'].  The
    # x is the sample's ACTUAL minimum redshift (0.073340 for DES, 0.025090 for
    # DES+legacy), not the rounded 0.073/0.025, so that the marker and the dashed line
    # coincide to the pixel; the manuscript's rounded quotes are unaffected.
    floor_x = {}
    for tag, _lab, _cl, _mk, fl, _zm in C['samples']:
        smp = C.get('baselines', {}).get(tag, (None,))[0]
        floor_x[tag] = float(dc.sn_dataset(smp)['z'].min()) if smp else fl
    for tag, (smp, jd) in C.get('baselines', {}).items():
        if tag not in rows:
            continue
        r = dict(_legacy_cell(smp, jd, 'cosmo'))
        r['z_floor'] = floor_x[tag]
        # drop any swept cell at or below z_min: it is the same data, hence the same fit
        rows[tag] = sorted([r] + [q for q in rows[tag] if q['z_floor'] > floor_x[tag]],
                           key=lambda q: q['z_floor'])

    # Several z_floor values can select an IDENTICAL supernova sample, because no
    # supernova lies between them: des_only 0.080 = 0.085 = 0.090 (n_sn = 1539) and
    # 0.120 = 0.130 (n_sn = 1533).  Those extra points are the SAME fit re-plotted, and
    # they draw a flat stretch that no cut produced -- exactly the stretch the text reads
    # stability off.  Keep the LOWEST floor of each identical-sample group and remember
    # the highest floor it stands for; the whisker below says "invariant across these
    # cuts" instead of "measured three times".  Grouped on n_sn, never on H0: two cuts
    # returning the same number is the claim, not the test.
    spans = {}
    for tag, rs in rows.items():
        kept, seen = [], None
        for r in rs:
            n = r.get('n_sn')
            if kept and n is not None and n == seen:
                spans[(tag, kept[-1]['z_floor'])] = r['z_floor']
                continue
            kept.append(r)
            seen = n
        rows[tag] = kept
    return rows, floor_x, spans


def _floorsweep_panels(C, axes, rows, floor_x, spans, legend_loc):
    """Draw the sweep onto whichever of the two panels is asked for.

    `axes` maps 'H0' and/or 'q0' to an Axes; `legend_loc` maps the same keys to a
    matplotlib loc, or to None for a panel that carries no legend.  Both the combined
    two-panel float and the two standalone single-column files go through here, so the
    split figures cannot drift from the one the manuscript used to carry."""
    axH, axq = axes.get('H0'), axes.get('q0')

    # reference bands, with the NAME below the band (above it, Breuval collided with the
    # rising DES points)
    if axH is not None:
        for val, sig, nm, cl, side, x in C['refs']:
            axH.axhspan(val - sig, val + sig, color=cl, alpha=0.15, zorder=0)
            axH.axhline(val, color=cl, ls=':', lw=1.3, zorder=1)
            x_lab, ha = ((C['ref_label_floor'] + 0.004, 'left') if x is None
                         else (x, 'center'))
            dy, va = ((sig + 0.12, 'bottom') if side == 'above' else (-sig - 0.12, 'top'))
            axH.text(x_lab, val + dy, nm, color=cl, va=va, ha=ha,
                     fontsize=C['fontsize_annot'], zorder=5)

    # q0 has no error bar here: it is q0 = 3/2 Om - 1 evaluated at the SPA Om, i.e. "where
    # a flat-LCDM CMB fit sits", not a measurement in this coordinate.  Hence a bare line.
    if axq is not None and C.get('q0_ref'):
        val, nm, cl = C['q0_ref']
        axq.axhline(val, color=cl, ls=':', lw=1.3, zorder=1)
        import matplotlib.transforms as mtransforms
        off = mtransforms.ScaledTranslation(0.0, C.get('q0_ref_dy_pt', 0.0) / 72.0,
                                            axq.figure.dpi_scale_trans)
        axq.text(C['ref_label_floor'] + 0.004, val, nm, color=cl, va='top', ha='left',
                 fontsize=C['fontsize_annot'], zorder=5,
                 transform=axq.transData + off)

    # the floor lines go down first, so no data point is drawn under one.  ONE line per
    # distinct floor: the Delta-sampled arm is the same supernovae as `full` and so has the
    # same 0.025 floor, and drawing it again would just paint blue over the red line and
    # invent a third survey limit that does not exist.
    for ax in axes.values():
        drawn = set()
        for tag, _lab, cl, _mk, _fl, _zm in C['samples']:
            fl = floor_x[tag]
            if tag in rows and fl not in drawn:
                ax.axvline(fl, color=cl, ls='--', lw=1.0, alpha=0.8, zorder=1)
                drawn.add(fl)

    # WHISKERS ARE OFF.
    # `spans` is still computed -- it is what collapses the identical-sample cuts down to
    # one plotted point each, which is the part that must stay: several z_floor values can
    # select the SAME supernovae, and plotting them all draws a flat stretch that no cut
    # produced.  What is no longer drawn is the thin horizontal marker that showed how far
    # such a point reaches; the line simply runs from the kept point to the next one.
    #
    # DRAWN in reverse so the baseline (DES, first in the config) ends up on TOP, but the
    # legend is built from the config order so it still reads DES first.
    handles = {}
    for tag, lab, cl, mk, _fl, _zm in reversed(C['samples']):
        if tag not in rows:
            continue
        z = [r['z_floor'] for r in rows[tag]]
        for key, ax in axes.items():
            h = ax.errorbar(z, [r[key + '_mean'] for r in rows[tag]],
                            yerr=[r[key + '_std'] for r in rows[tag]], fmt=mk + '-',
                            color=cl, capsize=C['capsize'], lw=C['lw'], ms=C['ms'],
                            zorder=4, label=lab)
            handles.setdefault(tag, h)
    order = [t for t, _l, _c, _m, _f, _z in C['samples'] if t in handles]
    leg_h = [handles[t] for t in order]
    leg_l = [dict((s[0], s[1]) for s in C['samples'])[t] for t in order]

    for key, ax in axes.items():
        ax.set_xlabel(C['xlabel'], fontsize=C['fontsize_label'])
        ax.set_ylabel(C['ylabel'] if key == 'H0' else C['q0_ylabel'],
                      fontsize=C['fontsize_label'])
        lim = C['ylim'] if key == 'H0' else C['q0_ylim']
        if lim:
            ax.set_ylim(*lim)
        if C['xlim']:
            ax.set_xlim(*C['xlim'])
        if legend_loc.get(key):
            ax.legend(leg_h, leg_l, loc=legend_loc[key], fontsize=C['fontsize_legend'],
                      framealpha=C['legend_framealpha'])
        # an extra tick at 0: with no gridlines, the panels otherwise start at 0.02 and
        # the eye has no origin to hang the trend on
        if C.get('extra_xtick') is not None:
            lo0, hi0 = ax.get_xlim()
            want = min(C['extra_xtick'], lo0), hi0
            # `get_xticks` hands back the LOCATOR's ticks, which run one step past the
            # view -- 0.35 for a view ending at 0.305 -- and `set_xticks` then pins them
            # as a FixedLocator whose own view limits reach that outermost tick.  That is
            # what quietly widened the panel back to 0.35 when xlim was first pinned to
            # the grid.  Ticks outside the window are dropped and BOTH limits are set
            # again afterwards, so the tick surgery cannot move the axis.
            ax.set_xticks(sorted(set([t for t in ax.get_xticks()
                                      if want[0] <= t <= want[1]] + [C['extra_xtick']])))
            ax.set_xlim(*want)


def fig_floorsweep():
    """Recovered (H0, q0) as the survey floor rises.  Only the DATA change between points.

    Two panels rather than two y-axes on one: H0 and q0 are different quantities on
    different scales, and overlaying them would invite reading a crossing as meaningful.

    This is the COMBINED two-panel float.  The manuscript now tiles the two standalone
    single-column files below instead; this one is kept so an un-updated
    \\includegraphics still builds, and as the reference the split must reproduce."""
    C = FLOORSWEEP
    rows, floor_x, spans = _floorsweep_rows(C, dc.BASELINE['P'])
    fig, (axH, axq) = plt.subplots(1, 2, figsize=C['figsize'], sharex=True)
    _floorsweep_panels(C, dict(H0=axH, q0=axq), rows, floor_x, spans,
                       dict(H0=C['legend_loc'], q0=C.get('q0_legend_loc')))
    fig.tight_layout()
    return _save(fig, C['out'])


def _fig_floorsweep_split(key):
    """One panel of the sweep as its OWN single-column file.

    The author tiles fig:floorsweep-real-new vertically with two \\includegraphics
    commands inside a one-column `figure`, so each panel has to stand on its own: same
    x range in both (z_floor is the shared independent variable, which is what makes the
    vertical stack read), and its own legend, since neither panel can be read through
    the other any more."""
    C = FLOORSWEEP
    rows, floor_x, spans = _floorsweep_rows(C, dc.BASELINE['P'])
    fig, ax = plt.subplots(figsize=C['figsize_split'])
    _floorsweep_panels(C, {key: ax}, rows, floor_x, spans,
                       {key: C['legend_loc_split'][key]})
    fig.tight_layout()
    return _save(fig, C['out_split'][key])


def fig_floorsweep_h0():
    """The H0 panel alone -- floorsweep_real_H0."""
    return _fig_floorsweep_split('H0')


def fig_floorsweep_q0():
    """The q0 panel alone -- floorsweep_real_q0."""
    return _fig_floorsweep_split('q0')


# =============================================================================
def _forest(ax, rows, xlim, xlabel, zero=None, top_pad=None, ms=None):
    """rows = list of (label, series, note), or None for a spacer row.

    series = list of (value, sigma, colour, marker, legend label or None); several entries
    put several backgrounds on ONE row, split vertically by LEG['h0_dy']."""
    C = LEG
    top_pad = C['top_pad'] if top_pad is None else top_pad
    ms = C['ms'] if ms is None else ms
    ys, labels, seen = [], [], set()
    y = 0
    for r in rows:
        if r is None:
            y -= 1
            continue
        lab, series, note = r
        dy = C['h0_dy'] if len(series) > 1 else 0.0
        y0 = y + dy * (len(series) - 1) / 2.0
        for i, (v, s, col, mk, leglab) in enumerate(series):
            kw = {}
            if leglab and leglab not in seen:
                kw['label'] = leglab
                seen.add(leglab)
            ax.errorbar(v, y0 - i * dy, xerr=s, fmt=mk, color=col, ms=ms,
                        capsize=C['capsize'], lw=C['lw'], zorder=3, **kw)
        if note:
            ax.annotate(note, (xlim[1], y), xytext=(-4, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=C['fontsize_annot'],
                        color=series[0][2])
        ys.append(y); labels.append(lab)
        y -= 1
    if zero is not None:
        ax.axvline(zero, color=C['col']['zero'], lw=1.0, ls='--', zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=C['fontsize_tick'])
    ax.set_ylim(y + 0.4, top_pad)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=C['fontsize_label'])
    ax.tick_params(axis='x', labelsize=C['fontsize_tick'])


def _efst_band(ax):
    ax.axvspan(LEG['efst']['val'] - LEG['efst']['half'],
               LEG['efst']['val'] + LEG['efst']['half'],
               color=LEG['col']['efst'], zorder=0)


def _legend_above(ax, ncol):
    """Outside the axes, sitting on the top spine ('lower left' is the corner of the
    legend pinned to the anchor), so the rows keep the full panel height."""
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0,
              frameon=False, fontsize=LEG['fontsize_legend'], ncol=ncol)


def _ladder_json():
    p = os.path.join(dc.RESULT_DIR, 'legacy_delta_ladder.json')
    if not os.path.exists(p):
        raise SystemExit('no Delta ladder.\n'
                         'Run: python dovekie_cosmography.py --task legacy')
    return json.load(open(p))


def fig_ladder():
    """TWO SEPARATE FILES: ladder_rungs (the four references) and ladder_splits (by
    subsample).

    They used to be one two-panel `ladder.pdf` stacked here with `subplots(2,1)`.  Panels
    are combined in LaTeX with two \\includegraphics, not externally, so that the article
    can re-lay them out.  The combined `ladder.pdf` is still written as well, unchanged.

    The four references agree on WHAT the offset is and disagree only on how significant
    it is, which is the point of drawing them together: the spread is bought by the
    assumed functional form below the DES floor, not by the data.

    The lower panel draws each subsample as its offset MINUS the offset of the rest, both
    fitted at once, so the DES sub-floor extrapolation error -- common to the two groups
    and dominant in each absolute sigma -- cancels instead of being counted twice.  Two
    absolute Deltas therefore cannot be differenced by eye, which is why this panel is not
    a redrawing of the one above it."""
    C = LEG
    lad = _ladder_json()

    def draw_rungs(ax):
        rows = []
        for key, lab, ckey in C['rungs']:
            st = lad[key]['splits']['all legacy']
            rows.append((lab, [(st['Delta'], st['sig'], C['col'][ckey], 'o', None)],
                         '%.1f$\\sigma$' % st['z']))
        _efst_band(ax)
        _forest(ax, rows, C['delta_xlim'], C['delta_xlabel'], zero=0.0)

    def draw_splits(ax):
        head = lad[C['headline_rung']].get('contrasts', {})
        if not head:
            raise SystemExit('the Delta ladder has no `contrasts` block -- it predates '
                             'the\njoint two-offset fit.  Rerun: '
                             'python dovekie_cosmography.py --task legacy')
        rows = []
        for skey, slab in C['splits']:
            ct = head.get(skey)
            if ct is None:
                continue                 # 'all legacy' is the reference, not a row
            rows.append((slab, [(ct['diff'], ct['sig'], C['col']['split'], 'o', None)],
                         '%.1f$\\sigma$' % abs(ct['z'])))
        _forest(ax, rows, C['contrast_xlim'], C['contrast_xlabel'], zero=0.0)

    out = []
    for name, draw, fs in (('ladder_rungs', draw_rungs, C['figsize']['ladder_rungs']),
                           ('ladder_splits', draw_splits, C['figsize']['ladder_splits'])):
        fig, ax = plt.subplots(figsize=fs)
        draw(ax)
        fig.tight_layout()
        out.append(_save(fig, name, tight_bbox=True))

    # the combined float, kept only so an un-updated \includegraphics{ladder} still builds
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=C['figsize']['ladder'],
        gridspec_kw=dict(height_ratios=C['height_ratio']['ladder']))
    draw_rungs(axT)
    draw_splits(axB)
    fig.tight_layout()
    out.append(_save(fig, 'ladder', tight_bbox=True))
    return out


def _legacy_cell(smp, joint, bg):
    cell = '%s_%s_%s_P%d' % (bg, smp, 'joint' if joint else 'fixed', dc.BASELINE['P'])
    c = dc.cache_load('legacy', cell)
    if c is None:
        raise SystemExit('no cached legacy cells.\n'
                         'Run: python dovekie_cosmography.py --task legacy')
    r = dc._unpack(c)
    dc.require_quality('chains/legacy/%s.npz' % cell, r)
    return r


def fig_legacy_h0():
    """Baseline H0 under each treatment of the legacy zero-point (main panel), with the
    jointly sampled Delta underneath.

    The Delta gets its own short panel rather than a separate figure because it is the
    offset carried by the row directly above it -- that row's nuisance parameter read off
    the same posterior, not a separate measurement -- so it keeps the same two series and
    the same two colours."""
    C = LEG

    def draw_h0(ax, legend=True):
        # bands first: they set the legend order (Breuval, SPA, then the two backgrounds)
        ax.axvspan(C['shoes']['val'] - C['shoes']['sig'],
                   C['shoes']['val'] + C['shoes']['sig'],
                   color=C['col']['shoes'], alpha=0.16, zorder=0,
                   label=C['shoes']['label'])
        ax.axvspan(C['planck']['val'] - C['planck']['sig'],
                   C['planck']['val'] + C['planck']['sig'],
                   color=C['col']['planck'], alpha=0.28, zorder=0,
                   label=C['planck']['label'])
        rows = []
        for row in C['h0_rows']:
            if row is None:
                rows.append(None)
                continue
            (smp, joint), rlab, ckey = row
            series = [(_legacy_cell(smp, joint, bg)['H0_mean'],
                       _legacy_cell(smp, joint, bg)['H0_std'],
                       mcol or C['col'][ckey], mk, mlab)
                      for bg, mlab, mk, mcol in C['h0_models']]
            rows.append((rlab, series, None))
        _forest(ax, rows, C['h0_xlim'], C['h0_xlabel'], ms=C['ms_legacy'])
        if legend:
            _legend_above(ax, ncol=2)

    def draw_delta(ax):
        smp, joint = C['jdelta_row']
        # colour by MODEL, exactly as the top panel: cosmography black, LCDM dark blue.
        series = [(_legacy_cell(smp, joint, bg)['Delta_mean'],
                   _legacy_cell(smp, joint, bg)['Delta_std'],
                   mcol or C['col']['joint'], mk, None)
                  for bg, _mlab, mk, mcol in C['h0_models']]
        _efst_band(ax)
        _forest(ax, [(C['jdelta_label'], series, None)], C['jdelta_xlim'],
                C['delta_xlabel'], zero=0.0, top_pad=C['top_pad_jdelta'],
                ms=C['ms_legacy'])

    # the two halves as standalone files, combined in LaTeX
    out = []
    for name, draw, fs in (('h0_rows', draw_h0, C['figsize']['h0_rows']),
                           ('h0_delta', draw_delta, C['figsize']['h0_delta'])):
        fig, ax = plt.subplots(figsize=fs)
        draw(ax)
        fig.tight_layout()
        out.append(_save(fig, name, tight_bbox=True))

    # the combined float, kept so an un-updated \includegraphics{h0} still builds
    fig, (axH, axD) = plt.subplots(
        2, 1, figsize=C['figsize']['h0'],
        gridspec_kw=dict(height_ratios=C['height_ratio']['h0']))
    draw_h0(axH)
    draw_delta(axD)
    fig.tight_layout()
    out.append(_save(fig, 'h0', tight_bbox=True))
    return out


# =============================================================================
def fig_ap():
    """F_AP divided by the SPA flat-LCDM reference.

    Dividing by a reference is not cosmetic here: F_AP^LCDM depends only on Omega_m --
    H0 cancels -- so the ratio panel is calibration-free in exactly the same sense as the
    test itself, and it amplifies differences invisible on the raw F_AP scale.

    Only the three in-range DESI redshifts are shown.  Beyond the SN z_max the prediction
    is an extrapolation whose D_H error exceeds D_H itself, F_AP diverges, and the
    near-zero chi2 it returns would measure a LACK of predictive power rather than
    agreement -- plotting it invites exactly that misreading."""
    C = AP
    p = os.path.join(dc.RESULT_DIR, 'ap_table.json')
    if not os.path.exists(p):
        raise SystemExit('no AP results.\nRun: python dovekie_cosmography.py --task ap')
    res = {r['sample']: r for r in json.load(open(p))}
    P = dc.BASELINE['P']

    zg = np.linspace(*C['zgrid'])
    ref_g = dc.fap_lcdm(zg, Om=C['ref_om'])
    fig, ax = plt.subplots(figsize=C['figsize'])
    ax.axhline(1.0, color=C['ref_line_color'], ls='--', lw=1.0, zorder=1)

    for name in C['order']:
        r = res.get(name)
        c = dc.cache_load('ap', '%s_P%d' % (name, P))
        if r is None or c is None:
            continue
        dc.require_quality('chains/ap/%s_P%d.npz' % (name, P),
                           json.loads(str(c['diag_json'])))
        # F_AP = H(z) * int_0^z dz'/H(z') is invariant under H -> lambda H, so the band
        # needs no calibration -- but the integral still has to be taken on the same fixed
        # grid the sampler used, not on the plotting grid.
        zz = np.linspace(0.0, zg.max() * 1.02, dc.BASELINE['NGRID'])
        coef = dc.ap_coefficients(c)          # each draw keeps its own H0: see there
        F = np.empty((len(coef), len(zg)))
        for i, cc in enumerate(coef):
            Hz = np.clip(dc.H_poly(zz, cc), dc.H_FLOOR, None)
            DM, DH, _ = dc.distances(zz, Hz, zg)
            F[i] = DM / DH
        lo, med, hi = np.percentile(F / ref_g[None, :], [16, 50, 84], axis=0)
        ax.plot(zg, med, color=C['colors'][name], lw=C['lw'], zorder=4,
                label=r'%s shape ($p{=}%.2f$)' % (C['labels'][name], r['p']))
        ax.fill_between(zg, lo, hi, color=C['colors'][name], alpha=C['band_alpha'],
                        lw=0, zorder=2)

    any_row = next(iter(res.values()))
    z = np.array(any_row['z'])
    ref_c = dc.fap_lcdm(z, Om=C['ref_om'])
    d = C['desi']
    ax.errorbar(z, np.array(any_row['fap_desi']) / ref_c,
                yerr=np.array(any_row['sig_desi']) / ref_c, fmt=d['marker'],
                color=d['color'], ms=d['ms'], capsize=d['capsize'], zorder=6,
                label=d['label'])

    ax.set_xlabel(C['xlabel'], fontsize=C['fontsize_label'])
    ax.set_ylabel(C['ylabel'], fontsize=C['fontsize_label'])
    ax.set_xlim(*C['xlim'])
    ax.tick_params(labelsize=C['fontsize_tick'])
    ax.legend(frameon=False, fontsize=C['fontsize_legend'], loc=C['legend_loc'])
    fig.tight_layout()
    return _save(fig, C['out'])


FIGURES = {'h0_q0': fig_h0_q0, 'floorsweep': fig_floorsweep,
           'floorsweep_H0': fig_floorsweep_h0, 'floorsweep_q0': fig_floorsweep_q0,
           'ladder': fig_ladder, 'legacy_h0': fig_legacy_h0, 'ap': fig_ap}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    ap.add_argument('--figure', choices=sorted(FIGURES), nargs='*')
    a = ap.parse_args(argv)
    print(dc.banner())
    for n in (a.figure or sorted(FIGURES)):
        try:
            FIGURES[n]()
        except SystemExit as e:
            print('[%s] %s' % (n, e))


if __name__ == '__main__':
    main()
