r"""
make_adot.py -- the adot(z) = H(z)/(1+z) reconstruction (Fig. 3).

READS ONLY.  It never samples.

    python make_adot.py                 # results/adot_reconstruction.{pdf,png}
    python make_adot.py --split         # ALSO the 3-panel version, adot_panels.{pdf,png}

WHAT THIS IS
The cosmographic fit carries H(z) = sum_k c_k y^k, y = z/(1+z), as a per-sample object, so
adot(z) = H(z)/(1+z) can be evaluated at any z for every posterior draw and reduced to a
median and a 1-sigma-equivalent (15.865 / 84.135 percentile) band.  Three cases:

    DES                          chains/legacy/cosmo_des_only_fixed_P4.npz   (no legacy)
    DES+legacy                   chains/legacy/cosmo_full_fixed_P4.npz       (face value)
    DES+legacy (recalibrated)    chains/legacy/cosmo_full_joint_P4.npz       (Delta free)

NOT DYNESTY.  The cosmography baseline is
NUTS -- dynesty is used for the EVIDENCE only (Table 2).  These chains are equal-weight, so
the percentiles below are unweighted, which is the correct reduction for them.  Nothing is
lost: the NUTS chains are the ones Table 1 and the H0-q0 figure are built from.

POLYNOMIAL ORDER.  This file reads the baseline P = k_max = 4 chains only; the other
orders are robustness tests (Section 4 and Appendix B of the paper).

BELOW THE SURVEY FLOOR.  Each sample's own lowest SN sets where the reconstruction is
supported by data.  DES alone starts at z=0.0733; legacy pulls that down to z=0.0251.
Below its floor a curve is an EXTRAPOLATION of the polynomial, and for DES alone that
extrapolation is precisely the prior-dominated regime.  It is therefore drawn faded and
dashed, never solid.  Do not "clean this up" by extending the solid line to z=0.

COLOURS come from make_figures.SERIES -- the one series palette (DES=black,
DES+legacy=red).  The recalibrated case used to be drawn in that same red, dashed, on the
grounds that it is the same data under a different treatment; it now has its OWN colour, the blue of SERIES['full_dleg'], which is the same
blue the h0_q0 figure and the appendix triangles give the Delta-varied contour.  It keeps
the dashed line as well, so the treatment is still readable in greyscale.  Its LABEL is
read off SERIES too (2026-08-20): it used to say `recalibrated $\Delta$`, a bare symbol and
a wording no other figure used, where every other figure says `DES+legacy
($\Delta M_\mathrm{legacy}$)`.

ONE PANEL.  This figure was a 1x2 pair -- absolute adot(z) on the left, the ratio to SPA
LCDM on the right.  Only the left panel is kept, so the ratio
panel is gone; `--split` (one case per panel, absolute) is untouched.

THE INSET.  The z < 0.2 regime --
the one the Hubble tension is decided in -- is compressed into the left edge of the full
z < 1.14 panel, so it is repeated in an inset over exactly that range.  The inset is drawn
by the SAME `_draw` as the main panel, so the solid/dotted floor split is identical and an
extrapolation still cannot be mistaken for a measurement.  Its y-range is read off the
curves themselves (INSET_ZMAX), never hard-coded, so it follows the chains if they move.

LEGEND ORDER.  SPA LCDM is plotted FIRST so it sits underneath the
reconstructions, but it comes LAST in the legend, since it is the lowest
curve at z -> 0.  The handles are therefore rotated after plotting; do not "fix" this by
plotting SPA last, which would put the orange line on top of the bands.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Stroke-count guard (strokecheck.py).  A page carrying
# thousands of SEPARATE stroked paths can defeat a printer's RIP even though it
# renders instantly and the file is small, and nothing in normal use surfaces
# that.  This patches Figure.savefig, so every savefig call below is left exactly
# as written and each PDF reports its stroke count against the limit as it is
# saved.  Warnings go to stderr.
import strokecheck
strokecheck.enable()

import dovekie_cosmography as dc
from make_figures import SPA_H0, SPA_OM, SPA_LABEL_TEX, SERIES, DPI

OUTDIR = dc.RESULT_DIR   # honours RESULTS_DIR like every other renderer (was 'results')

# 1-sigma-equivalent percentiles of a Gaussian, as asked for.
PCT_LO, PCT_HI = 15.865, 84.135

# The three cells.  `floor` is the lowest SN redshift of that sample -- the point below
# which the curve stops being data and starts being polynomial extrapolation.
# The cells, named through dc.BASELINE['P'] rather than a hard-coded P4, and loaded
# through dc.cache_load rather than a relative path and a raw np.load.  The old form read
# 'chains/legacy/..._P4.npz' from the current working directory, so it silently found
# nothing outside the repository root, kept reading P=4 after the baseline order changed,
# and accepted caches the main loader would have refused.
P = dc.BASELINE['P']
CASES = [
    dict(key='des_only',
         cell='cosmo_des_only_fixed_P%d' % P,
         lab=SERIES['des_only']['lab'],
         c=SERIES['des_only']['c'], ls='-', floor=0.0733, z=3),
    dict(key='full',
         cell='cosmo_full_fixed_P%d' % P,
         lab=SERIES['full']['lab'],
         c=SERIES['full']['c'], ls='-', floor=0.0251, z=2),
    dict(key='full_joint',
         cell='cosmo_full_joint_P%d' % P,
         lab=SERIES['full_dleg']['lab'],
         c=SERIES['full_dleg']['c'], ls='--', floor=0.0251, z=1),
]

ZMAX = 1.1442           # the SN z_max, shared by both samples.  No extrapolation above it.
NZ = 400

INSET_ZMAX = 0.2        # the inset range: the low-z regime, in detail.
# (x0, y0, w, h) in axes coords.  The inset reaches TOWARDS the panel's top right corner
# but deliberately stops short of it: a margin above and to the
# right, so the two top edges do not coincide and the inset's last x tick label ("0.20")
# stays inside the panel.  The legend lives INSIDE the inset, which is what frees the
# corner; nothing else has to be kept clear of it.
INSET_BBOX = (0.23, 0.26, 0.71, 0.67)
INSET_PAD = 0.04        # fraction of the inset y-span left as margin above and below
INSET_LABEL_FRAC = 0.06  # panel height taken by the x tick labels hanging below the inset
INSET_GAP_FRAC = 0.015   # and the clear space kept between those labels and any curve

ALPHA_BAND = 0.22
ALPHA_BAND_EXTRAP = 0.10
LW = 1.8

FS_LABEL, FS_TICK, FS_LEG = 14, 12, 11


def adot_draws(d, zg):
    """adot(z) = H(z)/(1+z) for every posterior draw.  Returns (ndraw, nz)."""
    H0 = np.asarray(d['H0'], float)
    c = H0[:, None] * np.column_stack([np.ones_like(H0),       # C_0 = 1
                                       np.asarray(d['C'], float)])   # C_1 .. C_P
    y = dc.y_of_z(zg)
    # H = H0 sum_k C_k y^k, vectorised over draws
    H = c @ np.power(y[None, :], np.arange(c.shape[1])[:, None])
    return H / (1.0 + zg)[None, :]


def band(draws):
    return (np.percentile(draws, PCT_LO, axis=0),
            np.median(draws, axis=0),
            np.percentile(draws, PCT_HI, axis=0))


def adot_spa(zg):
    return dc.H_lcdm(zg, SPA_H0[0], SPA_OM) / (1.0 + zg)


def z_acceleration(draws, zg):
    """adot has a MINIMUM where abar = 0, so the argmin is the acceleration-onset
    redshift of that draw -- read off the reconstruction, with no model assumed.
    Draws whose minimum sits at a grid EDGE have no interior turning point in range and
    are returned as NaN rather than being pinned to the boundary."""
    i = np.argmin(draws, axis=1)
    z = zg[i]
    z[(i == 0) | (i == len(zg) - 1)] = np.nan
    return z


def _draw(ax, zg, lo, med, hi, case, ratio=False):
    """One case, split at its floor: solid+filled where SNe support it, dashed+fainter
    below, so an extrapolation can never be mistaken for a measurement."""
    sup = zg >= case['floor']
    ext = zg <= case['floor']
    col, z = case['c'], case['z']
    ax.fill_between(zg[sup], lo[sup], hi[sup], color=col, alpha=ALPHA_BAND, lw=0, zorder=z)
    ax.plot(zg[sup], med[sup], color=col, ls=case['ls'], lw=LW, zorder=z + 10,
            label=case['lab'])
    if ext.sum() > 1:
        ax.fill_between(zg[ext], lo[ext], hi[ext], color=col, alpha=ALPHA_BAND_EXTRAP,
                        lw=0, zorder=z)
        ax.plot(zg[ext], med[ext], color=col, ls=':', lw=1.2, alpha=0.8, zorder=z + 10)


def _legend_spa_last(ax, **kw):
    """SPA is plotted first (so it lies under the bands) but goes
    LAST in the legend.  Rotate the handles rather than the plotting order."""
    h, l = ax.get_legend_handles_labels()
    if len(h) > 1:
        h, l = h[1:] + h[:1], l[1:] + l[:1]
    ax.legend(h, l, **kw)


def _inset_ylim(zg, ref, got):
    """The inset's y-range, read off the curves it will show rather than hard-coded, so it
    follows the chains if they move.  Returned separately from the drawing because the MAIN
    panel's y-range is derived from it -- see VERTICAL PARITY in `build`."""
    m = zg <= INSET_ZMAX
    ylo = min(float(lo[m].min()) for _, lo, _, _ in got)
    yhi = max(float(hi[m].max()) for _, _, _, hi in got)
    ylo, yhi = min(ylo, float(ref[m].min())), max(yhi, float(ref[m].max()))
    pad = INSET_PAD * (yhi - ylo)
    return ylo - pad, yhi + pad


def _inset_geometry(zg, ref, got, inset_ylim, ylo_m, yhi_need):
    """The inset's box and the panel's top, at the least panel range that gives BOTH
    (i) vertical parity >= 1 and (ii) a footprint -- its x tick labels included -- clear of
    every curve under it.  The box keeps INSET_BBOX's left edge, width and top;
    only its bottom may rise.  Parity alone wants the box tall,
    clearance wants its bottom high, so the least range is where the two requirements
    cross, D = d_in / (top - y) = d_c / (y - lab), unless INSET_BBOX's own bottom already
    clears."""
    x0, y0, w, h = INSET_BBOX
    top, lab = y0 + h, INSET_LABEL_FRAC + INSET_GAP_FRAC
    m = zg >= x0 * ZMAX
    under = max([float(hi[m].max()) for _, _, _, hi in got] + [float(ref[m].max())])
    d_in, d_c = inset_ylim[1] - inset_ylim[0], under - ylo_m
    yb = max(y0, (d_c * top + d_in * lab) / (d_c + d_in))
    span = max(d_in / (top - yb), d_c / (yb - lab), yhi_need - ylo_m)
    return (x0, yb, w, top - yb), ylo_m + span


def _add_inset(ax, zg, ref, got, ylim, bbox=INSET_BBOX):
    """The z < INSET_ZMAX blow-up, drawn by the same `_draw` as
    the main panel so the floor split (solid where SNe support it, dotted below) carries
    over unchanged."""
    m = zg <= INSET_ZMAX
    axin = ax.inset_axes(bbox)
    axin.plot(zg[m], ref[m], color='#e8a33d', lw=1.6, ls='-', zorder=0,
              label=SPA_LABEL_TEX + r' ($\Lambda$CDM)')
    for case, lo, med, hi in got:
        _draw(axin, zg[m], lo[m], med[m], hi[m], case)

    axin.set_xlim(0.0, INSET_ZMAX)
    axin.set_ylim(*ylim)
    # Coarser ticks than the default: the inset is now large enough that matplotlib offers
    # eight labels a side, which reads as clutter next to the main panel's five.
    axin.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    axin.yaxis.set_major_locator(MaxNLocator(5))
    axin.tick_params(labelsize=FS_TICK - 2, length=3, pad=2)
    axin.grid(alpha=0.15, lw=0.5)
    for sp in axin.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color('0.35')
    # No `indicate_inset_zoom` box: the inset's y-range is nearly the whole panel's, so the
    # indicator degenerates into a full-height rectangle drawn across the reconstructions.
    # The inset carries its own z axis, which says where it is.
    # The legend moved in here (2026-09-21): every case descends left to right over
    # z < INSET_ZMAX, so the inset's upper right is empty, and vacating the main panel's
    # upper right is what lets the inset reach the corner.
    _legend_spa_last(axin, fontsize=FS_LEG - 1, loc='upper right', frameon=True,
                     framealpha=0.92, edgecolor='0.55', fancybox=False, borderpad=0.6,
                     handlelength=_handle_whole_dashes(FS_LEG - 1, 3))
    return axin


def _handle_whole_dashes(fontsize, n):
    """Legend handle length (in font sizes) holding exactly `n` whole dashes of the '--'
    line at width LW.  The default 2.0 em gave the dashed DES+legacy (Delta M_legacy) key
    two dashes and a 0.9 pt stub, which reads as dot-dash.  Read
    off rcParams, so it follows any change to the dash pattern or line width."""
    on, off = matplotlib.rcParams['lines.dashed_pattern'][:2]
    if matplotlib.rcParams['lines.scale_dashes']:
        on, off = on * LW, off * LW
    return (n * on + (n - 1) * off) / fontsize


def build(split=False):
    zg = np.linspace(0.0, ZMAX, NZ)
    ref = adot_spa(zg)

    got, stats = [], {}
    for case in CASES:
        c = dc.cache_load('legacy', case['cell'])
        if c is None:
            print('[adot] MISSING or REFUSED: chains/legacy/%s.npz -- skipped'
                  % case['cell'])
            continue
        dc.require_quality('chains/legacy/%s.npz' % case['cell'],
                           json.loads(str(c['scalars_json'])))
        dr = adot_draws(c, zg)
        lo, med, hi = band(dr)
        zacc = z_acceleration(dr, zg)
        got.append((case, lo, med, hi))
        i0 = 0
        stats[case['key']] = dict(
            label=case['lab'], floor=case['floor'], ndraw=int(dr.shape[0]),
            adot0=[float(lo[i0]), float(med[i0]), float(hi[i0])],
            adot_at_floor=[float(np.interp(case['floor'], zg, lo)),
                           float(np.interp(case['floor'], zg, med)),
                           float(np.interp(case['floor'], zg, hi))],
            z_acc_median=(float(np.nanmedian(zacc)) if np.isfinite(zacc).any() else None),
            z_acc_1sig=([float(np.nanpercentile(zacc, PCT_LO)),
                         float(np.nanpercentile(zacc, PCT_HI))]
                        if np.isfinite(zacc).any() else None),
            frac_with_turning_point=float(np.isfinite(zacc).mean()))
    if not got:
        raise SystemExit('[adot] no chains found -- nothing to draw')

    # ---------------- the main figure: absolute adot(z), one panel ----------------
    # The ratio-to-SPA panel that used to sit at the right was dropped; the legend is unchanged, so the figure is a drop-in for the old left panel.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.plot(zg, ref, color='#e8a33d', lw=2.0, ls='-', zorder=0,
            label=SPA_LABEL_TEX + r' ($\Lambda$CDM)')
    for case, lo, med, hi in got:
        _draw(ax, zg, lo, med, hi, case)
    ax.set_xlabel(r'$z$', fontsize=FS_LABEL)
    ax.set_ylabel(r'$\dot{a}(z) = H(z)/(1+z)$  [km s$^{-1}$ Mpc$^{-1}$]', fontsize=FS_LABEL)
    ax.set_xlim(0.0, ZMAX)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.15, lw=0.6)

    # ---------------- VERTICAL PARITY ----------------
    # An inset must magnify, never shrink: one km s^-1 Mpc^-1 has to occupy at least as
    # many cm inside the inset as outside it.  The inset is INSET_BBOX[3] of the panel's
    # height, so the PANEL must span at least (inset span) / INSET_BBOX[3].  We buy that
    # with headroom at the top, which is empty in every case -- the curves all fall away
    # from adot(0) -- and which the inset itself then occupies.  The legend no longer
    # competes for it.  Never satisfy this by clipping the inset instead: the top of the
    # DES-only band at z -> 0 is the H_0 the paper is about.
    in_lo, in_hi = _inset_ylim(zg, ref, got)
    dat_lo = min([float(lo.min()) for _, lo, _, _ in got] + [float(ref.min())])
    dat_hi = max([float(hi.max()) for _, _, _, hi in got] + [float(ref.max())])
    pad = INSET_PAD * (dat_hi - dat_lo)
    ylo_m = dat_lo - pad
    bbox, yhi_m = _inset_geometry(zg, ref, got, (in_lo, in_hi), ylo_m, dat_hi + pad)
    ax.set_ylim(ylo_m, yhi_m)

    _add_inset(ax, zg, ref, got, (in_lo, in_hi), bbox)
    _check_inset(zg, ref, got, (ylo_m, yhi_m), (in_lo, in_hi), bbox)
    fig.tight_layout()
    _save(fig, 'adot_reconstruction')

    # ---------------- optional 3-panel version ----------------
    if split:
        fig, axs = plt.subplots(1, len(got), figsize=(4.4 * len(got), 4.6), sharey=True)
        axs = np.atleast_1d(axs)
        for a, (case, lo, med, hi) in zip(axs, got):
            a.plot(zg, ref, color='#e8a33d', lw=2.0, zorder=0,
                   label=SPA_LABEL_TEX + r' ($\Lambda$CDM)')
            _draw(a, zg, lo, med, hi, case)
            a.set_xlabel(r'$z$', fontsize=FS_LABEL)
            a.set_xlim(0.0, ZMAX)
            a.tick_params(labelsize=FS_TICK)
            a.grid(alpha=0.15, lw=0.6)
            _legend_spa_last(a, fontsize=FS_LEG, loc='upper right', frameon=False)
        axs[0].set_ylabel(r'$\dot{a}(z) = H(z)/(1+z)$  [km s$^{-1}$ Mpc$^{-1}$]',
                          fontsize=FS_LABEL)
        fig.tight_layout()
        _save(fig, 'adot_panels')

    p = os.path.join(OUTDIR, 'adot_stats.json')
    with open(p, 'w') as f:
        json.dump(stats, f, indent=2)
    print('[adot] wrote %s' % p)
    _report(stats)


def _check_inset(zg, ref, got, main_ylim, inset_ylim, bbox=INSET_BBOX):
    """Two things that can silently go wrong and that no eye catches reliably: the inset
    de-magnifying in adot, and the inset's footprint landing on top of the curves it is
    supposed to sit beside.  Both are printed on every run."""
    (ylo_m, yhi_m), (in_lo, in_hi) = main_ylim, inset_ylim
    parity = bbox[3] * (yhi_m - ylo_m) / (in_hi - in_lo)
    print('[adot] inset parity: %.2f cm per km/s/Mpc relative to the main panel %s'
          % (parity, 'OK' if parity >= 1.0 - 1e-9 else '-- TOO FLAT'))  # =1 by construction
    z_left = bbox[0] * ZMAX
    # the x tick labels hang below the frame, so they count as footprint too
    y_bot = ylo_m + (bbox[1] - INSET_LABEL_FRAC) * (yhi_m - ylo_m)
    m = zg >= z_left
    top = max([float(hi[m].max()) for _, _, _, hi in got] + [float(ref[m].max())])
    print('[adot] inset footprint: z > %.3f, adot > %.2f; highest curve under it %.2f (%s)'
          % (z_left, y_bot, top, 'clear' if top < y_bot else 'OVERLAPS'))


def _save(fig, base):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = os.path.join(OUTDIR, '%s.%s' % (base, ext))
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
        print('[adot] wrote %s' % p)
    plt.close(fig)


def _report(stats):
    print('\n%-28s %-22s %-22s %s' % ('case', 'adot(0) [=H0]',
                                      'adot(z_floor)', 'z_acc (adot minimum)'))
    for k, s in stats.items():
        lo, med, hi = s['adot0']
        flo, fmed, fhi = s['adot_at_floor']
        za = ('%.3f [%.3f, %.3f]  (%.0f%% of draws)'
              % (s['z_acc_median'], s['z_acc_1sig'][0], s['z_acc_1sig'][1],
                 100 * s['frac_with_turning_point'])) if s['z_acc_median'] else 'none in range'
        print('%-28s %6.2f -%.2f +%.2f      %6.2f -%.2f +%.2f      %s'
              % (k, med, med - lo, hi - med, fmed, fmed - flo, fhi - fmed, za))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', action='store_true',
                    help='also write the 3-panel (one case per panel) version')
    a = ap.parse_args()
    build(split=a.split)
