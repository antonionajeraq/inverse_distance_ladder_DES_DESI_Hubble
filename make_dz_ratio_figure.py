r"""
make_dz_ratio_figure.py -- the distance-redshift relation of the supernovae and the BAO,
divided by the SPA flat-LCDM prediction.

    python make_dz_ratio_figure.py          -> results/dz_ratio.pdf and .png

READS ONLY.  It never samples: the two numbers it takes from the posteriors are the
best-fit M_des of the two published ladder cells, read out of chains/ladder/.

WHAT IS PLOTTED

  supernovae   D_L(z) = (1 + z_HEL) * D_C(z_HD), reconstructed from the OBSERVED apparent
               magnitude with M_des held at the best fit of the relevant analysis:

                   D_L = 10 ** ( (m - M_des - 25) / 5 )   Mpc

               DES supernovae carry the DES-only M_des, legacy supernovae carry the
               DES+legacy M_des -- "the relevant analysis" in each case.  M_des errors are
               NEGLECTED, by instruction: they are an overall vertical translation of one
               series and would otherwise swamp the point-to-point structure the figure is
               about.  The legacy zero point is at FACE VALUE (Delta M_legacy = 0), as in
               the baseline ladder, so the legacy offset the paper measures is visible here
               rather than absorbed.

  BAO          D_V/r_d.  DESI DR2 publishes D_V/r_d directly at the BGS redshift and the
               (D_M/r_d, D_H/r_d) pair everywhere else; the pair is combined into
               D_V = (z D_M^2 D_H)^(1/3) with its own 2x2 covariance carried through the
               Jacobian of that combination.

  reference    Flat LCDM at SPA (SPT-3G D1 + ACT DR6 + Planck 2018; Camphuis et al. 2026):
               H0 = 67.19, Om = 0.3175, with r_d = 147.05 Mpc (Planck 2018) taken WITHOUT
               its error.  PLOTTING ONLY.  Nothing quantitative in the paper depends on it;
               it is the same reference fig:h0_q0 and fig:AP-new already draw against.

  green band   H0^SPA / H0^SH0ES = 67.19 / 73.17.  In a universe with the SPA shape but the
               SH0ES calibration every distance is smaller by that factor, so the band is
               where the points would sit if the local ladder set the scale.  Its width
               adds the two fractional errors in quadrature, treating them as independent.

BINNING.  Bin edges are LOGARITHMIC in z, separately per series, because the axis is, and
because equal-count bins put the first 129 DES supernovae into one bin running from 0.073
to 0.28 and leave the low-redshift end of DES -- the end that overlaps legacy and carries
the whole question -- unresolved.  Logarithmic edges cost a few sparse bins at that end, so
adjacent bins are MERGED forward until each holds at least MIN_COUNT supernovae; the merge
is reported, never silent.  The legacy series is binned finer, by instruction: it spans
0.025-0.087 and on a common grid would be a single point.

Within a bin the residuals are combined with inverse-variance weights from the diagonal,
and the error bar is then w^T C w on the FULL DES-Dovekie STAT+SYS covariance -- so the
weights are the well-behaved ones and the uncertainty is still the honest one.  A GLS solve
(optimal weights under the same C) is computed alongside and printed, as a check that the
choice of weights is not what the figure is showing.  They agree to < 0.4 sigma in every
bin.

Individual supernovae are the faint grey dots, drawn rasterised: 1738 separate stroked
paths would be most of a printer's per-page budget on their own (see strokecheck.py).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import strokecheck
strokecheck.enable()

import dovekie_cosmography as dc


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
# The CMB reference, and the local anchor, exactly as make_figures.py carries them.
SPA_H0 = (67.19, 0.38)
SPA_OM = 0.3175
SHOES_H0 = (73.17, 0.86)
SHOES_LEGEND = r'$H_0^{\rm SPA}/H_0^{\rm SH0ES}$'
PCT_LO, PCT_HI = 15.865, 84.135        # Gaussian-equivalent 1 sigma, as make_adot.py
RD_REF = dc.BASELINE['RD_PLANCK']        # 147.05 Mpc, no error -- plotting only

# The one series palette: DES black, DES+legacy red (make_figures.SERIES).
C_DES = 'k'
C_LEG = 'C3'
C_BAO = '#b5179e'                        # magenta.  NOT blue: blue is the recalibrated
                                         # legacy SNIa everywhere else in the paper, and
                                         # NOT green, which is the SH0ES band below.
C_DLEG = '#1f4e79'                       # SERIES['full_dleg'] -- the recalibrated legacy
                                         # blue used by every other figure in the paper

CFG = dict(
    out='dz_ratio',
    figsize=(7.6, 5.0),
    fontsize_label=15, fontsize_tick=12, fontsize_legend=11,
    ladder_cell_des='A_des_only_cosmo_P4',     # M_des for the DES points
    ladder_cell_full='A_full_cosmo_P4',        # M_des for the legacy points
    # The joint cell lives in chains/legacy/, not chains/ladder/, and carries BOTH its own
    # M_des and the offset `Delta` -- the recalibrated legacy series is built from the two
    # together, not by pasting one analysis's offset onto another's zero point.
    joint_cell=('legacy', 'cosmo_full_joint_P4'),
    nbin_des=12, nbin_leg=6, min_count=15,
    xlim=(0.02, 2.5),
    # Cropped deliberately.  The individual-supernova cloud has a ~0.17 mag scatter and
    # a handful of 3-sigma tails; showing all of it would squeeze the binned points, the
    # BAO and the SH0ES band -- everything the figure is for -- into the middle fifth of
    # the panel.  The dots are context, not the measurement.
    ylim=(0.87, 1.13),
    xlabel=r'Redshift',
    ylabel=r'$D \, / \, D_{\rm SPA, \Lambda CDM}$',
    # Individual supernovae, coloured by series: DES grey, legacy
    # a faint red keyed to its binned points.  Faintness is the alpha, not a washed-out
    # colour, so the two clouds stay the same weight on the page.
    dots=dict(s=2.2, alpha=0.30, lw=0),
    dots_des='0.62',
    dots_leg=C_LEG,
    unity_grey='0.25',
    # 🚨 SET BY HAND.  See place_legend(): the SN count is reported, never acted on.
    legend_z_left=0.06,
    sn_marker=dict(ms=6.5, capsize=3, lw=1.4, mew=1.2),
    # ls='none' is NOT optional: errorbar() with a `marker` and no `fmt` joins the points
    # with a solid line, which here drew a spurious D_V(z) "curve" through the DESI data.
    bao_marker=dict(marker='s', ls='none', ms=7.0, capsize=3.5,
                    elinewidth=1.3, mew=1.1),
    shoes=dict(color='green', alpha=0.18, lw=0.9),
    model=dict(lw=1.6, alpha=0.95),
    # The 1 sigma bands on the fitted curves are part of the article figure.  `--no-bands` reproduces the line-only version for comparison and
    # writes it under a DIFFERENT name, so it can never overwrite the figure the paper
    # uses.
    model_bands=True,
    xticks=[0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 2.5],
)

# The LAST DESI DR2 redshift, not the SN z_max.  The curves run
# solid all the way here even though the highest-redshift SNIa is at z = 1.1442: the BAO
# enter the same likelihood and constrain the expansion over 0.295 < z < 2.33, so nothing
# in this range is an extrapolation.  Only BELOW each sample's lowest supernova, where
# there is no datum of any kind, does the curve go dotted.
MODEL_ZMAX = 2.33
MODEL_FLOOR = dict(des=0.0733, full=0.0251)     # lowest SN redshift in each sample


# --------------------------------------------------------------------------- #
# The reference background
# --------------------------------------------------------------------------- #
def spa_reference(zmax=3.0, ngrid=4000):
    """(zgrid, D_C) for flat LCDM at the SPA parameters, in Mpc.

    Its own grid, NOT dc.BASELINE['NGRID']: this is a smooth analytic denominator, not a
    likelihood, and there is no cache key to match."""
    zg = np.linspace(0.0, zmax, ngrid)
    Hg = dc.H_lcdm(zg, SPA_H0[0], SPA_OM)
    return zg, Hg


def spa_dl(z_hd, z_hel):
    zg, Hg = spa_reference()
    DM, _, _ = dc.distances(zg, Hg, z_hd)
    return (1.0 + np.asarray(z_hel)) * DM


def spa_dv_over_rd(z):
    zg, Hg = spa_reference()
    _, _, DV = dc.distances(zg, Hg, z)
    return DV / RD_REF


# --------------------------------------------------------------------------- #
# Supernovae
# --------------------------------------------------------------------------- #
def _chain(cell, where='ladder'):
    c = dc.cache_load(where, cell)
    if c is None:
        raise SystemExit('no chain for %s/%s.\n'
                         'Run: python dovekie_cosmography.py --task ladder' % (where, cell))
    return c


def best_fit_M(cell, where='ladder'):
    return float(np.asarray(_chain(cell, where)['M']).mean())


def legacy_offset():
    """(shift, Delta) for the recalibrated legacy series, in magnitudes.

    `shift` is what the legacy magnitude residual moves by when the legacy supernovae are
    carried by the JOINT analysis instead of the DES+legacy one:

        shift = (M_des^joint + Delta^joint) - M_des^(DES+legacy)

    Both terms of the joint zero point are taken from the joint chain.  Using its Delta on
    top of the other cell's M_des would mix two analyses and is wrong by 0.4 mmag here --
    small, but there is no reason to carry it."""
    j = _chain(CFG['joint_cell'][1], CFG['joint_cell'][0])
    Delta = float(np.asarray(j['Delta']).mean())
    shift = (float(np.asarray(j['M']).mean()) + Delta) - best_fit_M(CFG['ladder_cell_full'])
    return shift, Delta


def model_ratio(cell, where='ladder', ngrid=1200, bands=False):
    """Median D(z) of the fitted cosmographic expansion in `cell`, over the SAME SPA
    reference the points are divided by.  Returns (z, ratio) with z[0] dropped.

    The ratio is formed PER DRAW and the median taken at each z, as
    fig:adot-reconstruction does, rather than plugging median parameters into the model:
    the median of a nonlinear function is not that function of the median.  Numerator and
    denominator use the same trapezoid on the same grid, so the factor of c, and most of
    the discretisation error, cancel in the ratio instead of being carried separately.
    D_L / D_L^SPA equals D_C / D_C^SPA -- the (1 + z_HEL) prefactor is common -- so this is
    the same quantity as the supernova points."""
    c = _chain(cell, where)
    zg = np.linspace(0.0, MODEL_ZMAX, ngrid)
    H0 = np.asarray(c['H0'], float)
    coef = H0[:, None] * np.column_stack([np.ones_like(H0),              # C_0 = 1
                                          np.asarray(c['C'], float)])    # C_1 .. C_P
    y = dc.y_of_z(zg)
    H = coef @ np.power(y[None, :], np.arange(coef.shape[1])[:, None])   # (ndraw, nz)
    Dc = _cumtrapz(1.0 / H, zg)
    Ds = _cumtrapz(1.0 / dc.H_lcdm(zg, SPA_H0[0], SPA_OM)[None, :], zg)
    r = Dc[:, 1:] / Ds[:, 1:]
    if bands:
        return (zg[1:], np.percentile(r, PCT_LO, axis=0), np.median(r, axis=0),
                np.percentile(r, PCT_HI, axis=0))
    return zg[1:], np.median(r, axis=0)


def _cumtrapz(y, x):
    """Cumulative trapezoid along the last axis, starting at zero.  Written out rather
    than imported so the two integrals above are provably the same operation."""
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x)[None, :]
    return np.concatenate([np.zeros(y.shape[:-1] + (1,)), np.cumsum(seg, axis=-1)], axis=-1)


def log_edges(z, nbin, min_count, tag=''):
    """Logarithmic bin edges over the range of z, merged forward to MIN_COUNT.

    Logarithmic because the figure's x axis is.  The merge walks left to right and drops
    the interior edge of any bin that is short, so a sparse tail is absorbed into its
    neighbour rather than plotted as a one-supernova point with a meaningless error bar;
    the last bin is merged BACKWARDS for the same reason.  What was merged is printed."""
    z = np.asarray(z)
    e = list(np.logspace(np.log10(z.min()), np.log10(z.max()), nbin + 1))
    e[0] *= (1.0 - 1e-9)
    e[-1] *= (1.0 + 1e-9)
    out = [e[0]]
    for k in range(1, len(e)):
        if int(((z >= out[-1]) & (z < e[k])).sum()) >= min_count:
            out.append(e[k])
    if out[-1] != e[-1]:
        out[-1] = e[-1]                      # sweep the short tail into the last bin
    n = [int(((z >= out[k]) & (z < out[k + 1])).sum()) for k in range(len(out) - 1)]
    print('[bin] %-7s %d log bins -> %d after the >=%d merge, counts %s'
          % (tag, nbin, len(n), min_count, n))
    return np.array(out)


def sn_residuals():
    """Per-supernova (z, ratio, is_legacy) and the full covariance, in catalogue order.

    ratio = D_L^obs / D_L^SPA with M_des fixed.  The (1 + z_HEL) prefactor is common to
    numerator and denominator and cancels exactly; it is written out anyway so that the
    expression here is the same one `sn_apparent_mag` defines and cannot drift from it."""
    d = dc.sn_dataset('full')
    leg = d['idsurvey'] != dc.DES_IDSURVEY
    M = np.where(leg, best_fit_M(CFG['ladder_cell_full']),
                 best_fit_M(CFG['ladder_cell_des']))
    mu_obs = d['m'] - M
    mu_ref = 5.0 * np.log10(spa_dl(d['z'], d['zhel'])) + 25.0
    return d['z'], mu_obs - mu_ref, leg, d['cov'], d['muerr']


def bin_series(dmu, cov, sel, edges, z, sigma):
    """Inverse-variance mean of the magnitude residual in each bin, with FULL-covariance
    errors.

    Returns (z_bin, dmu_bin, sigma_bin, n_bin).  The weights come from the diagonal and
    are therefore positive and bounded; the covariance of the weighted means is W C W^T,
    which keeps the DES-Dovekie systematics in the error bar."""
    idx = np.where(sel)[0]
    W = np.zeros((len(edges) - 1, len(dmu)))
    zb, nb = [], []
    for k in range(len(edges) - 1):
        m = idx[(z[idx] >= edges[k]) & (z[idx] < edges[k + 1])]
        w = 1.0 / sigma[m] ** 2
        w = w / w.sum()
        W[k, m] = w
        zb.append(float(np.dot(w, z[m])))
        nb.append(len(m))
    val = W @ dmu
    Cb = W @ cov @ W.T
    return np.array(zb), val, np.sqrt(np.diag(Cb)), np.array(nb), W


def gls_check(dmu, cov, W):
    """The optimal-weight (GLS) solution on the same bins, for comparison only.

    If the inverse-variance points and the GLS points agree inside their errors, the
    figure is showing the data and not the choice of weights."""
    X = (W > 0).astype(float).T                     # 1738 x nbin indicator
    Ci = np.linalg.inv(cov)
    A = X.T @ Ci @ X
    Ai = np.linalg.inv(A)
    return Ai @ (X.T @ Ci @ dmu), np.sqrt(np.diag(Ai))


# --------------------------------------------------------------------------- #
# BAO
# --------------------------------------------------------------------------- #
def bao_dv_points():
    """(z, D_V/r_d, sigma) for every DESI DR2 redshift.

    The BGS point is published as D_V/r_d and is taken as it stands.  Everywhere else
    D_V = (z D_M^2 D_H)^(1/3), and the error is the 2x2 (D_M, D_H) covariance pushed
    through d ln D_V = (2 d ln D_M + d ln D_H)/3 -- the cross term is NOT dropped: it is
    negative (rho = -0.35..-0.49) and shrinks sigma(D_V) by ~20 per cent."""
    zs, val, sig = [], [], []
    for z, (v, s) in sorted(dc.DESI_DV.items()):
        zs.append(z); val.append(v); sig.append(s)
    for z in sorted(dc.DESI_DM):
        dm = dc.DESI_DM[z][0]
        dh = dc.DESI_DH[z][0]
        C = np.array(dc.DESI_DMDH_COV[z])
        dv = (z * dm ** 2 * dh) ** (1.0 / 3.0)
        J = np.array([2.0 * dv / (3.0 * dm), dv / (3.0 * dh)])
        zs.append(z); val.append(dv); sig.append(float(np.sqrt(J @ C @ J)))
    o = np.argsort(zs)
    return np.array(zs)[o], np.array(val)[o], np.array(sig)[o]


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def place_legend(fig, ax, C, z_sn, r_sn, z_hard, r_hard):
    """Unframed legend pinned to the top of the panel, at the horizontal position set BY
    HAND in CFG['legend_z_left'].

    🚨 THIS POSITION IS A HUMAN DECISION, NOT AN OPTIMISATION.
    An earlier version sat the legend wherever it hid the fewest supernovae and moved it
    whenever the data changed.  That search survives only as a DIAGNOSTIC: every run prints
    what the chosen position costs, what the best available position would cost and where
    it is, so the choice can be revisited deliberately.  The figure goes where CFG says.
    Do not re-automate this.

    The legend is drawn once so its box can be measured with `get_window_extent()` rather
    than guessed at; the horizontal pad is constant, so the box for any anchor is the
    measured box translated.
    """
    # The green band is DRAWN first, so it sits under everything, but it is KEYED last, at
    # choice: the data series lead and the target level follows them.  The
    # sort is stable, so every other entry keeps its drawing order.
    h, l = ax.get_legend_handles_labels()
    k = sorted(range(len(l)), key=lambda i: l[i] == SHOES_LEGEND)
    leg = ax.legend([h[i] for i in k], [l[i] for i in k],
                    frameon=False, fontsize=C['fontsize_legend'],
                    loc='upper left', bbox_to_anchor=(0.0, 1.0),
                    bbox_transform=ax.transAxes)
    fig.canvas.draw()
    bb = leg.get_window_extent()
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    w = x1 - x0

    def box(a):
        """(z_lo, z_hi, r_lo, r_hi) of the legend in DATA coordinates for anchor `a`."""
        c = ax.transAxes.transform([[a + x0, y0], [a + x0 + w, y1]])
        (zlo, rlo), (zhi, rhi) = ax.transData.inverted().transform(c)
        return zlo, zhi, rlo, rhi

    def n_in(a, zz, rr):
        zlo, zhi, rlo, rhi = box(a)
        return int(((zz >= zlo) & (zz <= zhi) & (rr >= rlo) & (rr <= rhi)).sum())

    a_max = 1.0 - (x0 + w)
    a_set = float(np.interp(np.log10(C['legend_z_left']),
                            np.log10(ax.get_xlim()), [0.0, 1.0])) - x0
    if not 0.0 <= a_set <= a_max:
        print('[legend] WARNING: legend_z_left=%.3f puts the legend off the panel; '
              'clipped to the edge' % C['legend_z_left'])
        a_set = float(np.clip(a_set, 0.0, a_max))
    leg.set_bbox_to_anchor((a_set, 1.0), transform=ax.transAxes)

    zlo, zhi, rlo, _ = box(a_set)
    print('[legend] left edge z=%.3f, right edge z=%.2f, bottom %.3f -- set BY HAND in '
          "CFG['legend_z_left'], not optimised; hides %d of %d SNIa"
          % (zlo, zhi, rlo, n_in(a_set, z_sn, r_sn), len(z_sn)))
    nh = n_in(a_set, z_hard, r_hard)
    if nh:
        print('[legend] WARNING: it also covers %d binned SN or BAO point(s) -- those are '
              'the measurement.  Move legend_z_left.' % nh)
    cand = np.linspace(0.0, a_max, 241)
    cost = np.array([n_in(a, z_sn, r_sn) for a in cand])
    i = int(np.argmin(cost))
    print('[legend] diagnostic only: the fewest supernovae any position hides is %d, at '
          'left edge z=%.3f; over the 241 positions the count runs %d to %d'
          % (int(cost[i]), box(cand[i])[0], int(cost.min()), int(cost.max())))


def main(bands=True):
    C = CFG
    z, dmu, leg, cov, sigma = sn_residuals()

    e_des = log_edges(z[~leg], C['nbin_des'], C['min_count'], 'DES')
    e_leg = log_edges(z[leg], C['nbin_leg'], C['min_count'], 'legacy')
    zd, vd, sd, nd, Wd = bin_series(dmu, cov, ~leg, e_des, z, sigma)
    zl, vl, sl, nl, Wl = bin_series(dmu, cov, leg, e_leg, z, sigma)

    # the weight check, printed not plotted
    gd, gsd = gls_check(dmu, cov, Wd)
    gl, gsl = gls_check(dmu, cov, Wl)
    print('[sn] M_des  DES %.5f   DES+legacy %.5f'
          % (best_fit_M(C['ladder_cell_des']), best_fit_M(C['ladder_cell_full'])))
    print('[sn] bin      z      N     dmu(IVW)          dmu(GLS)        shift/sigma')
    for tag, zz, vv, ss, nn, gg, gs in (('DES', zd, vd, sd, nd, gd, gsd),
                                        ('legacy', zl, vl, sl, nl, gl, gsl)):
        for i in range(len(zz)):
            print('     %-7s %6.4f %5d  %+7.4f+-%6.4f  %+7.4f+-%6.4f   %+5.2f'
                  % (tag, zz[i], nn[i], vv[i], ss[i], gg[i], gs[i],
                     (vv[i] - gg[i]) / ss[i]))

    ratio_i = 10.0 ** (0.2 * dmu)
    rd_, rl_ = 10.0 ** (0.2 * vd), 10.0 ** (0.2 * vl)
    # the magnitude error bar is symmetric; the ratio's is not, and at these widths the
    # difference is visible on the legacy points.  Both ends are carried through.
    ed = np.vstack([rd_ - 10.0 ** (0.2 * (vd - sd)), 10.0 ** (0.2 * (vd + sd)) - rd_])
    el = np.vstack([rl_ - 10.0 ** (0.2 * (vl - sl)), 10.0 ** (0.2 * (vl + sl)) - rl_])

    # The same legacy bins carried by the joint (k_max = 4) analysis instead.  The offset
    # is a CONSTANT in magnitude, so the bin means move rigidly and the covariance -- and
    # therefore the error bars -- are untouched; only the asymmetry of the ratio changes.
    shift, Delta = legacy_offset()
    vlr = vl - shift
    rlr = 10.0 ** (0.2 * vlr)
    elr = np.vstack([rlr - 10.0 ** (0.2 * (vlr - sl)), 10.0 ** (0.2 * (vlr + sl)) - rlr])
    print('[sn] legacy recalibration: Delta %+.4f mag, series shift %+.4f mag '
          '-> ratio x %.4f' % (Delta, shift, 10.0 ** (-0.2 * shift)))

    zb, vb, sb = bao_dv_points()
    rb = vb / spa_dv_over_rd(zb)
    eb = sb / spa_dv_over_rd(zb)

    fig, ax = plt.subplots(figsize=C['figsize'])
    # Darker than the supernova dots (0.62), which at alpha 0.30 sit near 0.89 on white:
    # at 0.5 the unity line read as the same grey as the cloud it crosses.
    ax.axhline(1.0, color=C['unity_grey'], ls='--', lw=1.0, zorder=1)

    # H0^SPA / H0^SH0ES, fractional errors in quadrature, treated as independent
    r_sh = SPA_H0[0] / SHOES_H0[0]
    f_sh = r_sh * np.hypot(SPA_H0[1] / SPA_H0[0], SHOES_H0[1] / SHOES_H0[0])
    s = C['shoes']
    ax.axhspan(r_sh - f_sh, r_sh + f_sh, color=s['color'], alpha=s['alpha'], lw=0,
               zorder=0, label=SHOES_LEGEND)
    ax.axhline(r_sh, color=s['color'], lw=s['lw'], zorder=1)

    # z is zHD throughout -- `sn_residuals` returns dc.sn_dataset('full')['z'], which
    # read_hubble_diagram fills from the zHD column, and the (1 + z_HEL) of D_L is carried
    # separately in `spa_dl`.  Supernovae are therefore at their Hubble-diagram redshift,
    # as they must be.
    ax.scatter(z[~leg], ratio_i[~leg], color=C['dots_des'], zorder=2, rasterized=True,
               **C['dots'])
    ax.scatter(z[leg], ratio_i[leg], color=C['dots_leg'], zorder=2, rasterized=True,
               **C['dots'])

    # The fitted expansions themselves, DES black and DES + legacy red, NOT in the legend
    # the colours already key to the points they run through.
    # Split at each sample's lowest supernova, solid above and dotted below, the same
    # convention as fig:adot-reconstruction -- the black dive towards 0.88 at z -> 0 is a
    # sub-floor extrapolation and must not read as a measurement.
    for cell, where, col, floor in (
            (C['ladder_cell_des'], 'ladder', C_DES, MODEL_FLOOR['des']),
            (C['ladder_cell_full'], 'ladder', C_LEG, MODEL_FLOOR['full']),
            (C['joint_cell'][1], C['joint_cell'][0], C_DLEG, MODEL_FLOOR['full'])):
        if bands:
            zm, lom, rm, him = model_ratio(cell, where, bands=True)
        else:
            zm, rm = model_ratio(cell, where)
        sup, ext = zm >= floor, zm <= floor
        if bands:
            # Fainter below the floor than above it, matching the line going dotted there:
            # the band and the line have to weaken together or the extrapolation reads as
            # better determined than the measurement it hangs off.
            ax.fill_between(zm[sup], lom[sup], him[sup], color=col, alpha=0.16, lw=0,
                            zorder=2.5)
            ax.fill_between(zm[ext], lom[ext], him[ext], color=col, alpha=0.08, lw=0,
                            zorder=2.5)
        ax.plot(zm[sup], rm[sup], color=col, zorder=3, **C['model'])
        ax.plot(zm[ext], rm[ext], color=col, ls=':', lw=1.1, alpha=0.85, zorder=3)
        if bands:
            print('[model] %-22s 1sigma half-width: %.4f at z=0.05, %.4f at 0.2, '
                  '%.4f at 1.0' % (cell, 0.5 * (np.interp(0.05, zm, him) - np.interp(0.05, zm, lom)),
                                   0.5 * (np.interp(0.2, zm, him) - np.interp(0.2, zm, lom)),
                                   0.5 * (np.interp(1.0, zm, him) - np.interp(1.0, zm, lom))))
        print('[model] %-22s ratio at z=%.3f %.4f, 0.05 %.4f, 0.2 %.4f, 1.0 %.4f, '
              '2.33 %.4f' % (cell, zm[0], rm[0], np.interp(0.05, zm, rm),
                             np.interp(0.2, zm, rm), np.interp(1.0, zm, rm), rm[-1]))

    m = C['sn_marker']
    ax.errorbar(zd, rd_, yerr=ed, fmt='o', color=C_DES, mfc=C_DES, zorder=5,
                label=r'DES SNe Ia ($D_{\rm L}$)', **m)
    ax.errorbar(zl, rl_, yerr=el, fmt='o', color=C_LEG, mfc=C_LEG, zorder=6,
                label=r'legacy SNe Ia ($D_{\rm L}$)', **m)
    ax.errorbar(zl, rlr, yerr=elr, fmt='o', color=C_DLEG, mfc=C_DLEG, zorder=7,
                label='legacy SNe Ia ($D_{\\rm L}$)\n'
                      r'($\Delta M_\mathrm{legacy}$)', **m)
    b = C['bao_marker']
    ax.errorbar(zb, rb, yerr=eb, color=C_BAO, mfc=C_BAO, zorder=8,
                label=r'DESI DR2 BAO ($D_V/r_d$)', **b)

    ax.set_xscale('log')
    ax.set_xlim(*C['xlim'])
    if C['ylim']:
        ax.set_ylim(*C['ylim'])
    ax.set_xticks(C['xticks'])
    # Explicit strings, not ScalarFormatter: it pads every label to a common number of
    # decimals, which gave "0.20" and "1.00".  %g drops the
    # trailing zeros and leaves 0.03 / 0.1 / 1 / 2.5.
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FixedFormatter(['%g' % t for t in C['xticks']]))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(C['xlabel'], fontsize=C['fontsize_label'])
    ax.set_ylabel(C['ylabel'], fontsize=C['fontsize_label'])
    ax.tick_params(labelsize=C['fontsize_tick'])
    place_legend(fig, ax, C, z, ratio_i,
                 np.concatenate([zd, zl, zl, zb]),
                 np.concatenate([rd_, rl_, rlr, rb]))
    fig.tight_layout()

    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    p = os.path.join(dc.RESULT_DIR, C['out'] + ('' if bands else '_nobands'))
    fig.savefig(p + '.pdf')
    fig.savefig(p + '.png')
    plt.close(fig)
    print('wrote %s.pdf and .png' % p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-bands', dest='bands', action='store_false',
                    help='fitted curves as lines only; writes dz_ratio_nobands, never '
                         'the article figure')
    a = ap.parse_args()
    print(dc.banner())
    main(bands=a.bands and CFG['model_bands'])
