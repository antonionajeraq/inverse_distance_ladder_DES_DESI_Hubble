"""make_E_prior_graphic.py -- the envelope prior on E(y) (Fig. 1).

    python make_E_prior_graphic.py        # -> results/E_prior_graphic.{pdf,png}

The prior (dovekie_cosmography.py section 7) is uniform in C_1..C_K over every polynomial
E(y) = H(y)/H0 = 1 + sum_{k=1}^{k_max} C_k y^k with 0 <= E(y) <= 2 E_lin(y) on
[0, y_max = 0.70], E_lin = 1 + 3.5y being the chord of flat LCDM.  This figure shows that
envelope against the SPA flat-LCDM E(y), drawn as a single solid line: the Omega_m
uncertainty is NOT shown, since the band it induces is too narrow to
see, and the caption says so.  The dotted black line is E_lin, the solid black line is the upper limit 2 + 7y, and the region above
it is excluded.  The lower limit E = 0 is the bottom axis.  The legend carries only SPA and
"Excluded"; the two lines are described in the caption.

The SPA line is slightly thicker, and the best-fitting polynomial of the
baseline order (quartic) is overplotted as a thin dashed blue line.  "Best-fitting" means a
least-squares fit of E(y) = 1 + sum_{k=1}^{4} C_k y^k to the SPA curve itself, sampled
uniformly on [0, Y_MAX] with equal weights.  It is NOT a fit to data: showing the main result
this early would confuse the reader.  E(0) = 1 is imposed exactly by leaving out the constant
term, so only C_1..C_4 are fitted.  The fit is printed below, including its q0, which is far
from the SPA value although the curves look identical: a quartic in y matching LCDM over the
whole of [0, 0.7] buys that accuracy with the low-y slope.

Nothing here reads a chain; it is a schematic of the prior and can be rebuilt any time.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Stroke-count guard (strokecheck.py): patches Figure.savefig so the PDF
# reports its stroke count as it is written.
import strokecheck
strokecheck.enable()

from make_figures import SPA_OM, SPA_LABEL_TEX, DPI, MODEL_BLUE
import dovekie_cosmography as dc

OUTDIR = dc.RESULT_DIR   # honours RESULTS_DIR like every other renderer (was 'results')
BASE = 'E_prior_graphic'

# SPA (SPT + Planck + ACT) flat LCDM, arXiv:2506.20707.  The central value is the one
# every other figure uses (make_figures.SPA_OM).  SIG_OM is NOT drawn: it only feeds the
# band width printed at the end, which the caption quotes to justify leaving it out.
OM, SIG_OM = 0.3175, 0.0055
assert OM == SPA_OM, 'SPA Omega_m here (%g) disagrees with make_figures (%g)' % (OM, SPA_OM)

Y_MAX = 0.70                  # the envelope is enforced on [0, Y_MAX] (z = 2.33, Lya BAO)
E_LIN = (1.0, 3.5)            # 1 + 3.5y: the chord of flat LCDM on [0, Y_MAX]
E_UP = (2.0, 7.0)             # prior limit: twice E_LIN
E_TOP = 7.0                   # top of the plotted range

COL_SPA = '#e8a33d'           # the SPA orange of make_adot / make_figures
Z_TICKS = (0.0, 0.25, 0.5, 1.0, 2.0)   # top axis: redshift at the matching y
COL_EXCL = '0.80'             # grey of the excluded region
FS_LABEL, FS_TICK, FS_LEG = 14, 12, 11
LW_SPA = 3.0                  # slightly thicker than the 2.0 limit line
P_FIT = dc.BASELINE['P']      # the baseline order, i.e. a quartic
NFIT = 701                    # uniform y grid on [0, Y_MAX] for the equal-weight fit
COL_FIT = MODEL_BLUE          # the paper's one dark blue
LW_FIT = 1.4                  # thin, dashed, drawn over the SPA line
LS_FIT = (0, (5, 2.5))


def E_lcdm(y, Om):
    z = y / (1.0 - y)
    return np.sqrt(Om * (1.0 + z) ** 3 + 1.0 - Om)


def quartic_fit(Om=OM):
    """Least-squares C_1..C_P to the SPA E(y) on a uniform grid over [0, Y_MAX], equal
    weights.  No constant term, so E(0) = 1 holds exactly."""
    y = np.linspace(0.0, Y_MAX, NFIT)
    A = np.vstack([y ** k for k in range(1, P_FIT + 1)]).T
    C = np.linalg.lstsq(A, E_lcdm(y, Om) - 1.0, rcond=None)[0]
    return C


def E_poly(y, C):
    return 1.0 + sum(c * y ** (k + 1) for k, c in enumerate(C))


def main():
    y = np.linspace(0.0, Y_MAX, 701)
    upper = E_UP[0] + E_UP[1] * y
    if not np.all(upper <= E_TOP):
        raise SystemExit('the prior limit leaves the plotted range; raise E_TOP')

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Excluded region first, so every line sits on top of it.
    ax.fill_between(y, upper, E_TOP, color=COL_EXCL, lw=0, zorder=0)
    ax.plot(y, upper, color='k', lw=2.0, zorder=3)

    ax.plot(y, E_LIN[0] + E_LIN[1] * y, color='k', lw=1.6, ls=':', zorder=2)

    ax.plot(y, E_lcdm(y, OM), color=COL_SPA, lw=LW_SPA, zorder=2)

    C = quartic_fit()
    if E_poly(0.0, C) != 1.0:
        raise SystemExit('the fitted polynomial does not have E(0) = 1')
    ax.plot(y, E_poly(y, C), color=COL_FIT, lw=LW_FIT, ls=LS_FIT, zorder=4)

    ax.set_xlim(0.0, Y_MAX)
    ax.set_ylim(0.0, E_TOP)
    ax.set_xticks(np.round(np.arange(0.0, Y_MAX + 1e-9, 0.1), 1))
    ax.set_yticks(np.arange(0, int(E_TOP) + 1, 1))
    # The independent variable is a~ = 1 - a; the code keeps the name y.
    ax.set_xlabel(r'$\widetilde{a} \equiv 1 - a$', fontsize=FS_LABEL)
    ax.set_ylabel(r'$E(\widetilde{a}) \equiv H(\widetilde{a})/H_0$', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)

    # Redshift along the top, so a reader can place y without converting.
    top = ax.secondary_xaxis('top', functions=(lambda yy: yy / (1.0 - yy),
                                               lambda zz: zz / (1.0 + zz)))
    top.set_xticks(Z_TICKS)
    top.set_xticklabels(['%g' % zz for zz in Z_TICKS])
    # Centred on the axis the label sits next to the z = 0.5 tick (y = 0.35 vs 0.333), so it
    # is placed halfway between the z = 0.5 and z = 1 ticks instead.
    y_lab = 0.5 * (0.5 / 1.5 + 1.0 / 2.0)                   # y at z = 0.5 and z = 1
    top.set_xlabel(r'$z$', fontsize=FS_LABEL, x=y_lab / Y_MAX)
    top.tick_params(labelsize=FS_TICK)

    handles = [Line2D([], [], color=COL_SPA, lw=LW_SPA),
               Line2D([], [], color=COL_FIT, lw=LW_FIT, ls=LS_FIT),
               Patch(facecolor=COL_EXCL, lw=0)]
    labels = [SPA_LABEL_TEX + r' ($\Lambda$CDM)', 'Quartic fit', 'Excluded']
    # Handle length = exactly three dashes of LS_FIT (dash lengths scale with lw), so the
    # key's sample does not end on a stub.
    hl = (3 * LS_FIT[1][0] + 2 * LS_FIT[1][1]) * LW_FIT / FS_LEG
    ax.legend(handles, labels, loc='lower right', fontsize=FS_LEG, frameon=True,
              handlelength=hl,
              framealpha=1.0, edgecolor='0.8')

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = os.path.join(OUTDIR, '%s.%s' % (BASE, ext))
        fig.savefig(p, dpi=DPI, bbox_inches='tight', pad_inches=0.03)
        print('[E_prior] wrote %s' % p)
    plt.close(fig)

    # The numbers a caption would need, printed rather than typed by hand.
    for yy in (0.0, 0.35, Y_MAX):
        e = E_lcdm(np.array([yy]), OM)[0]
        u = E_UP[0] + E_UP[1] * yy
        print('[E_prior] y = %.2f (z = %.3f): E_SPA = %.3f, limit = %.2f, E_SPA/limit = %.2f'
              % (yy, yy / (1 - yy), e, u, e / u))
    yy = np.linspace(0.0, Y_MAX, NFIT)
    d = E_poly(yy, C) - E_lcdm(yy, OM)
    print('[E_prior] quartic fit (equal weights, E(0) = 1 fixed): C_1..C_%d = %s'
          % (P_FIT, ', '.join('%.4f' % c for c in C)))
    print('[E_prior] quartic fit: max |E_fit - E_SPA| = %.4f (%.2f%% of E), at y = %.3f; '
          'rms %.4f' % (abs(d).max(), 100 * abs(d / E_lcdm(yy, OM)).max(),
                        yy[np.argmax(abs(d))], np.sqrt(np.mean(d ** 2))))
    print('[E_prior] quartic fit q0 = C_1 - 1 = %.3f; SPA q0 = 1.5 Om - 1 = %.3f'
          % (C[0] - 1.0, 1.5 * OM - 1.0))
    for nsig in (1, 2):
        band = (E_lcdm(y, OM + nsig * SIG_OM) - E_lcdm(y, OM - nsig * SIG_OM)).max()
        print('[E_prior] full width of the SPA %d sigma band (not drawn), largest over the '
              'range: %.3f, i.e. +-%.3f, %.2f%% of E there' % (nsig, band, band / 2,
              100 * (band / 2 / E_lcdm(y, OM))[np.argmax(E_lcdm(y, OM + nsig * SIG_OM)
                                                         - E_lcdm(y, OM - nsig * SIG_OM))]))


if __name__ == '__main__':
    main()
