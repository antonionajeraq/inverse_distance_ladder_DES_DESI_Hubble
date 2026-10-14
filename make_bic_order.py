r"""make_bic_order.py -- the prior-free least-squares order sweep and the Delta BIC figure.

    python make_bic_order.py                 # P = 2..7, config A and B, both samples
    python make_bic_order.py --plist 2,3,4,5,6,7,8,9,10

READS ONLY.  Every chi2_min it reports comes from `dovekie_cosmography.task_bic`, which
serves the cached `chains/bic/chi2min_<config>_<sample>.npz` cells; nothing is refitted
here.  Pass --recompute to force the minimisation.

WHY THIS IS THE DEFENSIBLE FORM OF THE ORDER CHOICE.  The posterior means of the sweep
move with the c_k prior width -- that is established, and it is why an order chosen from
them would inherit an arbitrary knob.  chi2_min carries NO prior at all: it is the
unconstrained least-squares fit of the same likelihood, so

    BIC(P) = chi2_min(P) + k(P) ln N ,      k(P) = (P+1) + M_B + r_d = P + 3

is a statement about the data and the model alone.  Each order is seeded from the
solution of the previous one with c_P = 0 -- the identical model -- so
chi2_min(P) <= chi2_min(P-1) holds by construction and the ladder cannot be an artefact
of a minimiser that did better on one row than another.

WHAT THE FIGURE SHOWS.  Delta BIC(P) = BIC(P) - min_P BIC(P), per cell, so every curve
touches zero at its own selected order.  The star marks the ADOPTED order, k_max = 4 in
both samples, not the minimum.  The shaded band is Delta BIC < 2, the
conventional "not worth a mention" zone: it is what turns the DES+legacy P=4-vs-P=5 gap
into a non-difference and licenses ONE baseline order for both samples.

COLOURS come from make_figures.SERIES -- DES black, DES+legacy red -- so this figure sits
beside the others.  Configuration A (Planck r_d) is solid, configuration B (CC-calibrated,
r_d free) dashed; B is the control, and the point of drawing it is that the ladder does
not depend on which of the two r_d treatments is used.
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

import dovekie_cosmography as dc
import ls_errors as le
import make_figures as mf

DPI = mf.DPI
OUT = 'bic_order'

# The local reference the paper quotes its tensions against, from make_tables.SHOES_H0.
# Breuval et al. (2024) report H0 but publish no SNIa absolute magnitude, which is why the
# paper still cites Riess et al. (2022) for M_B; only H0 is compared here.
SHOES_H0 = (73.17, 0.86)
# Labelled SH0ES rather than 'Breuval et al.': both are
# correct -- the number IS Breuval et al. (2024) -- but the programme name is what the
# rest of the paper compares against, so it reads more clearly on the axis.
SHOES_LABEL = 'SH0ES'

CELLS = [('A', 'des_only'), ('A', 'full'), ('B', 'des_only'), ('B', 'full')]

# THE FIGURE IS A SUBSET OF THE TABLE, DELIBERATELY.
# Configuration B is dropped from the plot: its Delta BIC ladder
# sits within 0.3 units of configuration A's at every order, so the two pairs of curves
# overplot and the figure spends half its ink saying that the r_d treatment does not matter.
# The TABLE still carries all four cells, which is where that claim is checked, and it is
# also the only place configuration B's measured r_d can appear at all -- configuration A
# has none, since there r_d is a prior.
# k_max = 7 is dropped for a different reason: its curvature covariance fails the
# differencing-step plateau test (ls_errors), so its error bar is numerical rather than
# statistical and has no place on an axis whose point is the size of the error bars.  The
# order decision is untouched, since k_max = 7 is excluded by more than 13 units of BIC.
FIG_CONFIGS = ('A',)
FIG_PMAX = 6
CONFIG_LAB = {'A': r'A ($r_\mathrm{d}$ prior)', 'B': r'B (CC, $r_\mathrm{d}$ free)'}
CONFIG_LS = {'A': '-', 'B': '--'}
CONFIG_MK = {'A': 'o', 'B': 's'}


def _cells(out, plist):
    """(config, sample) -> the rows of `plist` in order, plus the flat-LCDM row."""
    got = {}
    for cfg, smp in CELLS:
        d = out['%s_%s' % (cfg, smp)]
        rows = sorted([r for r in d['rows'] if r['P'] in plist], key=lambda r: r['P'])
        if [r['P'] for r in rows] != sorted(plist):
            raise RuntimeError('cell %s_%s is missing orders: has %r, asked %r'
                               % (cfg, smp, [r['P'] for r in d['rows']], plist))
        bic_min = min(r['BIC'] for r in rows)
        for r in rows:
            r['dBIC'] = r['BIC'] - bic_min
            # theta = [c_0..c_P, M, r_d];  c_0 = H0 and, in the y-series,
            # c_1 = H0 (1 + q0), so q0 = c_1/c_0 - 1.
            th = r['theta']
            r['H0'] = th[0]
            r['q0'] = th[1] / th[0] - 1.0
            r['M'] = th[r['P'] + 1]
            r['rd'] = th[r['P'] + 2]
        got[(cfg, smp)] = dict(rows=rows, lcdm=d['lcdm'],
                               P_bic=min(rows, key=lambda r: r['BIC'])['P'])
    return got


def table(got, plist, err=None):
    L = []
    L.append('LEAST-SQUARES ORDER SWEEP.  chi2_min is the UNCONSTRAINED minimum of the same')
    L.append('chi2 the posterior uses -- no c_k prior -- so this ladder is prior-free.')
    L.append('BIC = chi2_min + k ln N, k = (P+1) + M_B + r_d.  dBIC is measured from the')
    L.append('minimum of each cell; dstep is the change from the previous order (negative =')
    L.append('the new coefficient paid for itself).  H0, q0, M_B, r_d are the LS fit itself.')
    L.append('')
    for cfg, smp in CELLS:
        c = got[(cfg, smp)]
        rows = c['rows']
        L.append('config %s   %s   N = %d   ln N = %.3f'
                 % (cfg, {'des_only': 'DES', 'full': 'DES+legacy'}[smp],
                    rows[0]['N'], rows[0]['lnN']))
        e = (err or {}).get((cfg, smp), {})
        hdr = ('  %4s %3s %10s %8s %8s | %15s %13s %17s %15s %8s'
               % ('kmax', 'k', 'chi2_min', 'dBIC', 'dstep', 'H0', 'q0', 'M_B', 'r_d',
                  'cond'))
        L.append(hdr)
        L.append('  ' + '-' * (len(hdr) - 2))
        prev = None
        for r in rows:
            step = '   -' if prev is None else '%+8.2f' % (r['BIC'] - prev['BIC'])
            s_ = e.get(str(r['P']), {})
            flag = '' if s_.get('reliable', True) else ' !'
            star = '  <== BIC min' if r['P'] == c['P_bic'] else ''
            L.append('  %4d %3d %10.3f %8.2f %8s | %7.2f+-%-6.2f %+6.2f+-%-6.2f '
                     '%9.4f+-%-6.4f %7.2f+-%-6.2f %8.1e%s%s'
                     % (r['P'], r['k'], r['chi2_min'], r['dBIC'], step,
                        r['H0'], s_.get('sig_H0', np.nan),
                        r['q0'], s_.get('sig_q0', np.nan),
                        r['M'], s_.get('sig_M', np.nan),
                        r['rd'], s_.get('sig_rd', np.nan),
                        s_.get('cond', np.nan), flag, star))
            prev = r
        lc, sl = c['lcdm'], e.get('lcdm', {})
        L.append('  %4s %3d %10.3f %8.2f %8s | %7.2f+-%-6.2f %+6.2f+-%-6.2f '
                 '%9.4f+-%-6.4f %7.2f+-%-6.2f %8.1e'
                 % ('LCDM', lc['k'], lc['chi2_min'],
                    lc['BIC'] - min(r['BIC'] for r in rows), '-',
                    sl.get('H0', np.nan), sl.get('sig_H0', np.nan),
                    sl.get('q0', np.nan), sl.get('sig_q0', np.nan),
                    sl.get('M', np.nan), sl.get('sig_M', np.nan),
                    sl.get('rd', np.nan), sl.get('sig_rd', np.nan),
                    sl.get('cond', np.nan)))
        L.append('')
    L.append('The flat-LCDM row is the caveat that travels with the choice: two shape')
    L.append('parameters beat five, which is what a parameter count should do.  It is not')
    L.append('evidence for LCDM -- the expansion is not offered as a rival model, it is')
    L.append('offered to leave H(z) below the survey floor free, and that freedom is')
    L.append('exactly what the criterion charges for.')
    txt = '\n'.join(L)
    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    p = os.path.join(dc.RESULT_DIR, OUT + '_table.txt')
    open(p, 'w').write(txt + '\n')
    print('\n' + txt)
    print('wrote %s' % p)
    return txt


def figure(got, plist, err=None):
    """Two panels on a shared order axis.

    The upper panel is the criterion itself.  The lower panel is the quantity the criterion
    is choosing on behalf of: H0 with the uncertainty implied by the curvature of the same
    chi2 (ls_errors), which is what makes the point that the order is far better determined
    than the H0 that DES alone returns at any of them.  The two panels share the x axis, so
    the selected order and the precision it costs are read off the same column."""
    plist = [p for p in plist if p <= FIG_PMAX]
    cells = [(c, s) for c, s in CELLS if c in FIG_CONFIGS]
    if err is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        axes = None
    else:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 7.8), sharex=True,
                                      gridspec_kw=dict(height_ratios=(1.35, 1.0),
                                                       hspace=0.08))
        axes = ax2

    # dBIC < 2 is the conventional threshold below which a difference is not worth
    # reporting.  It is drawn FIRST so every curve sits on top of it, and it is the whole
    # reason one order can serve both samples: the DES+legacy P=4/P=5 gap lives inside it.
    # No absolute value in the label: dBIC is measured from the
    # minimum, so it is >= 0 by construction and |.| only confuses.
    ax.axhspan(0.0, 2.0, color='0.85', zorder=0)
    # anchored in AXES coordinates and right-aligned, so it cannot run off the panel when
    # the order range changes
    ax.text(0.988, 1.0, r'$\Delta\mathrm{BIC}<2$', fontsize=10, color='0.45',
            va='center', ha='right', transform=ax.get_yaxis_transform(which='grid'))

    for cfg, smp in cells:
        c = got[(cfg, smp)]
        rows = [r for r in c['rows'] if r['P'] in plist]
        # Delta BIC is re-referenced to the minimum OVER THE PLOTTED ORDERS, so the curve
        # still touches zero at the order the figure is about.  Over 2..6 that minimum is
        # the same row the full sweep selects in both samples, so no number moves.
        bic_min = min(r['BIC'] for r in rows)
        P = np.array([r['P'] for r in rows], float)
        d = np.array([r['BIC'] - bic_min for r in rows], float)
        s = mf.SERIES[smp]
        ax.plot(P, d, CONFIG_LS[cfg], color=s['c'], marker=CONFIG_MK[cfg], ms=6.5,
                lw=1.8, mfc=('none' if cfg == 'B' else s['c']), mew=1.4,
                label=s['lab'], zorder=3)
        # The star marks the order ADOPTED, k_max = BASELINE['P'] = 4 in both samples, at
        # by choice -- not the BIC minimum.  The two coincide for DES; for
        # DES+legacy the minimum is k_max = 5 and the star sits on the red curve at its
        # k_max = 4 value, inside the dBIC < 2 band, which is the argument for the choice.
        i = int(np.flatnonzero(P == dc.BASELINE['P'])[0])
        ax.plot(P[i], d[i], marker='*', ms=15, color=s['c'], mec='w', mew=0.8, zorder=4)

    ax.axvline(dc.BASELINE['P'], color='0.55', lw=1.0, ls=':', zorder=1)
    ax.set_yscale('symlog', linthresh=10.0, linscale=1.0)
    ax.set_yticks([0, 2, 5, 10, 100, 400])
    ax.set_yticklabels(['0', '2', '5', '10', '100', '400'])
    ax.set_xticks(list(plist))
    ax.set_xlim(plist[0] - 0.25, plist[-1] + 0.55)
    # top headroom so the P=2 markers are not clipped by the axis they define
    ax.set_ylim(-1.5, 800.0)
    ax.set_xlabel(r'polynomial order $k_\mathrm{max}$', fontsize=13)
    ax.set_ylabel(r'$\Delta\mathrm{BIC} \equiv \mathrm{BIC}-\mathrm{BIC}_{\min}$', fontsize=13)
    ax.tick_params(labelsize=11)
    # The baseline order is marked by the dotted vertical line above and by the star
    # where each curve touches zero.  The text callout and its sloped leader line were
    # removed: a third way of saying the same thing, drawn over
    # the curves it is pointing at.
    ax.legend(fontsize=10.5, loc='upper center', frameon=True, framealpha=0.9)

    if axes is not None:
        ax2 = axes
        # The two external references the rest of the paper quotes tensions against, in
        # the convention fig_floorsweep already uses: a shaded 1 sigma band, a dotted line
        # on the central value, and the name set clear of the band.  They are drawn so the
        # panel can be READ against them, not because these fits are being tested on them.
        # SPA takes its label below the band and Breuval above, which is the same choice
        # fig_floorsweep makes and for the same reason, that the DES points rise through
        # the region directly under Breuval.
        for val, sig, nm, cl, side in ((mf.SPA_H0[0], mf.SPA_H0[1], mf.SPA_LABEL_LCDM,
                                        '#a8702a', 'below'),
                                       (SHOES_H0[0], SHOES_H0[1], SHOES_LABEL,
                                        'green', 'above')):
            ax2.axhspan(val - sig, val + sig, color=cl, alpha=0.15, zorder=0)
            ax2.axhline(val, color=cl, ls=':', lw=1.3, zorder=1)
            dy, va = ((sig + 0.25, 'bottom') if side == 'above' else (-sig - 0.25, 'top'))
            ax2.text(plist[0] - 0.15, val + dy, nm, color=cl, fontsize=10, va=va,
                     ha='left', zorder=5)
        flagged = False
        for cfg, smp in cells:
            rows = [r for r in got[(cfg, smp)]['rows'] if r['P'] in plist]
            e = err[(cfg, smp)]
            P = np.array([r['P'] for r in rows], float)
            H = np.array([r['H0'] for r in rows], float)
            S = np.array([e[str(r['P'])]['sig_H0'] for r in rows], float)
            ok = np.array([e[str(r['P'])].get('reliable', True) for r in rows], bool)
            sr = mf.SERIES[smp]
            dx = -0.06 if smp == 'des_only' else 0.06     # so the bars do not overlap
            ax2.errorbar(P[ok] + dx, H[ok], yerr=S[ok], fmt=CONFIG_MK[cfg], color=sr['c'],
                         ms=6.0, lw=1.6, capsize=3.5, label=sr['lab'], zorder=3)
            flagged = flagged or bool((~ok).any())
            # the unreliable curvature is drawn hollow with a dotted bar, never dropped
            ax2.errorbar(P[~ok] + dx, H[~ok], yerr=S[~ok], fmt=CONFIG_MK[cfg],
                         mfc='none', color=sr['c'], ms=6.0, lw=1.4, capsize=3.5,
                         ls='none', alpha=0.85, zorder=3)
            for x, y, sg in zip(P[~ok] + dx, H[~ok], S[~ok]):
                ax2.plot([x, x], [y - sg, y + sg], ':', color=sr['c'], lw=2.4, zorder=2)
        ax2.axvline(dc.BASELINE['P'], color='0.55', lw=1.0, ls=':', zorder=1)
        ax2.set_xlabel(r'polynomial order $k_\mathrm{max}$', fontsize=13)
        ax2.set_ylabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=13)
        ax2.tick_params(labelsize=11)
        # The k_max = 7 bar reaches 106, which would compress every informative point into
        # the middle third of the panel.  The axis is cut instead and the bar runs off it,
        # which is the honest picture of an error that is not a measurement anyway.
        lo, hi = [], []
        for cfg, smp in cells:
            e = err[(cfg, smp)]
            for r in got[(cfg, smp)]['rows']:
                if r['P'] in plist:
                    lo.append(r['H0'] - e[str(r['P'])]['sig_H0'])
                    hi.append(r['H0'] + e[str(r['P'])]['sig_H0'])
        for val, sig in (mf.SPA_H0, SHOES_H0):
            lo.append(val - sig)
            hi.append(val + sig)
        pad = 0.09 * (max(hi) - min(lo))
        ax2.set_ylim(min(lo) - pad, max(hi) + pad)
        # Lower LEFT: the DES series climbs into the upper right
        # as k_max grows, so the key was sitting on the points it labels.
        ax2.legend(fontsize=10, loc='lower left', frameon=True, framealpha=0.95)
        # The note appears only when there is something to note.  With k_max = 7 out of the
        # figure nothing is flagged, and a standing caveat about symbols the reader cannot
        # see would be worse than no caveat at all.
        if flagged:
            ax2.text(0.80, 0.955, 'open symbol, dotted bar:\ncurvature not reliable',
                     transform=ax2.transAxes, fontsize=9, color='0.35', ha='right',
                     va='top', linespacing=1.35)
        ax.set_xlabel('')
    fig.tight_layout()
    return mf._save(fig, OUT, tight_bbox=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plist', default='2,3,4,5,6,7')
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()
    plist = sorted(int(v) for v in a.plist.split(','))
    out = dc.task_bic(plist=plist, recompute=a.recompute)
    got = _cells(out, plist)
    err = le.sweep(got, plist)
    table(got, plist, err)
    figure(got, plist, err)


if __name__ == '__main__':
    main()
