r"""dh0_ensemble.py -- the null distribution of Delta H0 = H0(DES) - H0(DES+legacy).

THE QUESTION.  The published baseline (Table 2, configuration A, cosmography P=4) returns
H0 = 76.25 +- 2.87 from DES alone and 67.98 +- 0.69 once the legacy supernovae are added:
a shift of Delta H0 = 8.27 km/s/Mpc.  How often does an ensemble of CMB-calibrated flat
LCDM realisations -- in which the legacy supernovae are perfectly calibrated and there is
no late-time physics at all -- produce a shift that large or larger?

WHAT IS VARIED AND WHAT IS NOT.  Nothing but the data.  This module calls mock_idl's
build_mock/install_mock to serve one realisation of the fiducial to dovekie_cosmography's
three data readers, then calls mock_idl.ls_fit at z_floor = 0 twice: once for 'des_only'
and once for 'full'.  Those are the two PUBLISHED baseline cells -- the samples' own
redshift distributions, no floor imposed -- so the pair differs only in which supernovae
are in the fit, which is the contrast the question is about.  The seed depends on the
realisation index only, so realisation N here is the SAME universe as realisation N of the
floor-sweep ensemble in results_mock/lsweep_realN.json.

THE ESTIMATOR IS LEAST SQUARES, NOT NUTS, AND THAT IS WHY THE OBSERVED VALUE IS REFIT.
An ensemble of 10^4 x 2 NUTS cells is not affordable, so the prior-free chi2 minimum with
Gauss-Newton errors is used, exactly as in the floor-sweep null test.  The observed value
is therefore recomputed through THIS estimator (--real -1) and the p-value is quoted
against that, like for like; the published 8.27 is carried alongside as the number the
manuscript quotes, and the two are reported together so any estimator gap is visible
rather than assumed away.

ONE-SIDED BY CONSTRUCTION.  The alternative hypothesis is a positive Delta H0 of roughly
the size of the Hubble tension -- DES alone, which has no supernovae below z = 0.073,
reading high because the sub-floor extrapolation is unconstrained.  The reported fraction
is therefore P(Delta H0_mock >= Delta H0_obs), a right-tail probability.

USAGE
    python dh0_ensemble.py --reals 1-300      # the ensemble, one process
    python dh0_ensemble.py --real -1          # the observed catalogue, same estimator
    python dh0_ensemble.py --analyse          # statistics + histogram
"""
import argparse
import json
import os
import sys

import numpy as np
from mock_paths import rglob, rpath

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results_mock')

# The published Table 1 numbers this test exists to interpret (configuration A, P=4,
# cosmography), as posterior means from chains/ladder/.  Read, never recomputed here.
PUB = dict(H0_des=76.25, sig_des=2.87, H0_full=67.98, sig_full=0.69)   # envelope-prior rerun, 2026-10-01
PUB['dH0'] = PUB['H0_des'] - PUB['H0_full']


def path_for(tag):
    return rpath(OUT, 'dh0_%s.json' % tag, write=True)


def run(reals, fiducial='lcdm', delta=0.0, quiet=True):
    # dovekie_cosmography reads CHAINS_DIR/RESULTS_DIR at IMPORT, so the redirect has to
    # happen before mock_idl imports it.  Nothing here samples, but the quarantine is kept
    # so a mock can never be written where a real cell is looked up.
    os.environ['RESULTS_DIR'] = OUT
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    sys.path.insert(0, HERE)
    import mock_idl as mi
    os.environ['CHAINS_DIR'] = mi._chain_dir(mi.realisation_tag(reals[0], delta, fiducial))
    import dovekie_cosmography as dc
    print(dc.banner())
    fid = dict(mi.FID, background=fiducial)

    for real in reals:
        tag = mi.realisation_tag(real, delta, fiducial)
        dest = path_for(tag)
        if os.path.exists(dest):
            print('[dh0] %s cached' % tag)
            continue
        dc.CHAIN_DIR = mi._chain_dir(tag)
        if real < 0:
            mi.uninstall_mock(dc)
            print('\n=== REAL DATA (no mock installed)')
        else:
            mock = mi.build_mock(dc, real, delta=delta, fid=fid)
            mi.install_mock(dc, mock)
            print('\n=== mock %s  n_sn=%d (legacy %d)  sha_mu=%s'
                  % (tag, mock['spec']['n_sn'], mock['spec']['n_legacy'],
                     mock['spec']['sha_mu']))
        row = dict(tag=tag, real=int(real), delta_injected=float(delta),
                   fiducial=fiducial, P=dc.BASELINE['P'])
        for smp in ('des_only', 'full'):
            # z_floor = 0: each sample keeps its OWN redshift distribution, which is the
            # published baseline cell and the whole content of the contrast.
            r = mi.ls_fit(dc, smp, z_floor=0.0, joint_delta=False)
            for k in ('H0', 'q0', 'M', 'rd', 'chi2', 'n_sn', 'sig_H0', 'sig_q0', 'sig_M'):
                row['%s_%s' % (k, smp)] = r[k]
        row['dH0'] = row['H0_des_only'] - row['H0_full']
        row['dq0'] = row['q0_des_only'] - row['q0_full']
        os.makedirs(OUT, exist_ok=True)
        with open(dest, 'w') as fh:
            json.dump(row, fh, indent=1)
        print('[dh0] %-12s  DES %7.2f +- %5.2f   DES+leg %7.2f +- %5.2f   '
              'dH0 %+7.2f   N %d/%d'
              % (tag, row['H0_des_only'], row['sig_H0_des_only'], row['H0_full'],
                 row['sig_H0_full'], row['dH0'], row['n_sn_des_only'], row['n_sn_full']))
    return True


def load(fiducial='lcdm', delta=0.0):
    import glob
    rows = []
    for f in rglob(OUT, 'dh0_real[0-9][0-9][0-9]*.json'):
        r = json.load(open(f))
        if r['fiducial'] == fiducial and abs(r['delta_injected'] - delta) < 1e-12:
            rows.append(r)
    obs = json.load(open(path_for('realdata'))) if os.path.exists(path_for('realdata')) \
        else None
    nl = json.load(open(path_for('noiseless'))) if os.path.exists(path_for('noiseless')) \
        else None
    return rows, obs, nl


# =======================================================================================
# Analysis
# =======================================================================================
def _cp_interval(k, n, cl=0.6827):
    """Clopper-Pearson interval on a binomial fraction.

    NOT sqrt(p(1-p)/n).  The tail here holds a handful of events out of thousands, where
    the Gaussian error bar is both too narrow and capable of running below zero; the
    Clopper-Pearson construction is exact for any k, including k = 0, and stays inside
    [0, 1] by construction."""
    from scipy.stats import beta
    a = 0.5 * (1.0 - cl)
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def _tail(x, thr):
    """Right-tail fraction with the (k+1)/(n+1) convention.

    The plain k/n estimate returns EXACTLY ZERO when no realisation reaches the observed
    value, which is a statement the ensemble cannot make: with n draws the smallest
    resolvable probability is ~1/n.  The +1 correction is the standard unbiased bound for
    a Monte-Carlo p-value and degrades gracefully to 1/(n+1) instead of 0."""
    x = np.asarray(x, float)
    k = int((x >= thr).sum())
    n = len(x)
    return k, n, (k + 1.0) / (n + 1.0), k / float(n)


def _clear_key(fig, ax, leg, bins, cnt, pad=1.04):
    """Raise the top of the panel until the key clears the bars it sits over.

    The key moved to the upper LEFT on 2026-09-11, which is where the null's peak is --
    a fixed 30 per cent of headroom left the box resting on the tallest bins.  Measuring
    is what the fixed factor cannot do: the key's height depends on its type size and on
    how many of the four lines are drawn, and the peak moves with the ensemble.

    The key occupies a fixed FRACTION of the panel whatever the limits are, so one solve
    is exact rather than iterative: with the bottom fixed, clearing a bar of height `h`
    needs top = y0 + (h*pad - y0) / (1 - f), where f is the key's height as a fraction of
    the panel.  Only ever raises.
    """
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transData.inverted())
    mid = 0.5 * (bins[:-1] + bins[1:])
    under = cnt[(mid >= bb.x0) & (mid <= bb.x1)]
    if not len(under):
        return
    y0, y1 = ax.get_ylim()
    f = (y1 - bb.y0) / (y1 - y0)
    if f >= 1.0:
        return
    top = y0 + (under.max() * pad - y0) / (1.0 - f)
    if top > y1:
        ax.set_ylim(y0, top)


def analyse(fiducial='lcdm', delta=0.0, out='dh0_hist', onecol=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Stroke-count guard (strokecheck.py).  A page carrying
    # thousands of SEPARATE stroked paths can defeat a printer's RIP even though it
    # renders instantly and the file is small.  This patches Figure.savefig, so the
    # savefig calls below are left as written and each PDF reports its stroke count
    # against the limit as it is saved.  Warnings go to stderr.
    import strokecheck
    strokecheck.enable()

    rows, obs, nl = load(fiducial, delta)
    if not rows:
        raise SystemExit('no realisations in %s' % OUT)
    d = np.array([r['dH0'] for r in rows])
    hd = np.array([r['H0_des_only'] for r in rows])
    hf = np.array([r['H0_full'] for r in rows])
    n = len(d)

    obs_d = obs['dH0'] if obs else np.nan
    nl_d = nl['dH0'] if nl else np.nan
    thr = PUB['dH0']

    k_pub, _, p_pub, praw_pub = _tail(d, thr)
    lines = []
    A = lines.append
    A('Delta H0 = H0(DES) - H0(DES+legacy), baseline cells (config A, cosmography P=%d,'
      % rows[0]['P'])
    A('no redshift floor: each sample keeps its own redshift distribution).')
    A('Estimator: prior-free chi2 minimum (mock_idl.ls_fit).  Fiducial: flat LCDM at SPA,')
    A('H0 = 67.19, Om = 0.3175, rd = 147.05, Delta_legacy = 0.  N = %d realisations.' % n)
    A('')
    A('%-42s %10s %10s' % ('', 'H0', 'dH0'))
    A('%-42s %10s %10.2f' % ('OBSERVED, published NUTS posterior means', '', PUB['dH0']))
    A('%-42s %10.2f' % ('   DES', PUB['H0_des']))
    A('%-42s %10.2f' % ('   DES+legacy', PUB['H0_full']))
    if obs:
        A('%-42s %10s %10.2f' % ('OBSERVED, this estimator (least squares)', '', obs_d))
        A('%-42s %10.2f' % ('   DES', obs['H0_des_only']))
        A('%-42s %10.2f' % ('   DES+legacy', obs['H0_full']))
    if nl:
        A('%-42s %10s %10.2f' % ('NULL, noiseless (pure estimator bias)', '', nl_d))
        A('%-42s %10.2f' % ('   DES', nl['H0_des_only']))
        A('%-42s %10.2f' % ('   DES+legacy', nl['H0_full']))
    A('')
    A('NULL ENSEMBLE (n = %d)' % n)
    A('   dH0            mean %+7.3f   sd %6.3f   median %+7.3f   [min %+7.2f, max %+7.2f]'
      % (d.mean(), d.std(ddof=1), np.median(d), d.min(), d.max()))
    A('   H0 DES         mean %7.3f   sd %6.3f' % (hd.mean(), hd.std(ddof=1)))
    A('   H0 DES+legacy  mean %7.3f   sd %6.3f' % (hf.mean(), hf.std(ddof=1)))
    A('   corr(H0_DES, H0_DES+legacy) = %+.3f' % np.corrcoef(hd, hf)[0, 1])
    A('   percentiles of dH0: ' + '  '.join(
        '%g%%=%+.2f' % (q, np.percentile(d, q)) for q in (5, 16, 50, 84, 95, 99)))
    A('')
    A('ONE-SIDED TEST  P(dH0_mock >= dH0_obs)')
    lo_pub, hi_pub = _cp_interval(k_pub, n)
    A('   threshold %.2f (published)   : %d / %d = %.4f  (68%% CP [%.4f, %.4f])'
      '   [(k+1)/(n+1) = %.4f]'
      % (thr, k_pub, n, praw_pub, lo_pub, hi_pub, p_pub))
    stats = dict(n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                 median=float(np.median(d)), published_dH0=float(thr),
                 k_published=int(k_pub), p_published_raw=float(praw_pub),
                 p_published=float(p_pub),
                 z_published=float((thr - d.mean()) / d.std(ddof=1)),
                 p_published_cp68=[float(v) for v in _cp_interval(k_pub, n)],
                 H0_des_mean=float(hd.mean()), H0_des_sd=float(hd.std(ddof=1)),
                 H0_full_mean=float(hf.mean()), H0_full_sd=float(hf.std(ddof=1)),
                 corr=float(np.corrcoef(hd, hf)[0, 1]),
                 noiseless_dH0=(None if nl is None else float(nl_d)))
    if obs:
        k_ls, _, p_ls, praw_ls = _tail(d, obs_d)
        lo_ls, hi_ls = _cp_interval(k_ls, n)
        A('   threshold %.2f (this estimator): %d / %d = %.4f  (68%% CP [%.4f, %.4f])'
          '   [(k+1)/(n+1) = %.4f]'
          % (obs_d, k_ls, n, praw_ls, lo_ls, hi_ls, p_ls))
        stats.update(observed_dH0_ls=float(obs_d), k_ls=int(k_ls),
                     p_ls_raw=float(praw_ls), p_ls=float(p_ls),
                     z_ls=float((obs_d - d.mean()) / d.std(ddof=1)))
    A('')
    A('   Gaussian equivalent of the published threshold: z = %+.2f sigma'
      % stats['z_published'])
    if obs:
        A('   Gaussian equivalent of the LS threshold       : z = %+.2f sigma'
          % stats['z_ls'])
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(OUT, '%s.txt' % out), 'w') as fh:
        fh.write(txt + '\n')
    with open(rpath(OUT, '%s.json' % out, write=True), 'w') as fh:
        json.dump(dict(stats=stats, dH0=[float(v) for v in d],
                       H0_des=[float(v) for v in hd],
                       H0_full=[float(v) for v in hf],
                       reals=[int(r['real']) for r in rows]), fh, indent=1)

    # ---- figures ----------------------------------------------------------------------
    # TWO FILES, NOT ONE FLOAT.  The author asked for the panels to be shipped separately
    # and tiled with two \includegraphics commands, and for the SCATTER to come first, so
    # the joint distribution is met before the one-dimensional projection of it.  Nothing
    # here writes a combined figure any more.
    #
    # ONE palette, the manuscript's: DES = black, DES + legacy = red.  The histogram is the
    # null; the observed value is a vertical line and the shaded right tail is the quantity
    # the question asks for, drawn rather than only quoted.
    #
    # SIZED FOR THE PAGE.  What lands on the page is python_pt x displayed_in / canvas_in.
    #   onecol=True   each panel is its own \columnwidth (3.4 in) file, so the reduction is
    #                 1:1 and the default 10 pt type prints at 10 pt.  This is the paper.
    #   onecol=False  the wide 6.6 in diagnostic version of the same two files, for reading
    #                 on screen; the type is scaled so it still lands at 10-11 pt.
    PT = 1.0 if onecol else 1.6
    W, H = (3.4, 2.9) if onecol else (6.6, 4.6)
    plt.rcParams.update({'font.size': 10 * PT, 'axes.titlesize': 11 * PT,
                         'axes.labelsize': 10.5 * PT, 'xtick.labelsize': 9.5 * PT,
                         'ytick.labelsize': 9.5 * PT})

    UNIT = r'$\,{\rm km\,s^{-1}\,Mpc^{-1}}$'
    MOCK_LAB = r'SPA $\Lambda$CDM mocks'
    # The noiseless realisation is the estimator's own bias with the noise switched off.
    # It is "the noiseless MOCK", not "the noiseless null" -- the
    # more accurate wording: it is one realisation, not the distribution.  It is NOT black on
    # the scatter panel, where black already means the DES axis and the equality lines.
    NL_LAB = 'noiseless mock'
    C_NL = '#2e7d32'
    C_LS = '#1f4e79'                     # the least-squares reading of the real catalogue
    paths = {}

    # ================================================================================
    # 1.  THE SCATTER -- the joint distribution the histogram is a projection of
    # ================================================================================
    # METHOD IS THE AUTHOR'S `Large_scatter_plot.py`, applied to this ensemble.
    #
    # It is an EMPIRICAL 2-D histogram, not a kernel density estimate: bins are sorted by
    # count, cumulated, and the count at which the cumulative total first reaches the
    # requested fraction is the contour level.  So the 68.27% and 95.45% contours enclose
    # that fraction of the REALISATIONS, by construction and with no smoothing kernel to
    # choose -- which matters here, because the tail is the part of the panel the p-value
    # is about and a kernel would fill it in.
    #
    # Three layers: every mock faintly, the mocks
    # OUTSIDE the 95.45% contour again at higher opacity (which is what makes the tail
    # readable at n = 10^4, where a plain scatter saturates to one blob), then the
    # contours, then the reference lines, then each estimator's observed value, with the
    # hollow diamond last of all so nothing is hidden under it.
    from matplotlib.lines import Line2D

    # BINS: 22 per axis, not the template's 60.  `nbins` is the one knob the method has,
    # and it is a noise/resolution trade: at 60 these 10^4 realisations give 2.8 points per
    # bin, so the contour is Poisson noise -- it breaks into islands and grows spurious
    # lobes, which on a panel whose whole subject is the tail is actively misleading.  At
    # 22 it is ~21 points per bin and both contours close cleanly.  The CONTAINMENT is
    # unaffected: the level is defined as the count enclosing the requested fraction of the
    # realisations whatever the binning, so 68.27% and 95.45% stay exact by construction.
    NBINS = 22

    def _level(counts, fraction):
        """The bin count at or above which the bins hold `fraction` of all the points.

        Every bin has the same area, so a count IS a density here.  Straight from
        Large_scatter_plot.density_threshold_for_containment."""
        srt = np.sort(counts.ravel())[::-1]
        return srt[np.searchsorted(np.cumsum(srt), fraction * counts.sum())]

    fig, ax = plt.subplots(figsize=(W, H))

    xr = (hf.min(), hf.max())
    yr = (hd.min(), hd.max())
    counts, xed, yed = np.histogram2d(hf, hd, bins=NBINS, range=[xr, yr])
    lv68, lv95 = _level(counts, 0.6827), _level(counts, 0.9545)
    X, Y = np.meshgrid(0.5 * (xed[:-1] + xed[1:]), 0.5 * (yed[:-1] + yed[1:]),
                       indexing='ij')

    # which realisations fall outside the 95.45% contour, by their own bin's count
    ix = np.clip(np.searchsorted(xed, hf, side='right') - 1, 0, NBINS - 1)
    iy = np.clip(np.searchsorted(yed, hd, side='right') - 1, 0, NBINS - 1)
    outside = counts[ix, iy] < lv95

    # RASTERIZED: 10^4 markers are one PathCollection and so
    # cost almost nothing in strokes, but they are 10^4 objects in the file all the same.
    ax.scatter(hf, hd, s=3.5, color='0.35', alpha=0.08, linewidths=0, rasterized=True,
               zorder=1)
    ax.scatter(hf[outside], hd[outside], s=4.0, color='0.25', alpha=0.45, linewidths=0,
               rasterized=True, zorder=2)
    # contour() wants Z[j, i]
    ax.contour(X, Y, counts, levels=[lv68], colors='black', linewidths=1.0,
               linestyles='--', zorder=4)
    ax.contour(X, Y, counts, levels=[lv95], colors='black', linewidths=1.35,
               linestyles='-', zorder=4)

    xlim = [min(hf.min(), PUB['H0_full']) - 0.6, max(hf.max(), PUB['H0_full']) + 0.6]
    ylim = [min(hd.min(), PUB['H0_des']) - 1.0, max(hd.max(), PUB['H0_des']) + 1.0]
    xs = np.array(xlim)
    # The truth, then the two reference relations.  Solid navy IS the fiducial H0 in both
    # coordinates, which is the only place in the panel where a line means a value.
    ax.axhline(67.19, color='navy', lw=0.9, alpha=0.7, zorder=6)
    ax.axvline(67.19, color='navy', lw=0.9, alpha=0.7, zorder=6)
    h_zero, = ax.plot(xs, xs, color='0.5', lw=1.0, ls='--', zorder=6)
    h_thr, = ax.plot(xs, xs + thr, color='crimson', lw=1.0, ls=':', zorder=6)

    # the observed catalogue through each estimator, and the noiseless mock
    h_nl = None
    if nl:
        h_nl = ax.scatter([nl['H0_full']], [nl['H0_des_only']], marker='P', s=60,
                          color=C_NL, edgecolor='k', linewidth=0.5, alpha=0.95, zorder=7)
    h_pub = ax.scatter([PUB['H0_full']], [PUB['H0_des']], marker='*', s=150,
                       color='crimson', edgecolor='k', linewidth=0.5, alpha=0.75,
                       zorder=8)
    h_ls = None
    if obs:
        h_ls = ax.scatter([obs['H0_full']], [obs['H0_des_only']], marker='D', s=34,
                          facecolor='none', edgecolor=C_LS, linewidth=1.3, zorder=9)

    # INDEPENDENT limits, not a square frame.  The DES+legacy arm is four times the
    # narrower of the two, so equal limits spend half the panel on blank space; the two
    # reference lines stay exactly right, they simply stop being drawn at 45 degrees.
    ax.set_xlim(xlim)
    # HEADROOM FOR THE KEY.  The author asked for it at the top left, which is where the
    # observed star and diamond sit -- they are the highest DES values in the panel, which
    # is the whole point of them.  Raising the top of the axis opens a band above every
    # drawn thing, so the key can be upper left without covering the two points the figure
    # exists to place.
    #
    # PINNED at 95, rather than the 42 per cent of the
    # range it used to add: that put the top at ~90.3, which left the key sitting on the
    # crimson threshold line.  A fixed top also keeps the two D1 panels' headroom from
    # moving when the ensemble is regenerated.  max() so a mock ensemble that ever reaches
    # higher than this is not silently cropped.
    ax.set_ylim(ylim[0], max(95.0, ylim[1]))
    ax.set_xlabel(r'$H_0^{\rm DES+legacy}$  $[{\rm km\,s^{-1}\,Mpc^{-1}}]$')
    ax.set_ylabel(r'$H_0^{\rm DES}$  $[{\rm km\,s^{-1}\,Mpc^{-1}}]$')
    # NO TITLE, on either panel: the caption says what the panel is, and rho is in the text
    # and in the json.
    #
    # KEY ORDER IS THE AUTHOR'S: the mocks first -- as a GREY DOT, standing for the
    # individual points that are only visible in the outlying regions -- then the observed
    # value through each estimator, then the noiseless mock and the two reference lines.
    hs = [Line2D([], [], marker='o', color='0.25', markerfacecolor='0.25', markersize=3.2,
                 linestyle='None'), h_pub]
    ls_ = [MOCK_LAB, 'observed (NUTS)']
    if h_ls is not None:
        hs.append(h_ls); ls_.append('observed (LS)')
    if h_nl is not None:
        hs.append(h_nl); ls_.append(NL_LAB)
    hs += [h_zero, h_thr]
    # NO NUMBER IN THE KEY: the threshold entry used to print the
    # observed value, which went stale (8.81) when the results moved.  The caption and
    # the text carry the number.
    ls_ += [r'$\Delta H_0 = 0$', r'observed $\Delta H_0$']
    ax.legend(hs, ls_, fontsize=7.0 * PT, loc='upper left', framealpha=0.95,
              borderpad=0.4, handletextpad=0.6, labelspacing=0.35)
    fig.tight_layout()
    # No explicit dpi on the PDF: savefig.dpi (matplotlibrc, 600) sets the resolution of
    # the two rasterised scatter layers, and an argument here would override it.
    fig.savefig(os.path.join(OUT, 'dh0_scatter.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'dh0_scatter.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    paths['scatter'] = os.path.join(OUT, 'dh0_scatter.pdf')

    # ================================================================================
    # 2.  THE HISTOGRAM -- the one-dimensional statement, and the p-value
    # ================================================================================
    fig, ax = plt.subplots(figsize=(W, H))
    lo = min(d.min(), -1.0)
    hi = max(d.max(), thr) * 1.08
    bins = np.linspace(lo, hi, 34 if n < 2000 else 64)
    # NO SAMPLE SIZE IN THE KEY.  The ensemble size is a
    # fact about the test, not about the grey bars, and it is stated in the caption, in the
    # text and in results_mock/json/dh0_hist.json; in the key it was the entry that set the
    # width of the whole box.  `n` is still what the p-value and its interval are built on.
    cnt, _b, _p = ax.hist(d, bins=bins, color='0.72', edgecolor='0.35', linewidth=0.6,
                          label=MOCK_LAB)
    ylim = ax.get_ylim()
    if (d >= thr).any():
        ax.hist(d[d >= thr], bins=bins, color='crimson', alpha=0.55, edgecolor='crimson',
                linewidth=0.6)
    ax.axvspan(thr, bins[-1], color='crimson', alpha=0.07, zorder=0)
    # LINE STYLES: solid black = the null mean, dotted black = the
    # noiseless mock, solid red = the observed value through NUTS, dotted red = the same
    # catalogue through the least-squares estimator.  Nothing here is dashed, so no pair
    # of lines has to be told apart by dash length at print size.
    #
    # NO NUMBERS IN THE KEY.  Each entry used to carry its
    # own value and unit, which made the key four lines of arithmetic the reader has to
    # match back to four lines on the panel -- and the x axis already says where each line
    # falls, to better than the two decimals the key was quoting.  The values are in the
    # text, in Table 3 and in results_mock/json/dh0_hist.json; the key now only says which
    # line is which.
    ax.axvline(d.mean(), color='k', lw=1.6, ls='-', label='null mean')
    if nl:
        ax.axvline(nl_d, color='k', lw=1.4, ls=':', label=NL_LAB)
    ax.axvline(thr, color='crimson', lw=1.8, ls='-', label='observed (NUTS)')
    if obs:
        ax.axvline(obs_d, color='crimson', lw=1.4, ls=':', label='observed (LS)')
    # NO FIXED HEADROOM any more.
    # The key lost its two widest elements -- the four values and the sample size -- and
    # now sits over the far left of the panel, where the null has a handful of counts, so
    # the 30 per cent that used to be added on top of the bars buys nothing.  The panel
    # keeps the limits the histogram itself autoscaled to, ~5 per cent above the peak, and
    # _clear_key below opens it further only if the key would actually land on a bar.
    ax.set_ylim(*ylim)
    # `\equiv`, not `=`: the left-hand side is being DEFINED, not solved for.
    # TWO LINES.  Set on one, this label is wider than a \columnwidth panel, so it either
    # runs off the figure or forces the axes to shrink to fit it.
    ax.set_xlabel(r'$\Delta H_0 \equiv H_0^{\rm DES} - H_0^{\rm DES+legacy}$'
                  '\n' r'$[{\rm km\,s^{-1}\,Mpc^{-1}}]$')
    ax.set_ylabel('Number of mocks')
    # TOP LEFT.  With the values gone the key is
    # less than half its old width, and the left shoulder of the null -- which runs out to
    # DeltaH0 ~ -8 with a handful of counts -- is the emptiest corner of the panel; the
    # right one is where the observed lines and the crimson tail region are.
    leg = ax.legend(fontsize=6.3 * PT, loc='upper left', framealpha=0.95, borderpad=0.4,
                    handletextpad=0.6, labelspacing=0.35, handlelength=1.8)
    _clear_key(fig, ax, leg, bins, cnt)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, 'dh0_hist.%s' % ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    paths['hist'] = os.path.join(OUT, 'dh0_hist.pdf')

    print('\n[fig] %s\n[fig] %s' % (paths['scatter'], paths['hist']))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('USAGE')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--real', type=int, default=None,
                    help='0 = noiseless, -1 = the observed catalogue, >=1 = a realisation')
    ap.add_argument('--reals', default=None, help='a range such as 1-300')
    ap.add_argument('--delta', type=float, default=0.0)
    ap.add_argument('--fiducial', default='lcdm', choices=['lcdm', 'cosmo'])
    ap.add_argument('--shard', type=int, default=None,
                    help='(cluster) this task index; runs the realisations of --reals '
                         'with i %% NTASKS == SHARD.  Sharding by STRIDE and not by block '
                         'is deliberate: every shard then costs the same, so one slow '
                         'node cannot hold the array open on a tail of leftovers.')
    ap.add_argument('--ntasks', type=int, default=None,
                    help='(cluster) the FULL array size.  Must equal the whole grid, or '
                         'the unclaimed realisations are silently never run.')
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--onecol', action='store_true',
                    help='stack the panels on a \\columnwidth canvas, for a '
                         'plain one-column float')
    ap.add_argument('--out', default='dh0_hist')
    a = ap.parse_args(argv)
    if a.analyse:
        return analyse(a.fiducial, a.delta, a.out, onecol=a.onecol)
    sys.path.insert(0, HERE)
    import mock_idl as mi
    reals = mi.parse_reals(a.reals) if a.reals else [a.real if a.real is not None else 0]
    if a.shard is not None or a.ntasks is not None:
        if a.shard is None or a.ntasks is None:
            raise SystemExit('--shard and --ntasks go together')
        if not (0 <= a.shard < a.ntasks):
            # An out-of-range shard runs NO realisations and exits 0, which looks exactly
            # like success -- the failure mode _common.sh already guards for the samplers.
            raise SystemExit('SHARD=%d is not a valid shard of NTASKS=%d'
                             % (a.shard, a.ntasks))
        reals = [r for i, r in enumerate(reals) if i % a.ntasks == a.shard]
        print('[shard] %d/%d -> %d realisations: %s%s'
              % (a.shard, a.ntasks, len(reals), reals[:6], ' ...' if len(reals) > 6 else ''))
    run(reals, fiducial=a.fiducial, delta=a.delta)


if __name__ == '__main__':
    main()
