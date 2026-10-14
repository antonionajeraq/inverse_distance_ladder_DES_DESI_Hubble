r"""make_evidence_order.py -- the evidence-weighted order sweep: tables, figure, blend.

    python make_evidence_order.py            # k_max = 2..6, both configurations

READS ONLY.  The paper assesses the truncation order with the Bayesian evidence and
reports the Bayesian model average over it; make_bic_order.py supplies only the
least-squares chi2 column.

SOURCES.
  * The posterior at every order, and its evidence, are the nested-sampling cells of
    evidence_order.py (chains/evorder/<key>_P<k>.npz, collected in
    results/evidence_order.json).  All of them use the envelope prior on E(y)
    (dovekie_cosmography section 7), whose lnZ is already corrected for the envelope
    acceptance, lnZ_A = lnZ_sampled - ln acc(K).
  * chi2_min and Delta chi2 are the PRIOR-FREE least-squares minimum of the same
    likelihood (dovekie_cosmography.task_bic, ls_delta_order), kept because
    lnZ - lnL_max is the Occam factor and the reader can only see it with both.
  * The LCDM rows are the NUTS posteriors of the ladder / legacy tasks.  No LCDM evidence
    exists on the same footing as the order sweep (Table 3's dynesty runs marginalise
    M analytically, evidence_order samples it), so the LCDM rows carry no Delta lnZ and
    no weight.  LCDM is not part of the blend in any case: the blend is over the orders
    of one expansion.

THE BLEND IS BUILT FROM THE SAMPLES.  make_blend.py --weights evidence stands each order
in for a Gaussian with the same mean and variance.  That is exact for the blended mean and
standard deviation, but not for the 1-sigma HPD region, which is what the bimodal q0
entries print.  Here every order enters with its own weighted dynesty samples, so

    p(theta | D) = sum_k w(k) p(theta | D, k),     w(k) = Z(k) / sum_k' Z(k'),

is the actual Bayesian model average.  The HPD region is solved on a binned, kernel-
smoothed density (bandwidth = Silverman's rule with the Kish effective sample size of
each order), by bisecting on the density level and reading off every contiguous run.
The means and standard deviations agree with make_blend to every printed digit; the
intervals agree to the grid step except q0 in A, DES + legacy with Delta free, whose
upper piece the Gaussian route misplaces by ~0.3.

PRIOR SENSITIVITY.  The envelope 0 <= E(y) <= 2 + 7y is an affine image of itself about
E = 1 + 3.5y.  Halving it about that line divides the prior volume by 2^K, so
    lnZ_half(K) = lnZ(K) + K ln 2 + ln f_in(K),
with f_in the posterior mass inside the halved band.  This is exact, needs no new
sampling, and is reported for every cell.

Outputs (results/): table_ordersweep_evidence.tex, evidence_order.pdf/.png,
blend_evidence_samples.json, evidence_order_numbers.txt.
"""
import json
import math
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import strokecheck
strokecheck.enable()

import dovekie_cosmography as dc
import ls_delta_order as lsd
import make_bic_order as mbo
import make_figures as mf

PLIST = [2, 3, 4, 5, 6]
HPD_MASS = math.erf(1.0 / math.sqrt(2.0))
NGRID = 16384
SHOES_H0 = mbo.SHOES_H0
SPA_H0 = mf.SPA_H0
DASH = '---'
LN2 = math.log(2.0)

# (key, config, sample, Delta free, block heading)
BLOCKS = {
    'A': [('A_des_only', 'des_only', False, 'DES'),
          ('A_full', 'full', False, 'DES + legacy'),
          ('A_full_delta', 'full', True,
           r'DES + legacy ($\Delta M_\mathrm{legacy}$ free)')],
    'B': [('B_des_only', 'des_only', False, 'DES'),
          ('B_full', 'full', False, 'DES + legacy')],
}
LABEL = {'A': 'tab:ordersweep-ls-A', 'B': 'tab:ordersweep-ls-B'}
COL5_HEAD = {'A': r'$\Delta M_\mathrm{legacy}$ [mag]', 'B': r'$r_{\rm d}$ [cMpc]'}
# NUTS LCDM cell for each block (the ladder / legacy tasks)
LCDM_CELL = {'A_des_only': 'chains/ladder/A_des_only_lcdm_P4.npz',
             'A_full': 'chains/ladder/A_full_lcdm_P4.npz',
             'A_full_delta': 'chains/legacy/lcdm_full_joint_P4.npz',
             'B_des_only': 'chains/ladder/B_des_only_lcdm_P4.npz',
             'B_full': 'chains/ladder/B_full_lcdm_P4.npz'}
# The figure keeps configuration A only, like the BIC figure it replaces; B is in Table B2.
FIG_KEYS = [('A_des_only', 'des_only'), ('A_full', 'full'), ('A_full_delta', 'full_dleg')]


# ---------------------------------------------------------------------------------------
# 1.  The per-order posteriors and evidences
# ---------------------------------------------------------------------------------------
def load_order(key, P):
    """Weighted dynesty samples of one order, with q0 and C_k derived from v."""
    d = np.load(os.path.join(dc.HERE, 'chains', 'evorder', '%s_P%d.npz' % (key, P)),
                allow_pickle=True)
    row = json.loads(str(d['row_json']))
    if row.get('prior') != 'envelope-v1':
        raise SystemExit('%s_P%d was not sampled under the envelope prior' % (key, P))
    S = np.asarray(d['samples'], float)
    w = np.asarray(d['weights'], float)
    w = w / w.sum()
    q = {nm: S[:, i] for i, nm in enumerate(row['params'])}
    C = dc.shape_of_v(S[:, 1:P + 1], P)
    q['q0'] = C[:, 1] - 1.0
    return dict(row=row, q=q, w=w, C=C, P=P)


def weighted_ms(x, w):
    m = float(np.sum(w * x))
    return m, float(np.sqrt(np.sum(w * (x - m) ** 2)))


def halved_band_fraction(o, y_max=None):
    """Posterior mass inside the envelope halved about E = 1 + 3.5y (see header)."""
    y_max = dc.BASELINE['Y_MAX'] if y_max is None else y_max
    y = np.linspace(0.0, y_max, 701)
    lo, hi = dc.envelope_bounds(y)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    E = o['C'] @ np.power(y[None, :], np.arange(o['P'] + 1)[:, None])
    inside = np.all((E >= mid - 0.5 * half) & (E <= mid + 0.5 * half), axis=1)
    return float(o['w'][inside].sum())


def weights_from(lnz):
    lnz = np.asarray(lnz, float)
    w = np.exp(lnz - lnz.max())   # subtract the maximum: exp(-850) underflows to zero
    return w / w.sum()


# ---------------------------------------------------------------------------------------
# 2.  The model average, from the samples
# ---------------------------------------------------------------------------------------
def hpd_pieces(xs, ws, W, mass=HPD_MASS, ngrid=NGRID):
    """Every contiguous run of the 1-sigma HPD region of sum_k W_k p_k(x)."""
    live = [i for i, Wk in enumerate(W) if Wk > 1e-6]
    allx = np.concatenate([xs[i] for i in live])
    lo, hi = np.percentile(allx, [0.001, 99.999])
    pad = 0.1 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, ngrid + 1)
    dx = edges[1] - edges[0]
    f = np.zeros(ngrid)
    for i in live:
        x, w = xs[i], ws[i]
        neff = w.sum() ** 2 / np.sum(w ** 2)
        _m, s = weighted_ms(x, w)
        h = 1.06 * s * neff ** -0.2
        hst, _ = np.histogram(x, edges, weights=w)
        f += W[i] * gaussian_filter1d(hst / hst.sum(), h / dx, mode='constant')
    f /= f.sum()
    a, b = 0.0, f.max()
    for _ in range(80):
        lev = 0.5 * (a + b)
        if f[f >= lev].sum() > mass:
            a = lev
        else:
            b = lev
    on = f >= a
    xc = 0.5 * (edges[1:] + edges[:-1])
    pieces, i = [], 0
    while i < ngrid:
        if on[i]:
            j = i
            while j + 1 < ngrid and on[j + 1]:
                j += 1
            pieces.append(dict(lo=float(xc[i]), hi=float(xc[j]),
                               frac=float(f[i:j + 1].sum() / f[on].sum())))
            i = j + 1
        else:
            i += 1
    return pieces


def blend(orders, W):
    out = {}
    for qn in ('H0', 'M', 'rd', 'Delta', 'q0'):
        if qn not in orders[0]['q']:
            continue
        xs = [o['q'][qn] for o in orders]
        ws = [o['w'] for o in orders]
        m = float(sum(Wk * np.sum(w * x) for Wk, x, w in zip(W, xs, ws)))
        within = float(sum(Wk * np.sum(w * (x - np.sum(w * x)) ** 2)
                           for Wk, x, w in zip(W, xs, ws)))
        var = float(sum(Wk * np.sum(w * (x - m) ** 2) for Wk, x, w in zip(W, xs, ws)))
        out[qn] = dict(mean=m, std=math.sqrt(var), sig_within=math.sqrt(within),
                       sig_between=math.sqrt(max(var - within, 0.0)),
                       pieces=hpd_pieces(xs, ws, W))
    return out


def tension(m, s, ref):
    return (m - ref[0]) / math.hypot(s, ref[1])


# ---------------------------------------------------------------------------------------
# 3.  Assemble every block
# ---------------------------------------------------------------------------------------
def build_blocks():
    got = mbo._cells(dc.task_bic(plist=PLIST), PLIST)
    dl = lsd.ladder(plist=tuple(PLIST))
    res = {}
    for cfg, blocks in BLOCKS.items():
        for key, smp, dfree, _head in blocks:
            orders = [load_order(key, P) for P in PLIST]
            lnz = [o['row']['logz'] for o in orders]
            lnz_err = [o['row']['logz_err'] for o in orders]
            W = weights_from(lnz)
            fin = [halved_band_fraction(o) for o in orders]
            lnz_half = [z + o['P'] * LN2 + math.log(max(f, 1e-300))
                        for z, o, f in zip(lnz, orders, fin)]
            if dfree:
                chi2 = [dl[str(P)]['chi2_min'] for P in PLIST]
                chi2_lcdm = dl['lcdm']['chi2_min']
            else:
                rows = {r['P']: r for r in got[(cfg, smp)]['rows']}
                chi2 = [rows[P]['chi2_min'] for P in PLIST]
                chi2_lcdm = got[(cfg, smp)]['lcdm']['chi2_min']
            per = []
            for o in orders:
                s = {qn: weighted_ms(o['q'][qn], o['w'])
                     for qn in ('H0', 'M', 'rd', 'Delta', 'q0') if qn in o['q']}
                per.append(dict(P=o['P'], k=o['P'] + 3 + int(dfree), **{
                    qn: dict(mean=v[0], std=v[1]) for qn, v in s.items()}))
            lc = np.load(os.path.join(dc.HERE, LCDM_CELL[key]), allow_pickle=True)
            ls = json.loads(str(lc['scalars_json']))
            lcdm = dict(k=4 + int(dfree), chi2_min=chi2_lcdm)
            for qn, arr in (('H0', 'H0'), ('M', 'M'), ('rd', 'rd'), ('q0', 'q0'),
                            ('Delta', 'Delta')):
                if arr in lc.files:
                    a = np.asarray(lc[arr], float)
                    lcdm[qn] = dict(mean=float(a.mean()), std=float(a.std()))
            lcdm['quality_pass'] = ls.get('quality_pass')
            res[key] = dict(cfg=cfg, sample=smp, delta_free=dfree, plist=PLIST,
                            lnZ=lnz, lnZ_err=lnz_err, weight=W.tolist(),
                            dlnZ=[z - max(lnz) for z in lnz], chi2_min=chi2,
                            f_in_half=fin, lnZ_half=lnz_half,
                            weight_half=weights_from(lnz_half).tolist(),
                            orders=per, lcdm=lcdm, blend=blend(orders, W))
    return res


# ---------------------------------------------------------------------------------------
# 4.  LaTeX, in the manuscript's own layout (multicolumn block headings)
# ---------------------------------------------------------------------------------------
def _b(s, bold):
    return (r'\mathbf{%s}' % s) if bold else s


def _pm(d, fmt, bold=False):
    if d is None:
        return DASH
    sf = fmt.replace('%+', '%')
    return '$%s$' % _b((fmt + r' \pm ' + sf) % (d['mean'], d['std']), bold)


def _signed(v, bold, eps=1e-9):
    # The rule: the reference row's zeros print bare, every other
    # difference carries its sign
    return '$%s$' % _b('0.00' if abs(v) < eps else '%+.2f' % v, bold)


def _q0_cell(bq):
    pcs = sorted(bq['pieces'], key=lambda d: d['lo'])
    if len(pcs) == 1:
        return _pm(bq, '%+.2f')
    # top-aligned, so the single-number cells of the row sit on its FIRST line (IB's
    # caption note on Table B1); \shortstack is bottom-aligned and put them on the second.
    return (r'\begin{tabular}[t]{@{}c@{}}$[%+.2f, %+.2f]$ \\ $\cup \; [%+.2f, %+.2f]$\end{tabular}'
            % (pcs[0]['lo'], pcs[0]['hi'], pcs[1]['lo'], pcs[1]['hi']))


def block_lines(key, r):
    cfg, dfree = r['cfg'], r['delta_free']
    col5 = 'Delta' if dfree else ('rd' if cfg == 'B' else None)
    f5 = '%+.4f' if col5 == 'Delta' else '%.2f'
    # Configuration B: sigma(H0) >= 3.8 in every row, so the second decimal carries nothing
    # (3 s.f. in B, 4 s.f. in A).
    fH0 = '%.1f' if cfg == 'B' else '%.2f'
    ibest = int(np.argmax(r['lnZ']))
    chi2_ref = r['chi2_min'][ibest]
    L = []
    for i, o in enumerate(r['orders']):
        bold = i == ibest
        L.append(' & '.join([
            '$%s$' % _b('%d' % o['P'], bold), '$%s$' % _b('%d' % o['k'], bold),
            _pm(o['H0'], fH0, bold),
            _pm(o.get(col5), f5, bold) if col5 else DASH,
            _pm(o['M'], '%.3f', bold), _pm(o['q0'], '%+.2f', bold),
            '$%s$' % _b('%.2f' % r['chi2_min'][i], bold),
            _signed(r['chi2_min'][i] - chi2_ref, bold),
            _signed(r['dlnZ'][i], bold),
            '$%s$' % _b('%.3f' % r['weight'][i], bold)]) + r' \\')
    lc = r['lcdm']
    L.append(' & '.join([
        r'$\Lambda$CDM', '$%d$' % lc['k'], _pm(lc['H0'], fH0),
        _pm(lc.get(col5), f5) if col5 else DASH, _pm(lc['M'], '%.3f'),
        _pm(lc['q0'], '%+.2f'), '$%.2f$' % lc['chi2_min'],
        _signed(lc['chi2_min'] - chi2_ref, False), DASH, DASH]) + r' \\')
    b = r['blend']
    L.append(' & '.join([
        'Blend', DASH, _pm(b['H0'], fH0),
        _pm(b.get(col5), f5) if col5 else DASH, _pm(b['M'], '%.3f'),
        _q0_cell(b['q0']), DASH, DASH, DASH, '$1$']) + r' \\')
    return L


NCOL = 10

CAPTION_A = r"""Sweep over the truncation order $k_{\max}$ of our cosmographic expansion
(Equation~\ref{eqn:cosmographic-expansion}) in configuration~A. Every order is sampled with
\texttt{dynesty} under the same prior on the expansion history (Section~\ref{sec:setup}), so
the columns $H_0$, $\Delta M_\mathrm{legacy}$, $\Mdes$ and $q_0$ give the posterior mean and
standard deviation at that order, while $\Delta \ln \mathcal{Z}$ is the log-evidence relative
to the most probable order of the same block, which is the row in bold. The column $w_k$ is
the posterior probability of order $k$ (Equation~\ref{eqn:order_weight}). The $\chi^2_{\rm min}$
column is the minimum of the same $\chi^2$ without any prior, and $\Delta \chi^2$ is measured
from the bold row, so that the difference between $-\Delta \chi^2/2$ and $\Delta \ln
\mathcal{Z}$ is the change in the Occam factor between the two orders. The third block
repeats the DES + legacy analysis with the legacy zero point $\Delta M_\mathrm{legacy}$ free
(Section~\ref{sec:systematics-legacy}). We do not report $r_{\rm d}$ in this configuration,
where it is set by the \citetalias{Planck_2020} prior. The $\Lambda$CDM row of each block is
our flat $\Lambda$CDM fit (Equation~\ref{eqn:lcdm-expansion}), shown for reference only. We
give it neither an evidence nor a weight, because its parameters differ from those of the
cosmographic expansion, which does not include $\Omega_{\rm m}$, so it is not one of the
orders being averaged. The
row labelled ``Blend'' is the Bayesian model average over $k_{\max} = 2 - 6$
(Equation~\ref{eqn:bma}), whose uncertainty contains both the width of each posterior and
the spread of the central values between orders. At $k_{\max} = 4$ the nested-sampling
posteriors reproduce the NUTS results of Table~\ref{tab:baseline} to within 0.13 standard
deviations in $H_0$."""

CAPTION_B = r"""As Table~\ref{tab:ordersweep-ls-A}, but for configuration~B, in which
the sound horizon is inferred from the CC data with no CMB input, so $r_{\rm d}$ replaces the
$\Delta M_\mathrm{legacy}$ column. The legacy recalibration is run only against the
\citetalias{Planck_2020} prior and so has no counterpart here."""

BLOCK_PHRASE = {'A_full': r'with $\Delta M_\mathrm{legacy}$ held at zero',
                'A_full_delta': r'with $\Delta M_\mathrm{legacy}$ free',
                'B_full': r'for DES + legacy'}


def bimodal_note(cfg, res):
    hits = []
    for key, _s, _d, _h in BLOCKS[cfg]:
        pcs = sorted(res[key]['blend']['q0']['pieces'], key=lambda d: d['lo'])
        if len(pcs) > 1:
            hits.append((key, pcs))
    if not hits:
        return ''
    where = 'blocks' if len(hits) > 1 else 'block'
    out = [r"In the DES + legacy " + where + r", $k_{\max} = 4$ and 5 carry almost all of the "
           r"weight and place $q_0$ on opposite sides of zero, so the blended $q_0$ is "
           r"bimodal. We therefore print its $1\sigma$ highest-posterior-density region, "
           r"which is the union of two intervals, since the mean of such a mixture falls "
           r"between the two peaks."]
    for key, pcs in hits:
        lo_pc, hi_pc = 100 * pcs[0]['frac'], 100 * pcs[1]['frac']
        first, second = ('higher', 'lower') if hi_pc >= lo_pc else ('lower', 'higher')
        ph = BLOCK_PHRASE[key]
        out.append(r'%s, %.0f per cent of the enclosed probability lies in the %s-$q_0$ '
                   r'interval.' % (ph[0].upper() + ph[1:], max(lo_pc, hi_pc), first))
    return ' '.join(out)


def latex(res):
    L = [r'% Appendix tables -- built by make_evidence_order.py (evidence-weighted sweep).',
         r'% Posteriors and lnZ: chains/evorder (dynesty, envelope prior); chi2_min: the',
         r'% prior-free least-squares sweep; LCDM rows: NUTS ladder/legacy chains.', '']
    for cfg in ('A', 'B'):
        L += [r'\renewcommand{\arraystretch}{1.2}', r'\begin{table*}', r'    \centering',
              r'    \begin{tabular}{' + 'c' * NCOL + '}', r'    \hline',
              '    ' + ' & '.join([
                  r'$k_{\max}$', r'$N_\mathrm{params}$',
                  r'$H_0 \; [\rm{km} \, \rm{s}^{-1} \, \rm{Mpc}^{-1}]$', COL5_HEAD[cfg],
                  r'$\Mdes$', '$q_0$', r'$\chi^2_{\rm min}$', r'$\Delta \chi^2$',
                  r'$\Delta \ln \mathcal{Z}$', '$w_k$']) + r' \\', r'    \hline']
        for bi, (key, _s, _d, head) in enumerate(BLOCKS[cfg]):
            L.append(r'    \multicolumn{%d}{c}{%s} \\ \hline' % (NCOL, head))
            L += ['    ' + x for x in block_lines(key, res[key])]
            L.append(r'    \hline')
        cap = (CAPTION_A if cfg == 'A' else CAPTION_B) + ' ' + bimodal_note(cfg, res)
        L += [r'    \end{tabular}', r'    \caption{%s}' % ' '.join(cap.split()),
              r'    \label{%s}' % LABEL[cfg], r'\end{table*}',
              r'\renewcommand{\arraystretch}{1.0}', '']
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------------------
# 5.  Figure: Delta lnZ and H0 against k_max, configuration A
# ---------------------------------------------------------------------------------------
def figure(res):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 7.8), sharex=True,
                                  gridspec_kw=dict(height_ratios=(1.2, 1.0), hspace=0.08))
    xb = PLIST[-1] + 1.0      # the column on which the blend is drawn
    # |Delta lnZ| < 1: two orders are not distinguished (Jeffreys scale, Trotta 2008)
    ax.axhspan(-1.0, 0.0, color='0.85', zorder=0)
    ax.text(0.988, -0.5, r'$|\Delta \ln \mathcal{Z}| < 1$', fontsize=10, color='0.45',
            va='center', ha='right', transform=ax.get_yaxis_transform(which='grid'))
    dx = {'des_only': -0.08, 'full': 0.0, 'full_dleg': 0.08}
    for key, ser in FIG_KEYS:
        r, s = res[key], mf.SERIES[ser]
        # offset as in the lower panel, so the k_max = 4 stars of DES (0) and of the
        # Delta-free sample (-0.14) do not sit on top of each other
        P = np.array(r['plist'], float) + dx[ser]
        d = np.array(r['dlnZ'], float)
        ax.plot(P, d, '-', color=s['c'], marker='o', ms=6.0, lw=1.8, label=s['lab'],
                zorder=3)
        i = r['plist'].index(dc.BASELINE['P'])
        ax.plot(P[i], d[i], marker='*', ms=15, color=s['c'], mec='w', mew=0.8, zorder=4)
    ax.axvline(dc.BASELINE['P'], color='0.55', lw=1.0, ls=':', zorder=1)
    ax.set_yscale('symlog', linthresh=5.0, linscale=1.2)
    ax.set_yticks([0, -1, -2.5, -5, -10, -100])
    ax.set_yticklabels(['0', '$-1$', '$-2.5$', '$-5$', '$-10$', '$-100$'])
    ax.set_ylim(-400.0, 1.5)
    ax.set_ylabel(r'$\Delta \ln \mathcal{Z} \equiv \ln \mathcal{Z} - \ln \mathcal{Z}_{\max}$',
                  fontsize=13)
    ax.tick_params(labelsize=11)
    # 'best', not a fixed corner: the cubic-basis lnZ fall to about -9 at k_max = 5, right
    # where the old 'lower right' key sat (2026-10-05).
    ax.legend(fontsize=10, loc='best', frameon=True, framealpha=0.9)

    for val, sig, nm, cl, side in ((SPA_H0[0], SPA_H0[1], mf.SPA_LABEL_LCDM, '#a8702a',
                                    'below'),
                                   (SHOES_H0[0], SHOES_H0[1], mbo.SHOES_LABEL, 'green',
                                    'above')):
        ax2.axhspan(val - sig, val + sig, color=cl, alpha=0.15, zorder=0)
        ax2.axhline(val, color=cl, ls=':', lw=1.3, zorder=1)
        dy, va = ((sig + 0.25, 'bottom') if side == 'above' else (-sig - 0.25, 'top'))
        ax2.text(PLIST[0] - 0.15, val + dy, nm, color=cl, fontsize=10, va=va, ha='left',
                 zorder=5)
    lo, hi = [], []
    for key, ser in FIG_KEYS:
        r, s = res[key], mf.SERIES[ser]
        P = np.array(r['plist'], float) + dx[ser]
        H = np.array([o['H0']['mean'] for o in r['orders']])
        S = np.array([o['H0']['std'] for o in r['orders']])
        ax2.errorbar(P, H, yerr=S, fmt='o', color=s['c'], ms=5.5, lw=1.5, capsize=3.0,
                     label=s['lab'], zorder=3)
        b = r['blend']['H0']
        ax2.errorbar([xb + dx[ser]], [b['mean']], yerr=[b['std']], fmt='D', color=s['c'],
                     ms=6.5, lw=1.8, capsize=3.5, zorder=3)
        lo += list(H - S) + [b['mean'] - b['std']]
        hi += list(H + S) + [b['mean'] + b['std']]
    for val, sig in (SPA_H0, SHOES_H0):
        lo.append(val - sig)
        hi.append(val + sig)
    pad = 0.08 * (max(hi) - min(lo))
    ax2.set_ylim(min(lo) - pad, max(hi) + pad)
    ax2.axvline(dc.BASELINE['P'], color='0.55', lw=1.0, ls=':', zorder=1)
    ax2.axvline(PLIST[-1] + 0.5, color='0.3', lw=0.8, zorder=1)
    ax2.set_xticks(PLIST + [xb])
    ax2.set_xticklabels([str(p) for p in PLIST] + ['BMA'])
    ax2.set_xlim(PLIST[0] - 0.3, xb + 0.4)
    ax2.set_xlabel(r'polynomial order $k_\mathrm{max}$', fontsize=13)
    ax2.set_ylabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=13)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=9.5, loc='lower left', frameon=True, framealpha=0.95)
    fig.tight_layout()
    return mf._save(fig, 'evidence_order', tight_bbox=True)


# ---------------------------------------------------------------------------------------
# 6.  The numbers the prose quotes
# ---------------------------------------------------------------------------------------
def numbers(res):
    L = []
    for key, r in res.items():
        L.append('== %s' % key)
        L.append('  lnZ    ' + ' '.join('%9.2f' % v for v in r['lnZ']))
        L.append('  dlnZ   ' + ' '.join('%9.2f' % v for v in r['dlnZ']))
        L.append('  w      ' + ' '.join('%9.3f' % v for v in r['weight']))
        L.append('  w_half ' + ' '.join('%9.3f' % v for v in r['weight_half']))
        L.append('  f_in   ' + ' '.join('%9.4f' % v for v in r['f_in_half']))
        ib = int(np.argmax(r['lnZ']))
        L.append('  dchi2  ' + ' '.join('%9.2f' % (c - r['chi2_min'][ib])
                                        for c in r['chi2_min']))
        L.append('  Occam (dlnZ + dchi2/2 vs best) ' + ' '.join(
            '%7.2f' % (dz + 0.5 * (c - r['chi2_min'][ib]))
            for dz, c in zip(r['dlnZ'], r['chi2_min'])))
        for o in r['orders']:
            L.append('  k=%d H0 %.2f ± %.2f  q0 %+.2f ± %.2f' % (
                o['P'], o['H0']['mean'], o['H0']['std'], o['q0']['mean'], o['q0']['std']))
        for qn, b in r['blend'].items():
            pcs = ' U '.join('[%.3f, %.3f](%.0f%%)' % (p['lo'], p['hi'], 100 * p['frac'])
                             for p in sorted(b['pieces'], key=lambda d: d['lo']))
            L.append('  blend %-5s %.4f ± %.4f (within %.4f, between %.4f)  HPD %s'
                     % (qn, b['mean'], b['std'], b['sig_within'], b['sig_between'], pcs))
        bh = r['blend']['H0']
        L.append('  blend H0 vs SH0ES %+.2f sigma, vs SPA %+.2f sigma'
                 % (tension(bh['mean'], bh['std'], SHOES_H0),
                    tension(bh['mean'], bh['std'], SPA_H0)))
        wh = np.array(r['weight_half'])
        mu = np.array([o['H0']['mean'] for o in r['orders']])
        sg = np.array([o['H0']['std'] for o in r['orders']])
        mh = float(wh @ mu)
        sh = math.sqrt(float(wh @ (sg ** 2 + mu ** 2)) - mh ** 2)
        L.append('  halved band: blend H0 %.2f ± %.2f (vs SH0ES %+.2f, SPA %+.2f sigma)'
                 % (mh, sh, tension(mh, sh, SHOES_H0), tension(mh, sh, SPA_H0)))
    return '\n'.join(L) + '\n'


def main():
    res = build_blocks()
    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    tex = latex(res)
    open(os.path.join(dc.RESULT_DIR, 'table_ordersweep_evidence.tex'), 'w').write(tex)
    json.dump(res, open(os.path.join(dc.RESULT_DIR, 'blend_evidence_samples.json'), 'w'),
              indent=1)
    txt = numbers(res)
    open(os.path.join(dc.RESULT_DIR, 'evidence_order_numbers.txt'), 'w').write(txt)
    figure(res)
    print(txt)
    print(tex)


if __name__ == '__main__':
    main()
