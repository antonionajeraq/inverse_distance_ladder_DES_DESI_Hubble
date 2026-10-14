r"""analyse_mock.py -- read the mock floor sweeps and say whether the observed trend is
what a CMB-calibrated LCDM universe would have produced.

INPUTS (all written by mock_idl.py, plus the published chains)
    results_mock/lsweep_noiseless.json    the fiducial with NO noise -- the estimator's own
                                          bias as a function of z_floor
    results_mock/lsweep_real###.json      the noise ensemble
    results_mock/lsweep_realdata.json     the OBSERVED catalogue through the same estimator
    chains/floorsweep/*.npz               the published posteriors, for the NUTS-vs-LS check

OUTPUTS
    results_mock/floorsweep_mock.pdf/.png   the figure
    results_mock/mock_summary.json          every statistic below, machine readable
    results_mock/mock_summary.txt           the same, printed

WHAT IS BEING TESTED, PRECISELY.  The floors are NOT independent measurements: the sample at
z_floor = 0.090 is a subset of the sample at 0.030, so their errors are strongly correlated
and "is the trend significant" cannot be answered by eye from the per-point error bars.  The
ensemble answers it directly, because each realisation is one coherent draw truncated at
every floor exactly as the real catalogue is.  A trend statistic is therefore evaluated ON
each realisation, and the observed value is placed in that distribution.
"""
import glob
import re
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mock_paths import rglob, rpath

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results_mock')

SPA_H0, SPA_SIG = 67.19, 0.38
SPA_OM = 0.3175
SPA_Q0 = 1.5 * SPA_OM - 1.0
FID_M = -19.40

COL = dict(des_only='k', full='C3', full_joint='#1f4e79')
LAB = dict(des_only='DES', full='DES+legacy',
           full_joint=r'DES+legacy ($\Delta M_\mathrm{legacy}$)')


# =======================================================================================
def _arm(rows, sample, joint):
    """One curve: {z_floor: row} for a (sample, arm) pair."""
    return {round(r['z_floor'], 4): r
            for r in rows if r['sample'] == sample and bool(r['joint_delta']) == joint}


def load(tag):
    p = rpath(RES, 'lsweep_%s.json' % tag)
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    return {'des_only': _arm(rows, 'des_only', False),
            'full': _arm(rows, 'full', False),
            'full_joint': _arm(rows, 'full', True)}


def load_ensemble(suffix=''):
    r"""The noisy realisations for ONE null hypothesis.

    The suffix is the injected-offset tag (mock_idl.realisation_tag), '' for Delta = 0 and
    e.g. '_d+0.0678' for the injected-offset null.  It is matched EXACTLY and not by a
    prefix glob: 'lsweep_real[0-9]*.json' also matches 'lsweep_real007_d+0.0678.json', so a
    loose pattern silently pools two different nulls into one ensemble of twice the size.

    \d{3,} AND NOT \d{3}.  realisation_tag pads to a MINIMUM of three digits, so the
    10,000-member ensemble writes lsweep_real1000.json ... lsweep_real10000.json.  Against
    the old fixed-width \d{3} those names failed the match -- '900' then wanted '.json'
    and found '1' -- and every realisation past 999 was dropped SILENTLY, leaving a
    999-member ensemble that looked like a complete one."""
    out = []
    pat = re.compile(r'lsweep_real(\d{3,})' + re.escape(suffix) + r'\.json$')
    for p in rglob(RES, 'lsweep_real[0-9]*.json'):
        if not pat.search(os.path.basename(p)):
            continue
        rows = json.load(open(p))
        out.append({'des_only': _arm(rows, 'des_only', False),
                    'full': _arm(rows, 'full', False),
                    'full_joint': _arm(rows, 'full', True),
                    'tag': os.path.basename(p)[7:-5]})
    return out


def load_published():
    """The NUTS floor sweep that the manuscript figure is drawn from."""
    import sys
    sys.path.insert(0, HERE)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'chains', 'floorsweep', '*.npz'))):
        d = np.load(p, allow_pickle=False)
        s = json.loads(str(d['scalars_json']))
        rows.append(dict(sample=s['sample'], z_floor=s['z_floor'],
                         joint_delta=bool(s.get('joint_delta')),
                         H0=s['H0_mean'], sig_H0=s['H0_std'],
                         q0=s['q0_mean'], sig_q0=s['q0_std'], M=s['M_mean'],
                         Delta=s.get('Delta_mean'), sig_Delta=s.get('Delta_std'),
                         n_sn=s.get('n_sn'), n_leg=s.get('n_leg')))
    return {'des_only': _arm(rows, 'des_only', False),
            'full': _arm(rows, 'full', False),
            'full_joint': _arm(rows, 'full', True)}


# =======================================================================================
# The statistics.  Each is a function of ONE sweep, so it can be evaluated identically on
# the observed catalogue and on every mock realisation.
# =======================================================================================
def _get(sw, arm, fl, key):
    r = sw[arm].get(round(fl, 4))
    return None if r is None else r.get(key)


STATS = [
    # (name, description, function)
    ('H0_full_0030', 'H0, DES+legacy, z_floor=0.030',
     lambda s: _get(s, 'full', 0.030, 'H0')),
    ('H0_full_0090', 'H0, DES+legacy, z_floor=0.090',
     lambda s: _get(s, 'full', 0.090, 'H0')),
    ('trend_full', 'H0(0.090) - H0(0.030), DES+legacy  [THE TREND]',
     lambda s: _sub(_get(s, 'full', 0.090, 'H0'), _get(s, 'full', 0.030, 'H0'))),
    ('trend_full_073', 'H0(0.073) - H0(0.030), DES+legacy',
     lambda s: _sub(_get(s, 'full', 0.073, 'H0'), _get(s, 'full', 0.030, 'H0'))),
    ('q0trend_full', 'q0(0.090) - q0(0.030), DES+legacy',
     lambda s: _sub(_get(s, 'full', 0.090, 'q0'), _get(s, 'full', 0.030, 'q0'))),
    ('H0_des_0080', 'H0, DES only, z_floor=0.080',
     lambda s: _get(s, 'des_only', 0.080, 'H0')),
    ('trend_des', 'H0(0.300) - H0(0.080), DES only',
     lambda s: _sub(_get(s, 'des_only', 0.300, 'H0'), _get(s, 'des_only', 0.080, 'H0'))),
    ('legacy_jump_073', 'H0(DES,0.073) - H0(DES+legacy,0.073)  [the 10 legacy SNe]',
     lambda s: _sub(_get(s, 'des_only', 0.073, 'H0'), _get(s, 'full', 0.073, 'H0'))),
    ('Delta_0030', 'Delta, joint arm, z_floor=0.030',
     lambda s: _get(s, 'full_joint', 0.030, 'Delta')),
    ('Delta_trend', 'Delta(0.073) - Delta(0.030)',
     lambda s: _sub(_get(s, 'full_joint', 0.073, 'Delta'),
                    _get(s, 'full_joint', 0.030, 'Delta'))),
    ('M_full_0030', 'M_des, DES+legacy, z_floor=0.030',
     lambda s: _get(s, 'full', 0.030, 'M')),
    ('spearman_full', 'Spearman rho of H0 vs z_floor, DES+legacy arm',
     lambda s: _spearman(s, 'full')),
]


def _sub(a, b):
    return None if (a is None or b is None) else a - b


def _spearman(sw, arm):
    fl = sorted(sw[arm])
    if len(fl) < 4:
        return None
    h = [sw[arm][f]['H0'] for f in fl]
    rx = np.argsort(np.argsort(fl)).astype(float)
    ry = np.argsort(np.argsort(h)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def _cp_interval(k, n, cl=0.6827):
    """Clopper-Pearson interval on a binomial fraction.

    NOT sqrt(p(1-p)/n).  A 10,000-member ensemble resolves tails holding tens of events,
    where the Gaussian bar is both too narrow and capable of running below zero; the
    Clopper-Pearson construction is exact for any k, including k = 0.  Same function as
    dh0_ensemble._cp_interval, deliberately, so the two null tests quote intervals built
    the same way."""
    from scipy.stats import beta
    a = 0.5 * (1.0 - cl)
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def summarise(obs, noiseless, ens):
    out = []
    for name, desc, fn in STATS:
        o = fn(obs) if obs else None
        nl = fn(noiseless) if noiseless else None
        vals = np.array([v for v in (fn(e) for e in ens) if v is not None], float)
        row = dict(stat=name, description=desc, observed=o, noiseless=nl,
                   n_ens=int(len(vals)))
        if len(vals):
            row.update(ens_mean=float(vals.mean()), ens_std=float(vals.std(ddof=1)),
                       ens_p16=float(np.percentile(vals, 15.87)),
                       ens_p84=float(np.percentile(vals, 84.13)),
                       ens_p025=float(np.percentile(vals, 2.5)),
                       ens_p975=float(np.percentile(vals, 97.5)))
            if o is not None:
                # two-sided tail probability of a value at least this extreme, and the
                # Gaussian-equivalent z from the ensemble mean and spread
                frac = float(np.mean(np.abs(vals - vals.mean()) >= abs(o - vals.mean())))
                row['p_two_sided'] = frac
                row['z_vs_ensemble'] = float((o - vals.mean()) / vals.std(ddof=1))
                # ONE-SIDED, BY DIRECT COUNT, in the observed direction -- the statistic
                # the trend test is actually quoted as.  k is reported alongside p because
                # a tail of five events out of 300 and one of 170 out of 10,000 are the
                # same p and utterly different evidence, and only k shows which you have.
                n_e = len(vals)
                k1 = int((vals >= o).sum()) if o >= vals.mean() \
                    else int((vals <= o).sum())
                k2 = int((np.abs(vals - vals.mean()) >= abs(o - vals.mean())).sum())
                lo1, hi1 = _cp_interval(k1, n_e)
                row.update(k_one_sided=k1, p_one_sided=k1 / float(n_e),
                           # (k+1)/(n+1): the plain k/n returns EXACTLY ZERO when no
                           # realisation reaches the observed value, which is a claim no
                           # finite ensemble can make.
                           p_one_sided_plus1=(k1 + 1.0) / (n_e + 1.0),
                           p_one_sided_cp68=[lo1, hi1],
                           k_two_sided=k2,
                           p_two_sided_cp68=list(_cp_interval(k2, n_e)))
        out.append(row)
    return out


DELTA_INJECTED = [0.0]   # set by main(); the null's injected legacy offset


def print_summary(rows, fh=None):
    def p(s):
        print(s)
        if fh:
            fh.write(s + '\n')
    p('%-18s %10s %10s %10s %8s %7s %13s %8s  %s'
      % ('statistic', 'observed', 'noiseless', 'ens mean', 'ens sig', 'z',
         'k/n one-sided', 'p1', 'description'))
    p('-' * 150)
    for r in rows:
        def f(k, w=10, d=3):
            v = r.get(k)
            return ('%*s' % (w, '-')) if v is None else ('%*.*f' % (w, d, v))
        kn = ('%6d/%-6d' % (r['k_one_sided'], r['n_ens'])) if 'k_one_sided' in r else '-'
        p('%-18s %s %s %s %s %s %13s %8s  %s'
          % (r['stat'], f('observed'), f('noiseless'), f('ens_mean'), f('ens_std', 8),
             f('z_vs_ensemble', 7, 2), kn,
             ('%.4f' % r['p_one_sided']) if 'p_one_sided' in r else '-',
             r['description']))
    p('-' * 150)
    p('p1 = ONE-SIDED tail in the observed direction, by DIRECT COUNT.  68 per cent '
      'Clopper-Pearson intervals,')
    p('the (k+1)/(n+1) bound and the two-sided counts are in the json.  z is the '
      'Gaussian equivalent and is')
    p('NOT the quoted significance: these tails are not Gaussian.')
    p('ensemble = %d noisy realisations of flat LCDM at H0=%.2f, Om=%.4f, r_d=147.05, '
      'M=%.2f, Delta=%+.4f' % (rows[0]['n_ens'], SPA_H0, SPA_OM, FID_M, DELTA_INJECTED[0]))


# =======================================================================================
# DES-only and DES+legacy are the SAME FIT above z_floor = 0.087, where no legacy
# supernova survives the cut, so the two curves lie exactly on top of each other over half
# the axis.  Drawing DES-only wider and underneath is what keeps both readable; it is a
# plotting choice and not a weighting.
ZORDER = dict(des_only=dict(lw_data=3.4, lw_mock=3.0, ms=6.5, marker='o'),
              full=dict(lw_data=1.8, lw_mock=1.6, ms=4.0, marker='s'),
              full_joint=dict(lw_data=1.8, lw_mock=1.6, ms=4.0, marker='D'))


def figure(obs, noiseless, ens, pub, out='floorsweep_mock'):
    """Four panels.  Top row: H0 and q0 over the published z_floor range.  Bottom row: the
    same H0 zoomed onto the legacy range, where the whole effect lives, and the legacy
    zero-point Delta.

    Everything drawn as a line comes from ONE estimator -- the prior-free chi2 minimum --
    so the observed sweep and the mock band are directly comparable.  The published NUTS
    posterior means are overplotted as crosses; they agree with the least-squares curve to
    a median 0.12 km/s/Mpc and separate only at z_floor >= 0.26, where the coefficient box
    starts to bind (analyse_mock prints the comparison).

    DES and DES+legacy are the SAME FIT above z_floor = 0.087, where no legacy supernova
    survives the cut, so the black curve runs underneath the red one over half the axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    panels = [(axes[0][0], 'H0', r'recovered $H_0$  [km s$^{-1}$ Mpc$^{-1}$]', None),
              (axes[0][1], 'q0', r'recovered $q_0$', None),
              (axes[1][0], 'H0', r'recovered $H_0$  [km s$^{-1}$ Mpc$^{-1}$]',
               (0.023, 0.095)),
              (axes[1][1], 'Delta',
               r'recovered $\Delta M_\mathrm{legacy}$  [mag]', None)]
    style = dict(des_only=dict(lw_d=3.6, lw_m=3.0, ms=7.0, mk='o'),
                 full=dict(lw_d=1.8, lw_m=1.6, ms=4.2, mk='s'),
                 full_joint=dict(lw_d=1.8, lw_m=1.6, ms=4.2, mk='D'))
    handles, labels = [], []

    for ax, key, ylab, xlim in panels:
        use = ('full_joint',) if key == 'Delta' else ('des_only', 'full', 'full_joint')
        first = ax is axes[0][0]
        for arm in use:
            c, st = COL[arm], style[arm]
            fl = sorted(noiseless[arm]) if noiseless else sorted(obs[arm])
            if ens:
                M = [[e[arm].get(f, {}).get(key) for f in fl] for e in ens]
                M = np.asarray([m for m in M if all(x is not None for x in m)], float)
                if M.size:
                    lo95, lo, hi, hi95 = np.percentile(M, [2.5, 15.87, 84.13, 97.5],
                                                       axis=0)
                    ax.fill_between(fl, lo95, hi95, color=c, alpha=0.10, lw=0, zorder=1)
                    h = ax.fill_between(fl, lo, hi, color=c, alpha=0.24, lw=0, zorder=2)
                    if first:
                        handles.append(h)
                        labels.append(LAB[arm] + ': mock null, 68/95 per cent')
            if noiseless:
                v = [noiseless[arm].get(f, {}).get(key) for f in fl]
                h, = ax.plot(fl, v, color=c, ls='--', lw=st['lw_m'],
                             zorder=4 if arm == 'des_only' else 4.5)
                if first:
                    handles.append(h)
                    labels.append(LAB[arm] + ': mock null, no noise')
            if obs:
                fo = sorted(obs[arm])
                v = [obs[arm][f].get(key) for f in fo]
                fo = [f for f, x in zip(fo, v) if x is not None]
                v = [x for x in v if x is not None]
                h, = ax.plot(fo, v, color=c, ls='-', lw=st['lw_d'], marker=st['mk'],
                             ms=st['ms'], zorder=6 if arm == 'des_only' else 6.5)
                if first:
                    handles.append(h)
                    labels.append(LAB[arm] + ': DATA')
            if pub:
                fp = sorted(pub[arm])
                v = [pub[arm][f].get(key) for f in fp]
                fp = [f for f, x in zip(fp, v) if x is not None]
                v = [x for x in v if x is not None]
                if v:
                    h, = ax.plot(fp, v, color=c, ls='none', marker='x', ms=6.0, mew=1.4,
                                 zorder=7)
                    if first and arm == 'full':
                        handles.append(h)
                        labels.append('DATA: published NUTS posterior mean')

        if key == 'H0':
            ax.axhline(SPA_H0, color='#e8a33d', ls=':', lw=1.6)
            ax.axhspan(SPA_H0 - SPA_SIG, SPA_H0 + SPA_SIG, color='#e8a33d', alpha=0.35,
                       lw=0)
            ax.text(0.988, SPA_H0 + 0.55, 'SPA -- the truth the mocks were drawn from',
                    color='#8a5c10', fontsize=8.5, ha='right', va='bottom',
                    transform=ax.get_yaxis_transform())
            ax.axhline(73.17, color='green', ls=':', lw=1.2)
            ax.axhspan(73.17 - 0.86, 73.17 + 0.86, color='green', alpha=0.15, lw=0)
            ax.text(0.988, 73.17 - 1.25, 'Breuval et al.', color='green', fontsize=8.5,
                    ha='right', va='top', transform=ax.get_yaxis_transform())
        elif key == 'q0':
            ax.axhline(SPA_Q0, color='#e8a33d', ls=':', lw=1.6)
            ax.text(0.988, SPA_Q0 + 0.10, 'SPA (truth)', color='#8a5c10', fontsize=8.5,
                    ha='left', va='bottom', transform=ax.get_yaxis_transform())
        else:
            ax.axhline(0.0, color='#e8a33d', ls=':', lw=1.6)
            ax.text(0.988, 0.006, r'truth: $\Delta = 0$', color='#8a5c10', fontsize=8.5,
                    ha='right', va='bottom', transform=ax.get_yaxis_transform())

        ax.axvline(0.0251, color='0.55', ls='--', lw=0.9)
        ax.axvline(0.0733, color='0.55', ls='--', lw=0.9)
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_ylim(62.0, 82.0)
            ax.set_title('the same panel over the legacy redshift range', fontsize=10)
        ax.set_xlabel(r'$z_{\rm floor}$', fontsize=12)
        ax.set_ylabel(ylab, fontsize=12)
        ax.tick_params(labelsize=10)

    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8.6,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('Redshift-floor sweep against mocks drawn from flat $\\Lambda$CDM at the '
                 'SPA values\n'
                 '$H_0 = 67.19$, $\\Omega_{\\rm m} = 0.3175$, $r_{\\rm d} = 147.05$ cMpc, '
                 '$M = -19.40$, $\\Delta M_\\mathrm{legacy} = 0$ '
                 '-- same code, same covariances, same priors, only the data replaced',
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    os.makedirs(RES, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RES, '%s.%s' % (out, ext)), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print('[figure] %s.pdf/.png' % os.path.join(RES, out))


# =======================================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta', type=float, default=0.0,
                    help='injected legacy zero-point offset identifying the null to read '
                         '(0 = the plain LCDM null)')
    a = ap.parse_args()
    suffix = ('_d%+.4f' % a.delta) if a.delta else ''
    DELTA_INJECTED[0] = float(a.delta)

    obs = load('realdata')                       # the OBSERVED catalogue, never injected
    noiseless = load('noiseless' + suffix)
    ens = load_ensemble(suffix)
    try:
        pub = load_published()
    except Exception as exc:
        print('[warn] could not read the published chains (%s)' % exc)
        pub = None
    print('[load] observed %s | noiseless %s | ensemble %d | published %s'
          % ('yes' if obs else 'NO', 'yes' if noiseless else 'NO', len(ens),
             'yes' if pub else 'NO'))

    rows = summarise(obs, noiseless, ens)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, 'mock_summary%s.txt' % suffix), 'w') as fh:
        print_summary(rows, fh)
    with open(rpath(RES, 'mock_summary%s.json' % suffix, write=True), 'w') as fh:
        json.dump(dict(fiducial=dict(H0=SPA_H0, Om=SPA_OM, q0=SPA_Q0, rd=147.05,
                                     M=FID_M, Delta=float(a.delta)),
                       n_realisations=len(ens), stats=rows), fh, indent=2)
    if noiseless or obs:
        figure(obs, noiseless, ens, pub, out='floorsweep_mock' + suffix)

    # The NUTS-vs-least-squares check on the OBSERVED data, which is what licenses reading
    # the ensemble (least squares) against the published sweep (posteriors).
    if obs and pub:
        print('\n--- estimator check on the OBSERVED catalogue: LS minus NUTS posterior '
              'mean ---')
        d = []
        for arm in ('des_only', 'full', 'full_joint'):
            for fl in sorted(set(obs[arm]) & set(pub[arm])):
                dh = obs[arm][fl]['H0'] - pub[arm][fl]['H0']
                rs = obs[arm][fl].get('sig_H0', np.nan) / pub[arm][fl]['sig_H0']
                d.append((arm, fl, dh, dh / pub[arm][fl]['sig_H0'], rs))
        print('  cells %d | median |dH0| %.3f | max |dH0| %.3f | max |dH0|/sigma %.3f | '
              'sigma ratio %.3f-%.3f'
              % (len(d), np.median([abs(x[2]) for x in d]), max(abs(x[2]) for x in d),
                 max(abs(x[3]) for x in d), min(x[4] for x in d), max(x[4] for x in d)))
        for arm, fl, dh, dz, rs in d:
            if abs(dz) > 0.3:
                print('    %-10s %.3f  dH0 %+7.3f (%+.2f sigma)  sigma ratio %.3f'
                      % (arm, fl, dh, dz, rs))


if __name__ == '__main__':
    main()
