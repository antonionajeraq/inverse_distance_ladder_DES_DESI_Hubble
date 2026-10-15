"""
make_tables.py -- render every table in the paper from the cached chains.

READS ONLY.  It never samples.  With the shipped cache under chains/ this runs in a few
seconds and produces the LaTeX for Tables 1-4, so a referee can regenerate every number in
the manuscript without a cluster and without waiting for NUTS.

    python make_tables.py            # all four tables
    python make_tables.py --table 1  # just one

Output goes to results/table<N>.tex and is echoed to stdout.  Each table carries a
provenance comment naming the cells it was built from, so a number in the manuscript can
be traced back to the chain that produced it.
"""
import argparse
import json
import os

import numpy as np

import dovekie_cosmography as dc

# The external comparisons the paper quotes tensions against.
SHOES_MB = (-19.253, 0.027)      # Riess et al. (2022): M_B is set at the second rung, so
                                 # this remains the right reference for it even though the
                                 # local H0 reference is Breuval+2024, which publishes none
SPA_H0 = (67.19, 0.38)           # SPA = SPT-3G + ACT DR6 + Planck 2018 (Camphuis et al.
                                 # 2026, their table 1).  This -- NOT Planck 2018 alone --
                                 # is the CMB reference every tension in the paper is
                                 # quoted against.
SHOES_H0 = (73.17, 0.86)         # Breuval et al. (2024)


def tension(a, sa, b, sb):
    return abs(a - b) / np.hypot(sa, sb)


def _write(name, text):
    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    p = os.path.join(dc.RESULT_DIR, name)
    with open(p, 'w') as fh:
        fh.write(text)
    print(text)
    print('wrote %s\n' % p)
    return p


# =============================================================================
# Table 1 -- the baseline ladder
# =============================================================================
def table1():
    P = dc.BASELINE['P']
    cells, missing = {}, []
    for cfg, smp, bg in dc.LADDER_CELLS:
        cell = '%s_%s_%s_P%d' % (cfg, smp, bg, P)
        c = dc.cache_load('ladder', cell)
        if c is None:
            missing.append('%s/%s/%s' % (cfg, smp, bg))
        else:
            r = dc._unpack(c)
            # a chain that failed the publication gate does not reach a table.  Sampling
            # and publishing are separate decisions; this is where the second one is made.
            dc.require_quality('chains/ladder/%s.npz' % cell, r)
            cells[(cfg, smp, bg)] = r
    if missing:
        raise SystemExit('Table 1 needs these cells, which are not cached:\n  %s\n'
                         'Run: python dovekie_cosmography.py --task ladder'
                         % '\n  '.join(missing))

    def row(cfg, smp, bg):
        r = cells[(cfg, smp, bg)]
        label = {'des_only': 'DES', 'full': 'DES+legacy'}[smp]
        rd = ('$%.2f\\pm%.2f$ (prior)' % (dc.BASELINE['RD_PLANCK'], dc.BASELINE['SIG_RD'])
              if cfg == 'A' else '$%.2f\\pm%.2f$' % (r['rd_mean'], r['rd_std']))
        nd = 3 if cfg == 'A' else 2       # config B widens M_des by an order of magnitude
        return (' & %s        & $%.2f\\pm%.2f$ & %s & $%.*f\\pm%.*f$ & $%.2f\\pm%.2f$'
                ' & $%.2f$\\\\'
                % (label, r['H0_mean'], r['H0_std'], rd,
                   nd, r['M_mean'], nd, r['M_std'],
                   r['q0_mean'], r['q0_std'], r['chi2_dof']))

    L = [r'% Table 1 -- built by make_tables.py from chains/ladder/*.npz',
         r'% cells: ' + ', '.join('%s/%s/%s' % k for k in sorted(cells)),
         r'\begin{table*}', r'\centering',
         r'\caption{Baseline results at $P=%d$.}' % P,
         r'\label{tab:baseline-new}',
         r'\begin{tabular}{llccccc}', r'\hline',
         r'configuration & sample & $H_0 \; [\rm{km} \, \rm{s}^{-1} \, \rm{Mpc}^{-1}]$ '
         r'& $r_d$ [Mpc] & $\Mdes$ & $q_0$ & $\chi^2/{\rm dof}$\\', r'\hline']
    for cfg, head in (('A', r'(A) SNIa $+$ BAO $+$ Planck $r_d$'),
                      ('B', r'(B) SNIa $+$ BAO $+$ CC, $r_d$ free (no CMB)')):
        L += [r'\multicolumn{7}{l}{%s}\\' % head, r'\hline',
              r'\multicolumn{7}{l}{\textit{cosmography, $P=%d$}}\\' % P]
        L += [row(cfg, s, 'cosmo') for s in ('des_only', 'full')]
        L += [r'\noalign{\smallskip}',
              r'\multicolumn{7}{l}{\textit{flat $\Lambda$CDM}}\\']
        L += [row(cfg, s, 'lcdm') for s in ('des_only', 'full')]
        L += [r'\hline']
    L += [r'\end{tabular}', r'\end{table*}']

    print('\nDerived quantities quoted in the text:')
    for cfg in ('A', 'B'):
        for bg in ('cosmo', 'lcdm'):
            for smp in ('des_only', 'full'):
                r = cells[(cfg, smp, bg)]
                print('  %s %-6s %-9s  M_des vs SH0ES %.2f sigma | H0 vs SPA %.2f | '
                      'vs SH0ES %.2f'
                      % (cfg, bg, smp,
                         tension(r['M_mean'], r['M_std'], *SHOES_MB),
                         tension(r['H0_mean'], r['H0_std'], *SPA_H0),
                         tension(r['H0_mean'], r['H0_std'], *SHOES_H0)))
                if cfg == 'B':
                    print('%22s  r_d vs Planck %.2f sigma'
                          % ('', tension(r['rd_mean'], r['rd_std'],
                                         dc.BASELINE['RD_PLANCK'], dc.BASELINE['SIG_RD'])))

    print('\nDoes any posterior reach the prior envelope?')
    for k in sorted(cells):
        r = cells[k]
        if r.get('env_margin'):
            print('  %s/%s  0.1%% quantile of the distance to the envelope (in E): lower '
                  '%.3f, upper %.3f; at_bound %s'
                  % (k[0], k[1], r['env_margin']['lower'], r['env_margin']['upper'],
                     r['at_bound'] if r['at_bound'] else 'empty'))
    return _write('table1_baseline.tex', '\n'.join(L) + '\n')


# =============================================================================
# Table 2 -- the model comparison
# =============================================================================
def table2():
    import dovekie_evidence as de
    R = de.load_grid()
    if not R:
        raise SystemExit('Table 2 needs the evidence grid.\n'
                         'Run: python dovekie_evidence.py --task grid')
    L = [r'% Table 2 -- built by make_tables.py from chains/evidence/*.npz',
         r'\begin{table*}', r'\centering',
         r'\caption{Bayesian model comparison of a late-time solution (exp) and CPL '
         r'against $\Lambda$CDM.}',
         r'\label{tab:evidence}',
         r'\begin{tabular}{llccccccc}', r'\hline',
         r'Sample & Anchor & Model & $\ln B_{10}$ & $\Omega_{\rm m}$ & $w_0/A$ '
         r'& $w_a/z_1$ & $H_0 \, [\rm{km} \, \rm{s}^{-1} \, \rm{Mpc}^{-1}]$ '
         r'& $\Delta M_\mathrm{legacy} \, [\rm{mag}]$ \\',
         r'\hline']
    tex_sample = {'des_only': 'DES', 'full': 'DES\\,+\\,legacy',
                  'full_dleg': 'DES\\,+\\,legacy ($\\Delta$ free)'}
    tex_anchor = {'none': 'none', 'lfree3': 'SNIa-indep.', 'breuval': 'SH0ES'}
    for si, s in enumerate(('des_only', 'full', 'full_dleg')):
        first_s = True
        for ai, a in enumerate(('none', 'lfree3', 'breuval')):
            ref = R.get((s, a, 'lcdm'))
            if ref is None:
                continue
            first_a = True
            for m in ('exp', 'cpl'):
                r = R.get((s, a, m))
                if r is None:
                    continue
                p2, p3 = ('A', 'z1') if m == 'exp' else ('w0', 'wa')
                lnB = r['logz'] - ref['logz']
                err = np.hypot(r['logz_err'], ref['logz_err'])
                nd2 = 2 if m == 'exp' else 3
                # Delta exists ONLY where the legacy zero-point is free, i.e. the
                # `full_dleg` cells.  It is marginalised analytically and never sampled
                # (dovekie_evidence.delta_moments), so mean +/- std is the only quote
                # there is -- and the right one, the conditional being Gaussian.  A dash
                # elsewhere marks a parameter the model does not have, not a gap.
                dleg = ('$%.3f \\pm %.3f$' % (r['Delta_mean'], r['Delta_std'])
                        if 'Delta_mean' in r else '---')
                L.append('%s & %s & %s & $%+.2f \\pm %.2f$ & $%.3f \\pm %.3f$ & '
                         '$%.*f \\pm %.*f$ & $%.3f \\pm %.3f$ & $%.2f \\pm %.2f$ & %s \\\\'
                         % (tex_sample[s] if first_s else '',
                            tex_anchor[a] if first_a else '', m,
                            lnB, err, r['Om_mean'], r['Om_std'],
                            nd2, r[p2 + '_mean'], nd2, r[p2 + '_std'],
                            r[p3 + '_mean'], r[p3 + '_std'],
                            r['H0_at_0_mean'], r['H0_at_0_std'], dleg))
                first_s = first_a = False
            L.append(r'\noalign{\smallskip}')
        L.append(r'\hline')
    L += [r'\end{tabular}', r'\end{table*}']

    print('\nP(A>0) and the LCDM BACKGROUND H0.  P(A>0) belongs beside every lnB10 because')
    print('a two-sided A ~ U(-10,10) charges M1 about 0.9 nat of Occam volume the data')
    print('never constrain, so the parameter and the Bayes factor can say different things.')
    print('The background H0_LCDM is NOT in the table -- the H0 column there is H(0) =')
    print('H0_LCDM + A -- but it must not be lost: it stays near 68-70 even in cells whose')
    print('H(0) reaches 73, which is the whole point of the transition.')
    print('\n  %-26s %-13s %8s %16s %10s %10s'
          % ('sample', 'anchor', 'P(A>0)', 'A', 'H(0)', 'H0_LCDM'))
    for s in ('des_only', 'full', 'full_dleg'):
        for a in ('none', 'lfree3', 'breuval'):
            r = R.get((s, a, 'exp'))
            if r and 'P_A_gt_0' in r:
                print('  %-26s %-13s %8.3f %8.2f +- %-5.2f %10.2f %10.2f'
                      % (s, a, r['P_A_gt_0'], r['A_mean'], r['A_std'],
                         r['H0_at_0_mean'], r['H0_LCDM_mean']))
            d = R.get((s, a, 'exp'), {})
            if 'Delta_mean' in d:
                print('  %-26s %-13s   Delta = %+.4f +- %.4f mag'
                      % ('', '', d['Delta_mean'], d['Delta_std']))
    return _write('table2_evidence.tex', '\n'.join(L) + '\n')


# =============================================================================
# Table 3 -- the legacy zero-point and H0
# =============================================================================
def table3():
    P = dc.BASELINE['P']
    rows = {}
    for bg in ('cosmo', 'lcdm'):
        for smp, jd, label in (('des_only', False, 'Legacy excluded (DES only)'),
                               ('full', False, 'Uncorrected'),
                               ('full', True, 'Legacy zero-point sampled')):
            cell = '%s_%s_%s_P%d' % (bg, smp, 'joint' if jd else 'fixed', P)
            c = dc.cache_load('legacy', cell)
            if c is None:
                raise SystemExit('Table 3 needs the legacy cells.\n'
                                 'Run: python dovekie_cosmography.py --task legacy')
            r = dc._unpack(c)
            dc.require_quality('chains/legacy/%s.npz' % cell, r)
            rows[(label, bg)] = r

    def cell(label, bg, key='H0'):
        r = rows[(label, bg)]
        return '$%.2f \\pm %.2f$' % (r[key + '_mean'], r[key + '_std'])

    L = [r'% Table 3 -- built by make_tables.py from chains/legacy/*.npz',
         r'\begin{table}', r'\centering',
         r'\caption{Baseline $H_0$ after placing the legacy supernovae onto the DES '
         r'relation.}', r'\label{tab:legacy-h0}',
         r'\begin{tabular}{lcc}', r'\hline',
         r' & \multicolumn{2}{c}{$H_0$ [km s$^{-1}$ Mpc$^{-1}$]} \\',
         r'\cline{2-3}', r'Case & cosmography & $\Lambda$CDM \\', r'\hline']
    for label in ('Legacy excluded (DES only)', 'Uncorrected'):
        L.append('%s & %s & %s \\\\' % (label, cell(label, 'cosmo'), cell(label, 'lcdm')))
    L.append(r'\hline')
    lab = 'Legacy zero-point sampled'
    L.append('%s & %s & %s \\\\' % (lab, cell(lab, 'cosmo'), cell(lab, 'lcdm')))
    L.append(r'\quad $\Delta$ [mag] & $%+.4f \pm %.4f$ & $%+.4f \pm %.4f$ \\'
             % (rows[(lab, 'cosmo')]['Delta_mean'], rows[(lab, 'cosmo')]['Delta_std'],
                rows[(lab, 'lcdm')]['Delta_mean'], rows[(lab, 'lcdm')]['Delta_std']))
    L += [r'\hline', r'\end{tabular}', r'\end{table}']

    print('\nDerived quantities quoted in the text:')
    for lab in ('Legacy excluded (DES only)', 'Uncorrected', 'Legacy zero-point sampled'):
        r = rows[(lab, 'cosmo')]
        print('  %-28s cosmography H0 vs SH0ES %.2f sigma'
              % (lab, tension(r['H0_mean'], r['H0_std'], *SHOES_H0)))
    j = rows[('Legacy zero-point sampled', 'cosmo')]
    print('  Delta significance %.2f sigma;  corr(Delta,H0) = %+.2f;  '
          'corr(Delta,M_des) = %+.3f'
          % (abs(j['Delta_mean']) / j['Delta_std'], j['corr_Delta_H0'],
             j['corr_Delta_M']))

    lad_path = os.path.join(dc.RESULT_DIR, 'legacy_delta_ladder.json')
    if os.path.exists(lad_path):
        lad = json.load(open(lad_path))
        print('\nThe four-rung Delta ladder (prior-free least squares):')
        for k, v in lad.items():
            print('  %-34s Delta = %+.4f +- %.4f  (%.1f sigma)'
                  % (v['label'], v['Delta'], v['sig'], v['z']))
    return _write('table3_legacy.tex', '\n'.join(L) + '\n')


# =============================================================================
# Table 4 -- the Alcock-Paczynski consistency test
# =============================================================================
def table4():
    p = os.path.join(dc.RESULT_DIR, 'ap_table.json')
    if not os.path.exists(p):
        raise SystemExit('Table 4 needs the AP results.\n'
                         'Run: python dovekie_cosmography.py --task ap')
    res = json.load(open(p))
    # ap_table.json is a derived aggregate with no manifest of its own, so the gate is
    # applied to the CELL CACHES it was assembled from.  Checking the JSON alone would
    # accept a table built from a divergent chain.
    for r in res:
        cell = '%s_P%d' % (r['sample'], dc.BASELINE['P'])
        c = dc.cache_load('ap', cell)
        if c is None:
            raise SystemExit('results/ap_table.json has a %r row but chains/ap/%s.npz is '
                             'missing or stale -- the JSON predates the chains.  Re-run: '
                             'python dovekie_cosmography.py --task ap' % (r['sample'], cell))
        dc.require_quality('chains/ap/%s.npz' % cell,
                           json.loads(str(c['diag_json'])))
    L = [r'% Table 4 -- built by make_tables.py from results/ap_table.json',
         r'\begin{table}', r'\centering',
         r'\caption{Supernova-shape prediction of the DESI DR2 Alcock--Paczynski ratio '
         r'$F_{\rm AP}=D_M/D_H$.}', r'\label{tab:AP-new}',
         r'\begin{tabular}{lccccc}', r'\hline',
         r'Sample & $z$ & SN $F_{\rm AP}$ & DESI $F_{\rm AP}$ & pull & $p$ \\', r'\hline']
    for r in res:
        name = {'des_only': 'DES', 'full': 'DES+legacy'}[r['sample']]
        for i, z in enumerate(r['z']):
            L.append('%s & $%.3f$ & $%.4f \\pm %.4f$ & $%.4f \\pm %.4f$ & $%+.2f$ & %s \\\\'
                     % (name if i == 0 else '', z, r['fap_sn'][i], r['sig_sn'][i],
                        r['fap_desi'][i], r['sig_desi'][i], r['pull'][i],
                        ('$%.3f$' % r['p']) if i == 0 else ''))
        L.append(r'\noalign{\smallskip}')
    L += [r'\hline', r'\end{tabular}', r'\end{table}']
    for r in res:
        print('  %-9s joint chi2 = %.2f (dof=%d)  p = %.3f'
              % (r['sample'], r['chi2'], r['dof'], r['p']))
    return _write('table4_ap.tex', '\n'.join(L) + '\n')


def table5():
    r"""The prior on every sampled parameter, read from the code that imposes it.

    Since 2026-09-28 the cosmographic block is the envelope prior (dovekie_cosmography
    section 7): H0 is uniform, and the shape is uniform in C_1..C_P over every polynomial
    inside the envelope on E.  No C_k (nor q0 = C_1 - 1) has a flat marginal, so each is
    given its exact support and its prior standard deviation.

    Everything here is READ from BASELINE / dovekie_evidence rather than typed, so the
    table cannot drift away from the sampler.
    """
    import dovekie_evidence as de
    B = dc.BASELINE
    P = B['P']

    rows = []                       # (section, symbol, prior, note)
    rows.append(('Cosmographic background (Equation~\\ref{eqn:cosmographic-expansion})',
                 None, None, None))
    rows.append((None, r'$H_0$',
                 r'$\mathcal{U}(%.0f,\,%.0f)\kmsMpc$' % (B['H0_LO'], B['H0_HI']), ''))
    # The shape: uniform in C_1..C_P over every polynomial inside the envelope.  None of the
    # C_k has a flat marginal, so each gets its exact support (a linear programme over the
    # enforcement grid) and the standard deviation of exact prior draws.
    from scipy.optimize import linprog
    yg = dc.envelope_grid()
    elo, ehi = dc.envelope_bounds(yg)
    A = np.power(yg[:, None], np.arange(1, P + 1)[None, :])
    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([ehi - 1.0, 1.0 - elo])
    draws, _n = dc.sample_envelope_prior(P, 400000, seed=1)
    rows.append((None, r'$C_1, \dots, C_%d$' % P,
                 # the independent variable is printed as a~
                 r'uniform over $%s \leq E(\widetilde{a}) \leq %s$, $\widetilde{a} \leq %.2f$'
                 % (dc._poly_str(B['E_LOWER']).replace(' ', '').replace('y', r'\widetilde{a}'),
                    dc._poly_str(B['E_UPPER']).replace(' ', '').replace('y', r'\widetilde{a}'),
                    B['Y_MAX']), 'jointly'))
    for k in range(1, P + 1):
        cvec = np.zeros(P)
        cvec[k - 1] = 1.0
        lo_k = linprog(cvec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * P).fun
        hi_k = -linprog(-cvec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * P).fun
        sym = r'$q_0 = C_1 - 1$' if k == 1 else r'$C_%d$' % k
        off = -1.0 if k == 1 else 0.0
        rows.append((None, sym,
                     r'$[%.2f,\,%.2f]$, s.d.\ $%.2f$' % (lo_k + off, hi_k + off,
                                                       draws[:, k].std()),
                     'derived, non-uniform marginal' if k == 1 else ''))
    rows.append((None, r'$\Mdes$',
                 r'$\mathcal{U}(%.0f,\,%.0f)$~mag' % (B['M_LO'], B['M_HI']), ''))
    rows.append((None, r'$r_d$ (configuration~A)',
                 r'$\mathcal{N}(%.2f,\,%.2f)$~cMpc' % (B['RD_PLANCK'], B['SIG_RD']),
                 r'\citetalias{Planck_2020}'))
    rows.append((None, r'$r_d$ (configuration~B)',
                 r'$\mathcal{U}(%.0f,\,%.0f)$~cMpc' % (B['RD_FREE_LO'], B['RD_FREE_HI']),
                 r'derived from the CC'))
    rows.append((None, r'$\Delta$',
                 r'$\mathcal{U}(-%.0f,\,+%.0f)$~mag' % (B['DELTA_HALFWIDTH'],
                                                        B['DELTA_HALFWIDTH']),
                 r'${\sim}25\sigma$ wide, inert'))

    rows.append((r'Flat $\Lambda$CDM (Equation~\ref{eqn:lcdm-expansion})',
                 None, None, None))
    rows.append((None, r'$\Omega_{\rm m}$',
                 r'$\mathcal{U}(%.2f,\,%.2f)$' % (B['OM_LO'], B['OM_HI']), ''))
    rows.append((None, r'$H_0$',
                 r'$\mathcal{U}(%.0f,\,%.0f)\kmsMpc$' % (B['H0_LO'], B['H0_HI']), ''))

    # The model comparison samples (omega_b, Omega_bc, H0, ...) and derives r_d from the
    # compressed CMB (dovekie_evidence_cmb.make_ptform); there is no r_d or Omega_m prior.
    import dovekie_evidence_cmb as dec
    rows.append((r'Model comparison (Tables~\ref{tab:evidence} and \ref{tab:logz1})',
                 None, None, None))
    rows.append((None, r'$\omega_{\rm b}$',
                 r'$\mathcal{U}(%.3f,\,%.3f)$' % (dec.WB_LO, dec.WB_HI), ''))
    rows.append((None, r'$\Omega_{\rm bc}$',
                 r'$\mathcal{U}(%.2f,\,%.2f)$' % (de.OM_LO, de.OM_HI), ''))
    rows.append((None, r'$H_0$',
                 r'$\mathcal{U}(%.0f,\,%.0f)\kmsMpc$' % (de.H0_LO, de.H0_HI),
                 r'wider than the cosmographic box'))
    rows.append((None, r'$A$',
                 r'$\mathcal{U}(-%.0f,\,+%.0f)\kmsMpc$' % (de.A_MAX, de.A_MAX),
                 r'two-sided by construction'))
    rows.append((None, r'$z_1$',
                 r'$\mathcal{U}(%.3f,\,%.2f)$' % (de.Z1_MIN, de.Z1_MAX),
                 r'or log-uniform on the same range (Table~\ref{tab:logz1})'))
    rows.append((None, r'$w_0$',
                 r'$\mathcal{U}(%.0f,\,%.0f)$' % (de.W0_LO, de.W0_HI), ''))
    rows.append((None, r'$w_a$',
                 r'$\mathcal{U}(%.0f,\,%.0f)$' % (de.WA_LO, de.WA_HI), ''))

    L = [r'% Table 5 -- built by make_tables.py from dovekie_cosmography.BASELINE',
         r'%              and dovekie_evidence, i.e. from the code that imposes them.',
         r'\begin{table}', r'\centering',
         r'\caption{\nt{Priors on every sampled parameter, where $\mathcal{U}$ is '
         r'uniform and $\mathcal{N}$ is Gaussian. The cosmographic block is for '
         r'$k_{\max} = 4$. The prior on the shape coefficients $C_1, \dots, C_4$ is '
         r'uniform over every polynomial that satisfies the envelope of '
         r'Equation~\ref{eqn:envelope}, and zero elsewhere, so it has no width on any '
         r'individual coefficient. The marginals of $q_0$ and $C_2, \dots, C_4$ are '
         r'therefore not uniform, and we give their support and standard deviation, '
         r'computed from $4 \times 10^5$ draws of the prior. Importantly, no posterior in this '
         r'paper approaches a boundary of the envelope, so none of our constraints is '
         r'conferred by a prior edge.}}',
         r'\label{tab:priors}',
         r'\begin{tabular}{llp{0.30\linewidth}}', r'\hline',
         r'Parameter & Prior & Note \\', r'\hline']
    for sec, sym, pri, note in rows:
        if sec is not None:
            L.append(r'\multicolumn{3}{l}{\textit{%s}} \\' % sec)
            continue
        L.append(r'%s & %s & %s \\' % (sym, pri, note))
    L += [r'\hline', r'\end{tabular}', r'\end{table}']
    return _write('table5_priors.tex', '\n'.join(L) + '\n')


def _sci(v):
    """LaTeX scientific notation for the coefficient half-widths (1e3 .. 7.5e5)."""
    e = int(np.floor(np.log10(v)))
    m = v / 10.0 ** e
    return (r'10^{%d}' % e) if abs(m - 1.0) < 1e-9 else (r'%g \times 10^{%d}' % (m, e))


def census():
    r"""How many data points enter chi2, with and without the legacy supernovae.

    The data counts should not have to be reconstructed from the text.
    Counts are taken from `build_blocks`, i.e. AFTER the sigma_mu < 1 contaminant cut."""
    B = dc.BASELINE
    out = {}
    for smp in ('des_only', 'full'):
        b = dc.build_blocks(smp, True)      # use_cc=True so the CC block is built too
        z = np.asarray(b['z_sn'])
        out[smp] = dict(n_sn=len(z), zmin=float(z.min()), zmax=float(z.max()),
                        n_bao=len(b['O']), n_cc=len(b['cc'][0]))
    d, f = out['des_only'], out['full']
    bf = dc.build_blocks('full', True)
    zf = np.asarray(bf['z_sn'])
    # the legacy flag itself, not a z-cut: 10 legacy SNe lie ABOVE the DES floor and a
    # cut would miscount them into the DES sample
    is_leg = np.asarray(dc.legacy_indicator('full')) > 0.5
    z_leg = zf[is_leg]
    n_leg = int(is_leg.sum())

    lines = []
    A = lambda r: r['n_sn'] + r['n_bao'] + 1                 # + the r_d prior point
    Bc = lambda r: r['n_sn'] + r['n_bao'] + r['n_cc']
    lines.append('Data census (after the sigma_mu < %.1f cut):' % B['CLEAN_MUERR_MAX'])
    lines.append('  %-11s %5s %5s %5s | %-7s %-7s' %
                 ('sample', 'SNIa', 'BAO', 'CC', 'N(A)', 'N(B)'))
    for name, r in (('DES', d), ('DES+legacy', f)):
        lines.append('  %-11s %5d %5d %5d | %-7d %-7d'
                     % (name, r['n_sn'], r['n_bao'], r['n_cc'], A(r), Bc(r)))
    lines.append('  legacy alone: %d SNIa, z in [%.4f, %.4f]; %d of them lie below the '
                 'DES floor z = %.4f, so only %d overlap it in redshift'
                 % (n_leg, z_leg.min(), z_leg.max(), int((z_leg < d['zmin']).sum()),
                    d['zmin'], int((z_leg >= d['zmin']).sum())))
    lines.append('  N(A) = SNIa + BAO + 1 (the r_d prior).  N(B) = SNIa + BAO + CC.')
    lines.append('  N_params = (k_max + 1) + 2 = %d at k_max = %d.' % (B['P'] + 3, B['P']))
    txt = '\n'.join(lines)
    _write('data_census.txt', txt)
    return out


TABLES = {1: table1, 2: table2, 3: table3, 4: table4, 5: table5}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    ap.add_argument('--table', type=int, choices=sorted(TABLES), nargs='*')
    a = ap.parse_args(argv)
    print(dc.banner())
    if not a.table:
        print('\n' + '=' * 78 + '\n DATA CENSUS\n' + '=' * 78)
        census()
    for n in (a.table or sorted(TABLES)):
        print('\n' + '=' * 78 + '\n TABLE %d\n' % n + '=' * 78)
        try:
            TABLES[n]()
        except SystemExit as e:
            print(e)


if __name__ == '__main__':
    main()
