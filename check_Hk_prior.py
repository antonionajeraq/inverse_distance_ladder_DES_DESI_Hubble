r"""
check_Hk_prior.py -- is the prior on H_k = H(k y_max / k_max) flat where the data are?

READS ONLY.  It never samples the likelihood.

    python check_Hk_prior.py            # results/Hk_prior_check.txt
    python check_Hk_prior.py --n 400000 # quicker, noisier

The question, asked when the cosmography triangles switched from
C_2..C_4 to H_1..H_4: over the posterior ranges of the H_k, is the prior essentially flat,
once the rejection that keeps E(y) inside the envelope is accounted for?

WHAT IS EXACT.  The prior is H0 ~ U(50, 90) times uniform in C_1..C_K over the envelope set
A (0 <= E <= 2 + 7y on [0, y_max]).  H_k = H0 E(y_k) and E(y_k) = 1 + sum_j C_j y_k^j, so
at FIXED H0 the map C -> (H_1..H_K) is linear with Jacobian H0^K |det V|: the prior is
EXACTLY uniform in the H_k inside A.  Jointly with H0 the density is

    p(H0, H_1..H_K)  propto  H0^-K  x  1[E in A]  x  (positivity factor),

so any tilt across a posterior comes from (i) H0^-K, (ii) the edge of A, or (iii) the
positivity factor.  Sections 1 and 2 below measure (ii) and (iii) on the posterior draws;
section 3 shows how far (i) moves the answer by importance-reweighting each posterior by
H0^+K, i.e. to a prior flat in H_0..H_K jointly.

WHAT IS MEASURED.  Section 4: the 1D MARGINAL prior of each H_k -- what a reader would
call "the prior" on a triangle's diagonal -- from exact prior draws
(`prior_reference.cosmo_prior`: the same envelope rejection, the same positivity weight),
pushed through the same map.  Over each posterior's +/-3 sigma window it reports the
log-slope of that density in units of the posterior sigma, and the max/min density ratio
across the 1-sigma and 3-sigma windows.
"""
import argparse
import os

import numpy as np

import dovekie_cosmography as dc
import prior_reference as pr

CELLS = [('A', 'des_only'), ('A', 'full'), ('B', 'des_only'), ('B', 'full')]


def h_k(H0, C, y_max):
    """(N, K+1): H_0..H_K at y_k = k y_max / K, from H0 (N,) and C_1..C_K (N, K)."""
    K = C.shape[1]
    yk = y_max * np.arange(K + 1) / K
    E = 1.0 + C @ (yk[None, :] ** np.arange(1, K + 1)[:, None])
    return H0[:, None] * E


def wq(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return np.interp(np.asarray(q) * c[-1], c, x[i])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000000, help='prior draws per cell')
    a = ap.parse_args(argv)
    P = dc.BASELINE['P']
    y_max = dc.shape_y_max()
    out = []
    say = lambda s='': (print(s), out.append(s))

    say('H_k = H(k y_max / k_max), y_max = %.2f, k_max = %d  (y_k = %s; z_k = %s)'
        % (y_max, P, ', '.join('%.3f' % (y_max * k / P) for k in range(P + 1)),
           ', '.join('%.3f' % (y_max * k / P / (1 - y_max * k / P)) for k in range(P + 1))))
    say('prior: H0 ~ U(%g,%g), uniform in C_1..C_%d over 0 <= E <= 2+7y on [0, %.2f]; '
        'prior draws per cell: %d' % (dc.BASELINE['H0_LO'], dc.BASELINE['H0_HI'], P, y_max,
                                       a.n))

    for cfg, smp in CELLS:
        cell = '%s_%s_cosmo_P%d' % (cfg, smp, P)
        c = dc.cache_load('ladder', cell)
        H0p, Cp = np.asarray(c['H0']), np.asarray(c['C'])
        Hp = h_k(H0p, Cp, y_max)
        say()
        say('=' * 92)
        say('%s   (N_post = %d)' % (cell, len(H0p)))

        # 1. the envelope: how near does the posterior come?
        lo_m, hi_m = dc.envelope_margins(np.concatenate([np.ones((len(Cp), 1)), Cp], 1),
                                         y_max)
        say('1. envelope: min over posterior of (E - 0) = %.3f, of (2+7y - E) = %.3f  '
            '[ENV_TOL %.2f] -> %s'
            % (lo_m.min(), hi_m.min(), dc.ENV_TOL,
               'never active' if min(lo_m.min(), hi_m.min()) > dc.ENV_TOL else 'TOUCHED'))

        # 2. the positivity factor on the posterior draws
        zg = pr._zgrid(smp, cfg == 'B')
        yg = zg / (1 + zg)
        cdim = H0p[:, None] * np.concatenate([np.ones((len(Cp), 1)), Cp], 1)
        lw = pr._positivity_logw(cdim @ (yg[:, None] ** np.arange(P + 1)[None, :]).T, zg)
        say('2. positivity factor on the posterior: max |log w| = %.2e  -> %s'
            % (np.abs(lw).max(), 'inert' if np.abs(lw).max() < 1e-6 else 'ACTIVE'))

        # 3. the H0^-K Jacobian: reweight to a prior flat in (H_0..H_K) jointly
        w = (H0p / H0p.mean()) ** P
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        span = wq(H0p, np.ones_like(H0p), [0.00135, 0.99865])
        say('3. joint prior propto H0^-%d: across the posterior 3-sigma H0 window '
            '[%.1f, %.1f] it changes by x%.2f (1-sigma window: x%.2f).'
            % (P, span[0], span[1], (span[1] / span[0]) ** P,
               (wq(H0p, np.ones_like(H0p), 0.84135) / wq(H0p, np.ones_like(H0p), 0.15865)) ** P))
        say('   Reweighted by H0^+%d to a prior FLAT in H_0..H_%d (ESS %.0f of %d):'
            % (P, P, ess, len(H0p)))
        q0p = Cp[:, 0] - 1
        for name, x in [('H_0', Hp[:, 0]), ('q_0', q0p)] + \
                       [('H_%d' % k, Hp[:, k]) for k in range(1, P + 1)]:
            m0, s0 = x.mean(), x.std()
            m1 = np.sum(w * x)
            s1 = np.sqrt(np.sum(w * (x - m1) ** 2))
            say('     %-4s %9.3f +- %-7.3f -> %9.3f +- %-7.3f   shift %+.2f sigma, width x%.3f'
                % (name, m0, s0, m1, s1, (m1 - m0) / s0, s1 / s0))

        # 4. the 1D marginal prior of each H_k across the posterior window
        draws, wp = pr.cosmo_prior(smp, cfg, 'cosmo', P=P, n=a.n)
        Cpr = np.column_stack([draws['C[%d]' % (j + 1)] for j in range(P)])
        Hpr = h_k(draws['H0'], Cpr, y_max)
        say('4. 1D marginal prior density over each posterior window '
            '(slope = d ln p / d x x sigma_post; ratio = max/min of the fitted p):')
        say('     %-4s %18s %16s %10s %12s %12s %8s'
            % ('', 'posterior', '3sig window', 'slope', '1sig ratio', '3sig ratio', 'N_prior'))
        for k in range(P + 1):
            x, xp = Hp[:, k], Hpr[:, k]
            m, s = x.mean(), x.std()
            lo, hi = m - 3 * s, m + 3 * s
            nb = 12
            edges = np.linspace(lo, hi, nb + 1)
            cnt, _ = np.histogram(xp, edges, weights=wp)
            cnt2, _ = np.histogram(xp, edges, weights=wp ** 2)
            ok = cnt > 0
            mid = 0.5 * (edges[1:] + edges[:-1])
            # weighted least-squares quadratic in ln(density), Poisson errors
            t = (mid - m) / s
            sig = np.sqrt(cnt2[ok]) / cnt[ok]
            A = np.vander(t[ok], 3, increasing=True)
            coef = np.linalg.lstsq(A / sig[:, None], np.log(cnt[ok]) / sig, rcond=None)[0]
            f = lambda tt: np.polyval(coef[::-1], tt)
            tt3, tt1 = np.linspace(-3, 3, 61), np.linspace(-1, 1, 21)
            r3 = np.exp(f(tt3).max() - f(tt3).min())
            r1 = np.exp(f(tt1).max() - f(tt1).min())
            say('     H_%d  %8.2f +- %-7.2f [%6.1f,%6.1f] %+10.3f %12.3f %12.3f %8d'
                % (k, m, s, lo, hi, coef[1], r1, r3, int(((xp > lo) & (xp < hi)).sum())))

    p = os.path.join(dc.RESULT_DIR, 'Hk_prior_check.txt')
    with open(p, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print('\nwrote %s' % p)


if __name__ == '__main__':
    main()
