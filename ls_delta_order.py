r"""ls_delta_order.py -- the prior-free least-squares order ladder for DES + legacy with
the legacy zero-point offset Delta M_legacy FREE, in configuration A.

    python ls_delta_order.py            # k_max = 2..6, plus flat LCDM

This is the third block of the appendix order-sweep table (Table~\ref{tab:ordersweep-ls}).
The other two blocks come from `dovekie_cosmography.task_bic`, which has no hook for the
legacy offset, so that block is built here on exactly the same footing and cached the same
way.

HOW Delta ENTERS.  A single additive zero point on the legacy SNIa only:

    m_pred -> m_pred + Delta * u_leg   ==   m_sn -> m_sn - Delta * u_leg ,

with u_leg the legacy indicator.  The second form is what is implemented, because it leaves
`make_chi2`'s factorisations -- the Cholesky of the SN covariance above all -- untouched, so
the whitening, the covariance and the residual are bit-for-bit those of the fixed-Delta fit
and the only difference between the two blocks is the one extra parameter.  Delta enters
LINEARLY in the residual, which is why the Gauss-Newton curvature below is exact in that
direction.

BIC.  N_params = (k_max + 1) + M_des + r_d + Delta = k_max + 4, one more than the block
above it, and N_data is unchanged, so the two blocks are directly comparable: the offset
must buy itself ln N ~= 7.4 of chi2 to be worth carrying.

ERRORS come from `ls_errors`, on the AUGMENTED parameter vector, so sigma(H0) here is
marginalised over Delta rather than conditioned on Delta = 0.  That is the whole point of
the block, and it is why the H0 errors in it are wider than in the fixed-Delta block.

Output: results/ls_delta_order.json, and a cache under chains/bic/ so a re-run is free.
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares, minimize

import dovekie_cosmography as dc
import ls_errors as le

SAMPLE, CONFIG = 'full', 'A'
CELL = 'lsdelta_%s_%s' % (CONFIG, SAMPLE)
OUT = os.path.join('results', 'ls_delta_order.json')

# Delta is a magnitude zero point, so it takes M's finite-difference floor, not a
# coefficient's.  le._steps only knows the [shape..., M, r_d] layout, hence the override.
FLOOR_DELTA = le.FLOOR_M


def _u_leg(n):
    d = dc.sn_dataset(SAMPLE)
    u = (d['idsurvey'] != dc.DES_IDSURVEY).astype(float)
    assert len(u) == n, (len(u), n)
    return u


def augmented(background, P):
    """(resid_aug, n_shape, n_theta) with theta = [shape..., M, r_d, Delta]."""
    _f, meta = dc.make_chi2(SAMPLE, CONFIG, background, P=P)
    b, resid, ns = meta['blocks'], meta['resid'], meta['n_shape']
    m0 = np.array(b['m_sn'], float)
    u = _u_leg(len(m0))

    def R(th):
        b['m_sn'] = m0 - th[ns + 2] * u
        try:
            out = resid(th[:ns + 2])
        finally:
            b['m_sn'] = m0
        return np.asarray(out, float)

    return R, ns, ns + 3, meta


def _fit(background, P, seed=0, n_multistart=4):
    R, ns, nth, meta = augmented(background, P)
    F = lambda th: float(R(th) @ R(th))                                    # noqa: E731
    sh = (dc.lcdm_coeffs(0.315, 68.0, P) if background == 'cosmo'
          else np.array([68.0, 0.30]))
    th0 = np.concatenate([sh, [-19.40, dc.BASELINE['RD_PLANCK']], [0.0]])
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for i in range(n_multistart):
        t0 = th0 if i == 0 else th0 * (1.0 + 10.0 ** (-3 + i) * rng.standard_normal(nth))
        try:
            s = least_squares(R, t0, method='lm', max_nfev=200000,
                              xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except Exception:
            continue
        x, fv = s.x, float(s.fun @ s.fun)
        p = minimize(F, x, method='Nelder-Mead',
                     options=dict(maxiter=20000, maxfev=20000, xatol=1e-10, fatol=1e-9))
        if np.isfinite(p.fun) and np.all(np.isfinite(p.x)) and p.fun < fv:
            x, fv = p.x, float(p.fun)
        if fv < best[0]:
            best = (fv, x)
    if best[1] is None:
        raise RuntimeError('no start converged for %s P=%s with Delta free' % (background, P))
    return best[0], best[1], R, ns, meta


def _errors(R, theta, ns):
    """Curvature errors on the augmented vector, with le's own plateau check."""
    theta = np.asarray(theta, float)

    def jac(mult):
        s = np.abs(theta) * le.REL_STEP
        fl = np.concatenate([np.full(ns, le.FLOOR_SHAPE), [le.FLOOR_M, le.FLOOR_RD,
                                                           FLOOR_DELTA]])
        h = np.maximum(s, fl * le.REL_STEP) * mult
        r0 = R(theta)
        J = np.empty((len(r0), len(theta)))
        for i in range(len(theta)):
            tp, tm = theta.copy(), theta.copy()
            tp[i] += h[i]
            tm[i] -= h[i]
            J[:, i] = (R(tp) - R(tm)) / (2.0 * h[i])
        return J

    cov, cond = le.cov_from_jac(jac(1.0))
    var = np.diag(cov)
    if np.any(var <= 0):
        raise RuntimeError('non-positive variance with Delta free')
    sig = np.sqrt(var)
    sigs = [sig]
    for m in (1.0 / 3.0, 3.0):
        try:
            cm, _ = le.cov_from_jac(jac(m))
            vm = np.diag(cm)
            if np.all(vm > 0):
                sigs.append(np.sqrt(vm))
        except Exception:
            continue
    S = np.vstack(sigs)
    spread = float(np.max(S.max(axis=0) / S.min(axis=0) - 1.0)) if len(sigs) > 1 \
        else float('nan')
    return dict(sigma=sig, cov=cov, cond=cond, n_shape=ns, step_spread=spread,
                reliable=bool(np.isfinite(spread) and spread < 0.05))


def _row(background, P):
    chi2, th, R, ns, meta = _fit(background, P)
    e = _errors(R, th, ns)
    d = le.derived(th[:ns + 2], dict(sigma=e['sigma'][:ns + 2], cov=e['cov'][:ns + 2,
                                                                             :ns + 2],
                                     n_shape=ns), background)
    N = meta['n_sn'] + meta['n_bao'] + meta['n_cc'] + meta['n_rd']
    k = ns + 3
    return dict(P=(P if background == 'cosmo' else None), background=background,
                k=k, N=int(N), chi2_min=chi2, BIC=chi2 + k * np.log(N),
                H0=d['H0'][0], sig_H0=d['H0'][1],
                M=d['M'][0], sig_M=d['M'][1],
                rd=d['rd'][0], sig_rd=d['rd'][1],
                q0=d['q0'][0], sig_q0=d['q0'][1],
                Delta=float(th[ns + 2]), sig_Delta=float(e['sigma'][ns + 2]),
                cond=e['cond'], step_spread=e['step_spread'], reliable=e['reliable'],
                theta=[float(v) for v in th])


def ladder(plist=(2, 3, 4, 5, 6), recompute=False):
    cached = None if recompute else dc.cache_load('bic', CELL, ignore_p=True,
                                                  ignore=dc._SAMPLER_KEYS)
    have = json.loads(str(cached['rows_json'])) if cached is not None else {}
    rows, dirty = {}, False
    for P in plist:
        key = str(P)
        if key in have:
            rows[key] = have[key]
            continue
        print('   Delta-free least squares, cosmo k_max = %d ...' % P, flush=True)
        rows[key] = _row('cosmo', P)
        dirty = True
    if 'lcdm' in have:
        rows['lcdm'] = have['lcdm']
    else:
        print('   Delta-free least squares, flat LCDM ...', flush=True)
        rows['lcdm'] = _row('lcdm', None)
        dirty = True
    if dirty:
        dc.cache_save('bic', CELL, rows_json=json.dumps(rows),
                      plist_json=json.dumps(list(plist)))
    return rows


def main():
    rows = ladder()
    os.makedirs('results', exist_ok=True)
    json.dump(rows, open(OUT, 'w'), indent=1, sort_keys=True)
    bic_min = min(r['BIC'] for k, r in rows.items() if k != 'lcdm')
    print('\nDES + legacy, configuration A, Delta M_legacy FREE')
    print('%-6s %4s %10s %9s %8s %18s %10s' %
          ('k_max', 'k', 'chi2_min', 'dBIC', 'H0', 'Delta M_legacy', 'q0'))
    for key in sorted(rows, key=lambda s: (s == 'lcdm', s)):
        r = rows[key]
        print('%-6s %4d %10.2f %+9.2f %8.2f  %+8.4f +- %.4f %+6.2f +- %.2f'
              % (key, r['k'], r['chi2_min'], r['BIC'] - bic_min, r['H0'],
                 r['Delta'], r['sig_Delta'], r['q0'], r['sig_q0']))
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
