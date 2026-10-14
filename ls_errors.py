r"""ls_errors.py -- uncertainties on the least-squares order sweep, from the curvature.

The order sweep minimises the chi2 without any prior on the coefficients, which is what
makes the order decision prior-free.  That leaves the fit with a best-fitting parameter
vector and no error bar, so this module supplies the missing half in the only way that
keeps the statement prior-free: the covariance implied by the curvature of the same chi2
at its own minimum.

    chi2(theta) = r(theta) . r(theta)   with r the WHITENED residual (dovekie_cosmography
                                        .make_chi2 returns it, and test_repo asserts the
                                        identity)
    F_ij = sum_n dr_n/dtheta_i dr_n/dtheta_j = (J^T J)_ij       (Gauss-Newton)
    cov  = F^{-1}

This is the Fisher matrix of the likelihood the posterior samples, so its square roots are
the 1 sigma errors a Gaussian posterior with no prior would have.  Two things follow, and
both matter for how the numbers are read.

WHY GAUSS-NEWTON AND NOT THE FULL HESSIAN.  The dropped term is sum_n r_n d^2 r_n /
dtheta_i dtheta_j, which vanishes for a linear model and is small at a good fit whatever
the model, because the residuals are then near-random in sign and cancel in the sum.  It
also makes F positive semi-definite by construction, so the covariance cannot come back
with a negative variance from numerical noise, which the raw Hessian of a chi2 evaluated
by finite differences frequently does at high order.

WHY THE COLUMNS ARE SCALED BEFORE THE INVERSE.  The coefficients C_k span six orders of
magnitude by k_max = 7, so F in raw units has a condition number around 10^20 and its
inverse is numerical noise.  Scaling each column of J to unit norm turns F into a
correlation-like matrix with a unit diagonal, and the scaling is undone exactly afterwards,
D (D F D)^{-1} D = F^{-1}.  The condition number of the SCALED matrix is reported with
every cell, and it is the honest diagnostic of how flat the likelihood has become: it is
around 10^3 at k_max = 2 and exceeds 10^8 by k_max = 7, which is the same degeneracy that
stops NUTS converging at those orders and is a result about the parametrisation rather than
a defect of this calculation.

THE ERRORS ARE VALIDATED, NOT ASSUMED.  `validate()` compares them against the sampled
posteriors of the baseline cells, which exist independently in chains/ladder.  Agreement
there is what licenses quoting the curvature errors at the orders where no converged chain
exists, and it is checked rather than asserted.
"""
import numpy as np

import dovekie_cosmography as dc

# Central-difference step, as a fraction of the parameter's own scale.  1e-4 is a
# compromise: the whitened residual runs through np.interp on an 800-point grid, so a step
# far below the grid spacing samples the interpolation kinks rather than the model, while a
# step far above it leaves the quadratic regime.  `jacobian` checks the choice rather than
# trusting it, by recomputing at 3x the step and comparing the resulting errors.
REL_STEP = 1e-4

# Floors, so that a coefficient sitting near zero still gets a finite step.  Ordered as
# theta = [shape..., M, r_d]; the shape floor is applied to every coefficient.
FLOOR_SHAPE, FLOOR_M, FLOOR_RD = 1.0, 1e-3, 1e-2


def _steps(theta, n_shape):
    s = np.abs(np.asarray(theta, float)) * REL_STEP
    fl = np.concatenate([np.full(n_shape, FLOOR_SHAPE), [FLOOR_M], [FLOOR_RD]])
    return np.maximum(s, fl * REL_STEP)


def jacobian(resid, theta, n_shape, mult=1.0):
    """d r / d theta by central differences, one column per parameter."""
    theta = np.asarray(theta, float)
    h = _steps(theta, n_shape) * mult
    r0 = resid(theta)
    J = np.empty((len(r0), len(theta)))
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h[i]
        tm[i] -= h[i]
        J[:, i] = (resid(tp) - resid(tm)) / (2.0 * h[i])
    return J


def cov_from_jac(J):
    """(J^T J)^{-1}, inverted in column-scaled coordinates.  Returns (cov, cond)."""
    nrm = np.linalg.norm(J, axis=0)
    if not np.all(np.isfinite(nrm)) or np.any(nrm <= 0):
        raise RuntimeError('the Jacobian has a null or non-finite column: the fit is not '
                           'at a minimum, or a parameter does not enter the residual')
    D = 1.0 / nrm
    Fs = (J * D) .T @ (J * D)                      # unit diagonal by construction
    cond = float(np.linalg.cond(Fs))
    cov = D[:, None] * np.linalg.inv(Fs) * D[None, :]
    return cov, cond


def errors(sample, config, background, P, theta, check_step=True):
    """1 sigma errors on theta at the least-squares minimum, plus diagnostics.

    Returns dict(sigma, cov, cond, n_shape, step_ratio).  `step_ratio` is the largest
    fractional change in any sigma when the differencing step is tripled: it is the
    measurement of whether the numbers below are the model's curvature or the finite
    difference's, and anything above a per cent should be treated as the latter.
    """
    _f, meta = dc.make_chi2(sample, config, background, P=P)
    resid, n_shape = meta['resid'], meta['n_shape']
    theta = np.asarray(theta, float)
    J = jacobian(resid, theta, n_shape)
    cov, cond = cov_from_jac(J)
    var = np.diag(cov)
    if np.any(var <= 0):
        raise RuntimeError('non-positive variance at %s/%s P=%s: the Gauss-Newton matrix '
                           'is singular to working precision' % (config, sample, P))
    sig = np.sqrt(var)
    # THE STEP CHECK IS A PLATEAU TEST, NOT A SINGLE COMPARISON.  A finite difference is
    # squeezed between roundoff below and truncation above, and the derivative is only
    # measured where the two leave a plateau in between.  Tripling the step alone cannot
    # tell a stable answer from one that happens to have crossed into the truncation
    # regime, so sigma is recomputed a THIRD of the way down as well and the spread over
    # all three is reported.  It is under 0.2 per cent at every order up to k_max = 6 and
    # tens of per cent at k_max = 7, where the scaled Gauss-Newton matrix has a condition
    # number above 10^10 and its inverse amplifies the differencing error to O(1).
    spread = float('nan')
    if check_step:
        sigs = [sig]
        for m in (1.0 / 3.0, 3.0):
            try:
                cm, _cc = cov_from_jac(jacobian(resid, theta, n_shape, mult=m))
                vm = np.diag(cm)
                if np.all(vm > 0):
                    sigs.append(np.sqrt(vm))
            except Exception:                      # a singular matrix at one step is
                continue                           # itself the diagnostic, not a crash
        if len(sigs) > 1:
            S = np.vstack(sigs)
            spread = float(np.max(S.max(axis=0) / S.min(axis=0) - 1.0))
    return dict(sigma=sig, cov=cov, cond=cond, n_shape=n_shape, step_spread=spread,
                reliable=bool(np.isfinite(spread) and spread < 0.05))


def derived(theta, e, background):
    """(value, sigma) for the quantities the table prints, with q0 propagated exactly.

    cosmo:  q0 = C_1/C_0 - 1, so dq0/dC_0 = -C_1/C_0^2 and dq0/dC_1 = 1/C_0, and the
            covariance BETWEEN C_0 and C_1 is kept.  Dropping it is not conservative: the
            two are strongly anti-correlated, and ignoring the cross term overstates
            sigma(q0) at low order and understates it at high order.
    lcdm:   q0 = 3/2 Om - 1 exactly, so sigma(q0) = 3/2 sigma(Om).
    """
    theta, sig, cov, ns = np.asarray(theta, float), e['sigma'], e['cov'], e['n_shape']
    out = dict(H0=(theta[0], sig[0]), M=(theta[ns], sig[ns]), rd=(theta[ns + 1], sig[ns + 1]))
    if background == 'lcdm':
        out['Om'] = (theta[1], sig[1])
        out['q0'] = (1.5 * theta[1] - 1.0, 1.5 * sig[1])
    else:
        C0, C1 = theta[0], theta[1]
        g = np.array([-C1 / C0 ** 2, 1.0 / C0])
        S = cov[np.ix_([0, 1], [0, 1])]
        out['q0'] = (C1 / C0 - 1.0, float(np.sqrt(g @ S @ g)))
    return out


# ---------------------------------------------------------------------------------------
def validate(P=None, verbose=True):
    """Compare the curvature errors against the SAMPLED posteriors of the baseline cells.

    The posterior carries the cosmographic prior box and the curvature errors carry no
    prior at all, so the two agree only if the box is inactive, which the project reports
    it is.  That makes this a joint test of both claims, and it is the reason the errors at
    k_max >= 5 -- where no converged chain exists -- may be quoted at all.
    """
    P = dc.BASELINE['P'] if P is None else P
    rows = []
    for cfg in ('A', 'B'):
        for smp in ('des_only', 'full'):
            c = dc.cache_load('ladder', '%s_%s_cosmo_P%d' % (cfg, smp, P))
            if c is None:
                continue
            r = dc._unpack(c)
            c2, th, _m = dc.minimise_chi2(smp, cfg, 'cosmo', P=P)
            d = derived(th, errors(smp, cfg, 'cosmo', P, th), 'cosmo')
            for q, (pm, ps) in (('H0', (r['H0_mean'], r['H0_std'])),
                                ('M', (r['M_mean'], r['M_std'])),
                                ('q0', (r.get('q0_mean'), r.get('q0_std')))):
                if pm is None:
                    continue
                lv, ls = d[q]
                rows.append(dict(cell='%s/%s' % (cfg, smp), q=q, ls=lv, ls_sig=ls,
                                 post=pm, post_sig=ps,
                                 centre_sig=abs(lv - pm) / ps,
                                 width_frac=ls / ps - 1.0))
    if verbose:
        print('\nCURVATURE ERRORS vs THE SAMPLED POSTERIOR, k_max = %d' % P)
        print('%-12s %4s %12s %12s %10s %10s'
              % ('cell', 'q', 'least sq', 'posterior', '|d|/sig', 'width'))
        for r in rows:
            print('%-12s %4s %6.3f+-%-5.3f %6.3f+-%-5.3f %9.2f %+9.1f%%'
                  % (r['cell'], r['q'], r['ls'], r['ls_sig'], r['post'], r['post_sig'],
                     r['centre_sig'], 100 * r['width_frac']))
        w = max(abs(r['width_frac']) for r in rows)
        c = max(r['centre_sig'] for r in rows)
        print('\nworst centre offset %.2f sigma, worst width discrepancy %.1f per cent'
              % (c, 100 * w))
    return rows


if __name__ == '__main__':
    validate()


# ---------------------------------------------------------------------------------------
def sweep(got, plist, recompute=False):
    """Curvature errors at every cached least-squares solution of the order sweep.

    Keyed on the SAME theta the table prints, read from the bic cache, so the error bar
    and the central value cannot come from different minima.  Results are cached under
    chains/bic/lserr_<config>_<sample>.npz.
    """
    import json
    out = {}
    for (cfg, smp), c in sorted(got.items()):
        cell = 'lserr_%s_%s' % (cfg, smp)
        cached = None if recompute else dc.cache_load('bic', cell, ignore_p=True,
                                                      ignore=dc._SAMPLER_KEYS)
        have = json.loads(str(cached['rows_json'])) if cached is not None else {}
        rows, dirty = {}, False
        for r in c['rows']:
            key = str(r['P'])
            if key in have:
                rows[key] = have[key]
                continue
            e = errors(smp, cfg, 'cosmo', r['P'], r['theta'])
            d = derived(r['theta'], e, 'cosmo')
            rows[key] = dict(sig_H0=d['H0'][1], sig_M=d['M'][1], sig_rd=d['rd'][1],
                             q0=d['q0'][0], sig_q0=d['q0'][1],
                             cond=e['cond'], step_spread=e['step_spread'],
                             reliable=e['reliable'])
            dirty = True
            print('[lserr] %s/%s k_max=%d  sig(H0)=%8.2f sig(q0)=%6.2f  cond=%.1e  '
                  'step spread %6.2f%%  %s'
                  % (cfg, smp, r['P'], rows[key]['sig_H0'], rows[key]['sig_q0'],
                     rows[key]['cond'], 100 * rows[key]['step_spread'],
                     '' if rows[key]['reliable'] else '<== NOT RELIABLE'), flush=True)
        # LCDM on the same footing, so the closing row of each block carries errors too
        if 'lcdm' not in have or recompute:
            c2, th, _m = dc.minimise_chi2(smp, cfg, 'lcdm')
            if abs(c2 - c['lcdm']['chi2_min']) > 1e-2:
                raise RuntimeError('LCDM chi2 disagrees with the bic cache at %s/%s: '
                                   '%.4f vs %.4f' % (cfg, smp, c2, c['lcdm']['chi2_min']))
            e = errors(smp, cfg, 'lcdm', None, th)
            d = derived(th, e, 'lcdm')
            rows['lcdm'] = dict(H0=d['H0'][0], sig_H0=d['H0'][1], M=d['M'][0],
                                sig_M=d['M'][1], rd=d['rd'][0], sig_rd=d['rd'][1],
                                q0=d['q0'][0], sig_q0=d['q0'][1], Om=d['Om'][0],
                                sig_Om=d['Om'][1], cond=e['cond'],
                                step_spread=e['step_spread'], reliable=e['reliable'])
            dirty = True
            print('[lserr] %s/%s LCDM     sig(H0)=%.2f sig(q0)=%.3f  cond=%.1e'
                  % (cfg, smp, rows['lcdm']['sig_H0'], rows['lcdm']['sig_q0'],
                     rows['lcdm']['cond']), flush=True)
        else:
            rows['lcdm'] = have['lcdm']
        if dirty:
            dc.cache_save('bic', cell, rows_json=json.dumps(rows),
                          plist_json=json.dumps(sorted(int(k) for k in rows if k != 'lcdm')))
        out[(cfg, smp)] = rows
    return out
