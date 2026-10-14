r"""evidence_order.py -- Bayesian evidences for the cosmographic order ladder, by nested
sampling, so that the order blend can be weighted by Z(k) instead of by exp(-BIC(k)/2).

    python evidence_order.py --list                  # the grid, nothing else
    python evidence_order.py --cells A_full:4        # one cell, one order
    python evidence_order.py --nlive 200 --pilot     # a cheap smoke run
    NTASKS=25 SHARD=7 python evidence_order.py       # one task of the cluster array

WHAT THIS IS FOR.  `make_blend.py` currently weights the orders k_max = 2..6 by the
Schwarz weight w(k) ~ exp(-BIC(k)/2).  The Schwarz weight is the Laplace approximation to
the marginal likelihood with the volume term REPLACED by the parameter count, which is
precisely what lets it dodge the prior width on the high-order coefficients.  This module
computes the thing the Schwarz weight is approximating -- the marginal likelihood itself,

    Z(k) = int L(theta) pi(theta) d theta ,

with `dynesty`, so the blend can be repeated with w(k) ~ Z(k) and the two compared.  That
comparison is the deliverable: it measures how much of the blended H_0 is the data and how
much is the prior on coefficients the data barely constrain.

TWO REASONS THIS IS WORTH CLUSTER TIME RATHER THAN A LAPLACE FORMULA.

  1. THE LAPLACE APPROXIMATION IS LEAST TRUSTWORTHY WHERE THE WEIGHT SITS.  A Laplace
     evidence built on the cached least-squares minimum and its Gauss-Newton curvature
     (`--laplace`, seconds) assumes a Gaussian posterior wholly inside the prior.  Under
     the Bernstein prior it puts most of the weight on k_max = 5-6 in the DES + legacy
     cells (2026-09-28), exactly where the posterior is least Gaussian, and the envelope
     is known to bite at k_max = 6 for DES alone.  Nested sampling integrates the truncated
     posterior exactly.

  2. IT GIVES REAL POSTERIORS AT k_max >= 5, WHICH NOTHING ELSE HERE HAS.  Every chain in
     `chains/` is `_P4`; NUTS does not converge at k_max >= 5 because the coefficient
     likelihood has a genuine flat direction there, which is why `make_blend` has to use
     least-squares curvature Gaussians as its components.  Nested sampling does not have
     that failure mode -- it starts from the prior and contracts, so a flat direction is
     bounded by the box rather than unbounded -- so this run upgrades the blend components
     from curvature Gaussians to sampled posteriors.

THE PRIOR (2026-09-28): the envelope prior of dovekie_cosmography section 7 -- uniform in
the shape coefficients C_k of E(y) = H/H0 over every polynomial with 0 <= E(y) <= 2 + 7y for
y <= Y_MAX.  theta = [H0, v_1..v_k, M, r_d, (Delta)], with v_j = E(y_j) at fixed nodes,
uniform on a box that contains the envelope region A; log L = -inf outside A, and
    lnZ = lnZ_sampled - ln acc(k),     acc = vol(A)/vol(box)  (`dc.envelope_log_acceptance`),
because dynesty integrates the -inf region as zero likelihood over the whole box (asserted
in test_repo on a toy with a known answer).  The likelihood converts to the dimensional
chart c_k = H0 C_k and reuses `dc.make_chi2` unchanged.

It replaced a symmetric box on the dimensional c_k, whose half-widths
(1.1e7 on c_5) were a free choice that could move w(5) from 0.1% to 89% in the headline
cell.  The envelope leaves one choice, its
width: halving it could raise lnZ(k) by at most k ln 2 (the envelope region scales
affinely about 1 + 3.5y, so the bound is exact).  A
same-day predecessor imposed the envelope through a box on the Bernstein coefficients of E,
which is only 5% of A at k = 4 and 0.09% at k = 6; it was replaced because it cut
admissible shapes and handed the high orders 1.8-4 nats of spurious Occam relief.

WHAT CANCELS, AND WHAT DOES NOT.  Only lnZ DIFFERENCES between orders WITHIN a cell are
used.  The priors on H0, M, r_d and Delta are identical at every order, so their
normalisations cancel.  What does not cancel -- the entire content of an evidence weight
here -- is the volume of the shape prior each order adds, `dc.log_shape_volume(k)`, which is where
ln acc(k) enters.

r_d.  In configuration A the Planck measurement is a PRIOR, drawn in the prior transform,
and its residual term is dropped from the likelihood so it is not counted twice; in
configuration B r_d is uniform on the wide box and the likelihood has no such term.  The
treatment is identical at every order either way, so it cannot move a weight.

Output: chains/evorder/ (one .npz per cell, sampled posterior + lnZ) and
results/evidence_order.json when the run is not sharded.
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.special import ndtri

import dovekie_cosmography as dc
import ls_delta_order as lsd

OUT = 'evidence_order.json'   # write_json_atomic resolves this against RESULT_DIR
TASK = 'evorder'

# The same five cells make_blend.py mixes, in the same order and under the same keys, so a
# blend can be re-weighted without a second naming convention to keep in step.
CELLS = [('A_des_only', 'A', 'des_only', 'DES'),
         ('A_full', 'A', 'full', 'DES + legacy'),
         ('A_full_delta', 'A', 'full', 'DES + legacy, dM_legacy free'),
         ('B_des_only', 'B', 'des_only', 'DES'),
         ('B_full', 'B', 'full', 'DES + legacy')]
PLIST = (2, 3, 4, 5, 6)

NLIVE = int(os.environ.get('NLIVE', 1000))
DLOGZ = float(os.environ.get('DLOGZ', 0.1))
SEED = int(os.environ.get('SEED', dc.BASELINE['SEED']))
NPROC = int(os.environ.get('NPROC', 1))
SAMPLER = os.environ.get('SAMPLER', 'auto')
if 'CKSCALE' in os.environ:
    # retired with the coefficient box it rescaled; silently ignoring it would label a
    # default run with a setting it never had
    raise SystemExit('CKSCALE is retired (2026-09-28): the Bernstein prior has no '
                     'per-coefficient width to rescale.  Unset it.')

NTASKS = int(os.environ.get('NTASKS', 1))
SHARD = int(os.environ.get('SHARD', 0))


def grid():
    return [(key, cfg, smp, P) for (key, cfg, smp, _lab) in CELLS for P in PLIST]


def shard(items):
    if not NTASKS >= 1:
        raise ValueError('NTASKS must be >= 1, got %r' % (NTASKS,))
    if not 0 <= SHARD < NTASKS:
        raise ValueError('SHARD=%r is not in range(0, NTASKS=%r).  An out-of-range shard '
                         'runs no cells and exits 0, i.e. it looks like success.'
                         % (SHARD, NTASKS))
    return [x for i, x in enumerate(items) if i % NTASKS == SHARD]


def cell_name(key, P):
    return '%s_P%d' % (key, P)


# =============================================================================
# 1.  Prior and likelihood
# =============================================================================
def param_names(P, free_delta):
    nm = ['H0'] + ['v%d' % j for j in range(1, P + 1)] + ['M', 'rd']
    return nm + (['Delta'] if free_delta else [])


def prior_extra():
    """The cache key for the prior this grid samples under: a cell sampled under another
    prior on the shape must never be collected into the blend."""
    return dict(PRIOR=dc.cosmo_prior_signature())


def make_ptform(cfg, P, free_delta):
    """Unit cube -> theta = [H0, v_1..v_P, M, r_d, (Delta)]: v uniform on the node box.

    Config A draws r_d from the Planck measurement itself; the likelihood below drops the
    matching residual so the measurement enters exactly once."""
    _y, blo, bhi, _Vinv = dc.envelope_nodes(P)
    lo0, hi0 = dc.BASELINE['H0_LO'], dc.BASELINE['H0_HI']
    mlo, mhi = dc.BASELINE['M_LO'], dc.BASELINE['M_HI']
    rlo, rhi = dc.BASELINE['RD_FREE_LO'], dc.BASELINE['RD_FREE_HI']
    rmu, rsig = dc.BASELINE['RD_PLANCK'], dc.BASELINE['SIG_RD']
    w = dc.BASELINE['DELTA_HALFWIDTH']
    planck = (cfg == 'A')

    def ptform(u):
        th = np.empty(len(u))
        th[0] = lo0 + u[0] * (hi0 - lo0)
        th[1:P + 1] = blo + u[1:P + 1] * (bhi - blo)
        th[P + 1] = mlo + u[P + 1] * (mhi - mlo)
        th[P + 2] = rmu + rsig * ndtri(u[P + 2]) if planck else rlo + u[P + 2] * (rhi - rlo)
        if free_delta:
            th[P + 3] = -w + u[P + 3] * (2.0 * w)
        return th

    return ptform


def make_loglike(cfg, smp, P, free_delta):
    """-chi2/2 on the whitened residual, with the r_d prior term removed in config A.

    theta is in the SAMPLED basis [H0, v, M, r_d, (Delta)] and is carried into the
    dimensional chart [H0 C_0..H0 C_P, M, r_d, (Delta)] that `make_chi2` works in, so the
    likelihood is the one every other task uses, unchanged.  A shape outside the envelope
    is outside the prior: -inf, exactly as the acceptance in `dc.envelope_log_acceptance`
    counts it (same check, same grid).

    Returns -inf rather than a large finite penalty where H(z) <= 0: a plateau in the
    likelihood is a documented failure mode for nested sampling, whereas -inf is simply
    a point outside the support and is what the rejection step is for."""
    if free_delta:
        assert (cfg, smp) == ('A', 'full'), (cfg, smp)
        R, _ns, _nth, meta = lsd.augmented('cosmo', P)
    else:
        _f, meta = dc.make_chi2(smp, cfg, 'cosmo', P=P)
        R = meta['resid']
    n_rd = meta['n_rd']              # 1 in config A: the last residual IS the Planck term
    _y, _lo, _hi, Vinv = dc.envelope_nodes(P)
    yg = dc.envelope_grid()
    elo, ehi = dc.envelope_bounds(yg)
    pw = np.power(yg[:, None], np.arange(P + 1)[None, :])

    def loglike(theta):
        theta = np.asarray(theta, float)
        C = np.concatenate([[1.0], Vinv @ (theta[1:P + 1] - 1.0)])
        E = pw @ C
        if np.any(E < elo) or np.any(E > ehi):
            return -np.inf
        c = theta[0] * C
        r = np.asarray(R(np.concatenate([c, theta[P + 1:]])), float)
        if not np.all(np.isfinite(r)) or np.any(np.abs(r) >= 1e6):
            return -np.inf           # the make_chi2 guard value for H <= H_FLOOR
        if n_rd:
            r = r[:-1]
        return -0.5 * float(r @ r)

    return loglike, meta


def _wquant(x, w, q):
    """Weighted quantile.  The nested-sampling weights are wildly unequal, so an unweighted
    percentile over res.samples is a statement about the prior, not the posterior."""
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return float(np.interp(q * c[-1], c, np.asarray(x)[i]))


# Workers are forked AFTER the closures are built and reach them through these globals;
# dynesty pickles BOTH callables it is given, and neither closure is picklable.
_POOL_LL = None
_POOL_PT = None


def _pool_loglike(theta):
    return _POOL_LL(theta)


def _pool_ptform(u):
    return _POOL_PT(u)


def _blas_threads_pinned():
    return all(os.environ.get(k) == '1'
               for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))


# =============================================================================
# 2.  One cell
# =============================================================================
def run_cell(key, cfg, smp, P, nlive=None, dlogz=None, seed=None, nproc=None, quiet=False):
    import dynesty
    nlive = NLIVE if nlive is None else nlive
    dlogz = DLOGZ if dlogz is None else dlogz
    seed = SEED if seed is None else seed
    nproc = NPROC if nproc is None else nproc
    free_delta = key.endswith('_delta')

    if not _blas_threads_pinned():
        print('[evorder] WARNING: BLAS threads are not pinned to 1.  This likelihood is a '
              'single small triangular solve and is SLOWER threaded.  Re-launch with '
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.')

    ll, meta = make_loglike(cfg, smp, P, free_delta)
    pt = make_ptform(cfg, P, free_delta)
    names = param_names(P, free_delta)
    ndim = len(names)

    pool = None
    if nproc > 1:
        import multiprocessing as mp
        global _POOL_LL, _POOL_PT
        _POOL_LL, _POOL_PT = ll, pt
        pool = mp.get_context('fork').Pool(nproc)
        print('[evorder] likelihood queue over %d worker processes' % nproc)

    t0 = time.time()
    try:
        skw = {} if SAMPLER == 'auto' else dict(sample=SAMPLER)
        ns = dynesty.NestedSampler(_pool_loglike if pool else ll,
                                   _pool_ptform if pool else pt,
                                   ndim, nlive=nlive,
                                   rstate=np.random.default_rng(seed),
                                   pool=pool, queue_size=nproc if pool else None, **skw)
        ns.run_nested(dlogz=dlogz, print_progress=not quiet)
        res = ns.results
    finally:
        if pool is not None:
            pool.close()
            pool.join()
            _POOL_LL = _POOL_PT = None

    wt = np.exp(res.logwt - res.logz[-1])
    wt = wt / wt.sum()
    S = res.samples

    # How near the posterior comes to the envelope, asked of the WEIGHTED samples: a
    # nested run stores its whole prior-to-posterior shrinkage history, and its early
    # points are draws from the prior itself, so an unweighted reading reports the prior.
    C = dc.shape_of_v(S[:, 1:P + 1], P)
    marg, at = dc.envelope_activity(dict(C=C[:, 1:]), P, weights=wt)
    ln_acc, ln_acc_err = dc.envelope_log_acceptance(P)

    out = dict(key=key, config=cfg, sample=smp, P=int(P), ndim=int(ndim),
               free_delta=bool(free_delta), params=names,
               nlive=int(nlive), dlogz=float(dlogz), seed=int(seed),
               prior=dc.COSMO_PRIOR_NAME, proposal=SAMPLER,
               # the evidence of the prior UNIFORM ON THE ENVELOPE, not of the node box
               logz=float(res.logz[-1]) - ln_acc,
               logz_err=float(np.hypot(res.logzerr[-1], ln_acc_err)),
               logz_sampled=float(res.logz[-1]), ln_acc=ln_acc,
               information=float(res.information[-1]),
               niter=int(res.niter), ncall=int(np.sum(res.ncall)),
               seconds=float(time.time() - t0),
               env_margin=marg, at_bound=at)
    for i, nm in enumerate(names):
        m = float(np.sum(wt * S[:, i]))
        out[nm + '_mean'] = m
        out[nm + '_std'] = float(np.sqrt(np.sum(wt * (S[:, i] - m) ** 2)))
    # The shape in the reporting basis, C_1..C_P, and q0 = C_1 - 1.
    for k in range(1, P + 1):
        m = float(np.sum(wt * C[:, k]))
        out['C%d_mean' % k] = m
        out['C%d_std' % k] = float(np.sqrt(np.sum(wt * (C[:, k] - m) ** 2)))
    q0 = C[:, 1] - 1.0
    out['q0_mean'] = float(np.sum(wt * q0))
    out['q0_std'] = float(np.sqrt(np.sum(wt * (q0 - out['q0_mean']) ** 2)))
    return out, S, wt


def task(cells=None, recompute=False, quiet=False, **kw):
    want = shard(cells if cells is not None else grid())
    # 🚨 THE CACHE KEY RECORDS WHAT WAS USED, NOT WHAT THE MODULE DEFAULTS TO.  `--nlive`
    # and `--dlogz` override the module constants, so keying on NLIVE/DLOGZ here would
    # stamp a 200-live-point smoke run with nlive=1000 and let it be reused as a
    # production cell.  Provenance that records a request rather than the routing is worse
    # than none -- the same defect that made INIT=lcdm a no-op in 12 of 13 samplers.
    eff = dict(NLIVE=int(kw.get('nlive', NLIVE)),
               DLOGZ=float(kw.get('dlogz', DLOGZ)),
               SEED=int(kw.get('seed', SEED)),
               SAMPLER=SAMPLER, **prior_extra())
    rows = {}
    for (key, cfg, smp, P) in want:
        cell = cell_name(key, P)
        extra = dict(eff)
        cached = None if recompute else dc.cache_load(TASK, cell, ignore_p=True,
                                                      ignore=dc._SAMPLER_KEYS, extra=extra)
        if cached is not None:
            rows[cell] = json.loads(str(cached['row_json']))
            print('[evorder] %-22s (cached)  lnZ = %10.2f +- %.2f'
                  % (cell, rows[cell]['logz'], rows[cell]['logz_err']))
            continue
        print('[evorder] %-22s running  ndim=%d nlive=%d dlogz=%g'
              % (cell, P + 3 + key.endswith('_delta'), eff['NLIVE'], eff['DLOGZ']))
        row, S, wt = run_cell(key, cfg, smp, P, quiet=quiet, **kw)
        dc.cache_save(TASK, cell, extra=extra, row_json=json.dumps(row),
                      samples=S, weights=wt)
        rows[cell] = row
        print('[evorder] %-22s lnZ = %10.2f +- %.2f   H0 = %6.2f +- %.2f   '
              '%d calls in %.0f s%s'
              % (cell, row['logz'], row['logz_err'], row['H0_mean'], row['H0_std'],
                 row['ncall'], row['seconds'],
                 '   AT BOUND: ' + ','.join(row['at_bound']) if row['at_bound'] else ''))
    if NTASKS == 1:
        dc.write_json_atomic(OUT, rows)
        print('[evorder] wrote %s' % OUT)
    else:
        print('[evorder] shard %d/%d done; run unsharded (or --collect) to aggregate.'
              % (SHARD, NTASKS))
    return rows


def collect():
    """Re-read every cached cell into results/evidence_order.json.  No sampling."""
    rows = {}
    missing = []
    # The proposal changes how the constrained prior is explored, not the integral, so a
    # cell re-run with `SAMPLER=rslice` (the fix for the ellipsoid-inefficiency timeouts)
    # is the same lnZ.  Accept any proposal here -- the requested one first -- and say
    # which cells used another; each row still records its own `proposal`.  `task()` keeps
    # the strict key, so a sampling run never silently reuses a different proposal.
    proposals = [SAMPLER] + [s for s in ('auto', 'rslice', 'rwalk', 'slice', 'unif')
                             if s != SAMPLER]
    other = []
    for (key, cfg, smp, P) in grid():
        cell = cell_name(key, P)
        c = None
        for prop in proposals:
            extra = dict(NLIVE=NLIVE, DLOGZ=DLOGZ, SEED=SEED, SAMPLER=prop,
                         **prior_extra())
            c = dc.cache_load(TASK, cell, ignore_p=True, ignore=dc._SAMPLER_KEYS,
                              extra=extra)
            if c is not None:
                break
        if c is None:
            missing.append(cell)
            continue
        if prop != SAMPLER:
            other.append('%s(%s)' % (cell, prop))
        rows[cell] = json.loads(str(c['row_json']))
    if other:
        print('[evorder] %d cells sampled with a non-default proposal: %s'
              % (len(other), ' '.join(other)))
    if missing:
        # A silently short table is the documented failure mode of this repository's
        # collect steps; say so loudly and still write what there is.
        print('[evorder] %d of %d cells MISSING: %s'
              % (len(missing), len(grid()), ' '.join(missing)))
    dc.write_json_atomic(OUT, rows)
    print('[evorder] wrote %s with %d of %d cells' % (OUT, len(rows), len(grid())))
    return rows


# =============================================================================
# 3.  The local cross-checks
# =============================================================================
def laplace(plist=PLIST):
    """Laplace evidences from the cached least-squares sweep.  Seconds, no sampling.

    Valid only while the posterior is interior to the envelope and near-Gaussian; a best
    fit outside the envelope returns NaN.  The prior density in the dimensional chart the curvature
    lives in is H0^-P / Vol, Vol = `dc.log_shape_volume(P)`: the H0^-P is the Jacobian of
    c_k = H0 C_k, and it does not cancel between orders.  Its purpose is to forecast the
    grid and, afterwards, to be compared against it."""
    import ls_errors as le
    import make_bic_order as mbo
    import ls_delta_order as lsdo
    got = mbo._cells(dc.task_bic(plist=tuple(plist)), list(plist))
    rows = {}
    for (key, cfg, smp, _lab) in CELLS:
        if key.endswith('_delta'):
            # Same formula on the AUGMENTED parameter vector.  Delta's own prior width is
            # common to every order, so it is left out of lnV on both sides and cancels in
            # the weights exactly as c_0, M and r_d do.
            lad = lsdo.ladder(plist=tuple(plist))
            for P in plist:
                r = lad[str(P)]
                th = np.asarray(r['theta'], float)
                R, ns, _nth, _meta = lsdo.augmented('cosmo', P)
                e = lsdo._errors(R, th, ns)
                sign, logdet = np.linalg.slogdet(e['cov'])
                d = e['cov'].shape[0]
                inside = bool(dc.in_envelope(th[:P + 1] / th[0])[0])
                lnV = dc.log_shape_volume(P) + P * float(np.log(th[0]))
                if not inside:
                    sign = -1
                rows[(key, P)] = dict(
                    chi2=r['chi2_min'],
                    logz=(-0.5 * r['chi2_min'] + 0.5 * d * np.log(2 * np.pi)
                          + 0.5 * logdet - lnV) if sign > 0 else float('nan'))
            continue
        by_P = {r['P']: r for r in got[(cfg, smp)]['rows']}
        for P in plist:
            r = by_P[P]
            th = np.asarray(r['theta'], float)
            e = le.errors(smp, cfg, 'cosmo', P, th, check_step=False)
            sign, logdet = np.linalg.slogdet(e['cov'])
            if sign <= 0 or not dc.in_envelope(th[:P + 1] / th[0])[0]:
                rows[(key, P)] = dict(chi2=r['chi2_min'], logz=float('nan'))
                continue
            d = e['cov'].shape[0]
            lnV = dc.log_shape_volume(P) + P * float(np.log(th[0]))
            rows[(key, P)] = dict(chi2=r['chi2_min'],
                                  logz=(-0.5 * r['chi2_min'] + 0.5 * d * np.log(2 * np.pi)
                                        + 0.5 * logdet - lnV))
    return rows


def weights(rows, plist=PLIST):
    """w(k) from lnZ, per cell.  The block maximum is subtracted before exponentiating --
    lnZ ~ -750 here, so exp(lnZ) underflows to exactly zero, the same trap make_blend
    documents for exp(-BIC/2)."""
    out = {}
    for (key, _cfg, _smp, _lab) in CELLS:
        v = []
        for P in plist:
            r = rows.get(cell_name(key, P))
            v.append(float('nan') if r is None else r['logz'])
        a = np.asarray(v, float)
        if np.all(np.isfinite(a)):
            e = np.exp(a - a.max())
            out[key] = dict(zip(plist, e / e.sum()))
        else:
            out[key] = dict(zip(plist, [float('nan')] * len(plist)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cells', help='comma list of KEY:P, e.g. A_full:4,B_full:5')
    ap.add_argument('--plist', default=','.join(str(p) for p in PLIST))
    ap.add_argument('--nlive', type=int)
    ap.add_argument('--dlogz', type=float)
    ap.add_argument('--nproc', type=int)
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--list', action='store_true', help='print the grid and exit')
    ap.add_argument('--collect', action='store_true', help='aggregate cached cells only')
    ap.add_argument('--laplace', action='store_true', help='local Laplace forecast')
    a = ap.parse_args()

    print(dc.banner())
    print('[evorder] nlive=%d dlogz=%g seed=%d prior=%s proposal=%s  shard %d/%d'
          % (NLIVE, DLOGZ, SEED, dc.COSMO_PRIOR_NAME, SAMPLER, SHARD, NTASKS))
    pl = tuple(int(x) for x in a.plist.split(','))

    if a.list:
        g = grid()
        for i, (key, cfg, smp, P) in enumerate(g):
            print('%3d  %-14s cfg=%s sample=%-9s k_max=%d  ndim=%d %s'
                  % (i, key, cfg, smp, P, P + 3 + key.endswith('_delta'),
                     '(shard %d)' % (i % NTASKS)))
        print('%d cells' % len(g))
        return
    if a.laplace:
        rows = laplace(pl)
        print('\n%-16s %-3s %12s %12s   %s' % ('cell', 'k', 'chi2_min', 'lnZ_Laplace', 'w'))
        for (key, _c, _s, _l) in CELLS:
            v = np.array([rows[(key, P)]['logz'] for P in pl])
            w = np.exp(v - np.nanmax(v)) if np.all(np.isfinite(v)) else v * np.nan
            w = w / np.nansum(w)
            for i, P in enumerate(pl):
                print('%-16s %-3d %12.2f %12.2f   %6.3f'
                      % (key if i == 0 else '', P, rows[(key, P)]['chi2'], v[i], w[i]))
        return
    if a.collect:
        rows = collect()
    else:
        cells = None
        if a.cells:
            byk = {k: (k, c, s, l) for (k, c, s, l) in CELLS}
            cells = []
            for tok in a.cells.split(','):
                k, _, p = tok.partition(':')
                if k not in byk:
                    raise SystemExit('unknown cell %r; known: %s'
                                     % (k, ' '.join(sorted(byk))))
                cells += [(k, byk[k][1], byk[k][2], int(pp))
                          for pp in ([int(p)] if p else pl)]
        kw = {}
        for nm in ('nlive', 'dlogz', 'nproc'):
            if getattr(a, nm) is not None:
                kw[nm] = getattr(a, nm)
        rows = task(cells=cells, recompute=a.recompute, quiet=a.quiet, **kw)

    if rows:
        print('\n%-16s %-3s %12s %8s %10s %9s %s'
              % ('cell', 'k', 'lnZ', '+-', 'H0', 'w(Z)', 'at bound'))
        w = weights(rows, pl)
        for (key, _c, _s, _l) in CELLS:
            for i, P in enumerate(pl):
                r = rows.get(cell_name(key, P))
                if r is None:
                    continue
                print('%-16s %-3d %12.2f %8.2f %10.2f %9.3f %s'
                      % (key if i == 0 else '', P, r['logz'], r['logz_err'],
                         r['H0_mean'], w[key][P], ','.join(r['at_bound']) or '-'))


if __name__ == '__main__':
    main()
