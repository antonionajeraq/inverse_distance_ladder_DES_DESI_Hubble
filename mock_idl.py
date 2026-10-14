r"""mock_idl.py -- the NULL TEST behind the redshift-floor sweep.

THE QUESTION.  The floor sweep (`fig:floorsweep-real-new` in the paper) shows the DES+legacy
$H_0$ climbing monotonically from 67.85 at $z_{\rm floor}=0.030$ to 76.48 at 0.090 -- a
9 km/s/Mpc trend -- while DES alone sits flat and high.  Is a trend of that size a FEATURE
of the data, or is it what this estimator would return even if the Universe were exactly
the CMB-calibrated flat LCDM and the legacy supernovae were perfectly calibrated?

THE ANSWER IS OBTAINED BY REPLACING THE DATA AND NOTHING ELSE.  Every redshift, every
covariance, every prior, the sampler, the grid, the polynomial order and the cell list are
the real analysis, untouched: this module imports `dovekie_cosmography` and monkeypatches
the three DATA PROVIDERS -- `read_hubble_diagram`, `bao_vector`, `cc_block` -- to serve a
realisation drawn from a fiducial flat LCDM.  `dc.fit`, `dc.task_floorsweep`, `dc.joint_model`
and `dc.make_chi2` are then called UNMODIFIED.  Nothing in dovekie_cosmography.py is edited.

WHY PATCHING THE READERS AND NOT THE BLOCKS.  `sn_dataset('des_only')` is defined as a
row-subset of the same raw catalogue that gives `sn_dataset('full')`.  Substituting the
magnitudes at the reader keeps that relationship exact, so the mock DES sample is literally
the DES rows of the mock DES+legacy sample -- as in the real data.  Substituting them per
sample would let the two mocks disagree on the shared supernovae, and the whole point of
the sweep is what happens when the legacy rows are added to the DES rows.

THE FIDUCIAL.  Flat LCDM at the SPA (SPT-3G D1 + ACT DR6 + Planck; Camphuis et al. 2026)
values H0 = 67.19, Om = 0.3175, so q0 = 3/2 Om - 1 = -0.52625.  The sound horizon is set to
r_d = 147.05 cMpc, the CENTRE of the prior configuration A imposes, so the mock is exactly
self-consistent with the likelihood and no bias is planted by construction.  M = -19.40,
the value the real ladder returns, so the magnitudes are realistic and no prior edge is
approached.  The legacy supernovae are generated from the SAME fiducial with NO zero-point
offset: Delta_true = 0 is the null the Delta arm is tested against.

THREE REALISATION TYPES, AND THEY ANSWER DIFFERENT QUESTIONS.

  noiseless (--real 0)   Data set exactly to the fiducial model on the SAME z-grid and the
                         SAME trapezoid the likelihood uses.  Any residual trend here is
                         pure ESTIMATOR bias: the k_max=4 y-polynomial is not flat LCDM,
                         and how badly it fails depends on which redshifts survive the cut.
                         With background='lcdm' the truth is inside the model and recovery
                         must be exact -- that is the pipeline's own unit test.

  noisy (--real >= 1)    One coherent draw from the REAL STAT+SYS covariance over the whole
                         catalogue, then truncated at each floor exactly as the real data
                         are.  This is the only way to ask how often a 9 km/s/Mpc climb
                         happens by chance: the floors are not independent samples, they
                         are nested subsets of ONE realisation, and the trend is a property
                         of that realisation.

  injected (--delta X)   The same, with X mag added to the legacy rows only.  Used to check
                         the Delta arm recovers an offset it was given.

OUTPUTS ARE QUARANTINED.  CHAINS_DIR and RESULTS_DIR are redirected to
chains_mock/<tag>/ and results_mock/ before dovekie_cosmography is imported, so a mock cell
can never be served back to the real tables.  Each realisation directory carries a
MOCK_SPEC.json recording the fiducial, the seed and the SHA256 of the mock data vectors.

USAGE
    python mock_idl.py --real 0 --task floorsweep            # noiseless sweep
    python mock_idl.py --real 1 --task floorsweep            # one noisy realisation
    python mock_idl.py --real 0 --task baseline              # recovery at no floor
    python mock_idl.py --real 0 --task selftest              # LCDM background must be exact
    python mock_idl.py --reals 1-100 --task lsweep           # least-squares ensemble
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------
# The fiducial.  ONE place, and it is written into every MOCK_SPEC.json.
# --------------------------------------------------------------------------------------
FID = dict(
    H0=67.19,        # SPA, Camphuis et al. 2026 (make_figures.SPA_H0[0])
    Om=0.3175,       # SPA          (make_figures.SPA_OM)
    rd=147.05,       # = BASELINE['RD_PLANCK'], the centre of the configuration-A prior
    M=-19.40,        # the value the real ladder returns for M_des
    background='lcdm',   # 'lcdm' -> H = H0 sqrt(Om(1+z)^3+1-Om), the PHYSICAL null.
                         # 'cosmo' -> H = sum_{k<=P} c_k y^k with c_k = the exact y-Taylor
                         # coefficients of that same LCDM, i.e. the fiducial TRUNCATED into
                         # the fitted family.  See the note in build_mock.
)
FID['q0'] = 1.5 * FID['Om'] - 1.0


def fiducial_Hg(dc, zgrid, fid, P=None):
    """H on the grid for whichever fiducial background is asked for.

    THE TWO ARE NOT THE SAME UNIVERSE, and the difference is the whole point.  With
    background='lcdm' the truth is OUTSIDE the fitted k_max=4 family, so the noiseless
    sweep measures the estimator's TRUNCATION bias.  With background='cosmo' the truth
    IS a member of that family -- the degree-P y-polynomial whose coefficients are the
    exact Taylor coefficients of the SAME LCDM at u = 0, equivalently the (H0, q0, j0,
    s0, ...) that flat LCDM implies -- so truncation is switched off BY CONSTRUCTION and
    any residual drift with z_floor is something else (lever arm, covariance, priors).
    The difference between the two noiseless sweeps is therefore the truncation term,
    isolated.  It is NOT a more model-independent null: it is a null that has been
    chosen so the fitted model cannot be wrong."""
    P = dc.BASELINE['P'] if P is None else P
    if fid.get('background', 'lcdm') == 'lcdm':
        return dc.H_lcdm(zgrid, fid['H0'], fid['Om'])
    if fid['background'] != 'cosmo':
        raise ValueError("fiducial background must be 'lcdm' or 'cosmo', got %r"
                         % fid['background'])
    return dc.H_poly(zgrid, dc.lcdm_coeffs(fid['Om'], fid['H0'], P))


def _chain_dir(tag):
    return os.path.join(HERE, 'chains_mock', tag)


def realisation_tag(real, delta=0.0, background='lcdm'):
    """Directory name for a realisation.  The injected offset is part of the identity: a
    Delta=0.05 mock is a different data set from a Delta=0 mock at the same seed, and they
    must not share a cache."""
    if real < 0:
        return 'realdata'          # no mock installed: the observed catalogue itself
    t = 'noiseless' if real == 0 else 'real%03d' % real
    if delta:
        t += '_d%+.4f' % delta
    if background != 'lcdm':
        t += '_' + background
    return t


# =======================================================================================
# 1.  Generation
# =======================================================================================
def _mvn_draw(cov, rng, jitter=1e-9):
    """One draw from N(0, cov) by Cholesky.  The DES STAT+SYS matrix is dense and its
    correlations are the whole point -- a diagonal draw would delete exactly the
    survey-to-survey structure the sweep is probing."""
    n = cov.shape[0]
    L = np.linalg.cholesky(0.5 * (cov + cov.T) + jitter * np.eye(n))
    return L @ rng.standard_normal(n)


def build_mock(dc, real, delta=0.0, fid=None):
    """Return the mock data vectors, WITHOUT installing them.

    real == 0  -> noiseless.  real >= 1 -> that seed's noisy realisation.
    """
    fid = dict(FID if fid is None else fid)
    rng = np.random.default_rng(20260825 + 1000 * int(real))

    # --- the canonical z-grid --------------------------------------------------------
    # build_blocks makes zgrid = linspace(0, zmax + 1e-6, NGRID) with
    # zmax = max(BAO z, SN z, CC z).  The Lyman-alpha BAO point at z = 2.33 dominates both
    # samples and every floor (the floors cut the LOW end only), so ONE grid serves the
    # whole sweep and the mock is generated on precisely the grid the likelihood integrates.
    _O, _C, zs_b, kinds_b = dc.bao_vector()
    z_uni = np.unique(zs_b)
    zmax = max(float(z_uni.max()), float(dc.sn_dataset('full')['z'].max()))
    zgrid = np.linspace(0.0, zmax + 1e-6, dc.BASELINE['NGRID'])
    Hg = fiducial_Hg(dc, zgrid, fid)

    # --- supernovae ------------------------------------------------------------------
    cid, ids, z, zhel, mu, muerr = dc.read_hubble_diagram()
    cov_all = dc.read_covariance()
    keep = muerr < dc.BASELINE['CLEAN_MUERR_MAX']
    cov = 0.5 * (cov_all[np.ix_(keep, keep)] + cov_all[np.ix_(keep, keep)].T)

    DC_sn, _dh, _dv = dc.distances(zgrid, Hg, z[keep])     # flat: D_M == D_C
    m_true = dc.sn_apparent_mag(fid['M'], zhel[keep], DC_sn)
    leg = (ids[keep] != dc.DES_IDSURVEY).astype(float)
    if delta:
        m_true = m_true + delta * leg
    n_sn = int(keep.sum())
    noise_sn = np.zeros(n_sn) if real == 0 else _mvn_draw(cov, rng)
    m_mock = m_true + noise_sn

    # The reader hands back MU, and sn_dataset forms m = MU + M_B_FID.  Invert that here so
    # the substitution happens at the reader and every downstream path is untouched.
    mu_new = np.array(mu, float, copy=True)
    mu_new[keep] = m_mock - dc.BASELINE['M_B_FID']

    # --- BAO -------------------------------------------------------------------------
    DM_b, DH_b, DV_b = dc.distances(zgrid, Hg, zs_b)
    model_b = np.where(kinds_b == 'DM', DM_b,
                       np.where(kinds_b == 'DH', DH_b, DV_b)) / fid['rd']
    noise_b = np.zeros(len(model_b)) if real == 0 else _mvn_draw(_C, rng, jitter=1e-12)
    O_mock = model_b + noise_b
    bao_by_key = {(float(zz), str(kk)): float(vv)
                  for zz, kk, vv in zip(zs_b, kinds_b, O_mock)}

    # --- cosmic chronometers ---------------------------------------------------------
    # Not used by configuration A (the sweep), but mocked so configuration B is available
    # from the same realisation rather than being half real and half simulated.
    z_cc, H_cc, C_cc = dc.cc_block()
    H_cc_true = dc.H_poly(z_cc, dc.lcdm_coeffs(fid['Om'], fid['H0'],
                                                      dc.BASELINE['P'])) \
        if fid.get('background', 'lcdm') == 'cosmo' \
        else dc.H_lcdm(z_cc, fid['H0'], fid['Om'])
    noise_cc = np.zeros(len(z_cc)) if real == 0 else _mvn_draw(C_cc, rng)
    H_cc_mock = H_cc_true + noise_cc

    def _h(a):
        return hashlib.sha256(np.ascontiguousarray(a, float).tobytes()).hexdigest()[:16]

    spec = dict(
        fiducial=fid, real=int(real), delta_injected=float(delta),
        seed=int(20260825 + 1000 * int(real)),
        noiseless=bool(real == 0),
        n_sn=n_sn, n_legacy=int(leg.sum()), n_bao=len(O_mock), n_cc=len(z_cc),
        sha_mu=_h(mu_new[keep]), sha_bao=_h(O_mock), sha_cc=_h(H_cc_mock),
        zgrid=[0.0, float(zgrid[-1]), int(len(zgrid))],
        code_sha256=dc.source_sha256(), written=time.strftime('%Y-%m-%dT%H:%M:%S'),
    )
    return dict(cid=cid, ids=ids, z=z, zhel=zhel, mu=mu_new, muerr=muerr,
                bao_by_key=bao_by_key, cc=(z_cc, H_cc_mock, C_cc),
                keep=keep, leg=leg, m_true=m_true, zgrid=zgrid, Hg=Hg,
                spec=spec)


def install_mock(dc, mock):
    """Point dovekie_cosmography's three data readers at the mock and clear its caches.

    Everything downstream -- sn_dataset, sn_block, legacy_columns, build_blocks,
    joint_model, make_chi2, fit, task_floorsweep -- is the published code and is called
    with no arguments it would not receive on the real data."""
    cid, ids, z, zhel, mu, muerr = (mock['cid'], mock['ids'], mock['z'],
                                    mock['zhel'], mock['mu'], mock['muerr'])
    real_bao = dc.bao_vector
    by_key = mock['bao_by_key']
    z_cc, H_cc, C_cc = mock['cc']

    def mock_hd():
        return cid, ids, z, zhel, mu, muerr

    def mock_bao(use_dv=None, zmax=None):
        # Delegate for the ORDER, the redshifts, the kinds and the covariance -- the mock
        # replaces the measured values only.  Any cut the caller asks for is then applied
        # by the real function and the mock follows it automatically.
        O, Cov, zs, kinds = real_bao(use_dv=use_dv, zmax=zmax)
        O = np.array([by_key[(float(zz), str(kk))] for zz, kk in zip(zs, kinds)])
        return O, Cov, zs, kinds

    def mock_cc():
        return z_cc, H_cc, C_cc

    dc.read_hubble_diagram = mock_hd
    dc.bao_vector = mock_bao
    dc.cc_block = mock_cc
    dc._SN_CACHE.clear()
    return dc


# =======================================================================================
# 2.  Least-squares path -- the same chi2, with the floor and the offset it needs
# =======================================================================================
# dc.minimise_chi2 exists but is not usable here for two reasons that are properties of
# that function and not of this test: it drops the z_floor argument (it calls make_chi2
# without one) and it has no legacy-offset parameter.  Both are supplied here WITHOUT
# rewriting the model: `dc.make_chi2` builds the residual, and Delta is applied by
# offsetting the DATA vector the residual reads, which is algebraically identical to adding
# Delta to the model of the legacy rows.
from scipy.optimize import least_squares          # noqa: E402


def ls_fit(dc, sample, z_floor=0.0, joint_delta=False, P=None, n_multistart=3,
           want_errors=True):
    """chi2 minimum with no prior on the coefficients, at this floor, with or without a
    free legacy zero point.  Returns dict(H0, q0, M, rd, Delta, chi2, sigma..., cond)."""
    P = dc.BASELINE['P'] if P is None else P
    _f, meta = dc.make_chi2(sample, 'A', 'cosmo', P=P, z_floor=z_floor)
    resid, n_shape, b = meta['resid'], meta['n_shape'], meta['blocks']
    m0 = np.array(b['m_sn'], float, copy=True)

    if joint_delta:
        U, _names = dc.legacy_columns(sample, split=False, z_floor=z_floor)
        u = U[:, 0]
        if u.sum() == 0:
            raise ValueError('no legacy supernova survives z_floor=%.3f: Delta is not '
                             'identified and would return its starting value' % z_floor)

        def resid_ext(th):
            b['m_sn'] = m0 - th[-1] * u
            try:
                return resid(th[:-1])
            finally:
                b['m_sn'] = m0
    else:
        u = None

        def resid_ext(th):
            return resid(th)

    sh = dc.lcdm_coeffs(FID['Om'], FID['H0'], P)
    theta0 = np.concatenate([sh, [FID['M'], dc.BASELINE['RD_PLANCK']]]
                            + ([[0.0]] if joint_delta else []))
    rng = np.random.default_rng(0)
    best = (np.inf, None)
    for i in range(max(1, n_multistart)):
        t0 = theta0 if i == 0 else theta0 * (1.0 + 10.0 ** (-3 + i)
                                             * rng.standard_normal(len(theta0)))
        try:
            sol = least_squares(resid_ext, t0, method='lm', max_nfev=100000,
                                xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except Exception:
            continue
        if not sol.success:
            continue
        fv = float(sol.fun @ sol.fun)
        if np.isfinite(fv) and fv < best[0]:
            best = (fv, sol.x)
    if best[1] is None:
        raise RuntimeError('LS failed: sample=%s z_floor=%.3f joint=%s'
                           % (sample, z_floor, joint_delta))
    chi2, th = best
    c = th[:n_shape]
    out = dict(sample=sample, z_floor=float(z_floor), joint_delta=bool(joint_delta), P=P,
               H0=float(c[0]), q0=float(c[1] / c[0] - 1.0), M=float(th[n_shape]),
               rd=float(th[n_shape + 1]), chi2=float(chi2),
               n_sn=int(len(b['z_sn'])),
               n_leg=int(u.sum()) if u is not None else 0)
    if joint_delta:
        out['Delta'] = float(th[-1])

    if want_errors:
        # Gauss-Newton curvature, exactly ls_errors.cov_from_jac, on the EXTENDED residual.
        import ls_errors as le
        hs = le._steps(th[:n_shape + 2], n_shape)
        if joint_delta:
            hs = np.concatenate([hs, [max(abs(th[-1]), 1e-3) * le.REL_STEP]])
        r0 = resid_ext(th)
        J = np.empty((len(r0), len(th)))
        for i in range(len(th)):
            tp, tm = th.copy(), th.copy()
            tp[i] += hs[i]
            tm[i] -= hs[i]
            J[:, i] = (resid_ext(tp) - resid_ext(tm)) / (2.0 * hs[i])
        cov, cond = le.cov_from_jac(J)
        sig = np.sqrt(np.diag(cov))
        # q0 = c1/c0 - 1 by linear propagation through the (c0, c1) block.
        g = np.zeros(len(th))
        g[0], g[1] = -c[1] / c[0] ** 2, 1.0 / c[0]
        out.update(sig_H0=float(sig[0]), sig_M=float(sig[n_shape]),
                   sig_rd=float(sig[n_shape + 1]), cond=float(cond),
                   sig_q0=float(np.sqrt(g @ cov @ g)))
        if joint_delta:
            out['sig_Delta'] = float(sig[-1])
    return out


# =======================================================================================
# 3.  Tasks
# =======================================================================================
def task_selftest(dc, mock):
    """The truth is INSIDE the flat-LCDM background, so the noiseless mock must come back
    at the fiducial to the precision of the shared trapezoid.  If this fails, the mock and
    the likelihood are not using the same model and nothing below means anything."""
    print('\n--- selftest: LCDM background on the noiseless mock ---')
    ok = True
    for smp in ('des_only', 'full'):
        r = ls_fit_lcdm(dc, smp)
        dH0 = r['H0'] - FID['H0']
        dOm = r['Om'] - FID['Om']
        dM = r['M'] - FID['M']
        drd = r['rd'] - FID['rd']
        good = abs(dH0) < 0.02 and abs(dOm) < 0.002 and abs(dM) < 0.002
        ok &= good
        print('  %-9s H0 %8.4f (%+.4f)  Om %7.5f (%+.5f)  M %9.5f (%+.5f)  '
              'rd %8.4f (%+.4f)  chi2 %.3e  %s'
              % (smp, r['H0'], dH0, r['Om'], dOm, r['M'], dM, r['rd'], drd,
                 r['chi2'], 'OK' if good else 'FAIL'))
    print('  verdict: %s' % ('PASS' if ok else 'FAIL'))
    return ok


def ls_fit_lcdm(dc, sample, z_floor=0.0):
    """The same least squares with the flat-LCDM background -- the control in which the
    fitted family CONTAINS the fiducial."""
    _f, meta = dc.make_chi2(sample, 'A', 'lcdm', P=dc.BASELINE['P'], z_floor=z_floor)
    resid, b = meta['resid'], meta['blocks']
    theta0 = np.array([FID['H0'], FID['Om'], FID['M'], dc.BASELINE['RD_PLANCK']])
    sol = least_squares(resid, theta0, method='lm', max_nfev=100000,
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    x = sol.x
    return dict(sample=sample, z_floor=float(z_floor), H0=float(x[0]), Om=float(x[1]),
                q0=float(1.5 * x[1] - 1.0), M=float(x[2]), rd=float(x[3]),
                chi2=float(sol.fun @ sol.fun), n_sn=int(len(b['z_sn'])))


def task_baseline(dc, quiet=True):
    """Recovery at NO floor: the four published baseline cells, on the mock."""
    rows = []
    for smp, jd in (('des_only', False), ('full', False), ('full', True)):
        cell = 'baseline_%s%s_P%d' % (smp, '_joint' if jd else '', dc.BASELINE['P'])
        cached = dc.cache_load('mockbase', cell)
        if cached is not None:
            print('[baseline] %s (cached)' % cell)
            rows.append(dc._unpack(cached))
            continue
        print('[baseline] %s sampling...' % cell)
        r = dc.fit(smp, 'A', 'cosmo', joint_delta=jd, quiet=quiet)
        dc.cache_save('mockbase', cell, **dc._pack(r))
        rows.append(r)
    print('\n%-16s %8s %7s %8s %7s %10s %9s %8s'
          % ('cell', 'H0', 'sig', 'q0', 'sig', 'M', 'Delta', 'sig'))
    for r in rows:
        tag = r['sample'] + ('+joint' if r.get('joint_delta') else '')
        dl = (('%9.4f %8.4f' % (r['Delta_mean'], r['Delta_std']))
              if 'Delta_mean' in r else '%9s %8s' % ('-', '-'))
        print('%-16s %8.2f %7.2f %+8.2f %7.2f %10.4f %s'
              % (tag, r['H0_mean'], r['H0_std'], r['q0_mean'], r['q0_std'],
                 r['M_mean'], dl))
    print('\n  truth: H0 %.2f  q0 %+.3f  M %.4f  Delta 0' % (FID['H0'], FID['q0'], FID['M']))
    return rows


def task_lsweep(dc, tag, quiet=True):
    """The whole floor sweep by least squares.  Same cells, same floors, same arms as
    dc.floorsweep_grid, but prior-free and seconds per cell rather than minutes -- which is
    what makes an ENSEMBLE of realisations affordable.  Validated against the NUTS sweep on
    the same mock by --task compare."""
    # dc.floorsweep_grid() verbatim, plus ONE cell it cannot contain: des_only at
    # z_floor = 0.073.  The grid drops it because the DES minimum is 0.0733, so the cut is
    # a no-op there -- but the paper's headline sweep statistic is exactly the contrast
    # "at z_floor = 0.073, adding the 10 surviving legacy SNe moves H0 by 4.78", and that
    # needs the DES-only fit at the SAME floor to subtract.  It costs one extra cell.
    grid = list(dc.floorsweep_grid())
    if ('des_only', 0.073, False) not in grid:
        grid.append(('des_only', 0.073, False))
    out = []
    for smp, fl, jd in grid:
        try:
            r = ls_fit(dc, smp, z_floor=fl, joint_delta=jd)
        except Exception as exc:
            print('[lsweep] %s %.3f %s FAILED: %s' % (smp, fl, jd, exc))
            continue
        r['tag'] = tag
        out.append(r)
        if not quiet:
            print('[lsweep] %-9s %.3f %s N=%4d Nleg=%3d  H0 %7.2f +- %5.2f  q0 %+6.2f'
                  % (smp, fl, 'J' if jd else '-', r['n_sn'], r['n_leg'], r['H0'],
                     r.get('sig_H0', np.nan), r['q0']))
    # The order is part of the file's identity: the whole point of an order variation is
    # to compare it against k_max = 4, and a shared filename would overwrite it instead.
    suffix = '' if dc.BASELINE['P'] == 4 else '_P%d' % dc.BASELINE['P']
    path = os.path.join(dc.RESULT_DIR, 'lsweep_%s%s.json' % (tag, suffix))
    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('[lsweep] %-22s %3d cells -> %s' % (tag, len(out), os.path.basename(path)))
    return out


def task_nuts_cells(dc, spec, recompute=False, quiet=True):
    """NUTS on a NAMED SUBSET of the sweep, cached under the same cell names task_floorsweep
    uses so the two are interchangeable.

    The full 48-cell sweep at the published sampler settings is hours per cell on one box.
    The ensemble above is least squares, and the least-squares/posterior agreement is
    established on the OBSERVED catalogue by analyse_mock -- but it must also hold on the
    MOCK, where the data are different and the flat directions could in principle be
    populated differently.  These few cells are that check.

    spec: 'full:0.030,des_only:0.080,fullJ:0.073'  -- 'J' suffixes the joint-Delta arm.
    """
    rows = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, _, fl = item.partition(':')
        jd = name.endswith('J')
        smp = name[:-1] if jd else name
        fl = float(fl)
        cell = '%s%s_%.3f_P%d' % (smp, '_joint' if jd else '', fl, dc.BASELINE['P'])
        cached = None if recompute else dc.cache_load('floorsweep', cell)
        if cached is not None:
            print('[nuts] %s (cached)' % cell)
            rows.append(dc._unpack(cached))
            continue
        d = dc.sn_dataset(smp)
        keep = d['z'] >= fl
        print('[nuts] %s N=%d N_leg=%d sampling...'
              % (cell, int(keep.sum()),
                 int((keep & (d['idsurvey'] != dc.DES_IDSURVEY)).sum())))
        t0 = time.time()
        r = dc.fit(smp, 'A', 'cosmo', z_floor=fl, joint_delta=jd, quiet=quiet)
        r['n_sn'] = int(keep.sum())
        r['n_leg'] = int((keep & (d['idsurvey'] != dc.DES_IDSURVEY)).sum())
        r['wall_s'] = time.time() - t0
        dc.cache_save('floorsweep', cell, **dc._pack(r))
        rows.append(r)
        print('[nuts] %s  H0 %.3f +- %.3f  q0 %+.3f  (%.0f s)'
              % (cell, r['H0_mean'], r['H0_std'], r['q0_mean'], r['wall_s']))
    print('\n%-20s %8s %7s %8s %7s %10s %9s'
          % ('cell', 'H0', 'sig', 'q0', 'sig', 'M', 'Delta'))
    for r in rows:
        tag = r['sample'] + ('+J' if r.get('joint_delta') else '') + ' %.3f' % r['z_floor']
        print('%-20s %8.3f %7.3f %+8.3f %7.3f %10.5f %9s'
              % (tag, r['H0_mean'], r['H0_std'], r['q0_mean'], r['q0_std'], r['M_mean'],
                 ('%.5f' % r['Delta_mean']) if 'Delta_mean' in r else '-'))
    return rows


# =======================================================================================
# 4.  Entry point
# =======================================================================================
def parse_reals(s):
    out = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part.lstrip('-') and not part.startswith('-'):
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('USAGE')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True,
                    choices=['floorsweep', 'baseline', 'selftest', 'lsweep', 'lcdmsweep',
                             'nutscells'])
    ap.add_argument('--real', type=int, default=0,
                    help='0 = noiseless; >= 1 selects a noise realisation')
    ap.add_argument('--reals', default=None,
                    help='a range such as 1-100 or 1,5,9 (lsweep only)')
    ap.add_argument('--delta', type=float, default=0.0,
                    help='magnitudes of legacy zero-point offset to INJECT (default 0)')
    ap.add_argument('--cells', default='full:0.030,full:0.073,full:0.090,'
                                     'des_only:0.080,fullJ:0.030,fullJ:0.073',
                    help='(nutscells) which cells to sample')
    ap.add_argument('--fiducial', default='lcdm', choices=['lcdm', 'cosmo'],
                    help="'lcdm' (default) plants flat LCDM -- the truth is OUTSIDE the "
                         "fitted family and the noiseless sweep measures truncation bias. "
                         "'cosmo' plants the degree-P y-polynomial with LCDM's own Taylor "
                         "coefficients -- the truth is INSIDE the family, so truncation is "
                         "off by construction.")
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--recompute', action='store_true')
    # SHARDING IS BY STRIDE, NOT BY BLOCK (i % ntasks == shard), the same convention as
    # dh0_ensemble.py and slurm/dovekie_dh0.slurm.  Every shard then costs the same and no
    # single slow node holds the array open on a tail of leftovers.  The realisation INDEX
    # names the output file, so narrowing the array leaves the other realisations simply
    # unrun rather than renumbering and clobbering them.
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--no-spec', action='store_true',
                    help='do not create chains_mock/<tag>/ or write MOCK_SPEC.json.  For '
                         'the 10,000-member ensembles: the least-squares path never reads '
                         'a chain cache, and sha_mu is not a portable mock identity '
                         'anyway, so the only effect of writing them is 10,000 extra '
                         'directories and inodes on a shared filesystem.')
    a = ap.parse_args(argv)
    if not a.ntasks >= 1 or not 0 <= a.shard < a.ntasks:
        raise SystemExit('--shard %d is not a valid shard of --ntasks %d.  An out-of-range '
                         'shard runs NOTHING and exits 0, which looks like success.'
                         % (a.shard, a.ntasks))

    fid = dict(FID, background=a.fiducial)
    reals = parse_reals(a.reals) if a.reals else [a.real]
    if len(reals) > 1 and a.task not in ('lsweep', 'lcdmsweep'):
        raise SystemExit('--reals is for the least-squares tasks; NUTS tasks take --real')
    if a.ntasks > 1:
        reals = [r for i, r in enumerate(reals) if i % a.ntasks == a.shard]
        print('[shard] %d/%d -> %d realisations%s'
              % (a.shard, a.ntasks, len(reals),
                 ('  %s ... %s' % (reals[0], reals[-1])) if reals else '  (none)'))
        if not reals:
            return

    # RESULTS/CHAINS MUST BE REDIRECTED BEFORE dovekie_cosmography IS IMPORTED: it reads
    # both from the environment at import time.  The per-realisation chain directory is
    # what keeps two mocks from serving each other's cells -- the cache key knows about
    # BASELINE and the source files, and knows nothing about which data set was in memory.
    os.environ['RESULTS_DIR'] = os.path.join(HERE, 'results_mock')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    first_tag = realisation_tag(reals[0], a.delta, a.fiducial)
    os.environ['CHAINS_DIR'] = _chain_dir(first_tag)
    sys.path.insert(0, HERE)
    import dovekie_cosmography as dc                                   # noqa: E402
    print(dc.banner())
    P_fid = dc.BASELINE['P']
    if a.fiducial == 'cosmo':
        cg = dc.lcdm_cg(FID['Om'], P_fid)
        names = ['q0', 'j0', 's0', 'l0', 'm0'][:len(cg)]
        print('[fiducial] cosmo  H(y) = sum_{k<=%d} c_k y^k  with the EXACT y-Taylor '
              'coefficients of flat LCDM(Om=%.4f):' % (P_fid, FID['Om']))
        print('           H0 = %.4f   %s'
              % (FID['H0'], '  '.join('%s = %+.6f' % (n, v) for n, v in zip(names, cg))))
        print('           c_k = %s'
              % np.array2string(dc.lcdm_coeffs(FID['Om'], FID['H0'], P_fid),
                                precision=4, max_line_width=200))
        print('           TRUTH IS INSIDE THE FITTED FAMILY: truncation bias is zero by '
              'construction.\n           This is an estimator/coverage test, NOT a more '
              'model-independent null.')
    else:
        print('[fiducial] lcdm   H(z) = %.2f sqrt(%.4f (1+z)^3 + %.4f)   '
              '(truth OUTSIDE the k_max=%d family)'
              % (FID['H0'], FID['Om'], 1 - FID['Om'], P_fid))

    suffix_P = '' if dc.BASELINE['P'] == 4 else '_P%d' % dc.BASELINE['P']
    n_done = 0
    for real in reals:
        tag = realisation_tag(real, a.delta, a.fiducial)
        # RESUME.  A realisation is a fixed universe (the seed is the index alone), so a
        # result already on disk is the result this call would recompute.  Skipping it is
        # what makes a resubmission after a partial failure cost only the missing
        # realisations instead of the whole ensemble.
        if a.task == 'lsweep' and not a.recompute:
            # `results_mock/json/` is where the ensemble is kept (see mock_paths.py); the
            # flat directory is where THIS script writes.  Resume has to look in both, or a
            # tidied ensemble looks entirely missing and 10,000 realisations are recomputed.
            done = [os.path.join(HERE, 'results_mock', d,
                                 'lsweep_%s%s.json' % (tag, suffix_P))
                    for d in ('json', '')]
            if any(os.path.exists(d) for d in done):
                n_done += 1
                continue
        dc.CHAIN_DIR = _chain_dir(tag)
        if not a.no_spec:
            os.makedirs(dc.CHAIN_DIR, exist_ok=True)
        if real < 0:
            # THE OBSERVED CATALOGUE, through this module's least-squares path.  The mock
            # ensemble is compared against THIS and not against the published posteriors,
            # so that the comparison is like for like: same estimator, same cell list, same
            # error convention.  The published NUTS numbers are then a separate check that
            # the estimator agrees with the posterior, which --task baseline makes.
            mock = None
            print('\n=== REAL DATA (no mock installed)')
        else:
            mock = build_mock(dc, real, delta=a.delta, fid=fid)
            install_mock(dc, mock)
            if not a.no_spec:
                with open(os.path.join(dc.CHAIN_DIR, 'MOCK_SPEC.json'), 'w') as fh:
                    json.dump(mock['spec'], fh, indent=2)
            print('\n=== mock %s  n_sn=%d (legacy %d)  bao=%d  sha_mu=%s'
                  % (tag, mock['spec']['n_sn'], mock['spec']['n_legacy'],
                     mock['spec']['n_bao'], mock['spec']['sha_mu']))

        if a.task == 'selftest':
            task_selftest(dc, mock)
        elif a.task == 'baseline':
            task_baseline(dc, quiet=a.quiet)
        elif a.task == 'nutscells':
            task_nuts_cells(dc, a.cells, recompute=a.recompute, quiet=a.quiet)
        elif a.task == 'floorsweep':
            dc.task_floorsweep(recompute=a.recompute, quiet=a.quiet)
        elif a.task == 'lsweep':
            task_lsweep(dc, tag, quiet=a.quiet)
        elif a.task == 'lcdmsweep':
            rows = [ls_fit_lcdm(dc, smp, z_floor=fl)
                    for smp, fl, jd in dc.floorsweep_grid() if not jd]
            for r in rows:
                r['tag'] = tag
            os.makedirs(dc.RESULT_DIR, exist_ok=True)
            with open(os.path.join(dc.RESULT_DIR, 'lcdmsweep_%s.json' % tag), 'w') as fh:
                json.dump(rows, fh, indent=1)
            print('[lcdmsweep] %d cells' % len(rows))

    if n_done:
        print('[resume] %d realisation(s) already on disk were skipped' % n_done)


if __name__ == '__main__':
    main()
