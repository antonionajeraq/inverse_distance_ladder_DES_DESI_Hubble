"""
dovekie_evidence_cmb.py -- the Table 3 model comparison REPEATED WITH THE CMB.

The baseline comparison (`dovekie_evidence`) constrains exp, CPL and LCDM with DES-Dovekie,
DESI DR2 BAO, the cosmic chronometers and a Planck prior on r_d.  CPL was built to fit BAO
and uncalibrated SNe CONSISTENTLY WITH THE CMB, and without the CMB its (w0, wa) volume is
mostly spent where nothing pins it, so the baseline Bayes factor may be judging CPL off
its own ground.  This module asks the same question with the CMB in the likelihood.

WHAT CHANGES, AND NOTHING ELSE

  - The Planck r_d prior is REMOVED.  r_d is derived from (omega_b, omega_bc), which the
    CMB measures; keeping the prior as well would count the CMB twice.
  - The CMB enters as the compressed likelihood of Banik et al. (2026, appendix A):
    a Gaussian on
    (100 theta_*, omega_b, omega_bc) with the SPA mean and covariance (Camphuis et al.
    2026), z_* and z_drag from that code's quadratic fits in (omega_b, omega_bc), and the
    closed-form sound horizon.  Plus the BBN prior on omega_b (USE_BBN, default on, as in
    the protocol).
  - H(z) therefore carries photons and three neutrinos (50/10/0 meV, N_eff = 3.044), the
    same kernel as the protocol's relativistic branch, so that D_M(z_*) is right.  At the
    redshifts of the SNe, BAO and chronometers the radiation is a 1e-4 correction.
  - The sampled parameters become (omega_b, Omega_bc, H0, ...) instead of
    (Om, H0, ..., r_d).  omega_b ~ U(0.020, 0.025) as in the protocol; Omega_bc keeps the
    baseline Om box U(0.05, 0.70).  Om is reported as Omega_bc + Omega_nu.

Everything else -- the SN likelihood with its rank-2 legacy marginalisation, the BAO
vector, the chronometers, the forward-modelled anchors, and the exp/CPL priors -- is taken
from `dovekie_evidence` unchanged, so a difference from Table 3 is the CMB and nothing else.

THE exp MODEL AND THE CMB.  H(z) = H0_LCDM E_LCDM(z) + A exp(-z/z1).  The physical
densities belong to the LCDM background, omega_bc = Omega_bc h_LCDM^2, because the excess
has died away long before recombination.  It does NOT die away in D_M(z_*): the integral
runs through z ~ 0, so a late-time excess moves theta_* and the CMB sees it.  That is the
point of including the CMB, and it is why D_M(z_*) is integrated with the full H(z).

The CPL dark-energy density at z_* is included in H(z) for D_M, but the closed-form sound
horizon ignores it, exactly as in the protocol.  This matters only where w(a -> 0) > 0,
which the CMB already rejects.

lnZ is again comparable WITHIN a sample only (improper global SN offset); every constant
normalisation is dropped, as in `dovekie_evidence`, because it cancels in lnB10.

USAGE

    SAMPLER=rslice BOOTSTRAP=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python dovekie_evidence_cmb.py --task grid --only full/none/cpl
    python dovekie_evidence_cmb.py --task check     # protocol cross-checks, no sampling
    python dovekie_evidence_cmb.py --task table     # results/table_evidence_cmb.{json,tex}

Cells cache under chains/evidence_cmb/, never on top of the baseline chains/evidence/.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import norm

import dovekie_cosmography as dc
import dovekie_evidence as de

TASK = 'evidence_cmb'
c_km_s = de.c_km_s
ZGRID = de.ZGRID

# =============================================================================
# 1.  The CMB protocol, ported from Dovekie_pCPL/dynesty_sn_4nodes.py
# =============================================================================
# Constants and fits are copied verbatim; `check` reproduces the protocol's numbers.
G_SI, C_SI = 6.6743e-11, 299792458.0
H_PLANCK, K_B, EV = 6.62607e-34, 1.380649e-23, 1.602176634e-19
HBAR = H_PLANCK / (2 * np.pi)
AU = 149597870700.0
MPC = AU / (np.pi / 180 / 3600) * 1e6
H_100 = 100e3 / MPC

N_EFF = 3.044
MNU = (0.05, 0.01, 0.0)            # eV, descending; one massless species
T_CMB = 2.72548
RHO_CRIT_100 = 3 * H_100 ** 2 / (8 * np.pi * G_SI)
RHO_GAMMA = np.pi ** 2 / 15 * K_B ** 4 / (HBAR ** 3 * C_SI ** 5) * T_CMB ** 4
R_NU_GAMMA = N_EFF * 7 / 8 * (4 / 11) ** (4 / 3)
_n_gamma = 16 * np.pi * 1.202056903159594 * (K_B * T_CMB / (H_PLANCK * C_SI)) ** 3
_n_nu_i = 3 / 11 * _n_gamma * (N_EFF / 3) ** 0.68
_rho_nu_ur_i = RHO_GAMMA * R_NU_GAMMA / 3
_rho_nu = [_n_nu_i * m * EV / C_SI ** 2 for m in MNU]
# knu = 2 transition, per massive species; the massless one stays radiation
A_NR_SQINV = np.array([(r / _rho_nu_ur_i) ** 2 - 1 for r in _rho_nu if r > 0])
N_NU_MASSLESS = sum(1 for m in MNU if m == 0)
W_GAMMA = RHO_GAMMA / RHO_CRIT_100
W_NU = sum(_rho_nu) / RHO_CRIT_100                 # massive species, today
W_NU_UR_I = _rho_nu_ur_i / RHO_CRIT_100

# SPA compressed CMB on (100 theta_*, omega_b, omega_bc), Camphuis et al. 2026
MU_CMB = np.array([1.04161119, 0.02239789, 0.14265263])
COV_CMB = 1.0e-9 * np.array([[53.84815397, 1.1072339, -28.70053592],
                             [1.1072339, 9.20446757, -17.60162507],
                             [-28.70053592, -17.60162507, 843.13219472]])
CINV_CMB = np.linalg.inv(COV_CMB)

# BBN: Giovanetti et al. (2025), PRD 112, 063530 (arXiv:2408.14531), Table 1, BBN only,
# marginalised over nuclear rates, PRIMAT, fixed N_eff: 100 omega_b = 2.191 +0.022 -0.021,
# symmetrised to 2.1915 +- 0.0215.
USE_BBN = os.environ.get('USE_BBN', '1').lower() not in ('0', 'false', 'no')
WB_BBN, SIG_WB_BBN = 0.021915, 0.000215

# Quadratic fits for T(z_*) and T(z_drag) in (omega_b, omega_bc)
_T_REF = np.array([0.02250, 0.1450])
_TS0, _TS_G = 2970.2891639778, np.array([-3188.9167937363, 194.0918526352])
_TS_H = np.array([[265960.265674023, -7860.9268306572], [-7860.9268306572, -488.3600433765]])
_TD0, _TD_G = 2892.7434892901, np.array([5925.6388958146, 203.3428424024])
_TD_H = np.array([[-189076.8034166881, -4767.8740117157], [-4767.8740117157, -626.4853862271]])

WB_LO, WB_HI = 0.020, 0.025                       # the protocol's omega_b box


def z_star_drag(wb, wbc):
    d = np.array([wb, wbc]) - _T_REF
    t_s = _TS0 + _TS_G @ d + 0.5 * d @ _TS_H @ d
    t_d = _TD0 + _TD_G @ d + 0.5 * d @ _TD_H @ d
    return t_s / T_CMB - 1.0, t_d / T_CMB - 1.0


def sound_horizon(wb, wbc, z):
    """Closed-form comoving sound horizon [Mpc] at z, matter + radiation only."""
    R0 = 0.75 * wb * RHO_CRIT_100 / RHO_GAMMA
    a_eq = (1 + R_NU_GAMMA) / (wbc * RHO_CRIT_100 / RHO_GAMMA)
    R_eq = R0 * a_eq
    R = R0 / (1.0 + z)
    u = (np.sqrt(1 + R) + np.sqrt(R + R_eq)) / (1 + np.sqrt(R_eq))
    return c_km_s / 100.0 * 2 / 3 * np.sqrt(3 / wbc * a_eq / R_eq) * np.log(u)


def E2_background(z, Obc, h, rho_de=None):
    """E^2(z) of the LCDM-like background with photons and neutrinos.

    The protocol's relativistic branch: each massive species goes as
    a^-4 sqrt(1 + (a/a_nr)^2), the massless one as radiation, and Omega_DE closes E(0)=1."""
    ap = 1.0 + np.asarray(z, float)
    a2 = 1.0 / ap ** 2
    hs = 1.0 / h ** 2
    Og, Our_i = W_GAMMA * hs, W_NU_UR_I * hs
    nu = N_NU_MASSLESS + sum(np.sqrt(1.0 + a2 * s) for s in A_NR_SQINV)
    Ode = 1.0 - Obc - Og - W_NU * hs - N_NU_MASSLESS * Our_i
    de_term = Ode if rho_de is None else Ode * rho_de
    return ap ** 3 * Obc + ap ** 4 * (Og + Our_i * nu) + de_term


def _rho_cpl(z, w0, wa):
    a = 1.0 / (1.0 + np.asarray(z, float))
    return a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))


# =============================================================================
# 2.  Models: (omega_b, Omega_bc, H0, ...) replaces (Om, H0, ..., r_d)
# =============================================================================
MODELS = de.MODELS
NDIM = de.NDIM
PARAM_NAMES = {'lcdm': ('wb', 'Obc', 'H0'),
               'exp': ('wb', 'Obc', 'H0_LCDM', 'A', 'z1'),
               'cpl': ('wb', 'Obc', 'H0', 'w0', 'wa')}


def H_of(z, theta, model):
    """H(z) [km/s/Mpc] at any z, radiation and neutrinos included."""
    wb, Obc, H0 = theta[0], theta[1], theta[2]
    h = H0 / 100.0
    if model == 'lcdm':
        return H0 * np.sqrt(E2_background(z, Obc, h))
    if model == 'cpl':
        return H0 * np.sqrt(E2_background(z, Obc, h, _rho_cpl(z, theta[3], theta[4])))
    if model == 'exp':
        A, z1 = theta[3], theta[4]
        return H0 * np.sqrt(E2_background(z, Obc, h)) + A * np.exp(-np.asarray(z) / z1)
    raise ValueError('unknown model %r' % model)


# Above ZMAX the comoving distance to z_* is integrated in ln(1+z) by Simpson's rule.
# 513 nodes reproduce scipy.quad to < 1e-7 relative (`--task check`), against a theta_*
# measured to 2.2e-4.
_N_HI = 513


def comoving_to(zs, theta, model, Hg, Dc):
    """D_C(zs) [Mpc]: the ZGRID integral below ZMAX plus a log-spaced tail above it."""
    x = np.linspace(np.log1p(de.ZMAX), np.log1p(zs), _N_HI)
    z = np.expm1(x)
    f = (1.0 + z) / H_of(z, theta, model)
    h = x[1] - x[0]
    tail = h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
    return Dc[-1] + c_km_s * tail


def make_ptform(model):
    def ptform(u):
        x = np.asarray(u)
        wb = WB_LO + (WB_HI - WB_LO) * x[0]
        Obc = de.OM_LO + (de.OM_HI - de.OM_LO) * x[1]
        H0 = de.H0_LO + (de.H0_HI - de.H0_LO) * x[2]
        if model == 'lcdm':
            return np.array([wb, Obc, H0])
        if model == 'cpl':
            return np.array([wb, Obc, H0,
                             de.W0_LO + (de.W0_HI - de.W0_LO) * x[3],
                             de.WA_LO + (de.WA_HI - de.WA_LO) * x[4]])
        z1 = (de.Z1_MIN * (de.Z1_MAX / de.Z1_MIN) ** x[4] if de.Z1_PRIOR == 'log'
              else de.Z1_MIN + (de.Z1_MAX - de.Z1_MIN) * x[4])
        return np.array([wb, Obc, H0, -de.A_MAX + 2.0 * de.A_MAX * x[3], z1])
    return ptform


def derived(theta, model):
    """(Om, rd, 100 theta_*, z_*) for one draw."""
    wb, Obc, H0 = theta[0], theta[1], theta[2]
    wbc = Obc * (H0 / 100.0) ** 2
    zs, zd = z_star_drag(wb, wbc)
    Hg = H_of(ZGRID, theta, model)
    Dc = de._build_dc(Hg)
    return (Obc + W_NU / (H0 / 100.0) ** 2, sound_horizon(wb, wbc, zd),
            100.0 * sound_horizon(wb, wbc, zs) / comoving_to(zs, theta, model, Hg, Dc), zs)


# =============================================================================
# 3.  Likelihood: dovekie_evidence's SN + BAO + CC + anchors, with the CMB for r_d
# =============================================================================
def make_loglike(sample, model, anchor='none'):
    """chi2 = chi2_SN + chi2_BAO + chi2_CC + chi2_CMB (+ chi2_BBN) (+ chi2_anchor).

    The SN block is `dovekie_evidence.make_loglike`'s, line for line -- including the
    exact rank-2 marginalisation of the legacy zero-point in `full_dleg` -- so that the
    CMB is the only difference.  r_d is the drag-epoch sound horizon of the sampled
    (omega_b, omega_bc), not a draw from a Planck prior."""
    dsname, free_delta = de.SAMPLES[sample]
    d = dc.sn_dataset(dsname)
    z_sn, zhel_sn, mu = d['z'], d['zhel'], d['mu']
    invC = np.linalg.inv(d['cov'])
    one = np.ones_like(mu)
    Cinv_one = invC @ one
    C_oo = float(one @ Cinv_one)

    u_leg = G_inv = None
    ln_delta_norm = 0.0
    if free_delta:
        u_leg = (d['idsurvey'] != dc.DES_IDSURVEY).astype(float)
        V = np.column_stack([one, u_leg])
        G_inv = np.linalg.inv(V.T @ (invC @ V))
        sig_delta = float(np.sqrt(G_inv[1, 1]))
        ln_delta_norm = float(np.log(sig_delta * np.sqrt(2.0 * np.pi)
                                     / (2.0 * de.DELTA_MAX)))
        Cinv_u = invC @ u_leg

    bao, cc = de._static_blocks()
    terms = de.anchor_terms(anchor)

    def sn_resid(Dc):
        return mu - dc.sn_apparent_mag(0.0, zhel_sn, np.interp(z_sn, ZGRID, Dc))

    def loglike(theta):
        wb, Obc, H0 = theta[0], theta[1], theta[2]
        Hg = H_of(ZGRID, theta, model)
        if np.any(Hg <= 0) or np.any(~np.isfinite(Hg)):
            return -1e30
        Dc = de._build_dc(Hg)
        if np.any(np.diff(Dc) <= 0):
            return -1e30

        r = sn_resid(Dc)
        Cinv_r = invC @ r
        A_ = float(r @ Cinv_r)
        ln_nuis = 0.0
        if u_leg is None:
            B_ = float(r @ Cinv_one)
            chi2_sn = A_ - B_ * B_ / C_oo
        else:
            g = np.array([float(r @ Cinv_one), float(r @ Cinv_u)])
            Ginv_g = G_inv @ g
            chi2_sn = A_ - float(g @ Ginv_g)
            s_d = np.sqrt(G_inv[1, 1])
            frac = (norm.cdf((de.DELTA_MAX - Ginv_g[1]) / s_d)
                    - norm.cdf((-de.DELTA_MAX - Ginv_g[1]) / s_d))
            ln_nuis = ln_delta_norm + float(np.log(max(frac, 1e-300)))

        # ---- the CMB, and the sound horizon it implies ------------------------------
        wbc = Obc * (H0 / 100.0) ** 2
        zs, zd = z_star_drag(wb, wbc)
        Ds = comoving_to(zs, theta, model, Hg, Dc)
        if not (np.isfinite(Ds) and Ds > 0):
            return -1e30
        rcmb = np.array([100.0 * sound_horizon(wb, wbc, zs) / Ds, wb, wbc]) - MU_CMB
        chi2_cmb = float(rcmb @ CINV_CMB @ rcmb)
        chi2_bbn = ((wb - WB_BBN) / SIG_WB_BBN) ** 2 if USE_BBN else 0.0
        rd = sound_horizon(wb, wbc, zd)

        DM = np.interp(bao['z_uni'], ZGRID, Dc)
        DH = c_km_s / np.interp(bao['z_uni'], ZGRID, Hg)
        DV = (bao['z_uni'] * DM ** 2 * DH) ** (1.0 / 3.0)
        pred = np.where(bao['is_dm'], DM[bao['col']],
                        np.where(bao['is_dh'], DH[bao['col']], DV[bao['col']])) / rd
        db = pred - bao['O']
        chi2_bao = float(db @ bao['Cinv'] @ db)

        rc = np.interp(cc['z'], ZGRID, Hg) - cc['H']
        chi2_cc = float(rc @ cc['Cinv'] @ rc)

        chi2_anc = 0.0
        for comp, a_mean, a_sig in terms:
            chi2_anc += ((de.anchor_infer(Hg, comp) - a_mean) / a_sig) ** 2

        return -0.5 * (chi2_sn + chi2_bao + chi2_cc + chi2_cmb + chi2_bbn
                       + chi2_anc) + ln_nuis

    def delta_moments(theta):
        if u_leg is None:
            return np.nan, np.nan
        Hg = H_of(ZGRID, theta, model)
        if np.any(Hg <= 0) or np.any(~np.isfinite(Hg)):
            return np.nan, np.nan
        r = sn_resid(de._build_dc(Hg))
        g = np.array([float(r @ Cinv_one), float(r @ Cinv_u)])
        return float((G_inv @ g)[1]), float(np.sqrt(G_inv[1, 1]))

    loglike.delta_moments = delta_moments
    loglike.free_delta = bool(free_delta)
    return loglike


# =============================================================================
# 4.  One nested-sampling run, and the grid
# =============================================================================
def run_cell(sample, anchor, model, nlive=None, dlogz=None, seed=None, quiet=False,
             maxiter=None):
    import dynesty
    nlive = de.NLIVE if nlive is None else nlive
    dlogz = de.DLOGZ if dlogz is None else dlogz
    seed = de.SEED if seed is None else seed
    if not de._blas_threads_pinned():
        print('[evidence_cmb] WARNING: BLAS threads are not pinned to 1.')
    ll = make_loglike(sample, model, anchor)
    ns = dynesty.NestedSampler(ll, make_ptform(model), NDIM[model], nlive=nlive,
                               rstate=np.random.default_rng(seed), **de.sampler_kwargs())
    ns.run_nested(dlogz=dlogz, print_progress=not quiet, maxiter=maxiter)
    res = ns.results
    w = np.exp(res.logwt - res.logz[-1])
    w = w / w.sum()
    samples = res.samples

    der = np.array([derived(t, model) for t in samples])
    out = dict(sample=sample, anchor=anchor, model=model, cmb=True, bbn=USE_BBN,
               nlive=nlive, dlogz=dlogz, seed=seed,
               logz=float(res.logz[-1]), logz_err=float(res.logzerr[-1]),
               ncall=int(np.sum(res.ncall)), niter=int(res.niter),
               proposal=de.sampler_kwargs() or 'auto',
               params=list(PARAM_NAMES[model]),
               derived=['Om', 'rd', 'theta100', 'zstar'])

    def mom(name, x):
        m = float(np.sum(w * x))
        out[name + '_mean'] = m
        out[name + '_std'] = float(np.sqrt(np.sum(w * (x - m) ** 2)))

    for i, nm in enumerate(PARAM_NAMES[model]):
        mom(nm, samples[:, i])
    for i, nm in enumerate(out['derived']):
        mom(nm, der[:, i])
    if model == 'exp':
        h0 = samples[:, 2] + samples[:, 3]
        out['P_A_gt_0'] = float(np.sum(w * (samples[:, 3] > 0)))
    else:
        h0 = samples[:, 2]
    mom('H0_at_0', h0)

    if ll.free_delta:
        dm = np.array([ll.delta_moments(t) for t in samples])
        ok = np.isfinite(dm[:, 0])
        wm = w[ok] / w[ok].sum()
        mean = float(np.sum(wm * dm[ok, 0]))
        var = float(np.sum(wm * dm[ok, 1] ** 2) + np.sum(wm * (dm[ok, 0] - mean) ** 2))
        out['Delta_mean'], out['Delta_std'] = mean, float(np.sqrt(var))
    return out, samples, w, der


def cache_extra(anchor, model):
    e = de.cache_extra(anchor, model)
    e.update(CMB='SPA-Camphuis2026-compressed', MU_CMB=MU_CMB.tolist(),
             COV_CMB=COV_CMB.ravel().tolist(), USE_BBN=USE_BBN,
             WB_BBN=WB_BBN, SIG_WB_BBN=SIG_WB_BBN, WB_LO=WB_LO, WB_HI=WB_HI,
             MNU=list(MNU), N_EFF=N_EFF, RD='derived', N_HI=_N_HI)
    return e


def task_grid(only=None, recompute=False, quiet=False):
    cells = [(s, a, m) for s in ('des_only', 'full', 'full_dleg')
             for a in ('none', 'lfree3', 'breuval') for m in MODELS]
    if only:
        cells = [c for c in cells if '/'.join(c) in set(only)]
        if not cells:
            raise ValueError('no cell matched --only %s' % ', '.join(only))
    done = []
    for s, a, m in cells:
        cell = de.cell_name(s, a, m)
        extra = cache_extra(a, m)
        cached = None if recompute else dc.cache_load(TASK, cell, ignore_p=True,
                                                      extra=extra)
        if cached is not None:
            print('[evidence_cmb] %s (cached)' % cell)
            done.append(json.loads(str(cached['summary_json'])))
            continue
        print('[evidence_cmb] %s  nested sampling...' % cell, flush=True)
        out, draws, w, der = run_cell(s, a, m, quiet=quiet)
        dc.cache_save(TASK, cell, extra=extra, samples=draws, weights=w, derived=der,
                      summary_json=json.dumps(out))
        print('[evidence_cmb] %s  lnZ = %.3f +- %.3f  H(0) = %.2f +- %.2f'
              % (cell, out['logz'], out['logz_err'], out['H0_at_0_mean'],
                 out['H0_at_0_std']), flush=True)
        done.append(out)
    return done


# =============================================================================
# 5.  Checks against the protocol, and the table
# =============================================================================
def task_check():
    """Reproduce the protocol's numbers before anything is sampled."""
    from scipy.integrate import quad
    wb, wbc, H0 = 0.02240, 0.14265, 67.24
    th = np.array([wb, wbc / (H0 / 100) ** 2, H0])
    zs, zd = z_star_drag(wb, wbc)
    Hg = H_of(ZGRID, th, 'lcdm')
    Dc = de._build_dc(Hg)
    Ds = comoving_to(zs, th, 'lcdm', Hg, Dc)
    Dq = c_km_s * quad(lambda u: np.cosh(u) / H_of(np.sinh(u), th, 'lcdm'),
                       0, np.arcsinh(zs), epsabs=0, epsrel=1e-11, limit=500)[0]
    print('LCDM  wb=%.5f wbc=%.5f H0=%.2f' % (wb, wbc, H0))
    print('  z_* = %.2f   z_drag = %.2f' % (zs, zd))
    print('  r_* = %.3f   r_d = %.3f Mpc' % (sound_horizon(wb, wbc, zs),
                                            sound_horizon(wb, wbc, zd)))
    print('  D_M(z_*) = %.3f Mpc  (quad %.3f, rel diff %.1e)' % (Ds, Dq, Ds / Dq - 1))
    print('  100 theta_* = %.6f   (SPA %.6f +- %.6f)'
          % (100 * sound_horizon(wb, wbc, zs) / Ds, MU_CMB[0], np.sqrt(COV_CMB[0, 0])))
    print('  E(0)^2 = %.12f' % E2_background(0.0, th[1], H0 / 100))
    # exp with A = 0 and CPL at (-1, 0) must reproduce LCDM exactly
    for m, extra in (('exp', [0.0, 0.1]), ('cpl', [-1.0, 0.0])):
        t = np.concatenate([th, extra])
        ll_m = make_loglike('full_dleg', m, 'none')(t)
        ll_l = make_loglike('full_dleg', 'lcdm', 'none')(th)
        print('  %s at the LCDM point: lnL %.6f vs LCDM %.6f' % (m, ll_m, ll_l))


def load_grid():
    R = {}
    for s in ('des_only', 'full', 'full_dleg'):
        for a in ('none', 'lfree3', 'breuval'):
            for m in MODELS:
                c = dc.cache_load(TASK, de.cell_name(s, a, m), ignore_p=True,
                                  extra=cache_extra(a, m))
                if c is not None:
                    R[(s, a, m)] = json.loads(str(c['summary_json']))
    return R


def task_table():
    R = load_grid()
    if not R:
        raise SystemExit('no cached CMB evidence cells; run --task grid first')
    rows = []
    for s in ('des_only', 'full', 'full_dleg'):
        for a in ('none', 'lfree3', 'breuval'):
            ref = R.get((s, a, 'lcdm'))
            if ref is None:
                print('[table] missing LCDM reference for %s/%s' % (s, a))
                continue
            for m in ('exp', 'cpl'):
                r = R.get((s, a, m))
                if r is None:
                    continue
                p2, p3 = ('A', 'z1') if m == 'exp' else ('w0', 'wa')
                rows.append(dict(
                    sample=s, anchor=a, model=m,
                    lnB=r['logz'] - ref['logz'],
                    lnB_err=float(np.hypot(r['logz_err'], ref['logz_err'])),
                    Om=r['Om_mean'], Om_err=r['Om_std'],
                    p2=r[p2 + '_mean'], p2_err=r[p2 + '_std'],
                    p3=r[p3 + '_mean'], p3_err=r[p3 + '_std'],
                    H0=r['H0_at_0_mean'], H0_err=r['H0_at_0_std'],
                    rd=r['rd_mean'], rd_err=r['rd_std'],
                    P_A_gt_0=r.get('P_A_gt_0'),
                    Delta=r.get('Delta_mean'), Delta_err=r.get('Delta_std')))
    print('\n%-12s %-9s %-4s %12s %14s %16s %16s %14s %14s'
          % ('Sample', 'Anchor', 'Mod', 'lnB10', 'Om', 'w0/A', 'wa/z1', 'H(0)', 'r_d'))
    for r in rows:
        print('%-12s %-9s %-4s %+6.2f+-%.2f %7.3f+-%.3f %8.3f+-%-6.3f %8.3f+-%-6.3f '
              '%7.2f+-%.2f %7.2f+-%.2f'
              % (r['sample'], r['anchor'], r['model'], r['lnB'], r['lnB_err'], r['Om'],
                 r['Om_err'], r['p2'], r['p2_err'], r['p3'], r['p3_err'], r['H0'],
                 r['H0_err'], r['rd'], r['rd_err']))
    out = os.path.join(dc.RESULT_DIR, 'table_evidence_cmb.json')
    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print('\nwrote', out)
    return rows


# ---- mode + HPD, through print_cls_sigma's own per-parameter machinery -----------------
# `print_cls_sigma.main` is NOT run on this group: it rewrites results/mode_CLs.json from
# whatever groups it was given, which would drop every baseline record.  Its reader also
# rebuilds H(0) for `exp` as columns 1 + 2, which in THIS layout is Obc + H0_LCDM.  So the
# reader is swapped for one that knows the layout, and the records go to their own file.
def _read_cmb_chain(path, d):
    S = np.asarray(d['samples'], float)
    der = np.asarray(d['derived'], float)
    summ = json.loads(str(d['summary_json']))
    m = summ['model']
    cols = [('Om', der[:, 0]), ('rd', der[:, 1])]
    if m == 'exp':
        cols += [('A', S[:, 3]), ('z1', S[:, 4]), ('H0_at_0', S[:, 2] + S[:, 3])]
    elif m == 'cpl':
        cols += [('w0', S[:, 3]), ('wa', S[:, 4]), ('H0_at_0', S[:, 2])]
    else:
        cols += [('H0_at_0', S[:, 2])]
    return cols, np.asarray(d['weights'], float), 'dynesty (nested-sampling weights applied)'


def task_cls():
    import print_cls_sigma as pcs
    pcs.read_chain = _read_cmb_chain
    # The prior boxes are the evidence ones (A, z1, w0, wa, Om); resolve them under that
    # name.  Om here is Omega_bc + Omega_nu, 0.0014 above the Omega_bc box -- immaterial
    # to a KL test against a 0.65-wide box.
    _ref = pcs.prior_ref
    pcs.prior_ref = lambda name, group, cell, summ: _ref(name, 'evidence', cell, summ)
    recs = []
    d = os.path.join(dc.CHAIN_DIR, TASK)
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.npz'):
            res = pcs.analyse_chain(os.path.join(d, fn), TASK)
            if res is not None:
                recs.extend(res[1])
                print('  %-40s -> %s' % (fn, os.path.basename(res[0])))
    out = os.path.join(dc.RESULT_DIR, 'mode_CLs_cmb.json')
    with open(out, 'w') as fh:
        json.dump(recs, fh, indent=2)
    print('wrote', out)


# ---- the LaTeX rows, by the rules Table 3 already follows -------------------------------
#   NOT MEASURED -> Unc.;  one-sided -> < x;  NON-GAUSSIAN -> mode^{+}_{-};  else mean +- std
DIGITS = {'Om': 3, 'A': 2, 'z1': 3, 'w0': 3, 'wa': 3, 'H0_at_0': 2, 'H0': 2}


def fmt_cell(rec, nd):
    if rec is None:
        return '---'
    if rec.get('prior_dominated'):
        return 'Unc.'
    if rec.get('limit') == 'upper':
        return '$<%.*f$' % (nd, rec['limit_value'])
    if rec.get('limit') == 'lower':
        return '$>%.*f$' % (nd, rec['limit_value'])
    if any(n.startswith('NON-GAUSSIAN') for n in rec.get('notes', [])):
        return '$%.*f^{+%.*f}_{-%.*f}$' % (nd, rec['mode'], nd, rec['plus'], nd, rec['minus'])
    return '$%.*f \\pm %.*f$' % (nd, rec['mean'], nd, rec['std'])


def table_rows(R, recs, group, z1_prior='uniform'):
    """[(sample, anchor, model, [cells])] -- the Table 3 cells, in its order."""
    by = {(r['chain'], r['param']): r for r in recs if r['group'] == group}
    rows = []
    for s in ('des_only', 'full', 'full_dleg'):
        for a in ('none', 'lfree3', 'breuval'):
            ref = R.get((s, a, 'lcdm'))
            for m in ('exp', 'cpl'):
                r = R.get((s, a, m))
                if r is None or ref is None:
                    continue
                cell = '%s__%s__%s' % (s, a, m)
                if m == 'exp' and z1_prior != 'uniform':
                    cell += '__%sz1' % z1_prior
                p2, p3 = ('A', 'z1') if m == 'exp' else ('w0', 'wa')
                h0 = 'H0_at_0' if (m == 'exp' or group == TASK) else 'H0'
                lnB = r['logz'] - ref['logz']
                err = float(np.hypot(r['logz_err'], ref['logz_err']))
                dleg = ('$%.3f \\pm %.3f$' % (r['Delta_mean'], r['Delta_std'])
                        if 'Delta_mean' in r else '---')
                rows.append((s, a, m, ['$%+.2f \\pm %.2f$' % (lnB, err)]
                             + [fmt_cell(by.get((cell, p)), DIGITS[p])
                                for p in ('Om', p2, p3, h0)] + [dleg]))
    return rows


def task_tex(baseline=False):
    if baseline:
        R = de.load_grid()
        recs = json.load(open(os.path.join(dc.RESULT_DIR, 'mode_CLs.json')))
        group, out = 'evidence', 'table_evidence_baseline_check.tex'
    else:
        R = load_grid()
        recs = json.load(open(os.path.join(dc.RESULT_DIR, 'mode_CLs_cmb.json')))
        group, out = TASK, 'table_evidence_cmb.tex'
    lab_s = {'des_only': 'DES', 'full': 'DES + legacy', 'full_dleg': 'DES + legacy ($\\Delta$)'}
    lab_a = {'none': 'None', 'lfree3': 'Excl. SNIa', 'breuval': 'SH0ES'}
    L = []
    for s, a, m, cells in table_rows(R, recs, group):
        L.append('%s & %s & %s & %s \\\\' % (lab_s[s], lab_a[a],
                                             '\\texttt{exp}' if m == 'exp' else 'CPL',
                                             ' & '.join(cells)))
    p = os.path.join(dc.RESULT_DIR, out)
    with open(p, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('wrote', p)
    if not baseline:
        task_tex_logz1(recs)


def task_tex_logz1(recs):
    """Table 4 rows (the z1-prior comparison): lnB10, A and H(0) as mean +- std, z1 by
    the Table 3 rules.  Reads the exp cells under both z1 priors directly by file name, so
    it does not depend on Z1_PRIOR being set."""
    by = {(r['chain'], r['param']): r for r in recs}
    R = load_grid()
    L = []
    for s in ('des_only', 'full', 'full_dleg'):
        for a in ('none', 'lfree3', 'breuval'):
            ref = R[(s, a, 'lcdm')]
            for tag, suf in (('$\\mathcal{U}(z_1)$', ''), ('$\\mathcal{U}(\\ln z_1)$', '__logz1')):
                cell = '%s__%s__exp%s' % (s, a, suf)
                r = json.loads(str(np.load(dc.cache_path(TASK, cell))['summary_json']))
                L.append('%s & %s & %s & $%+.2f$ & $%+.2f \\pm %.2f$ & $%.2f \\pm %.2f$ & %s \\\\'
                         % (s, a, tag, r['logz'] - ref['logz'], r['A_mean'], r['A_std'],
                            r['H0_at_0_mean'], r['H0_at_0_std'],
                            fmt_cell(by.get((cell, 'z1')), 3)))
    p = os.path.join(dc.RESULT_DIR, 'table_logz1_cmb.tex')
    with open(p, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('wrote', p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('USAGE')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=('grid', 'table', 'check', 'cls', 'tex'),
                    default='check')
    ap.add_argument('--baseline', action='store_true',
                    help='with --task tex: render the CURRENT Table 3 from chains/evidence '
                         '(a check of the formatting rules; needs the CMB-free evidence chains)')
    ap.add_argument('--only', nargs='*', help='cells as sample/anchor/model')
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(argv)
    print(dc.banner())
    print('[evidence_cmb] CMB = SPA compressed (100 theta_*, omega_b, omega_bc); BBN %s; '
          'r_d derived; nlive=%d dlogz=%g  %s(%g,%g)  proposal %s'
          % ('on' if USE_BBN else 'OFF', de.NLIVE, de.DLOGZ,
             'ln z1~U' if de.Z1_PRIOR == 'log' else 'z1~U', de.Z1_MIN, de.Z1_MAX,
             de.sampler_kwargs() or 'auto'))
    if a.task == 'grid':
        task_grid(only=a.only, recompute=a.recompute, quiet=a.quiet)
    elif a.task == 'table':
        task_table()
    elif a.task == 'cls':
        task_cls()
    elif a.task == 'tex':
        task_tex(baseline=a.baseline)
    else:
        task_check()


if __name__ == '__main__':
    main()
