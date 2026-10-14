"""
dovekie_cosmography.py -- the inverse distance ladder of the DES-Dovekie / DESI DR2 paper.

ONE FILE.  It imports numpy, scipy, jax and numpyro, and NOTHING from this repository.
That is deliberate: the analysis this replaces was a fifteen-module import web driven by
environment variables whose defaults did not match the published baseline, and three
separate campaigns were silently wasted by that (a slurm `export P=3` overriding
`--export=ALL`; `INIT=lcdm` reaching the banner and the manifest but not the sampler; a
covariance handed to numpyro as if it were a Cholesky factor).  Everything that decides
what a number MEANS is in the BASELINE block below, in one place, and every result carries
a manifest of the settings that produced it.

WHAT IT COMPUTES  (--task, and where each lands in the manuscript)

  ladder      Table 1.  H0, r_d, M_des, q0 and chi2/dof for
              {config A, config B} x {DES, DES+legacy} x {cosmography P=4, flat LCDM}.
  floorsweep  Figure 2.  The same ladder with a rising redshift floor: only the DATA
              change, the estimator does not.
  bic         The order-selection protocol.  BIC(P) = chi2_min(P) + k ln N on the
              UNCONSTRAINED chi2 minimum, so the order decision carries no prior.
  legacy      Table 3 and Figure 3.  The four-rung Delta ladder (prior-free least
              squares) and the legacy zero-point sampled jointly with the cosmology.
  ap          Table 4 and Figure 4.  The Alcock-Paczynski consistency test: predict the
              DESI F_AP = D_M/D_H from the SN shape alone, with no r_d and no calibration.

  The Bayesian model comparison of Table 2 lives in `dovekie_evidence.py`, which imports
  this module for its data and likelihood blocks so there is one definition of each.

THE MODEL

    H(z) = H0 E(y) ,   E(y) = 1 + sum_{k=1}^{P} C_k y^k ,   y = z/(1+z) = 1 - a ,
    q0 = C_1 - 1

  H0 carries the normalisation and the DIMENSIONLESS C_k carry the shape.  (Before
  2026-09-28 the coefficients were dimensional, c_k = H0 C_k with c_0 = H0; the least-
  squares code below still works in that chart internally, which is harmless because a
  chi2 minimum does not depend on the chart.)  The prior bounds E itself -- see
  section 7.

  y < 1 for all z, so the series converges over the whole sample; a series in z itself has
  radius |z| < 1 and would diverge before the DESI Lyman-alpha point at z = 2.330.  This
  single expansion enters all three likelihoods -- the chronometers through H(z_j), the
  supernovae through D_M = c int dz/H, the BAO through D_M, D_H and D_V -- so the blocks
  are coupled through one background and cannot drift apart.  Flat LCDM replaces it as a
  fixed-shape cross-check; everything else is held identical.

USAGE

    python dovekie_cosmography.py --task ladder
    python dovekie_cosmography.py --task bic --recompute
    python dovekie_cosmography.py --list-cache

  Every task caches its posterior draws under chains/<task>/ and skips cells that are
  already there and whose manifest agrees with the current BASELINE.  A cache written
  under different settings is REFUSED, not silently reused.

Python 3.10+.  See README.md for the settings and the command-to-table map.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 0.  BASELINE -- every setting that decides what a number means, in one place
# =============================================================================
# Values here are the published configuration.  They may be overridden from the
# environment (the cluster needs that) but ONLY through this dict, and every override is
# printed on startup and recorded in the manifest of every result it touches.  There is no
# other env lookup anywhere in this file, so a setting cannot be changed without appearing
# in the provenance.

BASELINE = dict(
    # ---- expansion ---------------------------------------------------------
    P=4,                    # polynomial order.  Selected by BIC; see --task bic.
    NGRID=800,              # points in the z-grid carrying the trapezoid D_C integral.
                            # 400 doubled the low-z integration error once the grid was
                            # stretched to z=2.33 by the Lyman-alpha BAO point.

    # ---- data --------------------------------------------------------------
    BINNED=0,               # 0 = the full catalogue with its dense covariance (baseline).
    CLEAN_MUERR_MAX=1.0,    # drop BEAMS-deweighted contaminants; 1738 / 1541 SNe survive.
    DESI_ZMAX=float('inf'),  # no cut => the complete DR2 vector, 13 observables.
    USE_DV=1,               # include the BGS D_V(0.295) point.
    CC_SET='mm20',          # the 15 Moresco BC03 points that have a PUBLISHED covariance.

    # ---- priors ------------------------------------------------------------
    # H0 ~ U(H0_LO, H0_HI).  The shape E(y) = H/H0 is bounded as a FUNCTION: the prior
    # is uniform in the C_k over every polynomial with E_LOWER(y) <= E(y) <= E_UPPER(y)
    # on [0, Y_MAX] (section 7).  The envelopes are monomial coefficients in y:
    # 0 <= E <= 2 + 7y, i.e. twice the chord 1 + 3.5y of flat LCDM.
    # Y_MAX = 0.70 is the DESI Lyman-alpha BAO at z = 2.33, FIXED for the whole project:
    # it sets where the envelope is enforced, so every run whose evidences are compared
    # must share it.  (This replaced, on 2026-09-28, a symmetric box on the dimensional
    # coefficients pushed forward from q0/j0/s0/l0 in [-10,+9].)
    H0_LO=50.0, H0_HI=90.0,
    Y_MAX=0.70,
    # The AP test (--task ap) fits the supernova SHAPE alone, which stops at the highest
    # SN redshift, z = 1.144 (y = 0.5336), and predicts F_AP only at z <= 0.934.  It is a
    # separate question from the ladder, so its envelope is enforced up to the SN reach
    # instead.  Rounded UP, so it covers every supernova.
    AP_Y_MAX=0.534,
    E_LOWER=(0.0,),
    E_UPPER=(2.0, 7.0),
    M_LO=-21.0, M_HI=-18.0,
    RD_PLANCK=147.05, SIG_RD=0.30,          # configuration A: an imported measurement
    RD_FREE_LO=120.0, RD_FREE_HI=175.0,     # configuration B: r_d derived, no CMB
    OM_LO=0.05, OM_HI=0.70,
    DELTA_HALFWIDTH=1.0,    # legacy zero-point box, ~25 sigma wide: genuinely inert.

    # ---- sampler -----------------------------------------------------------
    N_WARM=1500, N_SAMP=2000, NUM_CHAINS=4, SEED=0,
    # Two tree depths, because the fits genuinely differ and collapsing them into one
    # would be a change of method, not a tidy-up.  The joint ladder carries the full
    # DESI vector and the chronometers and needs the deeper tree; the shape-only fits
    # (the AP test) carry the supernovae alone, converge at 10, and cost 4x more per
    # iteration at 12 for no gain.  A caller must name which it is -- `make_nuts` will
    # not guess.
    MAX_TREE_DEPTH_JOINT=12, MAX_TREE_DEPTH_SHAPE=10,
    DENSE_MASS=1,           # the induced c_k correlations reach +0.98; a diagonal mass
                            # matrix cannot precondition a ridge at 45 degrees to every
                            # axis, and that was the 2026-08 freeze.
    INIT='lcdm',            # start at flat LCDM, not at the posterior median.  The ML
                            # point is NOT usable: it slides along the flat directions to
                            # l0 ~ 1.9e4, far outside the stated prior support.
    INIT_OM=0.315, INIT_H0=68.0,
    TARGET_ACCEPT=0.8,      # numpyro's own default, and what produced every published
                            # chain.  It is a BASELINE entry and a cache key so that a
                            # cell rescued at 0.95 cannot be mistaken for one that never
                            # needed rescuing -- see the quality gate below.

    # ---- publication-quality thresholds ------------------------------------
    # A chain that fails these is still SAMPLED and still CACHED -- cluster hours are not
    # thrown away and the failure stays available for diagnosis -- but make_tables and
    # make_figures refuse to render it.  Sampling and publishing are separate decisions.
    # R-hat up to 1.02 passes, and divergences are NOT a failure on their own provided
    # R-hat and n_eff pass.
    # GATE_MAX_DIVERGENCES = -1 means reported, never gated; a value >= 0 re-enables it.
    GATE_MAX_DIVERGENCES=-1, GATE_MAX_RHAT=1.02, GATE_MIN_NEFF=400.0,

    # ---- constants ---------------------------------------------------------
    C_KM_S=299792.458,
    M_B_FID=-19.349,        # the fiducial absolute magnitude Dovekie's MU column assumes
)

# Only these may be overridden from the environment.  Anything else is a typo, and a typo
# that silently does nothing is the failure mode this whitelist exists to prevent.
_ENV_OVERRIDABLE = {
    'P', 'NGRID', 'BINNED', 'DESI_ZMAX', 'USE_DV', 'CC_SET',
    'N_WARM', 'N_SAMP', 'NUM_CHAINS', 'SEED', 'MAX_TREE_DEPTH_JOINT',
    'MAX_TREE_DEPTH_SHAPE', 'DENSE_MASS', 'TARGET_ACCEPT',
    'INIT', 'INIT_OM', 'INIT_H0',
    'GATE_MAX_DIVERGENCES', 'GATE_MAX_RHAT', 'GATE_MIN_NEFF',
}

OVERRIDES = {}
for _k in sorted(_ENV_OVERRIDABLE):
    if _k in os.environ:
        _raw = os.environ[_k]
        _cur = BASELINE[_k]
        try:
            BASELINE[_k] = type(_cur)(_raw) if not isinstance(_cur, str) else _raw
        except ValueError:
            raise ValueError('%s=%r cannot be read as %s' % (_k, _raw, type(_cur).__name__))
        # Record only settings that actually MOVED.  The cluster preamble exports several
        # of these at their baseline values, and listing those as "overrides" would fill
        # the provenance line with no-ops and train the reader to skip it.
        if BASELINE[_k] != _cur:
            OVERRIDES[_k] = (_cur, BASELINE[_k])

if BASELINE['INIT'] not in ('lcdm', 'median'):
    raise ValueError("INIT must be 'lcdm' (baseline) or 'median'; got %r.  Guessing an "
                     "init strategy is how the sampler settings drifted before."
                     % BASELINE['INIT'])
if BASELINE['CC_SET'] != 'mm20':
    raise ValueError("CC_SET='mm20' is the only set with a published covariance.  The "
                     "33-point compilation carries stat+met errors only -- its model "
                     "systematic is ABSENT, not merely uncorrelated -- so a diagonal CC "
                     "likelihood understates sigma(r_d) by a factor 2.7.  Got %r."
                     % BASELINE['CC_SET'])

# JAX wants these before it is imported.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS',
                      '--xla_force_host_platform_device_count=%d' % BASELINE['NUM_CHAINS'])


def banner():
    """One greppable line of provenance.  Printed on every run, including on the cluster,
    because the recurring failure mode in this analysis was a knob that silently defaulted
    to the wrong thing and was recoverable only from a slurm .out."""
    s = ('[dovekie] P=%d PRIOR=envelope(y_max=%g, E in [%s]..[%s]) H0_BOX=[%g,%g] '
         'init=%s(Om=%.3f) '
         'binned=%d dense_mass=%d tree=%d/%d CC=%s ngrid=%d %d/%d x%d seed=%d'
         % (BASELINE['P'], BASELINE['Y_MAX'],
            ','.join('%g' % v for v in BASELINE['E_LOWER']),
            ','.join('%g' % v for v in BASELINE['E_UPPER']),
            BASELINE['H0_LO'], BASELINE['H0_HI'], BASELINE['INIT'], BASELINE['INIT_OM'],
            BASELINE['BINNED'], BASELINE['DENSE_MASS'],
            BASELINE['MAX_TREE_DEPTH_JOINT'], BASELINE['MAX_TREE_DEPTH_SHAPE'],
            BASELINE['CC_SET'], BASELINE['NGRID'], BASELINE['N_WARM'], BASELINE['N_SAMP'],
            BASELINE['NUM_CHAINS'], BASELINE['SEED']))
    if OVERRIDES:
        s += '\n[dovekie] ENV OVERRIDES: ' + ' '.join(
            '%s %r->%r' % (k, a, b) for k, (a, b) in sorted(OVERRIDES.items()))
    if CHAIN_DIR != os.path.join(HERE, 'chains'):
        s += '\n[dovekie] CHAINS_DIR=%s' % CHAIN_DIR
    if RESULT_DIR != os.path.join(HERE, 'results'):
        s += '\n[dovekie] RESULTS_DIR=%s' % RESULT_DIR
    return s


# =============================================================================
# 1.  Provenance
# =============================================================================
DATA_DIR = os.path.join(HERE, 'data')
# CHAINS_DIR / RESULTS_DIR may be redirected -- to cluster scratch, or to a throwaway
# directory for a reduced-settings smoke run so that a short chain cannot land next to the
# published ones.  The cache signature would refuse to reuse it anyway, but keeping the
# directories apart means a wrong run never has to be cleaned up by hand.
CHAIN_DIR = os.environ.get('CHAINS_DIR') or os.path.join(HERE, 'chains')
RESULT_DIR = os.environ.get('RESULTS_DIR') or os.path.join(HERE, 'results')

# SHA256 of every vendored data file, so a silently swapped or truncated input is caught
# at read time rather than showing up as a moved number three tables later.
DATA_SHA256 = {
    'DES-Dovekie_HD.csv':
        '2f57019d783eaa976df80a41b0054171a2d994ee9808d715ce850c2df5720aaf',
    'STAT+SYS.npz':
        'ffd3124b32148b1372bd95fda9299269f0352a9f8eee02d416c610e38495463b',
    'CC_mm20/data_MM20.dat':
        '577ac2f346e346fe7cf94daa7b7000c05d04ebc8a029cda31e0d8643b956a485',
    'CC_mm20/HzTable_MM_BC03.dat':
        '32ce92caf251cb60a7a837c71f1856bea2b44fa5c1041f85410d11cb8164da98',
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def data_path(rel, verify=True):
    p = os.path.join(DATA_DIR, rel)
    if not os.path.exists(p):
        raise FileNotFoundError(
            '%s is missing.  Run data/fetch_data.sh to download it from the DES-SN5YR '
            'release (Zenodo DOI 10.5281/zenodo.12720778); see data/PROVENANCE.md.' % p)
    if verify and rel in DATA_SHA256:
        got = _sha256(p)
        if got != DATA_SHA256[rel]:
            raise ValueError(
                '%s has SHA256 %s, expected %s.  This is not the file the published '
                'numbers were computed from.' % (p, got, DATA_SHA256[rel]))
    return p


def git_sha():
    try:
        return subprocess.check_output(['git', '-C', HERE, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


# Bump this whenever a change to the MODEL or the LIKELIHOOD would make an existing cache
# wrong while leaving every entry of _CACHE_KEYS unchanged.  That is not hypothetical: the
# schema went 1 -> 2 for the z_HEL/z_HD repair and the removal of the duplicated Planck
# r_d factor, neither of which touches a single BASELINE value, so nothing else in the
# manifest could have refused the chains they invalidated.
#
#   1  the state audited on 2026-08-18 (D_L used z_HD twice; evidence imposed Planck twice)
#   2  z_HEL in the luminosity prefactor; exactly one Planck Gaussian in the evidence
CACHE_SCHEMA = 2

# The source files whose CONTENT defines a result.  Hashed into every manifest, because
# `git_sha` records the commit and not the working tree: an uncommitted edit to the
# likelihood is invisible to it.  Recorded for provenance, not compared -- comparing it
# would refuse every cache after any whitespace change.  CACHE_SCHEMA is the thing that
# refuses.
_SOURCE_FILES = ('dovekie_cosmography.py', 'dovekie_evidence.py')


def source_sha256():
    h = hashlib.sha256()
    for rel in _SOURCE_FILES:
        f = os.path.join(HERE, rel)
        if not os.path.exists(f):
            continue
        h.update(rel.encode())
        with open(f, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
    return h.hexdigest()


def manifest(extra=None, task=None):
    """Everything needed to say what produced a result.  Goes into every cached cell.

    `extra` carries settings that live OUTSIDE BASELINE and still change the answer -- the
    nested-sampling controls of dovekie_evidence, for instance.  They are compared on load
    exactly like the BASELINE signature, which is what stops a dlogz=1 scouting run from
    being served back as the dlogz=0.01 result."""
    vers = {}
    for mod in ('numpy', 'scipy', 'jax', 'numpyro', 'dynesty'):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = 'absent'
    return {
        'cache_schema': CACHE_SCHEMA,
        'cosmo_prior': cosmo_prior_signature(task),
        'baseline': {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in sorted(BASELINE.items())},
        'env_overrides': {k: [a, b] for k, (a, b) in sorted(OVERRIDES.items())},
        'git_sha': git_sha(),
        'source_sha256': source_sha256(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': vers,
        'extra': dict(extra or {}),
        'written': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


# The prior on the cosmographic SHAPE, as one comparable record.  It is deliberately NOT
# in _CACHE_KEYS: adding it there would refuse every flat-LCDM cell and every cell of
# dovekie_evidence (Table 2), none of which samples a coefficient.  Instead cache_load
# demands it of exactly the cells that hold coefficient draws -- see _COEFF_ARRAYS.
COSMO_PRIOR_NAME = 'envelope-v1'


def shape_y_max(task=None):
    """Where the envelope is enforced: Y_MAX, except for the AP test's SN-only fit."""
    return float(BASELINE['AP_Y_MAX'] if task == 'ap' else BASELINE['Y_MAX'])


def cosmo_prior_signature(task=None):
    B = BASELINE
    return {'name': COSMO_PRIOR_NAME, 'y_max': shape_y_max(task),
            'e_lower': [float(v) for v in B['E_LOWER']],
            'e_upper': [float(v) for v in B['E_UPPER']],
            'h0': [float(B['H0_LO']), float(B['H0_HI'])],
            'env_ngrid': ENV_NGRID, 'env_penalty': ENV_PENALTY}


# Array names under which a cell stores draws of the cosmographic coefficients: `v` and
# `C` under the envelope prior, `b` under its same-day Bernstein predecessor, `c_hi`
# (ladder) and `c` (AP) before 2026-09-28.  A cached cell holding any of them was sampled
# under SOME prior on the shape, and it is refused unless its manifest records the CURRENT
# one.  Cells written before 2026-09-28 carry no `cosmo_prior` at all, so every one of
# them is refused -- by the tasks and by every table and figure builder, because they all
# read through cache_load.
_COEFF_ARRAYS = ('v', 'b', 'C', 'c_hi', 'c')


# Settings a cached cell must agree on before it may be reused.  Deliberately narrower
# than the whole BASELINE: the package versions and the git SHA are recorded for
# provenance but must not invalidate a cache, or nothing would ever be reusable.
_CACHE_KEYS = ('P', 'NGRID', 'BINNED', 'CLEAN_MUERR_MAX', 'DESI_ZMAX', 'USE_DV', 'CC_SET',
               'H0_LO', 'H0_HI', 'M_LO', 'M_HI', 'RD_PLANCK', 'SIG_RD',
               'RD_FREE_LO', 'RD_FREE_HI', 'OM_LO', 'OM_HI', 'DELTA_HALFWIDTH',
               'N_WARM', 'N_SAMP', 'NUM_CHAINS', 'SEED',
               'MAX_TREE_DEPTH_JOINT', 'MAX_TREE_DEPTH_SHAPE', 'DENSE_MASS',
               'TARGET_ACCEPT',
               'INIT', 'INIT_OM', 'INIT_H0', 'C_KM_S', 'M_B_FID')


# Settings that describe the SAMPLER rather than the model or the data.  Tasks that use no
# sampler -- the BIC minimisation, the least-squares Delta ladder -- declare these
# irrelevant, so a change of tree depth does not force a pointless re-run of a result that
# could not possibly depend on it.
_SAMPLER_KEYS = ('N_WARM', 'N_SAMP', 'NUM_CHAINS', 'MAX_TREE_DEPTH_JOINT',
                 'MAX_TREE_DEPTH_SHAPE', 'DENSE_MASS', 'TARGET_ACCEPT',
                 'INIT', 'INIT_OM', 'INIT_H0')

# Settings for which a cache may legitimately disagree in ONE direction.  TARGET_ACCEPT is
# the only one: raising it shrinks the leapfrog step and reduces discretisation error
# WITHOUT changing the distribution being sampled, so a cell rescued at 0.95 is a valid --
# indeed a better -- answer to the question asked at 0.8, while the reverse is not true.
# Comparing it with == instead would make the remediation ladder unusable: rescuing the two
# configuration-B LCDM cells would produce caches that every default run refused, and the
# only way out would be to re-run the six healthy cells at the rescued setting too.
# MAX_TREE_DEPTH_* is monotone for the same reason and in the same direction: it caps how
# far one NUTS trajectory may travel, and raising the cap lets the sampler do MORE, never
# something different.  It is the right lever when the adapted step size has collapsed --
# a smaller step with the same depth cap means a SHORTER trajectory, so raising the target
# acceptance alone can make mixing worse rather than better, which is exactly what the AP
# cell did on 2026-08-18 (eps 0.0162 -> 0.0088, n_eff 912 -> 488).
# N_WARM joins them: more warmup gives the dense mass matrix more to adapt on and cannot
# change the target.  N_SAMP deliberately does NOT, even though a longer chain is also
# strictly better, because the gate counts divergences as an ABSOLUTE number -- doubling
# N_SAMP doubles the expected count at a fixed divergence RATE, so treating it as monotone
# would let a cell fail the gate merely for having been run longer.
_MONOTONE_KEYS = {'TARGET_ACCEPT', 'MAX_TREE_DEPTH_JOINT', 'MAX_TREE_DEPTH_SHAPE',
                  'N_WARM'}

_MISSING = object()


def _cache_signature(base=None):
    b = BASELINE if base is None else base
    return {k: (list(b[k]) if isinstance(b[k], (tuple, list)) else b[k])
            for k in _CACHE_KEYS if k in b}


def cache_path(task, cell):
    d = os.path.join(CHAIN_DIR, task)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, '%s.npz' % cell)


def _norm(v):
    """JSON round-trips tuples to lists and ints to ints; compare on that footing."""
    if isinstance(v, (tuple, list)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in sorted(v.items())}
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return v
    return v


def cache_load(task, cell, ignore_p=False, ignore=(), extra=None):
    """The cached cell, or None if absent or written under different settings.

    A cache whose signature disagrees is REFUSED rather than used, and a manifest that is
    missing a key the current code has counts as a disagreement rather than an error --
    it means the cache predates a setting, which is exactly when it must not be trusted.
    Silently reading a stale cache is how a P=3 figure was published under a P=6 caption.

    Three things are checked, in order of how badly they bite:

      CACHE_SCHEMA   the model/likelihood generation.  A change here means the cached
                     numbers answer a different question, and no BASELINE value moved.
      _CACHE_KEYS    the settings that describe the model and the data.
      extra          per-task settings that do not live in BASELINE at all -- the
                     nested-sampling controls, the evidence priors, the anchor windows.

    A corrupt or truncated file is refused, not raised on: an interrupted write during a
    cluster job must cost that cell alone, not the whole campaign."""
    p = cache_path(task, cell)
    if not os.path.exists(p):
        return None
    try:
        d = np.load(p, allow_pickle=False)
        man = json.loads(str(d['manifest_json']))
    except Exception as exc:
        print('[cache] %s is unreadable (%s: %s) -- refusing it'
              % (os.path.basename(p), type(exc).__name__, exc))
        return None
    got_schema = man.get('cache_schema', 1)
    if got_schema != CACHE_SCHEMA:
        print('[cache] %s was written under cache_schema=%s, this code is %d -- refusing it'
              % (os.path.basename(p), got_schema, CACHE_SCHEMA))
        return None
    if any(k in d.files for k in _COEFF_ARRAYS):
        got_prior = man.get('cosmo_prior')
        if _norm(got_prior) != _norm(cosmo_prior_signature(task)):
            print('[cache] %s holds coefficient draws sampled under a different shape '
                  'prior (%s) -- refusing it'
                  % (os.path.basename(p),
                     'none recorded: the pre-2026-09-28 coefficient box'
                     if got_prior is None else
                     '%s, y_max=%s; this code wants y_max=%s'
                     % (got_prior.get('name'), got_prior.get('y_max'), shape_y_max(task))))
            return None
    skip = set(ignore) | ({'P'} if ignore_p else set())
    want = _cache_signature()
    got = _cache_signature(man.get('baseline', {}))
    bad, rescued = [], []
    for k in want:
        if k in skip:
            continue
        g = got.get(k, _MISSING)
        if g == want[k]:
            continue
        if k in _MONOTONE_KEYS and g is not _MISSING and g > want[k]:
            rescued.append('%s=%r > %r' % (k, g, want[k]))
            continue
        bad.append(k)
    if rescued:
        print('[cache] %s was sampled at a STRICTER setting than requested (%s) -- '
              'accepting it' % (os.path.basename(p), ', '.join(rescued)))
    if bad:
        print('[cache] %s was written under different settings (%s) -- refusing it'
              % (os.path.basename(p),
                 ', '.join('%s=%s vs %r'
                           % (k, 'ABSENT' if k not in got else repr(got[k]), want[k])
                           for k in bad[:4])))
        return None
    if extra:
        gx = man.get('extra', {})
        badx = [k for k, v in sorted(extra.items())
                if k not in skip and _norm(gx.get(k, _MISSING)) != _norm(v)]
        if badx:
            print('[cache] %s was written under different task settings (%s) -- refusing it'
                  % (os.path.basename(p),
                     ', '.join('%s=%s vs %r'
                               % (k, 'ABSENT' if k not in gx else repr(gx[k]), extra[k])
                               for k in badx[:4])))
            return None
    return {k: d[k] for k in d.files}


def cache_save(task, cell, extra=None, **arrays):
    """Write a cell ATOMICALLY.

    np.savez_compressed straight onto the destination leaves a truncated file behind if
    the job is killed mid-write, and a truncated .npz is indistinguishable from a missing
    one only if something checks -- so write beside it and rename, which is atomic within
    a directory on POSIX."""
    p = cache_path(task, cell)
    tmp = p + '.tmp-%d' % os.getpid()
    try:
        np.savez_compressed(tmp, manifest_json=json.dumps(manifest(extra, task)),
                            **arrays)
        # numpy appends .npz when the name does not already end in it
        written = tmp if os.path.exists(tmp) else tmp + '.npz'
        os.replace(written, p)
    finally:
        for junk in (tmp, tmp + '.npz'):
            if os.path.exists(junk):
                os.remove(junk)
    return p


# =============================================================================
# 2.  Supernovae -- DES-Dovekie
# =============================================================================
# The Hubble diagram is SNANA-format despite the .csv name: comment lines, a `VARNAMES:`
# header, then rows prefixed `SN:`.  Row order matches the covariance and must be kept.
#
# IDSURVEY == 10 is DES; everything else is the low-z legacy anchor -- {61..66} the
# CfA/CSP samples and 150 Foundation.
DES_IDSURVEY = 10
FOUNDATION_IDSURVEY = 150
# The heterogeneous low-redshift remainder.  Taken from the IDSURVEY values that are
# actually PRESENT in the catalogue -- {5, 63, 64, 65, 66}, with counts 8/16/31/22/3 = 80 --
# not from the full survey dictionary in the upstream README, which also lists 61 and 62
# for samples this compilation does not contain.  Writing out the README's range instead
# would silently drop the 8 CSP supernovae of IDSURVEY 5 and give an 72-object subsample
# labelled as if it were the whole 80; test_repo.py pins the counts.
CFA_CSP_IDSURVEY = (5, 63, 64, 65, 66)


def read_hubble_diagram():
    """(cid, idsurvey, zHD, zHEL, mu, muerr) in covariance order."""
    names, rows = None, []
    with open(data_path('DES-Dovekie_HD.csv')) as fh:
        for line in fh:
            tok = line.strip().split()
            if not tok or tok[0].startswith('#'):
                continue
            if tok[0] == 'VARNAMES:':
                names = tok[1:]
                continue
            rows.append(tok[1:] if tok[0] == 'SN:' else tok)
    if names is None:
        raise ValueError('no VARNAMES: header in the Hubble diagram')
    arr = np.array(rows)
    col = {n: i for i, n in enumerate(names)}
    return (arr[:, col['CID']],
            arr[:, col['IDSURVEY']].astype(int),
            arr[:, col['zHD']].astype(float),
            arr[:, col['zHEL']].astype(float),
            arr[:, col['MU']].astype(float),
            arr[:, col['MUERR']].astype(float))


def read_covariance():
    """The forward STAT+SYS covariance.

    The release ships the packed upper triangle of the INVERSE covariance (its own README
    says so); this mirrors DES-Dovekie-SN_Likelihood.py -- unpack, symmetrise, invert."""
    d = np.load(data_path('STAT+SYS.npz'))
    n = int(d['nsn'][0])
    inv = np.zeros((n, n))
    inv[np.triu_indices(n)] = d['cov']
    lo = np.tril_indices(n, -1)
    inv[lo] = inv.T[lo]
    cov = np.linalg.inv(inv)
    return 0.5 * (cov + cov.T)


_SN_CACHE = {}


def sn_dataset(name):
    """The cleaned SN sample.  name in {'full', 'des_only'}.

    'full'     = DES + legacy, 1738 SNe, z in [0.025, 1.14]
    'des_only' = IDSURVEY == 10 alone, 1541 SNe, z in [0.073, 1.14]  -- the baseline

    The MUERR < 1 cut removes 82 objects whose BEAMS de-weighting inflates their variance
    to ~468^2.  They carry ~1e-5 of the total statistical weight, so no central value
    moves, but keeping them leaves the covariance ill-conditioned.  Applied HERE and
    nowhere else, so every task sees the same catalogue."""
    if name in _SN_CACHE:
        return _SN_CACHE[name]
    if name not in ('full', 'des_only'):
        raise ValueError("sample must be 'full' or 'des_only', got %r" % name)
    cid, idsurvey, z, zhel, mu, muerr = read_hubble_diagram()
    cov = read_covariance()
    keep = muerr < BASELINE['CLEAN_MUERR_MAX']
    if name == 'des_only':
        keep = keep & (idsurvey == DES_IDSURVEY)
    out = dict(cid=cid[keep], idsurvey=idsurvey[keep], z=z[keep], zhel=zhel[keep],
               mu=mu[keep], muerr=muerr[keep],
               cov=0.5 * (cov[np.ix_(keep, keep)] + cov[np.ix_(keep, keep)].T))
    # apparent magnitude, so that the sampled M is a genuine absolute magnitude directly
    # comparable to the SH0ES value rather than an offset against Dovekie's fiducial.
    out['m'] = out['mu'] + BASELINE['M_B_FID']
    _SN_CACHE[name] = out
    return out


def sn_block(name, z_floor=0.0):
    """(z_HD, z_HEL, m, cov) for the SN likelihood, optionally truncated at a floor.

    BOTH redshifts, deliberately.  An earlier version returned z_HD alone and every SN
    path then used it twice -- see `sn_apparent_mag`.  The floor cut is applied on z_HD,
    which is the Hubble-diagram redshift the survey selection is defined against.

    z_floor > 0 drops the SNe below it -- the sub-survey cut of the floor sweep.  It is
    the ONLY thing the floor sweep changes."""
    d = sn_dataset(name)
    keep = d['z'] >= z_floor
    return (d['z'][keep], d['zhel'][keep], d['m'][keep],
            d['cov'][np.ix_(keep, keep)])


def sn_apparent_mag(M, z_hel, dc_at_zhd, xp=np):
    """m = M + 5 log10[ (1 + z_HEL) * D_C(z_HD) ] + 25.

    THE ONE PLACE the supernova model magnitude is written down.  Every likelihood, chi2,
    least-squares residual and rung fit in this repository routes through it, in numpy or
    in jax, so the two redshifts cannot come apart in one path and not another.

    The two redshifts are NOT interchangeable and the distinction is the DES likelihood's
    own convention (DES-SN5YR/4_DISTANCES_COVMAT):

        D_L = (1 + z_HEL) * c * int_0^{z_HD} dz'/H(z')

    The integral runs over the CMB-frame Hubble-diagram redshift, because that is the one
    the expansion history is a function of.  The (1+z) prefactor is HELIOCENTRIC, because
    it is a photon-energy and time-dilation factor in the observer's own frame and knows
    nothing about the peculiar-velocity correction applied to z_HD.

    Using z_HD in both -- which this code did until 2026-08-18 -- biases the model
    magnitude by 5 log10[(1+z_HD)/(1+z_HEL)]: a mean of -0.00137 mag over DES and
    +0.00083 mag over the legacy sample.  A COMMON offset would be harmless, absorbed by
    the free M_des; the +0.0022 mag legacy-minus-DES DIFFERENTIAL is not, because it lands
    in exactly the low-redshift calibration contrast Delta measures.

    `xp` is numpy or jax.numpy.  Nothing else about the expression changes between them."""
    return M + 5.0 * xp.log10((1.0 + z_hel) * dc_at_zhd) + 25.0


def legacy_indicator(name='full'):
    """1 for the legacy supernovae, 0 for DES.  The lever the zero-point offset acts on."""
    d = sn_dataset(name)
    return (d['idsurvey'] != DES_IDSURVEY).astype(float)


# The split-offset model.  The legacy rung is NOT one survey: 117 supernovae come from
# Foundation and 80 from CfA/CSP (IDSURVEY 5, 63-66), the two partition the 197 legacy
# objects exactly, and they span the SAME redshift range (0.0251-0.0870 against
# 0.0251-0.0848).  That last fact is what makes the contrast Delta_Found - Delta_CfA
# clean: the DES sub-floor extrapolation is common to both and cancels from the
# difference, so it is not counted twice.
DELTA_GROUPS = {
    'Delta_found': lambda ids: ids == FOUNDATION_IDSURVEY,
    'Delta_cfa': lambda ids: np.isin(ids, list(CFA_CSP_IDSURVEY)),
}


def legacy_columns(name='full', split=False, z_floor=0.0):
    """(U, names) -- the offset design matrix and the site names that go with it.

    split=False gives one column, exactly `legacy_indicator`, site 'Delta'.
    split=True  gives one column per entry of DELTA_GROUPS.

    RAISES if the groups do not partition the legacy sample.  A silently uncovered
    legacy supernova would be modelled with NO offset while its neighbours carry one,
    which is a different model from the one the caller asked for and would show up only
    as an unexplained shift in Delta."""
    d = sn_dataset(name)
    keep = d['z'] >= z_floor
    ids = d['idsurvey'][keep]
    leg = ids != DES_IDSURVEY
    if not split:
        return leg.astype(float)[:, None], ['Delta']
    cols, names = [], []
    covered = np.zeros(len(ids), bool)
    for nm, fn in DELTA_GROUPS.items():
        sel = np.asarray(fn(ids), bool)
        if (sel & ~leg).any():
            raise ValueError('%s selects non-legacy supernovae' % nm)
        if (sel & covered).any():
            raise ValueError('%s overlaps an earlier group' % nm)
        covered |= sel
        cols.append(sel.astype(float))
        names.append(nm)
    if (leg & ~covered).any():
        import collections
        raise ValueError(
            'the split groups do not cover the legacy sample: %d supernova(e) left over, '
            'IDSURVEY %r.  Add them to DELTA_GROUPS or the split model quietly leaves '
            'them un-offset.'
            % (int((leg & ~covered).sum()),
               dict(collections.Counter(ids[leg & ~covered].tolist()))))
    return np.stack(cols, axis=1), names


# =============================================================================
# 3.  Baryon acoustic oscillations -- DESI DR2
# =============================================================================
# Means and covariance taken verbatim from the official DR2 likelihood files
# (CobayaSampler/bao_data, desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt),
# arXiv:2503.14738.  Different effective redshifts are independent tracers, so the
# covariance is block diagonal across z; within a bin D_M/r_d and D_H/r_d are correlated
# at rho = -0.35..-0.49 and that 2x2 is kept.
DESI_DM = {                    # z : (D_M/r_d, sigma)
    0.510: (13.58758, 0.16837),
    0.706: (17.35069, 0.17993),
    0.934: (21.57564, 0.16178),
    1.321: (27.60086, 0.32456),
    1.484: (30.51190, 0.76356),
    2.330: (38.98897, 0.53168),    # Lyman-alpha
}
DESI_DH = {                    # z : (D_H/r_d, sigma),  D_H = c/H(z)
    0.510: (21.86295, 0.42887),
    0.706: (19.45535, 0.33387),
    0.934: (17.64149, 0.20104),
    1.321: (14.17602, 0.22455),
    1.484: (12.81700, 0.51801),
    2.330: (8.63155, 0.10106),
}
# Note the DR2 file lists D_H before D_M at z = 2.330; the pair is reordered here so that
# every block is (D_M, D_H).
DESI_DMDH_COV = {              # z : [[var_DM, cov], [cov, var_DH]]
    0.510: [[2.83473742e-02, -3.26062007e-02], [-3.26062007e-02, 1.83928040e-01]],
    0.706: [[3.23752442e-02, -2.37445646e-02], [-2.37445646e-02, 1.11469198e-01]],
    0.934: [[2.61732816e-02, -1.12938006e-02], [-1.12938006e-02, 4.04183878e-02]],
    1.321: [[1.05336516e-01, -2.90308418e-02], [-2.90308418e-02, 5.04233092e-02]],
    1.484: [[5.83020277e-01, -1.95215562e-01], [-1.95215562e-01, 2.68336193e-01]],
    2.330: [[2.82685779e-01, -2.31395216e-02], [-2.31395216e-02, 1.02136194e-02]],
}
DESI_DV = {                    # z : (D_V/r_d, sigma) -- BGS, only D_V available
    0.295: (7.94168, 0.07609),
}

# The three DESI redshifts that lie inside the DES supernova range.  The AP test uses
# these and only these: beyond z_max the SN prediction is an extrapolation whose D_H error
# exceeds D_H itself, so F_AP diverges and a chi2 of ~0 means "uninformative", not
# "consistent".
AP_REDSHIFTS = (0.510, 0.706, 0.934)


def bao_vector(use_dv=None, zmax=None):
    """(O, Cov, zs, kinds) -- the stacked DESI observable vector and its block covariance.

    Order: D_V(0.295) if requested, then per redshift ascending the pair [D_M/r_d, D_H/r_d]
    with its DR2 2x2.  Thirteen observables at the baseline."""
    use_dv = BASELINE['USE_DV'] if use_dv is None else use_dv
    zcut = BASELINE['DESI_ZMAX'] if zmax is None else float(zmax)
    O, kinds, zs, blocks = [], [], [], []
    if use_dv:
        # the zcut applies HERE too.  It did not, so DESI_ZMAX < 0.295 still kept
        # D_V(0.295) and a 'BAO below z' cut silently was not one.  Dormant at the
        # baseline, where DESI_ZMAX is infinite.
        for z in [z for z in sorted(DESI_DV) if z <= zcut]:
            val, sig = DESI_DV[z]
            O.append(val); kinds.append('DV'); zs.append(z)
            blocks.append(np.array([[sig ** 2]]))
    for z in [z for z in sorted(DESI_DM) if z <= zcut]:
        O.extend([DESI_DM[z][0], DESI_DH[z][0]])
        kinds.extend(['DM', 'DH']); zs.extend([z, z])
        blocks.append(np.array(DESI_DMDH_COV[z]))
    O = np.asarray(O)
    Cov = np.zeros((len(O), len(O)))
    p = 0
    for b in blocks:
        m = b.shape[0]
        Cov[p:p + m, p:p + m] = b
        p += m
    return O, Cov, np.asarray(zs), np.asarray(kinds)


# =============================================================================
# 4.  Cosmic chronometers -- Moresco et al.
# =============================================================================
def cc_block():
    """(z, H, Cov) for the 15 BC03 D4000 points that have a published covariance.

        Cov = diag(stat+met) + Cov_IMF + Cov_SPSooo

    with the last two FULLY CORRELATED across redshift, Cov^X_ij = H_i x_i H_j x_j.  This
    is the combination prescribed by the authors' own CC_covariance notebook: it uses
    mod_ooo, not mod, and omits stlib.

    WHY THE FULL MATRIX MATTERS, stated correctly.  Per point the correlated part is
    sub-dominant -- it averages 4.9% of H(z) against a diagonal 13.4%.  But the diagonal
    part is independent between points and shrinks like 1/sqrt(N) on anything fixed by the
    overall SCALE of H(z), while the correlated part is one common multiplicative unknown
    and does not shrink at all.  The absolute scale of H(z) is exactly what calibrates r_d
    in configuration B, so a diagonal CC likelihood understates sigma(r_d) by a factor of
    2.7.  It is not the conservative choice it looks like.

    The other 18 points of the usual 33-point compilation have no published covariance and
    are DROPPED.  Leaving them diagonal would down-weight the best-characterised data and
    let the fit lean on the least-characterised; the MM20 percentages are properties of
    the BC03 D4000 modelling, so extending them to other groups would be an assumption,
    not a lookup."""
    z, H, err = np.genfromtxt(data_path('CC_mm20/HzTable_MM_BC03.dat'), comments='#',
                              usecols=(0, 1, 2), unpack=True, delimiter=',')
    zmod, imf, _stlib, _mod, spsooo = np.genfromtxt(
        data_path('CC_mm20/data_MM20.dat'), comments='#',
        usecols=(0, 1, 2, 3, 4), unpack=True)
    f_imf = np.interp(z, zmod, imf) / 100.0
    f_soo = np.interp(z, zmod, spsooo) / 100.0
    Cov = np.diag(err ** 2)
    for f in (f_imf, f_soo):
        Cov = Cov + np.outer(H * f, H * f)
    o = np.argsort(z)
    return z[o], H[o], Cov[np.ix_(o, o)]


# =============================================================================
# 5.  The cosmography <-> coefficient map
# =============================================================================
# The prior is stated on (H0, q0, j0, s0, l0) and imposed on the c_k, so the map between
# them has to be exact and has to be in this file: the per-coefficient half-widths quoted
# in the paper are a DERIVED quantity, and `--task bic --check-box` recomputes them here
# rather than trusting the numbers in BASELINE.
#
# Everything is truncated power-series arithmetic, coefficients lowest order first.

CG_NAMES = ('q0', 'j0', 's0', 'l0', 'm0', 'n0', 'o0')
CG_SCALE = {'m0': 5.0, 'n0': 20.0, 'o0': 100.0}


def cg_box(P):
    """[(lo, hi)] for the first P cosmographic parameters.

    🚨 NOT A PRIOR ANY MORE.  Until 2026-09-28 the coefficient prior was pushed forward
    from this box; it is now the Bernstein prior of section 7.  The box survives only for
    the chart diagnostics (ls_cosmography.py, make_cg_figure.py).

    q0/j0/s0/l0 in [-10, +9] -- width 19, centred on the standard -0.5.  m0 and n0 are
    that interval widened x5 and x20, and the widening is not cosmetic: flat LCDM itself
    sits at m0 = -7.8..-14.5 and n0 = +42..+84 over Om in [0.25, 0.35], so the plain box
    EXCLUDES the standard model at fifth and sixth order.  Only P <= 4 is used by the
    paper; the rest exist so the box can be recomputed at the orders the BIC sweeps."""
    if P > len(CG_NAMES):
        raise ValueError('no author range for the %dth cosmographic parameter' % P)
    return [(-10.0 * CG_SCALE.get(nm, 1.0), 9.0 * CG_SCALE.get(nm, 1.0))
            for nm in CG_NAMES[:P]]


def _mul(a, b, N):
    if a.ndim == 1 and b.ndim == 1:
        return np.convolve(a, b)[:N + 1]
    shape = (N + 1,) + np.broadcast_shapes(a.shape[1:], b.shape[1:])
    out = np.zeros(shape)
    for n in range(N + 1):
        for i in range(n + 1):
            out[n] += a[i] * b[n - i]
    return out


def _inv(a, N):
    out = np.zeros_like(a)
    out[0] = 1.0 / a[0]
    for n in range(1, N + 1):
        acc = np.zeros_like(a[0])
        for i in range(1, n + 1):
            acc = acc + a[i] * out[n - i]
        out[n] = -acc / a[0]
    return out


def _compose(f, g, N):
    """f(g(x)); requires g[0] == 0."""
    shape = (N + 1,) + np.broadcast_shapes(f.shape[1:], g.shape[1:])
    out = np.zeros(shape)
    pw = np.zeros(shape)
    pw[0] = 1.0
    for k in range(N + 1):
        out = out + f[k] * pw
        if k < N:
            pw = _mul(pw, g, N)
    return out


def _revert(z, N):
    """Compositional inverse T(z) of z(T), z[0]=0, z[1]!=0, by Newton on the composition."""
    T = np.zeros_like(z); T[1] = 1.0 / z[1]
    x = np.zeros_like(z); x[1] = 1.0
    dz = np.zeros_like(z)
    for k in range(N):
        dz[k] = (k + 1) * z[k + 1]
    for _ in range(int(math.ceil(math.log2(max(N, 2)))) + 2):
        num = _compose(z, T, N) - x
        den = _compose(dz, T, N)
        T = T - _mul(num, _inv(den, N), N)
        T[0] = 0.0
    return T


def coeffs(H0, cg, N=None):
    """c_0..c_P of H(z) = sum c_k y^k from H0 and cg = (q0, j0, s0, ...).

    Built from the scale factor a(T) = 1 + H0 T + sgn_n p_n H0^n T^n/n!, whence H = a'/a
    and z = 1/a - 1; reverting z(T) and composing gives H(z), and composing again with
    z(y) = y + y^2 + ... gives H(y).  Vectorised over trailing axes so the whole box can
    be pushed through at once."""
    P = len(cg)
    N = P + 2 if N is None else N
    signs = [-1.0] + [1.0] * (P - 1)
    H0 = np.asarray(H0, float)
    a = np.zeros((N + 1,) + H0.shape)
    a[0] = 1.0
    a[1] = H0
    for n, (p, sg) in enumerate(zip(cg, signs), start=2):
        if n <= N:
            a[n] = sg * np.asarray(p, float) * H0 ** n / math.factorial(n)
    da = np.zeros_like(a)
    for k in range(N):
        da[k] = (k + 1) * a[k + 1]
    inv_a = _inv(a, N)
    H_T = _mul(da, inv_a, N)
    z_T = inv_a.copy(); z_T[0] -= 1.0
    H_z = _compose(H_T, _revert(z_T, N), N)
    z_y = np.ones_like(a); z_y[0] = 0.0        # z = y/(1-y) = y + y^2 + ...
    return _compose(H_z, z_y, N)[:P + 1]


def lcdm_series(Om, P):
    """The y-series of sqrt(Om(1+z)^3 + 1 - Om) at H0 = 1, truncated at order P."""
    N = P + 2
    z_y = np.ones(N + 1); z_y[0] = 0.0
    E2_z = np.zeros(N + 1)
    E2_z[0], E2_z[1], E2_z[2], E2_z[3] = 1.0, 3 * Om, 3 * Om, Om
    E2 = _compose(E2_z, z_y, N)
    S = np.zeros(N + 1); S[0] = 1.0
    for _ in range(N + 2):                     # Newton: S <- (S + E2/S)/2
        S = 0.5 * (S + _mul(E2, _inv(S, N), N))
    return S[:P + 1]


def lcdm_coeffs(Om, H0, P):
    """The coefficient vector of flat LCDM.  c_k is exactly linear in H0."""
    return H0 * lcdm_series(Om, P)


def lcdm_cg(Om, P):
    """(q0, j0, s0, ...) of flat LCDM, solved order by order from the map itself.

    Nothing about the textbook q0 = 3Om/2 - 1, j0 = 1, s0 = 1 - 9Om/2 is assumed; that
    those come back is a test of the map, not an input to it."""
    N = P + 2
    target = lcdm_series(Om, P)
    cg = []
    for k in range(1, P + 1):
        lo = coeffs(1.0, cg + [0.0], N)[k]
        hi = coeffs(1.0, cg + [1.0], N)[k]
        cg.append(float((target[k] - lo) / (hi - lo)))   # linear in the k-th parameter
    return cg


def implied_ck_ranges(P, nsamp=200000, seed=0):
    """Per-k extremes of the cosmographic box, pushed through the exact map.

    Three routes, widest wins: every vertex, a vectorised Monte Carlo, and a coordinate
    refinement started from the MC extremes.  c_k is polynomial of degree up to k in q0
    and quadratic in some of the others, so the extreme need NOT sit at a vertex --
    `interior` flags where a corners-only calculation would have UNDER-stated the range,
    which happens at even k.

    Returns rows (k, lo, hi, half_width, interior).  Diagnostic of the RETIRED prior
    (see cg_box): the old coefficient box was these half-widths rounded outwards."""
    box = cg_box(P)
    rng = np.random.default_rng(seed)
    lo0, hi0 = BASELINE['H0_LO'], BASELINE['H0_HI']

    G = np.array(list(itertools.product([lo0, hi0], *[[a, b] for a, b in box]))).T
    cv = coeffs(G[0], list(G[1:]))

    H0 = rng.uniform(lo0, hi0, nsamp)
    cg = np.array([rng.uniform(a, b, nsamp) for a, b in box])
    vals = coeffs(H0, list(cg))

    def refine(x0, k, sign):
        x = x0.copy()
        bnds = [(lo0, hi0)] + box
        for _ in range(6):
            for d in range(len(x)):
                s_ = np.linspace(bnds[d][0], bnds[d][1], 401)
                X = np.repeat(x[:, None], len(s_), axis=1)
                X[d] = s_
                x = X[:, int(np.argmax(sign * coeffs(X[0], list(X[1:]))[k]))]
        return sign * float(coeffs(x[0], list(x[1:]))[k])

    rows = []
    for k in range(P + 1):
        lo_v, hi_v = float(cv[k].min()), float(cv[k].max())
        i_lo, i_hi = int(np.argmin(vals[k])), int(np.argmax(vals[k]))
        lo_r = -refine(np.concatenate([[H0[i_lo]], cg[:, i_lo]]), k, -1.0)
        hi_r = refine(np.concatenate([[H0[i_hi]], cg[:, i_hi]]), k, +1.0)
        lo = min(lo_v, float(vals[k].min()), lo_r)
        hi = max(hi_v, float(vals[k].max()), hi_r)
        rows.append((k, lo, hi, max(abs(lo), abs(hi)),
                     lo < lo_v - 1e-6 * abs(lo_v) or hi > hi_v + 1e-6 * abs(hi_v)))
    return rows


# =============================================================================
# 6.  Background evaluation (numpy side)
# =============================================================================
def y_of_z(z):
    """The expansion variable.  Single source: a coefficient vector produced in one
    variable and evaluated in the other is a silent wrong answer."""
    return np.asarray(z) / (1.0 + np.asarray(z))


def H_poly(z, c):
    """H(z) = sum c_k y^k."""
    y = y_of_z(z)
    return np.polyval(np.asarray(c)[::-1], y)


def H_lcdm(z, H0, Om):
    return H0 * np.sqrt(Om * (1.0 + np.asarray(z)) ** 3 + 1.0 - Om)


def distances(zgrid, Hg, z_out, c_km_s=None):
    """(D_M, D_H, D_V) at z_out from H on zgrid, by the same trapezoid the sampler uses."""
    c_km_s = BASELINE['C_KM_S'] if c_km_s is None else c_km_s
    invH = 1.0 / Hg
    trap = 0.5 * (invH[1:] + invH[:-1]) * np.diff(zgrid)
    DC = np.concatenate([[0.0], np.cumsum(trap)]) * c_km_s
    DM = np.interp(z_out, zgrid, DC)
    DH = c_km_s / np.interp(z_out, zgrid, Hg)
    DV = (np.asarray(z_out) * DM ** 2 * DH) ** (1.0 / 3.0)
    return DM, DH, DV


# =============================================================================
# 7.  Priors
# =============================================================================
# THE ENVELOPE PRIOR (2026-09-28).  The prior on the shape of E(y) = H/H0 =
# 1 + sum_{k=1}^{P} C_k y^k is UNIFORM IN THE C_k OVER EVERY POLYNOMIAL THAT STAYS INSIDE
# THE ENVELOPE
#
#     E_LOWER(y) <= E(y) <= E_UPPER(y)   for 0 <= y <= y_max      (0 <= E <= 2 + 7y)
#
# and nothing else.  That set A is convex and bounded, so the prior is proper, and it is
# defined by one statement about E -- no per-coefficient width anywhere.  y_max is Y_MAX,
# or AP_Y_MAX for the AP test's SN-only fit.
#
#   HOW IT IS SAMPLED.  The coordinates are v_j = E(y_j) at P fixed nodes (Chebyshev-
#   Lobatto on (0, y_max]); v is uniform on the box [E_LOWER(y_j), E_UPPER(y_j)] and
#   C = V^-1 (v - 1) is linear, so uniform in v is uniform in C.  Every member of A has
#   its node values in that box, so the box CONTAINS A; what is not in A is removed:
#     - nested sampling: log L = -inf outside A (checked on ENV_NGRID points), and
#       lnZ_A = lnZ_sampled - ln acc(P), acc = vol(A)/vol(box) from `envelope_log_
#       acceptance` (0.72 at P = 4, 0.56 at P = 6).  dynesty integrates -inf as zero
#       likelihood over the whole box -- asserted in test_repo on a toy with known Z.
#     - NUTS: a steep smooth wall, ENV_PENALTY x the integrated squared violation, because
#       -inf carries no gradient.  No posterior in this paper comes near the envelope;
#       `envelope_activity` records how near, per cell.
#   The nodes decide only how efficiently A is sampled, never the prior.
#
#   WHAT IT REPLACED, AND WHY.  The first version of this prior (same day) imposed the
#   envelope through a box on the Bernstein coefficients of E.  That is exact but only
#   SUFFICIENT: the box is 5% of A at P = 4, 0.8% at P = 5 and 0.09% at P = 6.  It cut
#   the low-H0 tail of the DES-alone posterior at P >= 5 and, by shrinking the prior
#   volume, handed the high orders 1.8-4 nats of evidence (the
#   Bernstein box is still the reference in test_repo, where it must lie inside A).
#
#   REPORTING.  The chains carry H0 and C_1..C_P (`C`); q0 = C_1 - 1.  The node values
#   `v` are sampling coordinates only.  q0 no longer has a flat prior: it is a marginal
#   of a uniform distribution over A, like every other C_k.
#
# r_d in configuration A is DELIBERATELY exempt from all of this: N(147.05, 0.30) is an
# external measurement being imported into the fit, not an ignorance prior, and imposing
# it is what defines the configuration.
import jax                                                          # noqa: E402
import jax.numpy as jnp                                             # noqa: E402
import numpyro                                                      # noqa: E402
import numpyro.distributions as dist                                # noqa: E402
from numpyro.infer import MCMC, NUTS                                # noqa: E402
from numpyro.diagnostics import summary                             # noqa: E402
from numpyro.infer.initialization import init_to_median, init_to_value  # noqa: E402

numpyro.set_host_device_count(BASELINE['NUM_CHAINS'])


ENV_NGRID = 1001           # points on [0, y_max] where the envelope is enforced
ENV_PENALTY = 1.0e8        # NUTS wall: log-density cost per unit of integral of violation^2 dy
ENV_TOL = 0.05             # at_bound: the posterior comes this close (in E) to the envelope
ENV_ACC_NDRAW = 2000000    # draws behind acc(P); sigma(ln acc) < 1e-3 at every P used
ENV_ACC_SEED = 20260928


def _y_max(y_max):
    y_max = float(BASELINE['Y_MAX'] if y_max is None else y_max)
    if not 0.0 < y_max < 1.0:
        raise ValueError('y_max must lie in (0, 1), got %r' % y_max)
    return y_max


def envelope_grid(y_max=None):
    return np.linspace(0.0, _y_max(y_max), ENV_NGRID)


def envelope_bounds(y, y_max=None):
    """(E_LOWER(y), E_UPPER(y)); the envelopes are monomial coefficients in y."""
    y = np.asarray(y, float)
    return (np.polyval(np.asarray(BASELINE['E_LOWER'], float)[::-1], y),
            np.polyval(np.asarray(BASELINE['E_UPPER'], float)[::-1], y))


def envelope_nodes(P, y_max=None):
    """(y_j, lo_j, hi_j, Vinv) for the sampling coordinates v_j = E(y_j), j = 1..P.

    Chebyshev-Lobatto nodes on (0, y_max], so the last node is y_max itself.  C_1..C_P
    = Vinv @ (v - 1).  The nodes set the sampling efficiency and nothing else."""
    y_max = _y_max(y_max)
    y = y_max * (1.0 - np.cos(np.arange(1, P + 1) * np.pi / P)) / 2.0
    lo, hi = envelope_bounds(y)
    if not np.all(hi > lo):
        raise ValueError('empty envelope at the nodes')
    V = np.power(y[:, None], np.arange(1, P + 1)[None, :])
    return y, lo, hi, np.linalg.inv(V)


def shape_of_v(v, P, y_max=None):
    """(1, C_1..C_P) from the node values; v may carry leading batch axes."""
    _y, _lo, _hi, Vinv = envelope_nodes(P, y_max)
    v = np.asarray(v, float)
    C = (v - 1.0) @ Vinv.T
    return np.concatenate([np.ones(C.shape[:-1] + (1,)), C], axis=-1)


def v_of_shape(C, P, y_max=None):
    """The node values of a shape vector C = (1, C_1..C_P)."""
    y, _lo, _hi, _Vinv = envelope_nodes(P, y_max)
    C = np.asarray(C, float)
    return C @ np.power(y[:, None], np.arange(P + 1)[None, :]).T


def envelope_margins(C, y_max=None):
    """(min_y (E - E_LOWER), min_y (E_UPPER - E)) on the envelope grid, per shape.

    C = (1, C_1..C_P), optionally batched.  Both >= 0 exactly when the shape is in A.
    Pass a DIMENSIONAL coefficient vector through C = c / c[0] first -- that is how a
    least-squares best fit, which lives in the old chart, is checked against this prior."""
    C = np.atleast_2d(np.asarray(C, float))
    y = envelope_grid(y_max)
    lo, hi = envelope_bounds(y)
    E = C @ np.power(y[None, :], np.arange(C.shape[1])[:, None])
    return (E - lo[None, :]).min(axis=1), (hi[None, :] - E).min(axis=1)


def in_envelope(C, y_max=None):
    a, b = envelope_margins(C, y_max)
    return (a >= 0.0) & (b >= 0.0)


def sample_envelope_prior(P, n, seed=0, y_max=None, chunk=20000):
    """(C, n_proposed): n exact draws from the prior on the shape, by rejection from the
    node box.  The single implementation behind the acceptance, prior_reference, Table 5
    and the tests."""
    y, lo, hi, _Vinv = envelope_nodes(P, y_max)
    rng = np.random.default_rng(seed)
    got, tried = [], 0
    while sum(len(g) for g in got) < n:
        C = shape_of_v(rng.uniform(lo, hi, (chunk, P)), P, y_max)
        tried += chunk
        got.append(C[in_envelope(C, y_max)])
    return np.concatenate(got)[:n], tried


_ACC_CACHE = {}


def envelope_log_acceptance(P, y_max=None):
    """(ln acc, sigma): acc = vol(A)/vol(node box), by Monte Carlo on a fixed seed.

    Enters every evidence: nested sampling runs over the node box with -inf outside A,
    so it returns acc x Z_A.  Deterministic (fixed seed and draw count), so every task
    and every rerun uses the identical number."""
    key = (int(P), _y_max(y_max))
    if key not in _ACC_CACHE:
        y, lo, hi, _Vinv = envelope_nodes(P, y_max)
        rng = np.random.default_rng(ENV_ACC_SEED + P)
        acc, done, chunk = 0, 0, 20000
        while done < ENV_ACC_NDRAW:
            C = shape_of_v(rng.uniform(lo, hi, (chunk, P)), P, y_max)
            acc += int(in_envelope(C, y_max).sum())
            done += chunk
        a = acc / done
        _ACC_CACHE[key] = (float(np.log(a)), float(np.sqrt((1.0 - a) / (a * done))))
    return _ACC_CACHE[key]


def log_shape_volume(P, y_max=None):
    """ln vol(A) in (C_1..C_P): ln prod(hi - lo) - ln |det V| + ln acc.

    The evidence charges exactly this per order.  In the dimensional chart the density of
    (H0, H0 C_1..H0 C_P) carries an extra H0^-P, the Jacobian of c_k = H0 C_k."""
    _y, lo, hi, Vinv = envelope_nodes(P, y_max)
    return float(np.sum(np.log(hi - lo)) + np.log(abs(np.linalg.det(Vinv)))
                 + envelope_log_acceptance(P, y_max)[0])


def _poly_str(coeffs):
    terms = [('%g' % c) + ('' if k == 0 else 'y' if k == 1 else 'y^%d' % k)
             for k, c in enumerate(coeffs) if c != 0 or k == 0]
    return ' + '.join(terms)


def prior_description():
    return ('H0 ~ U(%g,%g)  shape uniform in C_k over %s <= E(y) <= %s, y <= %g  '
            'M ~ U(%g,%g)'
            % (BASELINE['H0_LO'], BASELINE['H0_HI'], _poly_str(BASELINE['E_LOWER']),
               _poly_str(BASELINE['E_UPPER']), BASELINE['Y_MAX'],
               BASELINE['M_LO'], BASELINE['M_HI']))


def sample_coeffs(P, y_max=None):
    """(H0, C) with C = (1, C_1..C_P) the shape of E(y) = sum_k C_k y^k.

    Sampled sites: `H0` and `v` (E at the nodes).  `C` (C_1..C_P) is recorded as a
    deterministic, and the envelope between the nodes is the `envelope` factor."""
    H0 = numpyro.sample('H0', dist.Uniform(BASELINE['H0_LO'], BASELINE['H0_HI']))
    if P < 1:
        return H0, jnp.ones(1)
    _y, lo, hi, Vinv = envelope_nodes(P, y_max)
    v = numpyro.sample('v', dist.Uniform(jnp.asarray(lo), jnp.asarray(hi)).to_event(1))
    C = jnp.concatenate([jnp.ones(1), jnp.asarray(Vinv) @ (v - 1.0)])
    numpyro.deterministic('C', C[1:])
    yg = envelope_grid(y_max)
    elo, ehi = envelope_bounds(yg)
    E = jnp.asarray(np.power(yg[:, None], np.arange(P + 1)[None, :])) @ C
    viol2 = (jnp.clip(jnp.asarray(elo) - E, 0.0, None) ** 2
             + jnp.clip(E - jnp.asarray(ehi), 0.0, None) ** 2)
    area = jnp.sum(0.5 * (viol2[1:] + viol2[:-1])) * float(yg[1] - yg[0])
    numpyro.factor('envelope', -ENV_PENALTY * area)
    return H0, C


def sample_M():
    return numpyro.sample('M', dist.Uniform(BASELINE['M_LO'], BASELINE['M_HI']))


def sample_rd(rd_mode):
    """'planck' -> the imported Planck measurement; 'free' -> derived from the CC."""
    if rd_mode == 'planck':
        return numpyro.sample('rd', dist.Normal(BASELINE['RD_PLANCK'], BASELINE['SIG_RD']))
    if rd_mode != 'free':
        raise ValueError("rd_mode must be 'planck' or 'free', got %r" % rd_mode)
    return numpyro.sample('rd', dist.Uniform(BASELINE['RD_FREE_LO'], BASELINE['RD_FREE_HI']))


def sample_Om():
    return numpyro.sample('Om', dist.Uniform(BASELINE['OM_LO'], BASELINE['OM_HI']))


def sample_delta(name='Delta'):
    """The legacy-to-DES zero-point offset in magnitudes.

    U(-1, +1) is about 25 sigma wide against the measured sigma_Delta ~ 0.04, which is the
    point: the free-offset row must make NO assumption about the size of a possible
    calibration offset.

    `name` exists so that the split-offset model can draw one of these per legacy
    subsample (Foundation, CfA/CSP) from the SAME prior.  The default reproduces the
    single-offset site exactly, so every published chain is unaffected."""
    w = BASELINE['DELTA_HALFWIDTH']
    return numpyro.sample(name, dist.Uniform(-w, w))


# =============================================================================
# 8.  Sampler
# =============================================================================
def init_strategy(P, y_max=None):
    """Where the chain starts.

    At INIT='lcdm' the target is flat LCDM: its y-series truncated at order P, as node
    values.  It lies inside the envelope at every order 1..7 at both ceilings (asserted
    below), inside H(z) > 0 by construction, and deterministic given (Om, H0).  It is NOT tuned to the answer -- `full`'s posterior sits near it and
    `des_only`'s does not, and that asymmetry is the result.

    The prior-free maximum-likelihood point is NOT usable as an init and must not be
    reinstated: it slides along the two flat directions to j0 = 47.5, s0 = 829, l0 = 18737
    for des_only, i.e. outside the prior support by three orders of magnitude.

    Why this is a function and not a default: `init_to_median` froze 22 of 24 cells of an
    earlier campaign -- the step size collapsed and the chains never moved -- while the
    banner and the manifest both recorded the init that had been REQUESTED.  Provenance
    that records a request rather than the routing is worse than none, so `make_nuts`
    below refuses to run when INIT is non-default and no strategy was routed to it."""
    if BASELINE['INIT'] == 'median':
        return init_to_median
    vals = {'M': jnp.asarray(-19.40), 'rd': jnp.asarray(BASELINE['RD_PLANCK'])}
    if P >= 1:
        C = lcdm_series(BASELINE['INIT_OM'], P)
        m_lo, m_hi = envelope_margins(C, y_max)
        if not (m_lo[0] > 0.0 and m_hi[0] > 0.0):
            raise ValueError('flat LCDM (Om=%g) lies outside the envelope at P=%d; it '
                             'cannot be the initial point' % (BASELINE['INIT_OM'], P))
        vals['v'] = jnp.asarray(v_of_shape(C, P, y_max))
    vals['H0'] = jnp.asarray(float(BASELINE['INIT_H0']))
    vals['Om'] = jnp.asarray(float(BASELINE['INIT_OM']))
    return init_to_value(values=vals)


def make_nuts(model, P, kind='joint', y_max=None, **kw):
    """The project's one NUTS kernel.

    dense_mass is central and not a per-caller argument: the induced coefficient
    correlations reach +0.98, and a diagonal mass matrix cannot rescale a ridge that lies
    at 45 degrees to every axis.  Turning it on took the clean cells from n_eff ~2500 to
    ~6500 at identical cost."""
    if ('dense_mass' in kw or 'init_strategy' in kw or 'max_tree_depth' in kw
            or 'target_accept_prob' in kw):
        raise ValueError('dense_mass, init_strategy, max_tree_depth and '
                         'target_accept_prob are set centrally here; a per-caller '
                         'override is how the sampler settings drifted apart in the '
                         'first place.  Choose kind="joint" or "shape", and set '
                         'TARGET_ACCEPT in the environment if a cell needs rescuing.')
    if kind not in ('joint', 'shape'):
        raise ValueError("make_nuts kind must be 'joint' (the full ladder) or 'shape' "
                         "(supernovae alone); got %r.  Guessing is not an option -- the "
                         "two use different tree depths and always have." % kind)
    return NUTS(model, init_strategy=init_strategy(P, y_max),
                max_tree_depth=BASELINE['MAX_TREE_DEPTH_' + kind.upper()],
                target_accept_prob=float(BASELINE['TARGET_ACCEPT']),
                dense_mass=bool(BASELINE['DENSE_MASS']), **kw)


def run_mcmc(model, P_init, model_args=(), model_kwargs=None, seed=None, quiet=False,
             kind='joint', y_max=None):
    """Run NUTS and return (samples, diagnostics).

    One entry point, so every chain in the paper is adapted identically and every cell
    reports the same diagnostics.  `P_init` is the order the init strategy seeds; the
    model's own arguments go through `model_args`/`model_kwargs` so that a model keyword
    called `P` cannot collide with it."""
    seed = BASELINE['SEED'] if seed is None else seed
    mcmc = MCMC(make_nuts(model, P_init, kind=kind, y_max=y_max),
                num_warmup=BASELINE['N_WARM'], num_samples=BASELINE['N_SAMP'],
                num_chains=BASELINE['NUM_CHAINS'],
                progress_bar=not quiet, chain_method='parallel')
    mcmc.run(jax.random.PRNGKey(seed), *model_args, extra_fields=('diverging',),
             **(model_kwargs or {}))
    s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    diag = _diagnostics(mcmc)
    # The per-draw divergence flags, and enough shape information to REGROUP the flat
    # draws by chain: numpyro ravels (chain, draw) in C order, so .reshape(n_chains, -1)
    # recovers the grouping for any stored column.  Only the counts used to be kept, and
    # a count cannot tell you WHERE the sampler failed.
    diag['n_chains'] = int(BASELINE['NUM_CHAINS'])
    diag['n_draws_per_chain'] = int(BASELINE['N_SAMP'])
    try:
        dv = np.asarray(mcmc.get_extra_fields().get('diverging'))
        s['diverging'] = dv.reshape(-1).astype(np.int8)
    except Exception:
        pass
    ok, why = gate_verdict(diag)
    diag['quality_pass'] = bool(ok)
    diag['quality_why'] = list(why)
    if not ok:
        print('[gate] this chain FAILS the publication gate: %s' % '; '.join(why))
        print('[gate] it is cached anyway -- the tables will refuse it until it is '
              'either re-run (TARGET_ACCEPT=0.95, then 0.99) or explicitly allowed.')
    return s, diag


def gate_verdict(diag):
    """(ok, reasons) for one chain's diagnostics, against the BASELINE thresholds.

    Deliberately NOT enforced at sampling time.  A cell that took six hours on the cluster
    and came back with divergences is still worth keeping: it is the only evidence of what
    went wrong, and rejecting it before `cache_save` would mean paying for it twice to
    learn the same thing.  What must not happen is that it reaches a table -- so the gate
    is read by make_tables and make_figures, which refuse a failing chain unless
    ALLOW_BAD_CHAINS=1 says someone has looked at it and decided.

    Read the reasons in order: rhat and n_eff first, then the step size against the
    healthy reference of ~0.08-0.13, and only THEN divergences.  A divergence count is
    meaningless on a chain that never moved -- a collapsed step size never diverges, so
    div = 0 reads as reassurance on precisely the runs that failed.

    Divergences are reported but do not fail a chain whose
    R-hat and n_eff pass: GATE_MAX_DIVERGENCES = -1 switches that criterion off."""
    why = []
    div = diag.get('divergences', -1)
    rhat = diag.get('rhat_max', float('nan'))
    neff = diag.get('neff_min', float('nan'))
    if BASELINE['GATE_MAX_DIVERGENCES'] >= 0:
        if div is None or div < 0:
            why.append('divergence count unavailable')
        elif div > BASELINE['GATE_MAX_DIVERGENCES']:
            why.append('%d divergences (max %d)' % (div, BASELINE['GATE_MAX_DIVERGENCES']))
    if not np.isfinite(rhat):
        why.append('rhat unavailable')
    elif rhat > BASELINE['GATE_MAX_RHAT']:
        why.append('rhat_max = %.4f (max %.3f)' % (rhat, BASELINE['GATE_MAX_RHAT']))
    if not np.isfinite(neff):
        why.append('n_eff unavailable')
    elif neff < BASELINE['GATE_MIN_NEFF']:
        why.append('neff_min = %.0f (min %.0f)' % (neff, BASELINE['GATE_MIN_NEFF']))
    return (not why), why


ALLOW_BAD_CHAINS = os.environ.get('ALLOW_BAD_CHAINS', '0') not in ('0', '', 'no', 'false')


def require_quality(label, diag, allow=None):
    """Raise unless this chain is fit to publish.  Called by the table/figure builders."""
    ok, why = gate_verdict(diag)
    if ok:
        return True
    allow = ALLOW_BAD_CHAINS if allow is None else allow
    msg = ('%s fails the publication-quality gate: %s.\n'
           'Re-run it -- TARGET_ACCEPT=0.95 is the first lever, then 0.99 -- or set '
           'ALLOW_BAD_CHAINS=1 to render it anyway, deliberately and on the record.'
           % (label, '; '.join(why)))
    if not allow:
        raise SystemExit(msg)
    print('[gate] ALLOW_BAD_CHAINS: ' + msg.splitlines()[0])
    return False


def _diagnostics(mcmc):
    """rhat, n_eff, divergences, adapted step size and mass-matrix conditioning.

    The step size is here because a divergence count is MEANINGLESS on a chain that never
    moved: a collapsed step size never diverges, so `div = 0` reads as reassurance on
    precisely the runs that failed.  Read rhat and n_eff first, then eps against the
    healthy reference of ~0.08-0.13, and only then the divergences."""
    out = {}
    try:
        sm = summary(mcmc.get_samples(group_by_chain=True), prob=0.9)
        out['rhat_max'] = float(max(np.nanmax(v['r_hat']) for v in sm.values()))
        out['neff_min'] = float(min(np.nanmin(v['n_eff']) for v in sm.values()))
    except Exception:
        out['rhat_max'], out['neff_min'] = float('nan'), float('nan')
    try:
        out['divergences'] = int(np.sum(np.asarray(mcmc.get_extra_fields()
                                                   .get('diverging', 0))))
    except Exception:
        out['divergences'] = -1
    try:
        st = mcmc.last_state
        eps = st.adapt_state.step_size
        out['step_size'] = float(np.mean(np.asarray(eps)))
        imm = st.adapt_state.inverse_mass_matrix
        # with dense_mass, numpyro returns a dict keyed by a tuple of site names
        mats = list(imm.values()) if isinstance(imm, dict) else [imm]
        conds = []
        for m in mats:
            m = np.asarray(m)
            if m.ndim >= 2:
                # np.linalg.cond broadcasts over leading axes, which is what a
                # multi-chain dense mass matrix is.  The old fallback took
                # m.max()/m.min() on the batched case, and a positive-definite dense
                # matrix has negative off-diagonal entries, so every imm_cond ever
                # stored by this code was NEGATIVE -- a number no condition number can
                # take.  It was never read as a result, but it was printed as one.
                conds.extend(np.ravel(np.linalg.cond(m)).tolist())
            else:
                # genuinely diagonal: the mass matrix is a vector of variances
                conds.append(float(np.max(m) / np.min(m)))
        conds = [float(c) for c in conds]
        if not all(np.isfinite(c) and c >= 1.0 - 1e-6 for c in conds):
            raise AssertionError('inverse-mass condition numbers must be finite and '
                                 '>= 1; got %r' % (conds[:8],))
        out['imm_cond'] = float(max(conds))
    except Exception:
        out['step_size'], out['imm_cond'] = float('nan'), float('nan')
    return out


def envelope_activity(samples, P, weights=None, y_max=None):
    """How close the posterior comes to each side of the envelope.

    Returns ({'lower': m_lo, 'upper': m_hi}, at_bound): the 0.1% posterior quantiles of
    min_y (E - E_LOWER) and min_y (E_UPPER - E), in units of E, and the sides with a
    quantile below ENV_TOL -- the check that no constraint is bought at the prior edge.
    `weights` is for nested sampling, whose unweighted samples include the prior itself
    and would always look "at bound"."""
    if P < 1 or 'C' not in samples:
        return None, []
    Cs = np.asarray(samples['C'], float)
    C = np.column_stack([np.ones(len(Cs)), Cs])
    a, b = envelope_margins(C, y_max)
    if weights is None:
        q = [float(np.quantile(a, 0.001)), float(np.quantile(b, 0.001))]
    else:
        w = np.asarray(weights, float) / np.sum(weights)
        q = []
        for x in (a, b):
            i = np.argsort(x)
            cw = np.cumsum(w[i])
            q.append(float(np.interp(0.001 * cw[-1], cw, x[i])))
    at = [side for side, m in zip(('lower', 'upper'), q) if m < ENV_TOL]
    return {'lower': q[0], 'upper': q[1]}, at


# =============================================================================
# 9.  The likelihood
# =============================================================================
# The SN, BAO and CC blocks share ONE background.  H(z) is either the cosmographic
# polynomial or flat LCDM; nothing else differs between the two.
H_FLOOR = 1.0        # km/s/Mpc -- keeps 1/H finite
H_PENALTY = 1.0e4    # log-density penalty per (km/s/Mpc)^2 of violation, per DZ_REF of z
H_PENALTY_DZ_REF = 1.0e-3   # the redshift interval H_PENALTY is quoted per


def _h_positive(Hg, zgrid):
    """Clip H into the physical region and add a smooth penalty outside it.

    A hard rejection would be wrong here: -inf carries no gradient and breaks HMC.  The
    clip is an exact no-op wherever H >= H_FLOOR, so nothing in the supported region
    moves.  The check runs on the whole integration grid rather than at the data
    redshifts, because an interior sign flip can leave D_C positive and monotone at every
    observed z while the intermediate integral is unphysical.

    The penalty is an INTEGRAL over z, not a sum over grid points.  As a bare sum it grew
    linearly with NGRID, so doubling the integration resolution doubled the strength of a
    physicality constraint that ought to be a property of the model and not of the mesh.
    No draw in any supported posterior here has ever violated it -- checked on a denser
    grid -- so this changes no published number; it stops NGRID from being a physics knob.
    """
    viol = jnp.clip(H_FLOOR - Hg, 0.0, None)
    v2 = viol ** 2
    area = jnp.sum(0.5 * (v2[1:] + v2[:-1]) * jnp.diff(zgrid))
    return jnp.clip(Hg, H_FLOOR, None), -H_PENALTY * area / H_PENALTY_DZ_REF


def cc_to_cholesky(cc):
    """(z, H, Cov) -> (z, H, L).

    numpyro's MultivariateNormal takes `scale_tril`, so the factorisation MUST happen
    before the CC block reaches the model.  Handing the covariance straight in asserts
    Cov = C C^T and inflates the per-point CC sigma by 13-16x -- which switches the
    chronometers off and leaves r_d prior-dominated in configuration B.  Exactly that
    happened to a whole published table for four days.  ONE conversion, here, asserted by
    test_repo.py."""
    if cc is None:
        return None
    z, H, Cov = (np.asarray(a) for a in cc)
    return (jnp.asarray(z), jnp.asarray(H),
            jnp.asarray(np.linalg.cholesky(Cov + 1e-9 * np.eye(len(z)))))


def _selectors(zs, kinds, z_uni):
    idx = {z: k for k, z in enumerate(z_uni)}
    return (np.array([idx[z] for z in zs]),
            np.array([k == 'DM' for k in kinds]),
            np.array([k == 'DH' for k in kinds]),
            np.array([k == 'DV' for k in kinds]))


def joint_model(cc, sn, bao, zgrid, sel, z_uni, use_cc, rd_mode, P, background,
                u_leg=None, delta_names=None):
    """The inverse distance ladder.

    background='cosmo' -> H(z) = H0 sum_k C_k y^k     (free shape below the survey floor)
    background='lcdm'  -> H(z) = H0 sqrt(Om(1+z)^3 + 1 - Om)   (fixed-shape cross-check)

    u_leg, if given, is the legacy indicator: a single coherent zero-point offset Delta is
    then sampled INSIDE this likelihood alongside the cosmology, rather than measured
    first and applied afterwards.  It has to be joint because Delta and H0 are 92%
    degenerate, so a two-stage correction produces error bars that cannot know about the
    covariance they inherit.

    u_leg may be a single indicator (one offset, site 'Delta') or an (n_sn, K) matrix of
    indicators with `delta_names` naming the K offsets -- the split-offset model, in which
    Foundation and CfA/CSP each carry their own zero point.  With K = 1 the two paths are
    the same arithmetic."""
    z_sn, zhel_sn, m_sn, L_sn = sn
    O, L_bao = bao
    col, is_dm, is_dh, is_dv = sel

    if background == 'cosmo':
        H0, C = sample_coeffs(P)

        def H_of(z):
            y = z / (1.0 + z)
            return H0 * (jnp.power(y[:, None], jnp.arange(C.shape[0])[None, :]) @ C)
    elif background == 'lcdm':
        H0 = numpyro.sample('H0', dist.Uniform(BASELINE['H0_LO'], BASELINE['H0_HI']))
        Om = sample_Om()

        def H_of(z):
            return H0 * jnp.sqrt(Om * (1.0 + z) ** 3 + 1.0 - Om)
    else:
        raise ValueError("background must be 'cosmo' or 'lcdm', got %r" % background)

    M = sample_M()
    rd = sample_rd(rd_mode)

    Hg, pen = _h_positive(H_of(zgrid), zgrid)
    numpyro.factor('h_positive', pen)

    # ONE definition of H outside the physical region.  The chronometers and D_H used to
    # read the RAW H_of(z) while the distance integral read the CLIPPED grid, so below
    # H_FLOOR the code was two different models at once.  Clipping directly -- rather than
    # interpolating the clipped grid -- keeps this an EXACT no-op wherever H >= H_FLOOR,
    # so no supported draw moves by even one float32 ulp.
    def H_clipped(z):
        return jnp.clip(H_of(z), H_FLOOR, None)

    if use_cc:
        z_cc, H_cc, L_cc = cc
        numpyro.sample('cc', dist.MultivariateNormal(loc=H_clipped(z_cc), scale_tril=L_cc),
                       obs=H_cc)

    invH = 1.0 / Hg
    trap = 0.5 * (invH[1:] + invH[:-1]) * jnp.diff(zgrid)
    DC = jnp.concatenate([jnp.zeros(1), jnp.cumsum(trap)]) * BASELINE['C_KM_S']

    m_model = sn_apparent_mag(M, zhel_sn, jnp.interp(z_sn, zgrid, DC), xp=jnp)
    if u_leg is not None:
        U = u_leg if jnp.ndim(u_leg) == 2 else u_leg[:, None]
        names = list(delta_names) if delta_names else ['Delta']
        if U.shape[1] != len(names):
            raise ValueError('u_leg has %d column(s) but %d delta name(s) were given'
                             % (U.shape[1], len(names)))
        m_model = m_model + U @ jnp.stack([sample_delta(n) for n in names])
    numpyro.sample('sn', dist.MultivariateNormal(loc=m_model, scale_tril=L_sn), obs=m_sn)

    DM = jnp.interp(z_uni, zgrid, DC)
    DH = BASELINE['C_KM_S'] / H_clipped(z_uni)
    DV = (z_uni * DM ** 2 * DH) ** (1.0 / 3.0)
    D = jnp.where(is_dm, DM[col], jnp.where(is_dh, DH[col], DV[col]))
    numpyro.sample('bao', dist.MultivariateNormal(loc=D / rd, scale_tril=L_bao), obs=O)

    numpyro.deterministic('H0_out', H0)
    numpyro.deterministic('rd_out', rd)
    if background == 'cosmo':
        if P >= 1:
            numpyro.deterministic('q0', C[1] - 1.0)
    else:
        numpyro.deterministic('q0', 1.5 * Om - 1.0)


def build_blocks(sample, use_cc, z_floor=0.0):
    """Everything the model needs, assembled once so the tasks cannot assemble it
    differently from each other."""
    z_sn, zhel_sn, m_sn, cov_sn = sn_block(sample, z_floor=z_floor)
    O, Cov, zs, kinds = bao_vector()
    z_uni = np.unique(zs)
    cc = cc_block() if use_cc else None
    zmax = max(float(z_uni.max()), float(z_sn.max()),
               float(cc[0].max()) if use_cc else 0.0)
    zgrid = np.linspace(0.0, zmax + 1e-6, BASELINE['NGRID'])
    return dict(
        z_sn=z_sn, zhel_sn=zhel_sn, m_sn=m_sn, cov_sn=cov_sn,
        L_sn=np.linalg.cholesky(cov_sn + 1e-9 * np.eye(len(z_sn))),
        O=O, cov_bao=Cov,
        L_bao=np.linalg.cholesky(Cov + 1e-12 * np.eye(len(O))),
        zs=zs, kinds=kinds, z_uni=z_uni, sel=_selectors(zs, kinds, z_uni),
        cc=cc, zgrid=zgrid)


def fit(sample, config, background, P=None, z_floor=0.0, joint_delta=False,
        seed=None, quiet=False):
    """One cell of the ladder.

    config 'A' = SN + BAO + Planck r_d           (the baseline)
    config 'B' = SN + BAO + CC, r_d free          (no CMB input; the ruler is derived)
    """
    P = BASELINE['P'] if P is None else P
    if config == 'A':
        use_cc, rd_mode = False, 'planck'
    elif config == 'B':
        use_cc, rd_mode = True, 'free'
    else:
        raise ValueError("config must be 'A' or 'B', got %r" % config)

    b = build_blocks(sample, use_cc, z_floor=z_floor)
    # joint_delta: False = legacy at face value; True = ONE offset (the published row);
    # 'split' = one offset per legacy survey group (Foundation, CfA/CSP).
    if joint_delta not in (False, True, 'split'):
        raise ValueError("joint_delta must be False, True or 'split', got %r"
                         % (joint_delta,))
    u_leg, delta_names = None, None
    if joint_delta:
        U, delta_names = legacy_columns(sample, split=(joint_delta == 'split'),
                                        z_floor=z_floor)
        if U.shape[0] != len(b['z_sn']):
            raise ValueError('offset design matrix has %d rows against %d supernovae'
                             % (U.shape[0], len(b['z_sn'])))
        u_leg = jnp.asarray(U)

    args = (cc_to_cholesky(b['cc']),
            (jnp.asarray(b['z_sn']), jnp.asarray(b['zhel_sn']), jnp.asarray(b['m_sn']),
             jnp.asarray(b['L_sn'])),
            (jnp.asarray(b['O']), jnp.asarray(b['L_bao'])),
            jnp.asarray(b['zgrid']),
            tuple(jnp.asarray(a) for a in b['sel']),
            jnp.asarray(b['z_uni']))
    s, diag = run_mcmc(joint_model, P, model_args=args,
                       model_kwargs=dict(use_cc=use_cc, rd_mode=rd_mode, P=P,
                                         background=background, u_leg=u_leg,
                                         delta_names=delta_names),
                       seed=seed, quiet=quiet)

    out = dict(sample=sample, config=config, background=background, P=P,
               z_floor=float(z_floor), joint_delta=bool(joint_delta),
               delta_mode=('split' if joint_delta == 'split'
                           else 'joint' if joint_delta else 'fixed'),
               delta_names=list(delta_names) if delta_names else [], **diag)
    if 'diverging' in s:
        out['diverging'] = s['diverging']
    for k, key in (('H0', 'H0_out'), ('rd', 'rd_out'), ('M', 'M'), ('q0', 'q0')):
        if key in s:
            out[k] = s[key]
            out[k + '_mean'] = float(s[key].mean())
            out[k + '_std'] = float(s[key].std())
    if 'Om' in s:
        out['Om'] = s['Om']
        out['Om_mean'], out['Om_std'] = float(s['Om'].mean()), float(s['Om'].std())
    for nm in (delta_names or []):
        if nm not in s:
            continue
        out[nm] = s[nm]
        out[nm + '_mean'], out[nm + '_std'] = float(s[nm].mean()), float(s[nm].std())
        out['corr_%s_H0' % nm] = float(np.corrcoef(s[nm], s['H0_out'])[0, 1])
        out['corr_%s_M' % nm] = float(np.corrcoef(s[nm], s['M'])[0, 1])
    # The CONTRAST is the quantity the survey-split question is about, and it must be
    # formed per draw, never from the two marginal means: Delta_found and Delta_cfa share
    # the whole sub-floor extrapolation and are strongly correlated, so differencing the
    # summaries would report an error bar that has not cancelled anything.
    if joint_delta == 'split' and all(nm in s for nm in delta_names):
        dif = s[delta_names[0]] - s[delta_names[1]]
        out['Delta_diff'] = dif
        out['Delta_diff_mean'], out['Delta_diff_std'] = float(dif.mean()), float(dif.std())
        out['corr_delta_pair'] = float(np.corrcoef(s[delta_names[0]],
                                                   s[delta_names[1]])[0, 1])
    if background == 'cosmo' and P >= 1:
        out['v'] = s['v']
        out['C'] = s['C']
        marg, at = envelope_activity(s, P)
        out['env_margin'] = marg
        out['at_bound'] = at

    # goodness of fit at the posterior mean, with the same blocks the sampler used.  The
    # legacy offset has to travel with it: omitting it from the chi2 while the fit sampled
    # it would report a fit quality for a model that was not fitted.
    theta = _theta_at_mean(s, background, P, delta_names)
    chi2 = chi2_blocks(theta, b, use_cc, background, P,
                       u_leg=(np.asarray(u_leg) if u_leg is not None else None))
    n_par = ((P + 1 if background == 'cosmo' else 2) + 2
             + (len(delta_names) if delta_names else 0))
    n_dat = len(b['z_sn']) + len(b['O']) + (len(b['cc'][0]) if use_cc else 0)
    # Planck in configuration A is a PRIOR here -- sample_rd draws r_d from it and no
    # likelihood term repeats it -- so its residual is reported alongside the blocks but
    # is NOT folded into chi2_dof, and the datum is not counted in n_data.  The BIC path
    # counts it as a datum instead, and correctly so: that fit is prior-free least
    # squares, where the only way Planck can enter at all is as one more residual.  The
    # two conventions describe two different objects; the failure mode is calling them
    # both 'chi2/dof' and comparing them.
    chi2_planck = (((theta['rd'] - BASELINE['RD_PLANCK']) / BASELINE['SIG_RD']) ** 2
                   if rd_mode == 'planck' else 0.0)
    out.update(chi2_sn=chi2['sn'], chi2_bao=chi2['bao'], chi2_cc=chi2['cc'],
               chi2_planck=float(chi2_planck), chi2_tot=chi2['tot'],
               n_data=n_dat, n_par=n_par,
               chi2_dof=chi2['tot'] / (n_dat - n_par),
               chi2_eval_point='posterior mean',
               chi2_convention='SN+BAO+CC; Planck r_d is a prior and is excluded '
                               'from both chi2_tot and n_data')
    return out


def _theta_at_mean(s, background, P, delta_names=None):
    dvec = np.array([float(s[nm].mean()) for nm in (delta_names or []) if nm in s])
    if background == 'cosmo':
        # the mean of the SAMPLED parameters (H0, b), carried into the dimensional chart
        # chi2_blocks evaluates in.  C is linear in b, so this is also the mean of C.
        C = (np.concatenate([[1.0], s['C'].mean(axis=0)]) if P >= 1 else np.ones(1))
        return dict(c=float(s['H0_out'].mean()) * C, M=float(s['M'].mean()),
                    rd=float(s['rd_out'].mean()), Delta=dvec)
    return dict(H0=float(s['H0'].mean()), Om=float(s['Om'].mean()),
                M=float(s['M'].mean()), rd=float(s['rd_out'].mean()),
                Delta=dvec)


def chi2_blocks(theta, b, use_cc, background, P, u_leg=None):
    """chi2 per block, in numpy, sharing the model of `joint_model` exactly.

    Used both at the posterior mean (the fit-quality column of Table 1) and inside the BIC
    minimisation.  Note it solves against the covariance rather than a Cholesky factor,
    which is why the chi2 columns of an earlier table were right while its error bars were
    not: the two paths failed independently."""
    if background == 'cosmo':
        Hg = H_poly(b['zgrid'], theta['c'])
        Hcc = H_poly(b['cc'][0], theta['c']) if use_cc else None
        Huni = H_poly(b['z_uni'], theta['c'])
    else:
        Hg = H_lcdm(b['zgrid'], theta['H0'], theta['Om'])
        Hcc = H_lcdm(b['cc'][0], theta['H0'], theta['Om']) if use_cc else None
        Huni = H_lcdm(b['z_uni'], theta['H0'], theta['Om'])
    Hg = np.clip(Hg, H_FLOOR, None)

    invH = 1.0 / Hg
    DC = np.concatenate([[0.0], np.cumsum(0.5 * (invH[1:] + invH[:-1])
                                          * np.diff(b['zgrid']))]) * BASELINE['C_KM_S']
    m_pred = sn_apparent_mag(theta['M'], b['zhel_sn'],
                             np.interp(b['z_sn'], b['zgrid'], DC))
    dvec = np.atleast_1d(np.asarray(theta.get('Delta', 0.0), float))
    if u_leg is not None and dvec.size and np.any(dvec != 0.0):
        U = np.asarray(u_leg, float)
        U = U if U.ndim == 2 else U[:, None]
        if U.shape[1] != dvec.size:
            raise ValueError('chi2_blocks: %d offset column(s) against %d Delta value(s)'
                             % (U.shape[1], dvec.size))
        m_pred = m_pred + U @ dvec
    r_sn = b['m_sn'] - m_pred
    chi2_sn = float(r_sn @ np.linalg.solve(b['cov_sn'], r_sn))

    DM = np.interp(b['z_uni'], b['zgrid'], DC)
    DH = BASELINE['C_KM_S'] / Huni
    DV = (b['z_uni'] * DM ** 2 * DH) ** (1.0 / 3.0)
    col, is_dm, is_dh, _is_dv = b['sel']
    D = np.where(is_dm, DM[col], np.where(is_dh, DH[col], DV[col]))
    r_b = b['O'] - D / theta['rd']
    chi2_bao = float(r_b @ np.linalg.solve(b['cov_bao'], r_b))

    chi2_cc = 0.0
    if use_cc:
        r_c = b['cc'][1] - Hcc
        chi2_cc = float(r_c @ np.linalg.solve(b['cc'][2], r_c))

    return dict(sn=chi2_sn, bao=chi2_bao, cc=chi2_cc,
                tot=chi2_sn + chi2_bao + chi2_cc)


# =============================================================================
# 10.  The prior-free chi2 minimum, and the order-selection protocol
# =============================================================================
# The order decision must not inherit the coefficient box, or "which order the data ask
# for" becomes "which order the prior tolerates".  So the BIC is built on the
# UNCONSTRAINED chi2 minimum: no bounds, no prior term, only the three data blocks and the
# Planck r_d measurement where configuration A imports it.
from scipy.linalg import cho_factor, cho_solve                      # noqa: E402
from scipy.optimize import least_squares, minimize                  # noqa: E402
from scipy.stats import chi2 as chi2_dist                           # noqa: E402


def make_chi2(sample, config, background, P=None, z_floor=0.0):
    """chi2(theta) for theta = [shape..., M, r_d], with the factorisations done once.

    shape = the P+1 coefficients c_0..c_P, or (H0, Om) for flat LCDM.
    Returns (chi2_fn, meta) where meta carries the data counts the BIC needs."""
    P = BASELINE['P'] if P is None else P
    use_cc = (config == 'B')
    rd_planck = (config == 'A')
    b = build_blocks(sample, use_cc, z_floor=z_floor)

    cho_sn = cho_factor(b['cov_sn'], lower=True)
    cho_bao = cho_factor(b['cov_bao'], lower=True)
    cho_cc = cho_factor(b['cc'][2], lower=True) if use_cc else None
    z_cc = b['cc'][0] if use_cc else np.zeros(0)
    H_cc = b['cc'][1] if use_cc else np.zeros(0)
    zgrid, dz = b['zgrid'], np.diff(b['zgrid'])
    col, is_dm, is_dh, _ = b['sel']
    # HELIOCENTRIC, not z_HD -- the integral below is the one evaluated at z_HD.
    # See sn_apparent_mag; this is the same expression with the log10 hoisted out of the
    # inner loop, which the minimiser calls tens of thousands of times.
    log1pz = np.log10(1.0 + b['zhel_sn'])

    if background == 'cosmo':
        n_shape = P + 1
        ypow_g = np.power(y_of_z(zgrid)[:, None], np.arange(P + 1)[None, :])
        ypow_b = np.power(y_of_z(b['z_uni'])[:, None], np.arange(P + 1)[None, :])
        ypow_c = np.power(y_of_z(z_cc)[:, None], np.arange(P + 1)[None, :]) if use_cc \
            else None

        def H_of(sh):
            return ypow_g @ sh, ypow_b @ sh, (ypow_c @ sh if use_cc else None)
    else:
        n_shape = 2

        def H_of(sh):
            H0, Om = sh
            f = lambda z: H0 * np.sqrt(Om * (1.0 + z) ** 3 + 1.0 - Om)   # noqa: E731
            return f(zgrid), f(b['z_uni']), (f(z_cc) if use_cc else None)

    # The WHITENED RESIDUAL, not just the scalar chi2.  chi2 is a sum of squares, so
    # handing a minimiser the vector instead of the scalar lets it use the Gauss-Newton
    # structure: Levenberg-Marquardt converges in tens of iterations where Nelder-Mead on
    # the scalar needs tens of thousands, which is the difference between 10 seconds and
    # 40 minutes per order.  chi2(theta) == resid(theta) @ resid(theta) by construction,
    # and test_repo.py asserts it.
    L_sn = np.linalg.cholesky(b['cov_sn'] + 1e-9 * np.eye(len(b['z_sn'])))
    L_bao = np.linalg.cholesky(b['cov_bao'] + 1e-12 * np.eye(len(b['O'])))
    L_cc = np.linalg.cholesky(b['cc'][2] + 1e-9 * np.eye(len(z_cc))) if use_cc else None
    from scipy.linalg import solve_triangular

    def resid(theta):
        sh, M, rd = theta[:n_shape], theta[n_shape], theta[n_shape + 1]
        Hg, Hb, Hc = H_of(sh)
        n_out = (len(b['z_sn']) + len(b['O']) + (len(z_cc) if use_cc else 0)
                 + (1 if rd_planck else 0))
        if np.any(Hg <= H_FLOOR) or rd <= 0 or not np.all(np.isfinite(Hg)):
            return np.full(n_out, 1e6)
        invH = 1.0 / Hg
        DC = np.concatenate([[0.0], np.cumsum(0.5 * (invH[1:] + invH[:-1]) * dz)]) \
            * BASELINE['C_KM_S']
        r_sn = b['m_sn'] - (M + 5.0 * (log1pz + np.log10(
            np.interp(b['z_sn'], zgrid, DC))) + 25.0)
        DM = np.interp(b['z_uni'], zgrid, DC)
        DH = BASELINE['C_KM_S'] / Hb
        DV = (b['z_uni'] * DM ** 2 * DH) ** (1.0 / 3.0)
        r_b = b['O'] - np.where(is_dm, DM[col], np.where(is_dh, DH[col], DV[col])) / rd
        parts = [solve_triangular(L_sn, r_sn, lower=True),
                 solve_triangular(L_bao, r_b, lower=True)]
        if use_cc:
            parts.append(solve_triangular(L_cc, H_cc - Hc, lower=True))
        if rd_planck:
            parts.append(np.array([(rd - BASELINE['RD_PLANCK']) / BASELINE['SIG_RD']]))
        return np.concatenate(parts)

    def chi2(theta):
        sh, M, rd = theta[:n_shape], theta[n_shape], theta[n_shape + 1]
        Hg, Hb, Hc = H_of(sh)
        if np.any(Hg <= H_FLOOR) or rd <= 0:
            return 1e12
        invH = 1.0 / Hg
        DC = np.concatenate([[0.0], np.cumsum(0.5 * (invH[1:] + invH[:-1]) * dz)]) \
            * BASELINE['C_KM_S']
        r_sn = b['m_sn'] - (M + 5.0 * (log1pz + np.log10(
            np.interp(b['z_sn'], zgrid, DC))) + 25.0)
        DM = np.interp(b['z_uni'], zgrid, DC)
        DH = BASELINE['C_KM_S'] / Hb
        DV = (b['z_uni'] * DM ** 2 * DH) ** (1.0 / 3.0)
        r_b = b['O'] - np.where(is_dm, DM[col], np.where(is_dh, DH[col], DV[col])) / rd
        out = float(r_sn @ cho_solve(cho_sn, r_sn)) + float(r_b @ cho_solve(cho_bao, r_b))
        if use_cc:
            r_c = H_cc - Hc
            out += float(r_c @ cho_solve(cho_cc, r_c))
        if rd_planck:
            out += ((rd - BASELINE['RD_PLANCK']) / BASELINE['SIG_RD']) ** 2
        return out

    meta = dict(n_shape=n_shape, n_sn=len(b['z_sn']), n_bao=len(b['O']),
                n_cc=len(z_cc), n_rd=1 if rd_planck else 0, blocks=b, resid=resid)
    return chi2, meta


def minimise_chi2(sample, config, background, P=None, theta0=None, quiet=True,
                  n_multistart=3):
    """The UNCONSTRAINED minimum.

    No bounds and no prior term: this is the statistic the BIC is defined on, and it is
    what makes the order decision prior-free.  Levenberg-Marquardt on the whitened
    residual, restarted from a few jittered seeds because the coefficient likelihood has
    flat directions and a single start can stall along one of them; the best of the
    restarts wins."""
    P = BASELINE['P'] if P is None else P
    f, meta = make_chi2(sample, config, background, P=P)
    resid = meta['resid']
    if theta0 is None:
        sh = (lcdm_coeffs(0.315, 68.0, P) if background == 'cosmo'
              else np.array([68.0, 0.30]))
        theta0 = np.concatenate([sh, [-19.40, BASELINE['RD_PLANCK']]])
    theta0 = np.asarray(theta0, float)
    best_f, best_x = np.inf, None
    rng = np.random.default_rng(0)
    failures = []
    for i in range(max(1, n_multistart)):
        t0 = theta0 if i == 0 else theta0 * (1.0 + 10.0 ** (-3 + i)
                                             * rng.standard_normal(len(theta0)))
        # Every failure is RECORDED and re-raised if no start survives.  They used to be
        # swallowed by a bare `except: continue`, so a cell in which every start threw
        # returned (inf, theta0) -- the initial guess, labelled as the chi2 minimum, with
        # BIC = inf propagating silently into the order decision.
        try:
            sol = least_squares(resid, t0, method='lm', max_nfev=50000,
                                xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except Exception as exc:
            failures.append('start %d raised %s: %s' % (i, type(exc).__name__, exc))
            continue
        if not sol.success:
            failures.append('start %d did not converge: %s' % (i, sol.message))
            continue
        x, fv = sol.x, float(sol.fun @ sol.fun)
        if not np.isfinite(fv) or not np.all(np.isfinite(x)):
            failures.append('start %d returned a non-finite solution' % i)
            continue
        # polish on the scalar: LM stops on its own tolerances, and a few Nelder-Mead
        # sweeps from there costs seconds and occasionally finds a little more
        pol = minimize(f, x, method='Nelder-Mead',
                       options=dict(maxiter=4000, maxfev=4000, xatol=1e-10, fatol=1e-9))
        if np.isfinite(pol.fun) and np.all(np.isfinite(pol.x)) and pol.fun < fv:
            x, fv = pol.x, float(pol.fun)
        if fv < best_f:
            best_f, best_x = fv, x
    if best_x is None or not np.isfinite(best_f):
        raise RuntimeError(
            'chi2 minimisation failed for sample=%r config=%r background=%r P=%r: no '
            'start converged to a finite minimum.\n  %s'
            % (sample, config, background, P, '\n  '.join(failures) or 'no diagnostics'))
    if not quiet:
        print('   chi2_min = %.4f' % best_f)
    return float(best_f), np.asarray(best_x), meta


# =============================================================================
# 11.  --task ladder   (Table 1)
# =============================================================================

# Array sharding.  Every cell-looping task filters through this ONE helper, so a slurm
# array cannot renumber one task's cells against another's: the cell NAME is derived from
# the cell itself, never from the shard index, which is how an earlier version silently
# clobbered half a grid when the array was narrowed.
SHARD, NTASKS = 0, 1


def _shard(items):
    # An out-of-range shard used to select NOTHING and exit 0, which sacct reports as
    # COMPLETED.  A job that cannot run must not look like a job that ran.
    if not (isinstance(NTASKS, int) and NTASKS >= 1):
        raise ValueError('NTASKS must be a positive integer, got %r' % (NTASKS,))
    if not (isinstance(SHARD, int) and 0 <= SHARD < NTASKS):
        raise ValueError('SHARD=%r is not in range(0, NTASKS=%r).  An out-of-range shard '
                         'runs no cells and exits 0, i.e. it looks like success.'
                         % (SHARD, NTASKS))
    return [x for i, x in enumerate(items) if i % NTASKS == SHARD]


def _is_sharded():
    """True when this process is one shard of a grid and must not write an AGGREGATE."""
    return NTASKS > 1


LADDER_CELLS = [(cfg, smp, bg)
                for cfg in ('A', 'B')
                for bg in ('cosmo', 'lcdm')
                for smp in ('des_only', 'full')]


def task_ladder(recompute=False, cells=None, quiet=False):
    """The eight cells of Table 1."""
    want = _shard(cells or LADDER_CELLS)
    rows = []
    for cfg, smp, bg in want:
        cell = '%s_%s_%s_P%d' % (cfg, smp, bg, BASELINE['P'])
        cached = None if recompute else cache_load('ladder', cell)
        if cached is not None:
            print('[ladder] %s  (cached)' % cell)
            rows.append(_unpack(cached))
            continue
        print('[ladder] %s  sampling...' % cell)
        r = fit(smp, cfg, bg, quiet=quiet)
        cache_save('ladder', cell, **_pack(r))
        rows.append(r)
    if NTASKS > 1 and len(rows) < len(LADDER_CELLS):
        print('\n[ladder] shard %d of %d wrote %d cell(s); the table is printed by a '
              'full (unsharded) run once every shard has landed.'
              % (SHARD, NTASKS, len(rows)))
    else:
        _print_ladder(rows)
    return rows


_ARRAY_KEYS = ('H0', 'rd', 'M', 'q0', 'Om', 'Delta', 'Delta_found', 'Delta_cfa',
               'Delta_diff', 'v', 'C', 'diverging')


def write_json_atomic(name, obj):
    """Write an aggregate result to RESULT_DIR, atomically.

    Aggregates are the one place a SHARD can destroy another shard's work: every AP array
    task used to json.dump its own single-row `out` list over the shared ap_table.json, so
    the last writer won and the file that shipped was whatever a later unsharded run
    happened to leave behind.  Aggregates are now assembled from validated per-cell caches
    by an unsharded pass; this makes the write itself atomic too, so a killed job leaves
    the previous file intact rather than a truncated one."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, name)
    tmp = path + '.tmp-%d' % os.getpid()
    try:
        with open(tmp, 'w') as fh:
            json.dump(obj, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _pack(r):
    """Split a fit result into npz-storable arrays plus a JSON scalar blob."""
    out = {k: np.asarray(v) for k, v in r.items() if k in _ARRAY_KEYS}
    out['scalars_json'] = json.dumps({k: v for k, v in r.items()
                                      if k not in _ARRAY_KEYS})
    return out


def _unpack(d):
    r = json.loads(str(d['scalars_json']))
    for k in _ARRAY_KEYS:
        if k in d:
            r[k] = d[k]
    return r


def _print_ladder(rows):
    print('\n%-4s %-9s %-6s %8s %8s %9s %8s %7s   %6s %7s %4s'
          % ('cfg', 'sample', 'model', 'H0', 'sig', 'r_d', 'M_des', 'q0',
             'chi2/dof', 'rhat', 'div'))
    for r in rows:
        print('%-4s %-9s %-6s %8.2f %8.2f %9.2f %8.3f %+7.2f   %6.2f %7.4f %4d'
              % (r['config'], r['sample'], r['background'], r['H0_mean'], r['H0_std'],
                 r['rd_mean'], r['M_mean'], r.get('q0_mean', float('nan')),
                 r['chi2_dof'], r['rhat_max'], r['divergences']))
    box = [r for r in rows if r.get('at_bound')]
    if box:
        print('\n*** THE ENVELOPE IS ACTIVE in %d cell(s): %s' % (len(box), ', '.join(
            '%s/%s %s' % (r['config'], r['sample'], r['at_bound']) for r in box)))
    else:
        near = min((min(r['env_margin'].values()) for r in rows if r.get('env_margin')),
                   default=float('nan'))
        print('\nenvelope never active: the posterior stays %.2f (in E) from it, 0.1%% '
              'quantile, at_bound empty' % near)


# =============================================================================
# 12.  --task floorsweep   (Figure 2)
# =============================================================================
# Only the DATA change: SNe below z_floor are dropped and the sample is refitted with the
# identical estimator.  A monotonic dH0/dz_floor would mean the lowest-z supernovae carry
# signal away from the fiducial; a flat trend means every sub-sample prefers the same
# expansion rate.
# The tail (0.180 to 0.300) is the part of the published figure's x-range that lies above
# this repo's original grid: without it the sweep stops at 0.150 and the manuscript float
# cannot be reproduced end to end.  Those four floors are the ONLY cells that still need
# sampling; everything at or below 0.150 is already cached.
FLOORS = (0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070,
          0.073, 0.080, 0.085, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150,
          0.180, 0.220, 0.260, 0.300)


# THE DELTA ARM.  The same sweep run with the legacy zero-point SAMPLED JOINTLY
# (`joint_delta=True`), which asks the question the fixed-zero-point arm cannot: is the
# legacy correction itself a property of the lowest-redshift supernovae?  One fit per floor
# yields both curves -- H0 with the offset marginalised, and Delta(z_floor) itself.
#
# 🚨 IT STOPS AT 0.085, AND NOT BECAUSE OF COST.  Delta is identified only where legacy SNe
# SURVIVE the cut, and the legacy sub-sample spans z = 0.0251-0.0870 (197 SNe).  A floor
# above 0.087 leaves `u_leg` ALL-ZERO: Delta then samples its U(-1, +1) prior, which would
# plot as a wide flat band that reads exactly like a measured null.  The surviving legacy
# count falls 197 -> 10 -> 3 -> 1 over floors 0.030 / 0.073 / 0.080 / 0.085, so the last
# three points are where the correction visibly stops being determined -- that is the
# informative end of the curve, not padding.  `n_leg` is cached with every cell so the
# figure can label it.
FLOORS_DELTA_MAX = 0.085


def floorsweep_grid():
    """(sample, z_floor, joint_delta) for every cell of the sweep.

    The Delta arm is APPENDED, so the 36 fixed-zero-point cells keep their positions.  It
    would not matter if they did not: a cell is named from its own (sample, floor, arm) and
    a cached cell is skipped whichever shard reaches it -- the shard index never enters a
    cell name."""
    grid = [(smp, fl, False) for smp in ('full', 'des_only')
            for fl in FLOORS if fl >= sn_dataset(smp)['z'].min() - 1e-9]
    zmin = sn_dataset('full')['z'].min() - 1e-9
    grid += [('full', fl, True) for fl in FLOORS
             if zmin <= fl <= FLOORS_DELTA_MAX + 1e-9]
    return grid


def task_floorsweep(recompute=False, quiet=True):
    rows = []
    for smp, fl, jd in _shard(floorsweep_grid()):
        cell = '%s%s_%.3f_P%d' % (smp, '_joint' if jd else '', fl, BASELINE['P'])
        cached = None if recompute else cache_load('floorsweep', cell)
        if cached is not None:
            print('[floorsweep] %s (cached)' % cell)
            rows.append(_unpack(cached))
            continue
        d = sn_dataset(smp)
        keep = d['z'] >= fl
        n = int(keep.sum())
        n_leg = int((keep & (d['idsurvey'] != DES_IDSURVEY)).sum())
        print('[floorsweep] %s  N=%d  N_leg=%d  sampling...' % (cell, n, n_leg))
        r = fit(smp, 'A', 'cosmo', z_floor=fl, joint_delta=jd, quiet=quiet)
        r['n_sn'] = n
        r['n_leg'] = n_leg
        cache_save('floorsweep', cell, **_pack(r))
        rows.append(r)
    print('\n%-16s %7s %6s %5s %8s %8s %8s %8s %9s %9s %8s'
          % ('cell', 'z_floor', 'N', 'N_leg', 'H0', 'sig', 'q0', 'sig', 'M_des',
             'Delta', 'sig'))
    for r in rows:
        tag = r['sample'] + ('+joint' if r.get('joint_delta') else '')
        dl = (('%9.4f %8.4f' % (r['Delta_mean'], r['Delta_std']))
              if 'Delta_mean' in r else '%9s %8s' % ('-', '-'))
        print('%-16s %7.3f %6d %5d %8.2f %8.2f %+8.2f %8.2f %9.4f %s'
              % (tag, r['z_floor'], r.get('n_sn', -1), r.get('n_leg', -1),
                 r['H0_mean'], r['H0_std'], r['q0_mean'], r['q0_std'], r['M_mean'], dl))
    return rows


# =============================================================================
# 13.  --task bic   (order selection)
# =============================================================================
def task_bic(plist=(2, 3, 4, 5, 6, 7, 8, 9, 10), recompute=False, check_box=False):
    """BIC(P) = chi2_min(P) + k ln N,  k = (P+1) + 2,  minimised over P.

    Two properties are deliberate.  The chi2 is minimised DIRECTLY rather than read at the
    posterior mean, because the comparison between orders is nested only at the minimum --
    chi2(E[theta]) is not E[chi2(theta)] and is not chi2_min, and a parameter-counting
    criterion evaluated off the minimum produced "improvements" of +0.0 from adding a
    parameter in an earlier version of this analysis.  And each order is SEEDED from the
    solution of the previous one with c_P = 0, so chi2_min(P) <= chi2_min(P-1) holds by
    construction; the run asserts it rather than hoping."""
    plist = [int(v) for v in plist]
    if len(plist) < 2:
        raise ValueError('--plist needs at least two orders to compare, got %r' % (plist,))
    if any(b <= a for a, b in zip(plist, plist[1:])):
        raise ValueError('--plist must be STRICTLY INCREASING (each order is seeded from '
                         'the previous one with the new coefficient at zero, which is what '
                         'makes the comparison nested); got %r' % (plist,))
    if plist[0] < 1:
        raise ValueError('--plist orders must be >= 1, got %r' % (plist,))
    out = {}
    for cfg in ('A', 'B'):
        for smp in ('des_only', 'full'):
            key = '%s_%s' % (cfg, smp)
            cell = 'chi2min_%s' % key
            cached = None if recompute else cache_load('bic', cell, ignore_p=True,
                                                       ignore=_SAMPLER_KEYS)
            if cached is not None and set(json.loads(str(cached['plist_json']))) >= set(plist):
                print('[bic] %s (cached)' % key)
                cachedrows = json.loads(str(cached['rows_json']))
                # a cached SUPERSET is reusable, but only after being cut down to what was
                # asked for -- returning every stored order under a narrower --plist made
                # the request look honoured when it was not
                want = set(plist)
                out[key] = dict(cachedrows,
                                rows=[r for r in cachedrows['rows'] if r['P'] in want])
                continue
            rows, theta, P_prev = [], None, None
            for P in plist:
                if theta is not None:
                    # seed order P from the solution of P_prev with the new coefficients
                    # set to zero, so the higher order CONTAINS the lower one and
                    # chi2_min(P) <= chi2_min(P_prev) holds by construction
                    theta = np.concatenate([theta[:P_prev + 1],
                                            np.zeros(P - P_prev), theta[P_prev + 1:]])
                c2, theta, meta = minimise_chi2(smp, cfg, 'cosmo', P=P, theta0=theta)
                P_prev = P
                N = meta['n_sn'] + meta['n_bao'] + meta['n_cc'] + meta['n_rd']
                k = (P + 1) + 2
                rows.append(dict(P=P, chi2_min=c2, k=k, N=N, lnN=float(np.log(N)),
                                 BIC=c2 + k * float(np.log(N)),
                                 theta=list(map(float, theta))))
                print('[bic] %s P=%d  chi2_min=%.3f  BIC=%.3f' % (key, P, c2, rows[-1]['BIC']))
            for a, b in zip(rows, rows[1:]):
                if b['chi2_min'] > a['chi2_min'] + 1e-3:
                    # the docstring says this is asserted, and now it is.  It was printed,
                    # which on a --quiet cluster job is indistinguishable from not
                    # happening, and the cell was cached and used regardless.
                    raise RuntimeError(
                        'NESTING VIOLATED at %s P=%d->%d: chi2_min rose by %.3f.  The '
                        'higher order CONTAINS the lower one with c_P = 0, so chi2_min '
                        'cannot rise; this is a failed minimisation, not a result.'
                        % (key, a['P'], b['P'], b['chi2_min'] - a['chi2_min']))
            # flat LCDM on the same footing, for the caveat that must travel with the choice
            c2l, _tl, ml = minimise_chi2(smp, cfg, 'lcdm')
            Nl = ml['n_sn'] + ml['n_bao'] + ml['n_cc'] + ml['n_rd']
            lcdm = dict(chi2_min=c2l, k=4, N=Nl, BIC=c2l + 4 * float(np.log(Nl)))
            out[key] = dict(rows=rows, lcdm=lcdm)
            cache_save('bic', cell, plist_json=json.dumps(list(plist)),
                       rows_json=json.dumps(out[key]))
    _print_bic(out)
    if check_box:
        _print_box_check(out)
    return out


def _print_bic(out):
    keys = sorted(out)
    print('\nDelta BIC per step (negative = the order is worth its parameter)')
    print('%-8s %s' % ('step', ' '.join('%10s' % k for k in keys)))
    plist = [r['P'] for r in out[keys[0]]['rows']]
    for a, b in zip(plist, plist[1:]):
        cells = []
        for k in keys:
            rows = {r['P']: r for r in out[k]['rows']}
            cells.append('%+10.1f' % (rows[b]['BIC'] - rows[a]['BIC']))
        print('%-8s %s' % ('%d->%d' % (a, b), ' '.join(cells)))
    print('\n%-8s %s' % ('BIC min', ' '.join(
        '%10d' % min(out[k]['rows'], key=lambda r: r['BIC'])['P'] for k in keys)))
    print('%-8s %s' % ('vs LCDM', ' '.join(
        '%+10.1f' % (min(r['BIC'] for r in out[k]['rows']) - out[k]['lcdm']['BIC'])
        for k in keys)))
    print('\nThe last row is the caveat that must travel with the choice: flat LCDM beats\n'
          'every polynomial order, which is what four free parameters against seven should\n'
          'do.  It is not evidence for LCDM over the expansion -- the expansion is not\n'
          'offered as a rival model of the data.  Its purpose is to leave the shape below\n'
          'the survey floor free, and that freedom is exactly what the criterion penalises.')


def _print_box_check(out, used=(2, 3, 4, 5, 6)):
    """Whether every least-squares best fit lies inside the envelope.

    The chi2 minimum does not depend on the chart, so the BIC sweep, the Schwarz blend and
    the least-squares mock nulls carry over to the envelope prior UNCHANGED exactly when
    each best fit lies inside the prior's support.  An order outside `used` is reported
    but not held against the sweep: it enters no result."""
    print('\nEnvelope margins of each least-squares best fit, min_y(E - lower) and '
          'min_y(upper - E), y_max = %g:' % BASELINE['Y_MAX'])
    bad = []
    for key in sorted(out):
        for r in out[key]['rows']:
            Pk = r['P']
            c = np.asarray(r['theta'][:Pk + 1], float)
            a, b = envelope_margins(c / c[0])
            inside = bool(a[0] >= 0.0 and b[0] >= 0.0)
            if not inside and Pk in used:
                bad.append('%s P=%d' % (key, Pk))
            print('  %-12s P=%-2d  %7.3f %7.3f  %s' % (key, Pk, a[0], b[0], '' if inside else
                                                    ('OUTSIDE' if Pk in used
                                                     else 'outside (order not used)')))
    if bad:
        raise SystemExit('least-squares best fit OUTSIDE the envelope at %s: those '
                         'results do NOT carry over to the new prior unchanged'
                         % ', '.join(bad))
    print('every order in %s is inside the envelope, so its least-squares results carry '
          'over to the envelope prior unchanged' % (list(used),))


# =============================================================================
# 14.  --task legacy   (Table 3, Figure 3)
# =============================================================================
# Two questions, and they need different machinery.
#
#   (a) DO the legacy supernovae follow the DES distance-redshift relation?  Fit the SHAPE
#       of the DES Hubble diagram with a rigid one-parameter family, extrapolate below the
#       DES floor to the legacy redshifts, and take Delta = the GLS mean of the residuals
#       under the full STAT+SYS covariance including the legacy-DES cross-block.  This is
#       prior-free least squares, no sampler.
#
#       It is calibration-free BY CONSTRUCTION, and that is what makes Delta comparable
#       across the four references: rescaling H -> lambda H sends D_C -> D_C/lambda and
#       hence m -> m - 5 log10 lambda, so (c_k -> lambda c_k, M -> M + 5 log10 lambda) is
#       an EXACT flat direction.  Fixing c_0 = 70 as a gauge and letting M float leaves
#       every prediction m(z*) invariant, so no absolute scale, no sound horizon and no
#       distance ladder enters Delta.
#
#       The redshift geometry makes this a DOWNWARD EXTRAPOLATION, not an in-overlap
#       comparison: legacy spans z in [0.025, 0.087] and DES begins at 0.073, so about two
#       DES supernovae lie in the populated overlap.  sigma_Delta is therefore dominated
#       by how much freedom the DES reference retains below its own floor, which is why
#       the four references agree on WHAT the offset is and disagree only on how
#       significant it is.
#
#   (b) WHAT does it do to H0?  Not by adding Delta to the legacy magnitudes afterwards --
#       Delta and H0 are 92% correlated, so a two-stage correction produces error bars
#       that cannot know about the covariance they inherit.  Delta is sampled INSIDE the
#       ladder instead (`fit(..., joint_delta=True)`).
H0_GAUGE = 70.0
LEGACY_RUNGS = ('cosmo', 'cosmo_fap', 'lcdm', 'lcdm_fap')
RUNG_LABEL = {
    'cosmo': 'cosmography, SN shape only',
    'cosmo_fap': 'cosmography + DESI F_AP (r_d-free)',
    'lcdm': 'rigid LCDM (Om free), SN shape only',
    'lcdm_fap': 'rigid LCDM + DESI F_AP (r_d-free)',
}
FAP_ZMAX = 1.965          # the reach over which the y-series stays convergent


def desi_fap():
    """DESI F_AP = D_M/D_H and its sigma, by linear propagation of the published 2x2.

    BOTH DESI entries are D/r_d, so r_d cancels EXACTLY in the ratio: this constraint is
    sound-horizon-free and calibration-free, which is the whole point of using it."""
    z_list = [z for z in sorted(DESI_DM) if z <= FAP_ZMAX]
    F, S = [], []
    for z in z_list:
        dm, dh = DESI_DM[z][0], DESI_DH[z][0]
        g = np.array([1.0 / dh, -dm / dh ** 2])
        F.append(dm / dh)
        S.append(float(np.sqrt(g @ np.array(DESI_DMDH_COV[z]) @ g)))
    return np.array(z_list), np.array(F), np.array(S)


def legacy_blocks():
    """The legacy/DES split of the full sample with every covariance block the test needs."""
    d = sn_dataset('full')
    leg = d['idsurvey'] != DES_IDSURVEY
    des = ~leg
    return dict(z_leg=d['z'][leg], zhel_leg=d['zhel'][leg], ids_leg=d['idsurvey'][leg],
                m_leg=d['m'][leg],
                z_des=d['z'][des], zhel_des=d['zhel'][des], m_des=d['m'][des],
                C_leg=d['cov'][np.ix_(leg, leg)], C_des=d['cov'][np.ix_(des, des)],
                C_legdes=d['cov'][np.ix_(leg, des)])


def _rung_H(shape, theta, zgrid, P):
    """(H on the grid, M) for a shape family.  c_0 is the gauge and is NOT a parameter."""
    if shape.startswith('lcdm'):
        Om, M = theta[0], theta[1]
        return H0_GAUGE * np.sqrt(Om * (1.0 + zgrid) ** 3 + 1.0 - Om), M
    c = np.concatenate([[H0_GAUGE], theta[:P]])
    return H_poly(zgrid, c), theta[P]


def _rung_m(shape, theta, zstar, zhelstar, zgrid, P):
    """Model magnitude at (z_HD, z_HEL) = (zstar, zhelstar) -- the SAME objects, paired.

    Both are required rather than defaulted, so a caller cannot pass one array and
    silently get the z_HD-twice model back.  See `sn_apparent_mag`."""
    Hg, M = _rung_H(shape, theta, zgrid, P)
    Hg = np.clip(Hg, H_FLOOR, None)
    invH = 1.0 / Hg
    DC = np.concatenate([[0.0], np.cumsum(0.5 * (invH[1:] + invH[:-1])
                                          * np.diff(zgrid))]) * BASELINE['C_KM_S']
    return sn_apparent_mag(M, zhelstar, np.interp(zstar, zgrid, DC))


def _rung_fap_resid(shape, theta, zgrid, P):
    if not shape.endswith('_fap'):
        return None
    zf, obs, sig = desi_fap()
    Hg, _ = _rung_H(shape, theta, zgrid, P)
    Hg = np.clip(Hg, H_FLOOR, None)
    invH = 1.0 / Hg
    DC = np.concatenate([[0.0], np.cumsum(0.5 * (invH[1:] + invH[:-1])
                                          * np.diff(zgrid))]) * BASELINE['C_KM_S']
    DM = np.interp(zf, zgrid, DC)
    DH = BASELINE['C_KM_S'] / np.interp(zf, zgrid, Hg)
    return (obs - DM / DH) / sig


def _fd_jac(f, theta, step=1e-5):
    base = np.atleast_1d(f(theta))
    J = np.empty((len(base), len(theta)))
    for k in range(len(theta)):
        h = step * max(abs(theta[k]), 1.0)
        tp = np.array(theta, float)
        tp[k] += h
        J[:, k] = (np.atleast_1d(f(tp)) - base) / h
    return J


def fit_des_shape(shape, P=None):
    """Whitened least squares of one shape family to the DES supernovae alone."""
    P = BASELINE['P'] if P is None else P
    b = legacy_blocks()
    zgrid = np.linspace(0.0, max(float(b['z_des'].max()) * 1.05, FAP_ZMAX + 0.05),
                        BASELINE['NGRID'])
    L = np.linalg.cholesky(b['C_des'] + 1e-10 * np.eye(len(b['z_des'])))
    from scipy.linalg import solve_triangular

    theta0 = (np.array([0.30, -19.4]) if shape.startswith('lcdm')
              else np.concatenate([np.zeros(P), [-19.4]]))

    def r_sn(th):
        return solve_triangular(L, b['m_des'] - _rung_m(shape, th, b['z_des'],
                                                       b['zhel_des'], zgrid, P),
                                lower=True)

    def r_all(th):
        ex = _rung_fap_resid(shape, th, zgrid, P)
        return r_sn(th) if ex is None else np.concatenate([r_sn(th), ex])

    sol = least_squares(r_all, theta0, method='lm', max_nfev=20000)
    # Jacobians built with OUR differencing, not least_squares' internal one: mixing the
    # two breaks the M_eff <-> P_sn identity asserted below.
    J = _fd_jac(lambda th: _rung_m(shape, th, b['z_des'], b['zhel_des'], zgrid, P),
                sol.x)
    Jw = solve_triangular(L, J, lower=True)
    A = Jw.T @ Jw
    n_ex = 0
    if _rung_fap_resid(shape, sol.x, zgrid, P) is not None:
        Jx = _fd_jac(lambda th: _rung_fap_resid(shape, th, zgrid, P), sol.x)
        A = A + Jx.T @ Jx                                # already whitened
        n_ex = Jx.shape[0]
    # chi2 and dof MUST describe the same residual vector.  They did not: chi2 counted
    # r_sn alone while dof added the FAP points, so the F_AP rungs reported a numerator
    # and a denominator for different datasets.  Both pairs are returned now, named.
    r_full = r_all(sol.x)
    r_snx = r_sn(sol.x)
    return dict(shape=shape, P=P, theta=sol.x, cov=np.linalg.inv(A), zgrid=zgrid, L=L,
                J=J, A_sn=Jw.T @ Jw,
                chi2=float(r_full @ r_full), dof=len(r_full) - len(sol.x),
                chi2_sn=float(r_snx @ r_snx), dof_sn=len(r_snx) - len(sol.x),
                n_extra=n_ex,
                cond=float(np.linalg.cond(A)), blocks=b)


def delta_residual(shape, P=None, with_cross=True):
    """r = m_legacy - m_DES_pred(z_legacy) and its covariance.

    The cross term is the legacy-DES shared systematic, propagated through the EXACT
    linear map M_eff = d m_pred / d m_DES.  It is checked, not assumed: propagating the
    DES covariance through M_eff must reproduce the SN-sourced part of the predictive
    covariance identically, and the run raises if it does not."""
    from scipy.linalg import solve_triangular
    f = fit_des_shape(shape, P=P)
    b, P = f['blocks'], f['P']
    m_pred = _rung_m(shape, f['theta'], b['z_leg'], b['zhel_leg'], f['zgrid'], P)
    Jstar = _fd_jac(lambda th: _rung_m(shape, th, b['z_leg'], b['zhel_leg'],
                                       f['zgrid'], P), f['theta'])
    P_pred = Jstar @ f['cov'] @ Jstar.T
    LiJ = solve_triangular(f['L'], f['J'], lower=True)
    Li = solve_triangular(f['L'], np.eye(f['L'].shape[0]), lower=True)
    M_eff = Jstar @ f['cov'] @ LiJ.T @ Li
    P_sn = Jstar @ (f['cov'] @ f['A_sn'] @ f['cov']) @ Jstar.T
    P_chk = M_eff @ b['C_des'] @ M_eff.T
    if not np.allclose(P_chk, P_sn, rtol=1e-3,
                       atol=1e-8 + 1e-3 * np.abs(P_sn).max()):
        raise AssertionError(
            'M_eff is inconsistent with the predictive covariance it must reproduce '
            '(max dev %.2e, cond %.1e).  The cross-covariance would be propagated through '
            'the wrong map.' % (np.abs(P_chk - P_sn).max(), f['cond']))
    C = 0.5 * (b['C_leg'] + P_pred + (b['C_leg'] + P_pred).T)
    if with_cross:
        X = b['C_legdes'] @ M_eff.T
        C = 0.5 * ((C - (X + X.T)) + (C - (X + X.T)).T)
    return dict(r=b['m_leg'] - m_pred, C=C, z_leg=b['z_leg'], ids_leg=b['ids_leg'],
                fit=f, sig_pred_lowz=float(np.sqrt(P_pred[np.argmin(b['z_leg']),
                                                          np.argmin(b['z_leg'])])))


def offset_stats(r, C):
    """Delta = the GLS-weighted mean offset, its sigma, and the shape chi2 about it."""
    W = np.linalg.inv(C)
    one = np.ones_like(r)
    A = one @ W @ one
    Delta = (one @ W @ r) / A
    res = r - Delta
    return dict(N=len(r), Delta=float(Delta), sig=float(1.0 / np.sqrt(A)),
                z=float(Delta * np.sqrt(A)),
                chi2_shape=float(res @ W @ res), dof_shape=len(r) - 1)


def split_contrast(r, C, sel):
    """Offset of a subsample MINUS the offset of the rest, both fitted AT ONCE.

    r = V beta + noise with V = [1_S, 1_rest]; GLS gives beta and its 2x2 covariance, and
    the contrast is beta_0 - beta_1 with sigma^2 = C00 + C11 - 2*C01.

    This is NOT the difference of the two separately-weighted means.  Each ABSOLUTE Delta
    is dominated by the DES sub-floor extrapolation error, which is COMMON to the two
    groups; fitting both at once cancels that common mode exactly, while differencing two
    separate fits counts it twice and is not reference-independent.  p_same is the
    likelihood-ratio probability that one offset describes the same rows (nested K=1
    model on the identical r and C), so the question is settled on one dataset rather
    than by eye from two overlapping error bars.

    Reading the rows: the SURVEY splits (Foundation, CfA/CSP) span a common redshift
    range, so the DES reference cancels exactly out of their difference whatever family
    fitted it.  'z > 0.05' does NOT -- a redshift split leaves the two groups on
    different parts of the extrapolation, so its value depends on where the cut is
    placed.  Read that row as consistent with zero; do not quote its value.
    """
    from scipy.stats import chi2 as _chi2dist
    V = np.column_stack([sel.astype(float), (~sel).astype(float)])
    W = np.linalg.inv(C)
    cov = np.linalg.inv(V.T @ W @ V)
    beta = cov @ (V.T @ W @ r)
    res = r - V @ beta
    chi2_2 = float(res @ W @ res)
    one = np.ones_like(r)
    d1 = float((one @ W @ r) / (one @ W @ one))
    chi2_1 = float((r - d1) @ W @ (r - d1))
    diff = float(beta[0] - beta[1])
    sig = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1], 0.0)))
    return dict(N=int(sel.sum()), N_rest=int((~sel).sum()), diff=diff, sig=sig,
                z=float(diff / sig) if sig > 0 else float('nan'),
                p_same=float(_chi2dist.sf(chi2_1 - chi2_2, 1)))


LEGACY_SPLITS = {
    'all legacy': lambda z, ids: np.ones(len(z), bool),
    'Foundation': lambda z, ids: ids == FOUNDATION_IDSURVEY,
    'CfA/CSP': lambda z, ids: np.isin(ids, list(CFA_CSP_IDSURVEY)),
    'z > 0.05': lambda z, ids: z > 0.05,
    'z >= DES floor': lambda z, ids: z >= 0.073,
}


def task_legacy(recompute=False, quiet=True):
    print('\n(a) the Delta ladder -- prior-free least squares, no sampler\n')
    print('%-34s %10s %10s %7s %9s %9s %9s'
          % ('reference for the DES relation', 'Delta', 'sigma', 'signif',
             'shape X2/n', 'DESref X2/n', 'sig[m(0.025)]'))
    ladder = {}
    for rung in LEGACY_RUNGS:
        b = delta_residual(rung)
        st = offset_stats(b['r'], b['C'])
        st['label'] = RUNG_LABEL[rung]
        # TWO different statistics, and they were interchanged.  `des_ref` is the
        # quality of the DES reference fit -- the curve the legacy SNe are compared
        # AGAINST.  `legacy_shape` is the quality of the legacy residuals about their own
        # fitted offset, which is the statistic this section is actually about, and the
        # one that says whether the legacy SNe depart from the DES curve in SHAPE as well
        # as in normalisation.  Reporting the first under the second's name made the
        # legacy residuals look better than they are (0.92 against 1.05).  Never one
        # ambiguous `chi2_dof` again.
        st['des_ref_chi2_dof'] = b['fit']['chi2'] / b['fit']['dof']
        st['legacy_shape_chi2_dof'] = st['chi2_shape'] / st['dof_shape']
        st['sig_pred_lowz'] = b['sig_pred_lowz']
        st['splits'] = {}
        st['contrasts'] = {}
        for nm, fn in LEGACY_SPLITS.items():
            sel = np.asarray(fn(b['z_leg'], b['ids_leg']), bool)
            if sel.sum() >= 5:
                st['splits'][nm] = offset_stats(b['r'][sel], b['C'][np.ix_(sel, sel)])
            # 'all legacy' is the reference, not a row: it has no complement to fit
            # against, so it gets an absolute Delta and no contrast.
            if 5 <= sel.sum() < len(sel) - 4:
                st['contrasts'][nm] = split_contrast(b['r'], b['C'], sel)
        ladder[rung] = st
        print('%-34s %+10.4f %10.4f %7.1f %9.3f %11.3f %9.3f'
              % (RUNG_LABEL[rung], st['Delta'], st['sig'], st['z'],
                 st['legacy_shape_chi2_dof'], st['des_ref_chi2_dof'],
                 st['sig_pred_lowz']))
    print('\nThe four references agree on WHAT the offset is and disagree only on how\n'
          'significant it is.  The spread is the price of extrapolating from z=0.073 down\n'
          'to z=0.025, i.e. it is bought by the assumed functional form below the DES\n'
          'floor and not by the data.  The LEGACY SHAPE chi2/dof is ~1.05 for every\n'
          'reference -- an acceptable fit for 196 dof -- so the legacy supernovae follow\n'
          'the DES curve and depart from it in normalisation.  Do not quote the DES\n'
          'reference column in its place: that one is ~0.92 and describes a different\n'
          'fit to a different dataset.')
    # the Delta ladder is prior-free least squares over ALL four rungs in one pass, so
    # it is never partial -- but it is still an aggregate, and aggregates are written
    # atomically and by unsharded runs only.
    if not _is_sharded():
        write_json_atomic('legacy_delta_ladder.json', ladder)
    else:
        print('[legacy] sharded run: legacy_delta_ladder.json left to the unsharded pass')

    print('\n(b) the legacy zero-point sampled jointly with the ladder\n')
    rows = []
    grid = [(bg, smp, jd, label)
            for bg in ('cosmo', 'lcdm')
            for smp, jd, label in (('des_only', False, 'legacy excluded (DES only)'),
                                   ('full', False, 'uncorrected'),
                                   ('full', True, 'legacy zero-point sampled'))]
    for bg, smp, jd, label in _shard(grid):
        cell = '%s_%s_%s_P%d' % (bg, smp, 'joint' if jd else 'fixed', BASELINE['P'])
        cached = None if recompute else cache_load('legacy', cell)
        if cached is not None:
            print('[legacy] %s (cached)' % cell)
            r = _unpack(cached)
        else:
            print('[legacy] %s sampling...' % cell)
            r = fit(smp, 'A', bg, joint_delta=jd, quiet=quiet)
            cache_save('legacy', cell, **_pack(r))
        r['label'] = label
        rows.append(r)
    if NTASKS > 1 and len(rows) < 6:
        print('\n[legacy] shard %d of %d wrote %d cell(s); the table is printed by a '
              'full (unsharded) run once every shard has landed.'
              % (SHARD, NTASKS, len(rows)))
        return ladder, rows
    print('\n%-30s %20s %20s' % ('case', 'H0 (cosmography)', 'H0 (LCDM)'))
    for label in ('legacy excluded (DES only)', 'uncorrected',
                  'legacy zero-point sampled'):
        pair = [r for r in rows if r['label'] == label]
        c = next(r for r in pair if r['background'] == 'cosmo')
        l = next(r for r in pair if r['background'] == 'lcdm')
        print('%-30s %12.2f +- %-5.2f %12.2f +- %-5.2f'
              % (label, c['H0_mean'], c['H0_std'], l['H0_mean'], l['H0_std']))
        if 'Delta_mean' in c:
            print('%-30s %12.4f +- %-5.4f %12.4f +- %-5.4f'
                  % ('   Delta [mag]', c['Delta_mean'], c['Delta_std'],
                     l['Delta_mean'], l['Delta_std']))
            print('%-30s %12.2f %19.2f'
                  % ('   corr(Delta, H0)', c['corr_Delta_H0'], l['corr_Delta_H0']))
            print('%-30s %12.2f %19.2f'
                  % ('   corr(Delta, M_des)', c['corr_Delta_M'], l['corr_Delta_M']))
    print('\ncorr(Delta, H0) is why the offset had to be sampled jointly rather than\n'
          'applied as a correction, and corr(Delta, M_des) is why M_des is immune to it:\n'
          'M_des is pinned by the BAO at z = 0.29-2.33, far above any legacy redshift.')
    return ladder, rows


# =============================================================================
# 15.  --task ap   (Table 4, Figure 4)
# =============================================================================
# The consistency test.  Using the supernovae ALONE, predict the F_AP = D_M/D_H that DESI
# measures at its three in-range redshifts, assuming only that the SN distance-redshift
# relation is smooth.
#
# It carries no absolute calibration and no sound horizon on EITHER side: H -> lambda H
# scales D_M and D_H together so F_AP is invariant, and both DESI entries are D/r_d so r_d
# cancels in the ratio.  The test therefore probes the SHAPE of the reconstruction
# independently of the calibration and of H0.
#
# Only the three in-range redshifts are used, deliberately.  Beyond the SN z_max ~ 1.14
# the prediction is an extrapolation in which sigma(D_H) exceeds D_H itself, so F_AP
# diverges and a chi2 of ~0 measures a LACK of predictive power rather than agreement.
def sn_only_model(z_sn, zhel_sn, m_sn, L_sn, zgrid, P, M_fixed, y_max):
    """The SN shape alone.  M is FIXED and H0 free -- the gauge choice is immaterial
    because every quantity read off this fit is scale-invariant.  The shape carries the
    envelope prior enforced up to `y_max` = AP_Y_MAX, the supernovae's own reach, not the
    ladder's Y_MAX -- see BASELINE."""
    H0, C = sample_coeffs(P, y_max)

    def H_of(z):
        y = z / (1.0 + z)
        return H0 * (jnp.power(y[:, None], jnp.arange(C.shape[0])[None, :]) @ C)

    Hg, pen = _h_positive(H_of(zgrid), zgrid)
    numpyro.factor('h_positive', pen)
    invH = 1.0 / Hg
    DC = jnp.concatenate([jnp.zeros(1),
                          jnp.cumsum(0.5 * (invH[1:] + invH[:-1])
                                     * jnp.diff(zgrid))]) * BASELINE['C_KM_S']
    m_model = sn_apparent_mag(M_fixed, zhel_sn, jnp.interp(z_sn, zgrid, DC), xp=jnp)
    numpyro.sample('sn', dist.MultivariateNormal(loc=m_model, scale_tril=L_sn), obs=m_sn)


def ap_coefficients(d):
    """(ndraw, P+1) DIMENSIONAL coefficients H0 (1, C_1..C_P) of an AP cell.

    F_AP is invariant under H -> lambda H, so any positive scale would do for the ratio --
    but NOT for the H_FLOOR clip the distance integral applies.  On the shape alone
    (H0 = 1) that clip at 1 km/s/Mpc would cut every draw with E < 1, which is every draw
    with q0 < -1 at low z, DES alone included.  Hence each draw keeps its own H0."""
    H0 = np.asarray(d['H0'], float)
    return H0[:, None] * np.column_stack([np.ones_like(H0), np.asarray(d['C'], float)])


def task_ap(recompute=False, quiet=True):
    zf, fap_d, sig_d = desi_fap()
    keep = np.isin(zf, AP_REDSHIFTS)
    zc, fap_d, sig_d = zf[keep], fap_d[keep], sig_d[keep]
    P = BASELINE['P']
    AP_CELLS = ['des_only', 'full']
    out = []
    for smp in _shard(AP_CELLS):
        cell = '%s_P%d' % (smp, P)
        cached = None if recompute else cache_load('ap', cell)
        if cached is not None:
            print('[ap] %s (cached)' % cell)
            c_draws = ap_coefficients(cached)
        else:
            print('[ap] %s sampling...' % cell)
            z_sn, zhel_sn, m_sn, cov_sn = sn_block(smp)
            L = np.linalg.cholesky(cov_sn + 1e-9 * np.eye(len(z_sn)))
            zgrid = np.linspace(0.0, float(z_sn.max()) + 1e-6, BASELINE['NGRID'])
            y_ap = shape_y_max('ap')
            if float(y_of_z(zgrid.max())) > y_ap:
                raise SystemExit('the %s supernovae reach y = %.4f, beyond AP_Y_MAX = %g: '
                                 'the envelope would not cover the data'
                                 % (smp, float(y_of_z(zgrid.max())), y_ap))
            s, diag = run_mcmc(
                sn_only_model, P,
                model_args=(jnp.asarray(z_sn), jnp.asarray(zhel_sn), jnp.asarray(m_sn),
                            jnp.asarray(L), jnp.asarray(zgrid)),
                model_kwargs=dict(P=P, M_fixed=-19.40, y_max=y_ap), quiet=quiet,
                kind='shape', y_max=y_ap)
            marg, at = envelope_activity(s, P, y_max=y_ap)
            diag['env_margin'], diag['at_bound'] = marg, at
            if at:
                print('[ap] %s: the posterior reaches the envelope (%s)' % (cell, at))
            cache_save('ap', cell, H0=s['H0'], v=s['v'], C=s['C'],
                       diag_json=json.dumps(diag))
            c_draws = ap_coefficients(s)
        # F_AP = H(z) * int_0^z dz'/H(z') -- invariant under H -> lambda H
        zg = np.linspace(0.0, float(zc.max()) * 1.05, BASELINE['NGRID'])
        F = np.empty((len(c_draws), len(zc)))
        for i, c in enumerate(c_draws):
            Hg = np.clip(H_poly(zg, c), H_FLOOR, None)
            DM, DH, _ = distances(zg, Hg, zc)
            F[i] = DM / DH
        fap_s, cov_s = F.mean(0), np.cov(F.T)
        sig_s = np.sqrt(np.diag(cov_s))
        pull = (fap_s - fap_d) / np.hypot(sig_s, sig_d)
        d = fap_s - fap_d
        chi2v = float(d @ np.linalg.solve(cov_s + np.diag(sig_d ** 2), d))
        p = float(chi2_dist.sf(chi2v, len(zc)))
        out.append(dict(sample=smp, z=zc.tolist(), fap_sn=fap_s.tolist(),
                        sig_sn=sig_s.tolist(), fap_desi=fap_d.tolist(),
                        sig_desi=sig_d.tolist(), pull=pull.tolist(),
                        chi2=chi2v, dof=len(zc), p=p))
        print('\n %s (P=%d):' % ({'full': 'DES+legacy', 'des_only': 'DES'}[smp], P))
        for i, z in enumerate(zc):
            print('   z=%.3f  SN F_AP = %.4f +- %.4f   DESI = %.4f +- %.4f   pull = %+.2f'
                  % (z, fap_s[i], sig_s[i], fap_d[i], sig_d[i], pull[i]))
        print('   joint chi2 = %.2f (dof=%d)   p = %.3f' % (chi2v, len(zc), p))
    # ap_table.json is an AGGREGATE over both cells and is written ONLY by a pass that
    # holds both of them.  A shard writing its own one-row list over the shared file is
    # how a two-row table became whatever the last process to finish had computed.
    if _is_sharded():
        print('\n[ap] shard %d of %d wrote %d cell cache(s); ap_table.json is written by '
              'a full (unsharded) run once every shard has landed:\n'
              '     python dovekie_cosmography.py --task ap'
              % (SHARD, NTASKS, len(out)))
        return out
    missing = [c for c in AP_CELLS
               if cache_load('ap', '%s_P%d' % (c, P)) is None]
    if missing:
        raise SystemExit('refusing to write a PARTIAL ap_table.json: no validated cache '
                         'for %s.  Run those cells first.' % ', '.join(missing))
    if len(out) != len(AP_CELLS):
        raise SystemExit('expected %d AP rows, assembled %d' % (len(AP_CELLS), len(out)))
    write_json_atomic('ap_table.json', out)
    return out


def fap_lcdm(z, Om=0.3175):
    """F_AP^LCDM(z) = E(z) int_0^z dz'/E(z').  H0 cancels: it depends only on Om.

    The reference the paper figure divides by, so the ratio panel is calibration-free in
    exactly the same sense as the test itself.

    The default is SPA's Om (SPT-3G + ACT DR6 + Planck 2018; Camphuis et al. 2026), which
    is the CMB reference every comparison in the paper is made against.  It was Planck
    2018's 0.3153 up to 2026-08-17.  PLOTTING ONLY -- no chi2 or p-value in the AP test
    references a fiducial cosmology, so nothing quantitative depends on this number."""
    zg = np.linspace(0.0, float(np.max(z)) * 1.05, 4000)
    E = np.sqrt(Om * (1.0 + zg) ** 3 + 1.0 - Om)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / E[1:] + 1.0 / E[:-1])
                                         * np.diff(zg))])
    return np.interp(z, zg, I) * np.sqrt(Om * (1.0 + np.asarray(z)) ** 3 + 1.0 - Om)


# =============================================================================
# 16.  CLI
# =============================================================================

# =============================================================================
# 15.  --task deltaorder   (order robustness of the legacy zero-point result,
#                           and the two-offset survey-split model)
# =============================================================================
#
# TWO questions, one grid, because they share every cell:
#
#   (a) ORDER.  The published legacy result (H0 = 71.19 +- 1.96 with the zero point
#       sampled, against 68.02 +- 0.70 with it held at face value) is quoted at P = 4.
#       P = 4 is the BIC minimum for des_only, but for `full` the BIC minimum is P = 5,
#       with P = 4 behind by only 1.79 -- inside the |dBIC| < 2 band.  The prior-free
#       chi2 minimum with Delta free reads 71.14 at P = 4 and 68.61 at P = 5, and 68.6
#       to 68.8 at every order from 5 to 8, so P = 4 is the outlier.  This task puts a
#       POSTERIOR behind that ladder instead of a least-squares minimum.
#
#   (b) SPLIT.  The legacy rung is Foundation (117) plus CfA/CSP (80), and Section
#       'systematics-legacy' already reports that their offsets differ by
#       +0.0216 +- 0.0214 mag when measured against a fixed reference.  That measurement
#       does not say what the split does to H0, because it is not sampled jointly with
#       the cosmology.  delta_mode='split' does exactly that.
#
# Cell name: <background>_<sample>_<delta_mode>_<config>_P<order>.  The published
# `legacy` cells keep their own names under chains/legacy/, so nothing here can clobber
# them.
DELTAORDER_CELLS = (
    # (background, sample, config, P, delta_mode)
    [('cosmo', 'full', 'A', P, dm)
     for P in (3, 4, 5, 6) for dm in ('fixed', 'joint', 'split')]
    + [('cosmo', 'des_only', 'A', P, 'fixed') for P in (3, 5, 6, 7)]
    + [('cosmo', 'full', 'B', P, dm) for P in (4, 5) for dm in ('fixed', 'joint', 'split')]
    + [('lcdm', 'full', 'A', 4, dm) for dm in ('fixed', 'joint', 'split')]
)
_DELTA_MODE = {'fixed': False, 'joint': True, 'split': 'split'}


def task_deltaorder(recompute=False, quiet=True):
    rows = []
    for bg, smp, cfg, P, dm in _shard(list(DELTAORDER_CELLS)):
        cell = '%s_%s_%s_%s_P%d' % (bg, smp, dm, cfg, P)
        cached = None if recompute else cache_load('deltaorder', cell, ignore_p=True,
                                                   extra={'order': P, 'delta_mode': dm})
        if cached is not None:
            print('[deltaorder] %s (cached)' % cell)
            r = _unpack(cached)
        else:
            print('[deltaorder] %s sampling...' % cell, flush=True)
            r = fit(smp, cfg, bg, P=P, joint_delta=_DELTA_MODE[dm], quiet=quiet)
            cache_save('deltaorder', cell, extra={'order': P, 'delta_mode': dm},
                       **_pack(r))
        rows.append(r)
    if NTASKS > 1 and len(rows) < len(DELTAORDER_CELLS):
        print('\n[deltaorder] shard %d of %d wrote %d cell(s); the table is printed by a '
              'full (unsharded) run once every shard has landed.'
              % (SHARD, NTASKS, len(rows)))
        return rows
    _print_deltaorder(rows)
    write_json_atomic('deltaorder_table.json',
                      [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                       for r in rows])
    return rows


def _print_deltaorder(rows):
    print('\n%-6s %-9s %-4s %-6s %-3s %16s %19s %19s %8s %7s %4s'
          % ('bg', 'sample', 'cfg', 'delta', 'P', 'H0', 'Delta / Delta_found',
             'Delta_cfa', 'M_des', 'rhat', 'div'))
    for r in rows:
        d1 = ('%9.4f +-%-7.4f' % (r['Delta_mean'], r['Delta_std'])
              if 'Delta_mean' in r else
              '%9.4f +-%-7.4f' % (r['Delta_found_mean'], r['Delta_found_std'])
              if 'Delta_found_mean' in r else '%18s' % '-')
        d2 = ('%9.4f +-%-7.4f' % (r['Delta_cfa_mean'], r['Delta_cfa_std'])
              if 'Delta_cfa_mean' in r else '%18s' % '-')
        print('%-6s %-9s %-4s %-6s %-3d %7.2f +-%-6.2f %s %s %8.3f %7.4f %4d'
              % (r['background'], r['sample'], r['config'], r['delta_mode'], r['P'],
                 r['H0_mean'], r['H0_std'], d1, d2, r['M_mean'], r['rhat_max'],
                 r['divergences']))
    # The contrast, per draw.  This is the number the survey-split question is about.
    print('\nsurvey contrast, Delta_found - Delta_cfa (formed per draw, not from the '
          'marginal means)')
    for r in rows:
        if 'Delta_diff_mean' in r:
            sig = r['Delta_diff_std']
            print('  %-6s %-9s %-4s P=%-3d %+8.4f +- %.4f   (%.2f sigma from zero, '
                  'corr(pair) = %+.2f)'
                  % (r['background'], r['sample'], r['config'], r['P'],
                     r['Delta_diff_mean'], sig,
                     abs(r['Delta_diff_mean']) / sig if sig > 0 else float('nan'),
                     r['corr_delta_pair']))
    bad = [r for r in rows if r.get('at_bound')]
    if bad:
        print('\n*** AT BOUND: %s' % ', '.join(
            '%s_%s_%s_P%d %s' % (r['background'], r['sample'], r['delta_mode'], r['P'],
                                 r['at_bound']) for r in bad))


TASKS = {'ladder': task_ladder, 'floorsweep': task_floorsweep, 'bic': task_bic,
         'legacy': task_legacy, 'ap': task_ap, 'deltaorder': task_deltaorder}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('WHAT IT COMPUTES')[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=sorted(TASKS) + ['all'], required=False)
    ap.add_argument('--recompute', action='store_true',
                    help='resample even where a valid cached chain exists')
    ap.add_argument('--check-box', action='store_true',
                    help='(bic) check every least-squares best fit lies inside the '
                         'envelope, so the prior-free results carry over')
    ap.add_argument('--plist', default='2,3,4,5,6,7,8,9,10',
                    help='(bic) orders to sweep')
    ap.add_argument('--list-cache', action='store_true')
    ap.add_argument('--quiet', action='store_true', help='no sampler progress bars')
    ap.add_argument('--shard', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID',
                                                                   os.environ.get('SHARD', 0))),
                    help='run only every NTASKS-th cell, starting here')
    ap.add_argument('--ntasks', type=int,
                    default=int(os.environ.get('NTASKS', 1)),
                    help='total number of shards')
    a = ap.parse_args(argv)
    global SHARD, NTASKS
    SHARD, NTASKS = a.shard, max(1, a.ntasks)
    if not 0 <= SHARD < NTASKS:
        raise SystemExit('--shard %d is not in range(0, --ntasks %d)' % (SHARD, NTASKS))
    if NTASKS > 1:
        print('[dovekie] shard %d of %d' % (SHARD, NTASKS))

    print(banner())
    os.makedirs(RESULT_DIR, exist_ok=True)

    if a.list_cache:
        for task in sorted(os.listdir(CHAIN_DIR)) if os.path.isdir(CHAIN_DIR) else []:
            d = os.path.join(CHAIN_DIR, task)
            if not os.path.isdir(d):
                continue
            files = sorted(f for f in os.listdir(d) if f.endswith('.npz'))
            print('\n%s: %d cached cell(s)' % (task, len(files)))
            for f in files:
                ok = cache_load(task, f[:-4]) is not None
                print('   %-44s %s' % (f, 'ok' if ok else 'STALE -- would be refused'))
        return

    if not a.task:
        ap.error('--task is required (or --list-cache)')
    tasks = sorted(TASKS) if a.task == 'all' else [a.task]
    for t in tasks:
        print('\n' + '=' * 78 + '\n %s\n' % t.upper() + '=' * 78)
        kw = dict(recompute=a.recompute)
        if t == 'bic':
            kw['plist'] = tuple(int(x) for x in a.plist.split(','))
            kw['check_box'] = a.check_box
        else:
            kw['quiet'] = a.quiet
        TASKS[t](**kw)


if __name__ == '__main__':
    main()
