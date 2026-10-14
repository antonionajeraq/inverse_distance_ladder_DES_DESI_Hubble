"""
test_repo.py -- acceptance tests.  No network, no cluster, ~2 minutes.

    python test_repo.py            # everything
    python test_repo.py -k legacy  # substring filter on the test name

These are not unit tests for their own sake.  Every one of them is a defect this analysis
actually shipped at some point, turned into something that now fails loudly instead:

  * a covariance handed to numpyro as if it were a Cholesky factor, which switched the
    chronometers off and left r_d prior-dominated for four days in a published table;
  * a chi2 minimisation that silently inherited the coefficient box, making a
    "prior-free" order decision a no-op;
  * an `INIT=lcdm` that reached the banner and the manifest but not the sampler;
  * a cached grid from a superseded configuration read straight into a new figure;
  * a two-stage legacy correction whose error bars could not know about the covariance
    they inherited.

A passing run means the repository is self-contained, reads the data it says it reads,
and that the identities the analysis relies on actually hold.
"""
import ast
import io
import json
import os
import sys
import traceback
import contextlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def close(a, b, tol, what):
    if not np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tol):
        raise AssertionError('%s: %r vs %r (tol %g)' % (what, a, b, tol))


# =============================================================================
# self-containment and provenance
# =============================================================================
@test
def test_cosmography_module_imports_nothing_from_this_repo():
    """Parsed with ast, not grepped: a grep for 'import dovekie_' would miss a deferred
    import inside a function, which is exactly where the old code hid them."""
    local = {f[:-3] for f in os.listdir(HERE) if f.endswith('.py')}
    src = open(os.path.join(HERE, 'dovekie_cosmography.py')).read()
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            bad += [a.name.split('.')[0] for a in node.names
                    if a.name.split('.')[0] in local]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split('.')[0] in local:
                bad.append(node.module)
    if bad:
        raise AssertionError('dovekie_cosmography.py imports from the repo: %s' % bad)


@test
def test_every_vendored_data_file_matches_its_checksum():
    import dovekie_cosmography as dc
    for rel, want in dc.DATA_SHA256.items():
        got = dc._sha256(dc.data_path(rel, verify=False))
        if got != want:
            raise AssertionError('%s: %s != %s' % (rel, got, want))


@test
def test_only_whitelisted_settings_can_be_overridden_from_the_environment():
    """A typo in a slurm export must not silently do nothing.  Everything that decides
    what a number means lives in BASELINE, and only the listed keys are env-readable."""
    import dovekie_cosmography as dc
    unknown = dc._ENV_OVERRIDABLE - set(dc.BASELINE)
    if unknown:
        raise AssertionError('overridable keys not in BASELINE: %s' % unknown)
    for k in ('Y_MAX', 'AP_Y_MAX', 'E_LOWER', 'E_UPPER', 'H0_LO', 'M_LO', 'RD_PLANCK',
              'CLEAN_MUERR_MAX'):
        if k in dc._ENV_OVERRIDABLE:
            raise AssertionError('%s must not be env-overridable: it defines the '
                                 'published prior/data selection' % k)


# =============================================================================
# data
# =============================================================================
@test
def test_the_catalogue_is_the_published_one():
    import dovekie_cosmography as dc
    for name, n, zlo in (('full', 1738, 0.025), ('des_only', 1541, 0.073)):
        d = dc.sn_dataset(name)
        if len(d['z']) != n:
            raise AssertionError('%s has %d SNe, expected %d' % (name, len(d['z']), n))
        close(d['z'].min(), zlo, 0.001, '%s redshift floor' % name)
    full, des = dc.sn_dataset('full'), dc.sn_dataset('des_only')
    if len(full['z']) - len(des['z']) != 197:
        raise AssertionError('the legacy anchor should contribute 197 SNe, got %d'
                             % (len(full['z']) - len(des['z'])))


@test
def test_the_legacy_subsamples_partition_the_legacy_anchor():
    """Foundation plus CfA/CSP must be the whole legacy sample and nothing else.

    The survey-ID list is easy to get wrong from the upstream README, which lists surveys
    this compilation does not contain: writing out 61-66 rather than the IDs actually
    present drops the 8 CSP supernovae of IDSURVEY 5 and yields a 72-object subsample
    presented as if it were the 80-object one."""
    import dovekie_cosmography as dc
    d = dc.sn_dataset('full')
    leg = d['idsurvey'] != dc.DES_IDSURVEY
    found = d['idsurvey'] == dc.FOUNDATION_IDSURVEY
    cfa = np.isin(d['idsurvey'], list(dc.CFA_CSP_IDSURVEY))
    if found.sum() != 117:
        raise AssertionError('Foundation has %d SNe, expected 117' % found.sum())
    if cfa.sum() != 80:
        raise AssertionError('CfA/CSP has %d SNe, expected 80' % cfa.sum())
    if np.any(found & cfa):
        raise AssertionError('the two legacy subsamples overlap')
    if not np.array_equal(found | cfa, leg):
        raise AssertionError('Foundation + CfA/CSP do not partition the legacy anchor')


@test
def test_the_desi_vector_is_complete():
    import dovekie_cosmography as dc
    O, Cov, zs, kinds = dc.bao_vector()
    if len(O) != 13:
        raise AssertionError('expected 13 DESI observables, got %d' % len(O))
    close(zs.min(), 0.295, 1e-9, 'lowest BAO redshift')
    close(zs.max(), 2.330, 1e-9, 'highest BAO redshift (the Lyman-alpha point)')
    if not np.allclose(Cov, Cov.T):
        raise AssertionError('the BAO covariance is not symmetric')
    np.linalg.cholesky(Cov)          # raises if not positive definite


@test
def test_the_chronometer_covariance_is_full_not_diagonal():
    """A diagonal CC likelihood understates sigma(r_d) by a factor 2.7, because the
    correlated Moresco term is an uncertainty on the ABSOLUTE SCALE of H(z) and does not
    average down."""
    import dovekie_cosmography as dc
    z, H, C = dc.cc_block()
    if len(z) != 15:
        raise AssertionError('expected 15 chronometer points, got %d' % len(z))
    off = C[~np.eye(len(z), dtype=bool)]
    if np.max(np.abs(off)) < 1e-6:
        raise AssertionError('the CC covariance came back diagonal')
    # the correlated part must not average down: compare a pure-scale fit both ways
    Hf = H / H[0]
    var_full = 1.0 / (Hf @ np.linalg.solve(C, Hf))
    var_diag = 1.0 / (Hf @ np.linalg.solve(np.diag(np.diag(C)), Hf))
    ratio = np.sqrt(var_full / var_diag)
    if not 1.5 < ratio < 3.5:
        raise AssertionError('the full matrix should inflate the scale error by ~2x, '
                             'got %.2f' % ratio)


# =============================================================================
# the cosmography map and the prior
# =============================================================================
@test
def test_the_map_reproduces_the_textbook_lcdm_cosmography():
    """Nothing about q0 = 3Om/2 - 1, j0 = 1, s0 = 1 - 9Om/2 is put into the map; that
    they come back out of it is the test."""
    import dovekie_cosmography as dc
    for Om in (0.25, 0.315, 0.35):
        cg = dc.lcdm_cg(Om, 4)
        close(cg[0], 1.5 * Om - 1.0, 1e-9, 'q0 at Om=%g' % Om)
        close(cg[1], 1.0, 1e-9, 'j0 at Om=%g' % Om)
        close(cg[2], 1.0 - 4.5 * Om, 1e-9, 's0 at Om=%g' % Om)
    close(dc.coeffs(70.0, dc.lcdm_cg(0.315, 4)),
          dc.lcdm_coeffs(0.315, 70.0, 4), 1e-10,
          'coeffs(H0, lcdm_cg) vs H0 * lcdm_series')


@test
def test_the_envelope_prior_is_the_envelope_and_its_volume_is_right():
    """The shape prior is uniform in the C_k over A = {0 <= E(y) <= 2 + 7y on [0, y_max]}.
    Four things must hold, at both ceilings:
      - the node map is exact: v -> C -> v round-trips;
      - every prior draw is inside A, on a grid FINER than the enforcement grid;
      - the old Bernstein box (Bernstein_E_prior.py) lies inside A -- A is the larger set;
      - the evidence normalisation is right: vol(A)/vol(Bernstein box), from the two
        volume formulas, equals the share of A-draws that land in the Bernstein box,
        which is measured independently of both formulas."""
    import dovekie_cosmography as dc
    from Bernstein_E_prior import BernsteinEPrior
    rng = np.random.default_rng(11)
    for y_max in (dc.BASELINE['Y_MAX'], dc.shape_y_max('ap')):
        for P in (2, 4, 6):
            C, tried = dc.sample_envelope_prior(P, 20000, seed=P, y_max=y_max)
            # relative: prior draws of C_k reach ~1e4 at P = 6 (the monomial basis is
            # ill-conditioned), so an absolute 1e-9 is below double precision there
            close(dc.shape_of_v(dc.v_of_shape(C, P, y_max), P, y_max), C,
                  1e-11 * np.abs(C).max(), 'v <-> C round trip at P=%d' % P)
            y = np.linspace(0.0, y_max, 4 * dc.ENV_NGRID)
            E = C @ np.power(y[None, :], np.arange(P + 1)[:, None])
            if E.min() < -1e-4 or np.max(E - (2.0 + 7.0 * y)) > 1e-4:
                raise AssertionError('a prior draw leaves the envelope between grid '
                                     'points at P=%d, y_max=%g' % (P, y_max))
            ref = BernsteinEPrior(P, y_max)
            Cb = ref.coeffs(ref.prior_transform(rng.random((5000, ref.ndim))))
            if not np.all(dc.in_envelope(Cb, y_max)):
                raise AssertionError('the Bernstein box is not inside A at P=%d' % P)
            if y_max == dc.BASELINE['Y_MAX'] and P == 4:
                lnVb = (np.log(abs(np.linalg.det(ref.M[1:, 1:])))
                        + np.sum(np.log(ref.b_hi - ref.b_lo)))
                Minv = np.linalg.inv(ref.M)
                bb = C @ Minv.T
                share = np.mean(np.all((bb[:, 1:] >= ref.b_lo) & (bb[:, 1:] <= ref.b_hi),
                                       axis=1))
                d = (dc.log_shape_volume(P, y_max) - lnVb) + np.log(share)
                if abs(d) > 0.08:
                    raise AssertionError('ln vol(A) - ln vol(box) disagrees with the '
                                         'measured share by %.3f nats' % d)


@test
def test_dynesty_integrates_a_minus_inf_region_as_zero_likelihood():
    """The evidence correction lnZ_A = lnZ_sampled - ln acc assumes dynesty integrates
    over the WHOLE sampled box with L = 0 where log L = -inf.  Checked on a toy with a
    known answer, so a dynesty release that changes this cannot silently shift every
    evidence weight.  (dynesty 3.0.0: -5.216 against the truth -5.175, 0.4 sigma.)"""
    try:
        import dynesty
    except ImportError:
        return
    s = 0.03

    def ll(t):
        if t[0] + t[1] > 1.0:
            return -np.inf
        return -0.5 * ((t[0] - 0.2) ** 2 + (t[1] - 0.3) ** 2) / s ** 2
    ns = dynesty.NestedSampler(ll, lambda u: u, 2, nlive=300,
                               rstate=np.random.default_rng(1))
    ns.run_nested(dlogz=0.05, print_progress=False)
    got, err = ns.results.logz[-1], ns.results.logzerr[-1]
    truth = np.log(2 * np.pi * s ** 2)          # w.r.t. the square; the triangle is ln 2 higher
    if abs(got - truth) > 4 * err + 0.05:
        raise AssertionError('dynesty returned %.3f +- %.3f, truth over the whole box is '
                             '%.3f (over the allowed region %.3f): the lnZ correction '
                             'assumes the former' % (got, err, truth, truth + np.log(2)))


@test
def test_a_coefficient_cell_from_another_shape_prior_is_refused():
    """Every cell that stores coefficient draws must carry the current shape prior.  A
    chain from the retired coefficient box -- no `cosmo_prior` in its manifest -- must be
    refused everywhere, while a cell with no coefficient draws (flat LCDM, Table 2) must
    NOT be caught by this, or cells nothing changed would be re-run for nothing.  An AP
    cell is keyed on its own ceiling, so it cannot be read as a ladder cell or vice versa."""
    import dovekie_cosmography as dc
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        saved = dc.CHAIN_DIR
        try:
            dc.CHAIN_DIR = tmp
            man = dc.manifest()
            man.pop('cosmo_prior')
            os.makedirs(os.path.join(tmp, 'ladder'), exist_ok=True)
            np.savez(os.path.join(tmp, 'ladder', 'old.npz'),
                     manifest_json=json.dumps(man), H0=np.ones(3), c_hi=np.ones((3, 4)))
            np.savez(os.path.join(tmp, 'ladder', 'lcdm.npz'),
                     manifest_json=json.dumps(man), H0=np.ones(3), Om=np.ones(3))
            with contextlib.redirect_stdout(io.StringIO()):
                if dc.cache_load('ladder', 'old') is not None:
                    raise AssertionError('a pre-Bernstein coefficient chain was accepted')
                if dc.cache_load('ladder', 'lcdm') is None:
                    raise AssertionError('a cell with no coefficient draws was refused')
            dc.cache_save('ladder', 'new', H0=np.ones(3), v=np.ones((3, 4)))
            dc.cache_save('ap', 'new', H0=np.ones(3), v=np.ones((3, 4)))
            if dc.cache_load('ladder', 'new') is None or dc.cache_load('ap', 'new') is None:
                raise AssertionError('a cell written under the current prior was refused')
            import shutil
            shutil.copy(os.path.join(tmp, 'ap', 'new.npz'),
                        os.path.join(tmp, 'ladder', 'fromap.npz'))
            with contextlib.redirect_stdout(io.StringIO()):
                if dc.cache_load('ladder', 'fromap') is not None:
                    raise AssertionError('an AP-ceiling cell was accepted as a ladder cell')
        finally:
            dc.CHAIN_DIR = saved


@test
def test_the_init_point_is_inside_the_box_and_physical():
    """INIT=lcdm seeds the sampler at the node values of the LCDM series, so it must lie
    inside the envelope, at both ceilings."""
    import dovekie_cosmography as dc
    P = dc.BASELINE['P']
    if not dc.BASELINE['H0_LO'] <= dc.BASELINE['INIT_H0'] <= dc.BASELINE['H0_HI']:
        raise AssertionError('the init H0 is outside the box')
    C = dc.lcdm_series(dc.BASELINE['INIT_OM'], P)
    for y_max in (None, dc.shape_y_max('ap')):
        if not dc.in_envelope(C, y_max)[0]:
            raise AssertionError('init shape outside the envelope (y_max=%r)' % y_max)
    z = np.linspace(0.0, 2.33, 500)
    if np.any(dc.H_poly(z, dc.BASELINE['INIT_H0'] * C) <= dc.H_FLOOR):
        raise AssertionError('the init point violates H(z) > 0 within the data range')


# =============================================================================
# the likelihood
# =============================================================================
@test
def test_the_cc_block_reaches_the_model_as_a_cholesky_factor():
    """numpyro's MultivariateNormal takes scale_tril.  Handing it the covariance asserts
    Cov = C C^T and inflates the per-point CC sigma 13-16x, i.e. switches the
    chronometers off.  This is the single most expensive defect this analysis has had."""
    import dovekie_cosmography as dc
    cc = dc.cc_block()
    out = dc.cc_to_cholesky(cc)
    L = np.asarray(out[2], dtype=float)
    if not np.allclose(L, np.tril(L)):
        raise AssertionError('the CC block is not lower triangular -- a covariance was '
                             'passed straight through')
    # Tolerance is RELATIVE and loose because jax is float32 by default, which is what
    # produced the published numbers; the factorisation itself is done in float64 by numpy
    # and the cast is the only loss.  Absolute tolerances are meaningless here: the CC
    # covariance spans 90 to 2477 in its diagonal alone.
    rel = np.max(np.abs(L @ L.T - cc[2])) / np.max(np.abs(cc[2]))
    if rel > 1e-4:
        raise AssertionError('L L^T does not reproduce the CC covariance (rel %.2e)' % rel)
    # and the defect itself: feeding the covariance in as scale_tril would imply Cov Cov^T
    wrong = np.sqrt(np.diag(cc[2] @ cc[2].T))
    right = np.sqrt(np.diag(cc[2]))
    if np.median(wrong / right) < 5.0:
        raise AssertionError('the two paths are not distinguishable, so this test would '
                             'not catch the defect it exists for')


@test
def test_chi2_equals_the_squared_whitened_residual():
    """The BIC minimises the residual vector for speed; the reported chi2 comes from the
    scalar path.  They must be the same number, or the BIC ladder and the fit-quality
    column would be measuring different things."""
    import dovekie_cosmography as dc
    for cfg in ('A', 'B'):
        f, meta = dc.make_chi2('des_only', cfg, 'cosmo', P=3)
        th = np.concatenate([dc.lcdm_coeffs(0.315, 68.0, 3), [-19.4, 147.05]])
        r = meta['resid'](th)
        close(float(r @ r), f(th), 1e-4 * abs(f(th)), 'config %s chi2 vs |resid|^2' % cfg)


@test
def test_the_order_decision_carries_no_prior():
    """The BIC is defined on the UNCONSTRAINED minimum.  If the box entered, "which order
    the data ask for" would quietly become "which order the prior tolerates" -- and a
    version of this module once hardcoded the box in every mode, making a whole
    prior-sensitivity run a no-op."""
    import dovekie_cosmography as dc
    f, meta = dc.make_chi2('des_only', 'A', 'cosmo', P=4)
    _y, _lo, hi, _Vinv = dc.envelope_nodes(4)
    c = 70.0 * dc.shape_of_v(3.0 * hi, 4)                      # far outside the envelope
    if dc.in_envelope(c / c[0])[0]:
        raise AssertionError('the test point is not outside the envelope')
    th = np.concatenate([c, [-19.4, 147.05]])
    v = f(th)
    if not np.isfinite(v):
        raise AssertionError('chi2 is not finite outside the box')
    if v >= 1e12:
        raise AssertionError('chi2 rejects points outside the envelope, so the '
                             'minimum is NOT prior-free')
    src = open(os.path.join(HERE, 'dovekie_cosmography.py')).read()
    body = src[src.index('def make_chi2'):src.index('def minimise_chi2')]
    for token in ('envelope', 'Y_MAX', 'E_UPPER', 'H0_LO', 'M_LO'):
        if token in body:
            raise AssertionError('make_chi2 references %s -- the chi2 must not see the '
                                 'prior' % token)


@test
def test_a_planted_offset_is_recovered_exactly():
    """The legacy zero-point estimator, checked against a truth we set.  Shifting the
    legacy magnitudes by a known amount must move Delta by exactly that amount."""
    import dovekie_cosmography as dc
    b0 = dc.delta_residual('lcdm')
    s0 = dc.offset_stats(b0['r'], b0['C'])
    plant = 0.037
    r = b0['r'] + plant          # the legacy residual is what Delta is the mean of
    s1 = dc.offset_stats(r, b0['C'])
    close(s1['Delta'] - s0['Delta'], plant, 1e-9, 'planted offset recovery')
    close(s1['sig'], s0['sig'], 1e-12, 'sigma_Delta must not move with a pure offset')


@test
def test_delta_is_calibration_free_by_construction():
    """Rescaling H -> lambda H with M free is an EXACT flat direction of the shape fit,
    so Delta must not move.  That invariance is what makes Delta comparable across the
    four references and what keeps the distance ladder out of it."""
    import dovekie_cosmography as dc
    b = dc.legacy_blocks()
    zg = np.linspace(0.0, 2.0, dc.BASELINE['NGRID'])
    th = np.concatenate([np.zeros(4), [-19.4]])
    m0 = dc._rung_m('cosmo', th, b['z_leg'], b['zhel_leg'], zg, 4)
    lam = 1.13
    th2 = np.concatenate([np.zeros(4), [-19.4 + 5.0 * np.log10(lam)]])
    saved = dc.H0_GAUGE
    try:
        dc.H0_GAUGE = saved * lam
        m1 = dc._rung_m('cosmo', th2, b['z_leg'], b['zhel_leg'], zg, 4)
    finally:
        dc.H0_GAUGE = saved
    close(m1, m0, 1e-8, 'predicted magnitudes under H -> lambda H with M free')


@test
def test_the_evidence_rank2_reduces_to_rank1():
    """The legacy marginalisation is a rank-2 version of the published rank-1 projection.
    With no legacy supernovae in the indicator the two must give the same chi2, or the
    published Bayes factors would not be reproduced by this code path."""
    import dovekie_cosmography as dc
    import dovekie_evidence as de
    d = dc.sn_dataset('full')
    invC = np.linalg.inv(d['cov'])
    one = np.ones_like(d['mu'])
    rng = np.random.default_rng(0)
    r = rng.standard_normal(len(one)) * 0.1
    B = float(r @ (invC @ one))
    chi2_rank1 = float(r @ (invC @ r)) - B * B / float(one @ (invC @ one))
    u = np.zeros_like(one)
    u[0] = 1.0                                    # a single-SN offset: nearly rank-1
    V = np.column_stack([one, u])
    G_inv = np.linalg.inv(V.T @ (invC @ V))
    g = np.array([float(r @ (invC @ one)), float(r @ (invC @ u))])
    chi2_rank2 = float(r @ (invC @ r)) - float(g @ (G_inv @ g))
    if chi2_rank2 > chi2_rank1 + 1e-9:
        raise AssertionError('projecting out a larger subspace must not raise chi2')
    if not np.isfinite(de.DELTA_MAX):
        raise AssertionError('the Delta prior must be proper for lnZ to be a number')


@test
def test_retired_anchors_cannot_be_used():
    import dovekie_evidence as de
    for a in sorted(de.RETIRED_ANCHORS):
        try:
            de.anchor_terms(a)
        except ValueError:
            continue
        raise AssertionError('retired anchor %r was accepted' % a)
    if de.LADDER_ANCHORS & set(de.ANCHOR_SETS['lfree3']):
        raise AssertionError('the SN Ia-independent set contains a distance ladder')


@test
def test_every_anchor_has_a_verified_redshift_window():
    """A wrong window silently biases every anchored evidence, so an anchor without one
    must raise rather than fall back to identifying the anchor with H(0)."""
    import dovekie_evidence as de
    Hg = 70.0 * np.sqrt(0.3 * (1 + de.ZGRID) ** 3 + 0.7)
    for name in sorted(set(de.ANCHORS) - de.RETIRED_ANCHORS - {'none'}):
        h = de.anchor_infer(Hg, name)
        if not 60.0 < h < 80.0:
            raise AssertionError('%s forward model returned H0 = %g' % (name, h))
        close(h, 70.0, 0.7, '%s should recover the planted H0 on flat LCDM' % name)


@test
def test_the_anchor_windows_are_the_published_samples():
    """The redshift windows are DATA, inlined rather than read from a catalogue.  Inlined
    data can be mistyped, and a wrong window biases every anchored evidence silently, so
    the count and the first two moments of each are pinned here.

    This is not hypothetical: an early draft of dovekie_evidence.py in this repository had
    237 of the 277 Breuval redshifts, with the mean shifted from 0.05359 to 0.05665."""
    import dovekie_evidence as de
    expect = {                    # n, min, max, mean
        'jensen':   (63, 0.004810, 0.024454, 0.015026),
        'vogl':     (10, 0.010839, 0.039579, 0.023122),
        'gw170817': (1, 0.009800, 0.009800, 0.009800),
        'breuval':  (277, 0.023430, 0.148980, 0.053586),
    }
    for name, (n, lo, hi, mean) in expect.items():
        z = de.ANCHOR_Z[name]
        if len(z) != n:
            raise AssertionError('%s window has %d redshifts, expected %d'
                                 % (name, len(z), n))
        close(z.min(), lo, 1e-5, '%s min' % name)
        close(z.max(), hi, 1e-5, '%s max' % name)
        close(z.mean(), mean, 1e-5, '%s mean' % name)


@test
def test_a_sub_window_transition_is_invisible_to_the_anchor():
    """The reason anchors are forward-modelled at all: a transition completing below an
    anchor's redshift window must NOT be credited with fitting it."""
    import dovekie_evidence as de
    Om, H0L, A = 0.3, 68.0, 5.0
    wide = de.H_of_grid(np.array([Om, H0L, A, 0.30, 147.05]), 'exp')
    narrow = de.H_of_grid(np.array([Om, H0L, A, 0.005, 147.05]), 'exp')
    h_wide = de.anchor_infer(wide, 'breuval')
    h_narrow = de.anchor_infer(narrow, 'breuval')
    if not h_wide > h_narrow + 1.0:
        raise AssertionError('a z1 = 0.30 transition should raise the inferred H0 well '
                             'above a z1 = 0.005 one (got %.2f vs %.2f)'
                             % (h_wide, h_narrow))
    if h_narrow > H0L + A - 1.0:
        raise AssertionError('a sub-window transition was credited with its full '
                             'amplitude: inferred %.2f against H(0) = %.2f'
                             % (h_narrow, H0L + A))


# =============================================================================
# caching
# =============================================================================
@test
def test_a_cache_from_different_settings_is_refused():
    """The standing failure mode: a stale cache read silently into a new figure."""
    import dovekie_cosmography as dc
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        saved = dc.CHAIN_DIR
        try:
            dc.CHAIN_DIR = tmp
            dc.cache_save('unit', 'cell', x=np.arange(3.0))
            if dc.cache_load('unit', 'cell') is None:
                raise AssertionError('a cache written under the current settings was '
                                     'refused')
            dc.BASELINE['P'] += 1
            with contextlib.redirect_stdout(io.StringIO()):
                got = dc.cache_load('unit', 'cell')
            if got is not None:
                raise AssertionError('a cache from a different P was reused')
            with contextlib.redirect_stdout(io.StringIO()):
                got = dc.cache_load('unit', 'cell', ignore_p=True)
            if got is None:
                raise AssertionError('ignore_p should allow a P-independent reuse')
        finally:
            dc.BASELINE['P'] -= 1
            dc.CHAIN_DIR = saved


@test
def test_a_cache_without_a_manifest_is_refused():
    import dovekie_cosmography as dc
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        saved = dc.CHAIN_DIR
        try:
            dc.CHAIN_DIR = tmp
            os.makedirs(os.path.join(tmp, 'unit'), exist_ok=True)
            np.savez(os.path.join(tmp, 'unit', 'cell.npz'), x=np.arange(3.0))
            with contextlib.redirect_stdout(io.StringIO()):
                if dc.cache_load('unit', 'cell') is not None:
                    raise AssertionError('a cache with no manifest was accepted')
        finally:
            dc.CHAIN_DIR = saved


@test
def test_the_manifest_records_every_cache_relevant_setting():
    import dovekie_cosmography as dc
    man = dc.manifest()
    for k in dc._CACHE_KEYS:
        if k not in man['baseline']:
            raise AssertionError('%s is cache-relevant but absent from the manifest' % k)
    for k in ('git_sha', 'packages', 'python', 'written'):
        if k not in man:
            raise AssertionError('the manifest is missing %s' % k)


# =============================================================================
# the sampler
# =============================================================================
@test
def test_the_sampler_refuses_a_per_caller_mass_matrix_or_init():
    """Both were per-caller once, and both drifted.  dense_mass is central because a
    diagonal one cannot precondition a ridge at 45 degrees to every axis; the init is
    central because 12 of 13 call sites took a default while the banner claimed
    otherwise."""
    import dovekie_cosmography as dc
    for kw in ({'dense_mass': False}, {'init_strategy': None}):
        try:
            dc.make_nuts(lambda: None, 4, **kw)
        except ValueError:
            continue
        raise AssertionError('make_nuts accepted %s' % list(kw))


@test
def test_every_nuts_kernel_goes_through_the_one_factory():
    """Parsed, not grepped."""
    src = open(os.path.join(HERE, 'dovekie_cosmography.py')).read()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'NUTS']
    if len(calls) != 1:
        raise AssertionError('NUTS is constructed %d times; it must be exactly once, '
                             'inside make_nuts' % len(calls))


# Tests that need only numpy + the data.  Everything else imports the sampling stack,
# and one missing module used to turn 24 of 26 checks into the same ImportError printed
# 24 times -- which says nothing about the analysis and buries whatever else broke.
_NO_SAMPLER_OK = ('module_imports_nothing', 'checksum', 'overridden_from_the_environment',
                  'cache', 'manifest', 'slurm', 'shard', 'aggregate')


def preflight():
    """One line per missing dependency, before any test runs."""
    need = [('numpy', 'always'), ('scipy', 'always'),
            ('jax', 'sampler'), ('numpyro', 'sampler'), ('dynesty', 'evidence')]
    missing = []
    for mod, why in need:
        try:
            __import__(mod)
        except Exception as exc:
            missing.append((mod, why, '%s: %s' % (type(exc).__name__, exc)))
    if missing:
        print('DEPENDENCY PREFLIGHT -- %d module(s) missing:' % len(missing))
        for mod, why, msg in missing:
            print('  %-10s (%s)  %s' % (mod, why, msg))
        print('  pip install -r requirements.txt')
        print('  Tests that need them will FAIL ON IMPORT and say nothing about the\n'
              '  analysis.  Run with -k to select the ones that do not:  %s\n'
              % ', '.join(_NO_SAMPLER_OK))
    return [m for m, _w, _e in missing]


# =============================================================================
# the 2026-08-18 audit.  Every test below is a defect this repository shipped.
# =============================================================================
@test
def test_the_sn_model_uses_zhel_in_the_prefactor_and_zhd_in_the_integral():
    """D_L = (1 + z_HEL) c int_0^{z_HD} dz/H, not (1 + z_HD) twice.

    The catalogue reader always read both columns; sn_block threw one away, so every SN
    path evaluated the same wrong prefactor.  Perturb one redshift at a time: the model
    magnitude must respond to BOTH, and to z_HEL through the prefactor alone."""
    import dovekie_cosmography as dc
    zhd = np.array([0.10, 0.50])
    zhel = np.array([0.11, 0.51])
    dcz = np.array([400.0, 2000.0])
    base = dc.sn_apparent_mag(-19.4, zhel, dcz)
    close(base, -19.4 + 5.0 * np.log10((1.0 + zhel) * dcz) + 25.0, 1e-12,
          'sn_apparent_mag is not the stated expression')
    wrong = dc.sn_apparent_mag(-19.4, zhd, dcz)
    if np.allclose(base, wrong):
        raise AssertionError('the model magnitude does not depend on z_HEL at all')
    close(wrong - base, 5.0 * np.log10((1.0 + zhd) / (1.0 + zhel)), 1e-12,
          'the z_HD-twice bias is not 5 log10[(1+zHD)/(1+zHEL)]')

    # and the block that feeds it must carry both, paired and same-length
    z, zh, m, cov = dc.sn_block('full')
    if not (len(z) == len(zh) == len(m) == len(cov)):
        raise AssertionError('sn_block returned ragged (z_HD, z_HEL, m, cov)')
    if np.allclose(z, zh):
        raise AssertionError('z_HD and z_HEL are identical -- the catalogue was misread')
    b = dc.build_blocks('des_only', use_cc=False)
    if 'zhel_sn' not in b:
        raise AssertionError('build_blocks does not carry z_HEL to the likelihood')


@test
def test_no_sn_path_writes_the_magnitude_expression_for_itself():
    """Parsed, not grepped.  Eight copies of `5 log10((1+z) D)` is how one of them stayed
    wrong after the others were fixed; they now all route through sn_apparent_mag."""
    import dovekie_cosmography as dc  # noqa: F401
    # the helper's OWN definition is where the expression is allowed to live; the anchor
    # templates are a different model (a kinematic mu(z), not the SN likelihood); and the
    # BIC's two closures hoist log10(1+z_HEL) out of a loop the minimiser runs tens of
    # thousands of times, which is a real speed difference and is checked NUMERICALLY
    # against the helper below rather than by parsing.
    ALLOWED = {'sn_apparent_mag', '_mu_kin', '_mu_model', 'resid', 'chi2'}

    # the hoisted form the BIC uses IS the helper, algebraically:
    #     M + 5 (log10(1+z_HEL) + log10 D) + 25 == M + 5 log10((1+z_HEL) D) + 25
    zhel = np.array([0.0231, 0.4210, 1.0553])
    dcz = np.array([98.7, 1611.2, 3402.6])
    import dovekie_cosmography as _dc
    close(-19.4 + 5.0 * (np.log10(1.0 + zhel) + np.log10(dcz)) + 25.0,
          _dc.sn_apparent_mag(-19.4, zhel, dcz), 1e-11,
          'the BIC hoisted magnitude form has drifted from sn_apparent_mag')
    for fn in ('dovekie_cosmography.py', 'dovekie_evidence.py'):
        src = open(os.path.join(HERE, fn)).read()
        tree = ast.parse(src)
        inside = {}
        for f in ast.walk(tree):
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # plain assignment, not setdefault: ast.walk is breadth-first, so the
                # INNERMOST enclosing function is the last to write and therefore wins.
                # With setdefault a nested closure was attributed to its outer function.
                for sub in ast.walk(f):
                    inside[id(sub)] = f.name
        for node in ast.walk(tree):
            if inside.get(id(node)) in ALLOWED:
                continue
            # a literal + 25.0 in an arithmetic expression is the signature of the
            # distance-modulus constant being written out by hand
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                    and isinstance(node.right, ast.Constant)
                    and node.right.value == 25.0):
                seg = ast.get_source_segment(src, node) or ''
                if 'log10' in seg:
                    raise AssertionError(
                        '%s line %d writes the SN magnitude expression inline instead of '
                        'calling sn_apparent_mag:\n    %s'
                        % (fn, node.lineno, seg.replace('\n', ' ')[:120]))


@test
def test_planck_rd_is_imposed_exactly_once_in_the_evidence():
    """The prior transform draws r_d from N(147.05, 0.30) and the likelihood must NOT
    repeat it.  Imposing it twice is equivalent to sigma -> sigma/sqrt(2), which is
    visible in every chain written before this test existed as sigma(r_d) ~ 0.212.

    Measured, not grepped: hold everything but r_d fixed and read the curvature of
    log posterior = loglike + log prior in r_d.  For an r_d-only quadratic that curvature
    is -1/sigma^2 with ONE Gaussian and -2/sigma^2 with two."""
    import dovekie_evidence as de
    import dovekie_cosmography as dc
    from scipy.stats import norm

    ll = de.make_loglike('des_only', 'lcdm', anchor='none')
    mu, sig = dc.BASELINE['RD_PLANCK'], dc.BASELINE['SIG_RD']
    Om, H0 = 0.315, 68.0

    def logpost(rd):
        return ll(np.array([Om, H0, rd])) + norm.logpdf(rd, mu, sig)

    # the BAO term also constrains r_d, so difference against a run with the SN and BAO
    # blocks left alone but the Gaussian removed -- the DIFFERENCE is pure prior
    h = 0.02
    d2_with = (logpost(mu + h) - 2 * logpost(mu) + logpost(mu - h)) / h ** 2
    d2_like = (ll(np.array([Om, H0, mu + h])) - 2 * ll(np.array([Om, H0, mu]))
               + ll(np.array([Om, H0, mu - h]))) / h ** 2
    n_gauss = (d2_like - d2_with) * sig ** 2
    if not (0.9 < n_gauss < 1.1):
        raise AssertionError(
            'the posterior carries %.3f Planck Gaussian factors, not 1.  2 means the '
            'likelihood is adding chi2_rd on top of the prior transform.' % n_gauss)


@test
def test_the_legacy_shape_statistic_is_not_the_des_reference_statistic():
    """Two chi2/dof live side by side and were interchanged: task_legacy reported the DES
    reference fit's (~0.92) under the name of the legacy residual shape's (~1.05)."""
    import dovekie_cosmography as dc
    b = dc.delta_residual('lcdm')
    st = dc.offset_stats(b['r'], b['C'])
    shape = st['chi2_shape'] / st['dof_shape']
    ref = b['fit']['chi2'] / b['fit']['dof']
    if abs(shape - ref) < 0.05:
        raise AssertionError('the two statistics have converged (%.4f vs %.4f); this test '
                             'can no longer tell them apart' % (shape, ref))
    if not (0.8 < shape < 1.3):
        raise AssertionError('legacy shape chi2/dof = %.4f is not plausible' % shape)


@test
def test_the_des_reference_fit_reports_chi2_and_dof_for_the_same_data():
    """fit_des_shape returned chi2 over r_sn alone with a dof that counted the extra F_AP
    residuals, so its numerator and denominator described different datasets."""
    import dovekie_cosmography as dc
    for shape in ('lcdm', 'lcdm_fap', 'cosmo', 'cosmo_fap'):
        f = dc.fit_des_shape(shape, P=dc.BASELINE['P'])
        npar = len(f['theta'])
        if f['dof'] != f['dof_sn'] + f['n_extra']:
            raise AssertionError('%s: dof=%d is not dof_sn=%d + n_extra=%d'
                                 % (shape, f['dof'], f['dof_sn'], f['n_extra']))
        if f['dof'] + npar - f['n_extra'] != f['dof_sn'] + npar:
            raise AssertionError('%s: the two dof do not describe nested datasets' % shape)
        if f['n_extra'] and not f['chi2'] > f['chi2_sn']:
            raise AssertionError('%s: chi2 including the F_AP residuals is not larger '
                                 'than the SN-only chi2' % shape)


@test
def test_a_cache_from_an_older_schema_is_refused():
    """CACHE_SCHEMA is what refuses a cache when NO BASELINE value moved -- which is
    exactly the case for the z_HEL repair and the duplicated Planck factor."""
    import dovekie_cosmography as dc
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir, saved_schema = dc.CHAIN_DIR, dc.CACHE_SCHEMA
        try:
            dc.CHAIN_DIR = tmp
            dc.cache_save('unit', 'cell', x=np.arange(3.0))
            if dc.cache_load('unit', 'cell') is None:
                raise AssertionError('a cache at the current schema was refused')
            dc.CACHE_SCHEMA = saved_schema + 1
            with contextlib.redirect_stdout(io.StringIO()):
                if dc.cache_load('unit', 'cell') is not None:
                    raise AssertionError('a cache from an older schema was reused')
        finally:
            dc.CACHE_SCHEMA, dc.CHAIN_DIR = saved_schema, saved_dir


@test
def test_a_cache_written_under_different_task_settings_is_refused():
    """The nested-sampling controls live in dovekie_evidence, not in BASELINE, so
    _CACHE_KEYS could not see them: a scouting run at DLOGZ=1 was reloadable as a
    production run at DLOGZ=0.01."""
    import dovekie_cosmography as dc
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        saved = dc.CHAIN_DIR
        try:
            dc.CHAIN_DIR = tmp
            dc.cache_save('unit', 'cell', extra=dict(DLOGZ=0.01, NLIVE=800),
                          x=np.arange(3.0))
            if dc.cache_load('unit', 'cell',
                             extra=dict(DLOGZ=0.01, NLIVE=800)) is None:
                raise AssertionError('a cache with matching task settings was refused')
            with contextlib.redirect_stdout(io.StringIO()):
                got = dc.cache_load('unit', 'cell', extra=dict(DLOGZ=1.0, NLIVE=800))
            if got is not None:
                raise AssertionError('a cache written at a different DLOGZ was reused')
        finally:
            dc.CHAIN_DIR = saved


@test
def test_a_stricter_target_accept_is_accepted_but_a_looser_one_is_not():
    """The remediation ladder writes cells at a HIGHER TARGET_ACCEPT than the baseline.
    Raising it shrinks the step size without changing the distribution being sampled, so a
    rescued cell answers the baseline question too.  The reverse must still be refused, or
    a cheap scouting run would pass for a rescued one."""
    import dovekie_cosmography as dc
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir, saved_ta = dc.CHAIN_DIR, dc.BASELINE['TARGET_ACCEPT']
        try:
            dc.CHAIN_DIR = tmp
            dc.BASELINE['TARGET_ACCEPT'] = 0.95
            dc.cache_save('unit', 'rescued', x=np.arange(3.0))
            dc.BASELINE['TARGET_ACCEPT'] = 0.6
            dc.cache_save('unit', 'loose', x=np.arange(3.0))

            dc.BASELINE['TARGET_ACCEPT'] = 0.8
            with contextlib.redirect_stdout(io.StringIO()):
                if dc.cache_load('unit', 'rescued') is None:
                    raise AssertionError('a cell sampled at 0.95 was refused at 0.8')
                if dc.cache_load('unit', 'loose') is not None:
                    raise AssertionError('a cell sampled at 0.6 was accepted at 0.8')
        finally:
            dc.BASELINE['TARGET_ACCEPT'] = saved_ta
            dc.CHAIN_DIR = saved_dir


@test
def test_a_cache_write_is_atomic_and_leaves_no_temporary_behind():
    import dovekie_cosmography as dc
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        saved = dc.CHAIN_DIR
        try:
            dc.CHAIN_DIR = tmp
            dc.cache_save('unit', 'cell', x=np.arange(3.0))
            left = [f for f in os.listdir(os.path.join(tmp, 'unit')) if '.tmp' in f]
            if left:
                raise AssertionError('cache_save left temporaries behind: %s' % left)
        finally:
            dc.CHAIN_DIR = saved


@test
def test_a_truncated_cache_is_refused_rather_than_raised_on():
    """An interrupted cluster write must cost that cell, not the whole campaign."""
    import dovekie_cosmography as dc
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        saved = dc.CHAIN_DIR
        try:
            dc.CHAIN_DIR = tmp
            p = dc.cache_save('unit', 'cell', x=np.arange(1000.0))
            raw = open(p, 'rb').read()
            open(p, 'wb').write(raw[:len(raw) // 2])
            with contextlib.redirect_stdout(io.StringIO()):
                if dc.cache_load('unit', 'cell') is not None:
                    raise AssertionError('a truncated cache was accepted')
        finally:
            dc.CHAIN_DIR = saved


@test
def test_an_out_of_range_shard_raises_instead_of_running_nothing():
    """An invalid shard selected no cells and exited 0, which sacct reports COMPLETED."""
    import dovekie_cosmography as dc
    saved = (dc.SHARD, dc.NTASKS)
    try:
        dc.SHARD, dc.NTASKS = 0, 2
        if dc._shard(list(range(4))) != [0, 2]:
            raise AssertionError('_shard no longer selects i %% NTASKS == SHARD')
        dc.SHARD, dc.NTASKS = 2, 2
        try:
            dc._shard(list(range(4)))
        except ValueError:
            pass
        else:
            raise AssertionError('SHARD == NTASKS did not raise')
        dc.SHARD, dc.NTASKS = -1, 2
        try:
            dc._shard(list(range(4)))
        except ValueError:
            pass
        else:
            raise AssertionError('a negative SHARD did not raise')
    finally:
        dc.SHARD, dc.NTASKS = saved


@test
def test_an_aggregate_json_is_never_written_by_a_shard():
    """Every AP array task used to json.dump its own one-row list over the shared
    ap_table.json, so the last writer won."""
    import dovekie_cosmography as dc
    src = open(os.path.join(HERE, 'dovekie_cosmography.py')).read()
    tree = ast.parse(src)
    inside = {}
    for f in ast.walk(tree):
        if isinstance(f, ast.FunctionDef):
            for sub in ast.walk(f):
                inside[id(sub)] = f.name      # innermost wins; see the note above
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'dump'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'json'
                and inside.get(id(node)) != 'write_json_atomic'):
            raise AssertionError('json.dump at line %d (in %s) writes an aggregate '
                                 'directly; use write_json_atomic, which unsharded runs '
                                 'alone reach' % (node.lineno, inside.get(id(node))))
    saved = (dc.SHARD, dc.NTASKS)
    try:
        dc.SHARD, dc.NTASKS = 0, 2
        if not dc._is_sharded():
            raise AssertionError('_is_sharded() is False for a 2-way shard')
        dc.SHARD, dc.NTASKS = 0, 1
        if dc._is_sharded():
            raise AssertionError('_is_sharded() is True for an unsharded run')
    finally:
        dc.SHARD, dc.NTASKS = saved


@test
def test_the_inverse_mass_condition_number_is_a_condition_number():
    """The batched fallback computed m.max()/m.min(), and a positive-definite dense matrix
    has negative off-diagonal entries -- so every stored imm_cond was NEGATIVE."""
    import dovekie_cosmography as dc
    A = np.array([[4.0, -1.0], [-1.0, 3.0]])
    stack = np.stack([A, 2.0 * A, 5.0 * A])          # a 4-chain-style batch
    conds = np.ravel(np.linalg.cond(stack))
    if not np.all(np.isfinite(conds)) or np.any(conds < 1.0 - 1e-9):
        raise AssertionError('np.linalg.cond over a stack gave %r' % conds)
    if (A.max() / A.min()) > 0:
        raise AssertionError('the old fallback is no longer negative on this matrix, so '
                             'this test would not have caught the bug')
    src = open(os.path.join(HERE, 'dovekie_cosmography.py')).read()
    if 'm.max() / m.min()' in src:
        raise AssertionError('the max/min fallback is still in _diagnostics')


@test
def test_the_publication_gate_rejects_an_unconverged_chain():
    """A failing chain is still SAMPLED and CACHED -- cluster hours are not thrown away
    -- but it must not reach a table without a deliberate override.
    R-hat up to 1.02 passes, and divergences alone do not fail a chain whose R-hat and
    n_eff pass."""
    import dovekie_cosmography as dc
    good = dict(divergences=0, rhat_max=1.001, neff_min=3000.0)
    for fine in (good, dict(good, divergences=260), dict(good, rhat_max=1.0199)):
        ok, why = dc.gate_verdict(fine)
        if not ok:
            raise AssertionError('a passing chain failed the gate: %s' % why)
    for bad, what in ((dict(good, rhat_max=1.03), 'rhat'),
                      (dict(good, neff_min=120.0), 'n_eff'),
                      (dict(good, divergences=260, rhat_max=1.05), 'rhat with divergences')):
        ok, why = dc.gate_verdict(bad)
        if ok:
            raise AssertionError('the gate passed a chain failing on %s' % what)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                dc.require_quality('unit', bad, allow=False)
            except SystemExit:
                pass
            else:
                raise AssertionError('require_quality did not stop a %s failure' % what)
            if dc.require_quality('unit', bad, allow=True):
                raise AssertionError('ALLOW_BAD_CHAINS should return False, not True')


@test
def test_the_desi_redshift_cut_applies_to_the_dv_point_too():
    """DESI_ZMAX filtered the D_M/D_H pairs and not the D_V point, so a cut below 0.295
    still kept D_V(0.295)."""
    import dovekie_cosmography as dc
    O, _C, zs, kinds = dc.bao_vector(use_dv=True, zmax=0.2)
    if len(O):
        raise AssertionError('a cut at z=0.2 kept %d observable(s) at z=%s'
                             % (len(O), zs))
    O, _C, zs, kinds = dc.bao_vector(use_dv=True, zmax=0.5)
    if 'DV' not in kinds:
        raise AssertionError('a cut at z=0.5 dropped D_V(0.295)')


@test
def test_the_positivity_penalty_does_not_depend_on_the_grid_resolution():
    """As a bare sum it grew with NGRID, so doubling the integration resolution doubled
    the strength of a physicality constraint."""
    import dovekie_cosmography as dc
    import jax.numpy as jnp
    pens = []
    for n in (400, 800, 1600):
        zg = jnp.linspace(0.0, 2.0, n)
        H = jnp.where(zg < 1.0, -5.0, 70.0)          # violating on half the range
        _clip, pen = dc._h_positive(H, zg)
        pens.append(float(pen))
    if not (abs(pens[1] / pens[0] - 1.0) < 0.02 and abs(pens[2] / pens[1] - 1.0) < 0.02):
        raise AssertionError('the penalty moved with NGRID: %r' % pens)


@test
def test_the_chi2_minimiser_raises_rather_than_returning_its_starting_point():
    """Every start's exception used to be swallowed, so a cell in which all of them failed
    returned (inf, theta0) -- the initial guess, labelled as the minimum."""
    import dovekie_cosmography as dc
    src = open(os.path.join(HERE, 'dovekie_cosmography.py')).read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == 'minimise_chi2')
    seg = ast.get_source_segment(src, fn)
    if 'sol.success' not in seg:
        raise AssertionError('minimise_chi2 ignores sol.success')
    if 'raise RuntimeError' not in seg:
        raise AssertionError('minimise_chi2 can still return a non-minimum silently')


@test
def test_the_order_list_must_be_strictly_increasing():
    """Each order is seeded from the previous one with c_P = 0, which is what makes the
    comparison nested; a repeated or decreasing --plist breaks that silently."""
    import dovekie_cosmography as dc
    for bad in ([4, 4], [5, 4], [4], [0, 1]):
        try:
            dc.task_bic(plist=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('--plist %r was accepted' % (bad,))


@test
def test_no_output_script_raw_loads_a_chain():
    """make_adot and print_cls_sigma hard-coded relative paths and np.load, so they read
    from the wrong directory outside the repo root and accepted caches the main loader
    would have refused."""
    for fn in ('make_adot.py', 'print_cls_sigma.py', 'make_tables.py', 'make_figures.py',
               'make_triangles.py'):
        src = open(os.path.join(HERE, fn)).read()
        if 'allow_pickle=True' in src:
            raise AssertionError('%s still loads caches with allow_pickle=True' % fn)
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'load'
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'np'):
                seg = (ast.get_source_segment(src, node) or '').replace('\n', ' ')
                raise AssertionError('%s line %d raw-loads a chain: %s'
                                     % (fn, node.lineno, seg[:100]))


def main(argv=None):
    keyword = None
    argv = argv if argv is not None else sys.argv[1:]
    if '-k' in argv:
        keyword = argv[argv.index('-k') + 1]
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    preflight()
    sel = [t for t in TESTS if keyword is None or keyword in t.__name__]
    n_pass = 0
    fails = []
    for t in sel:
        name = t.__name__.replace('test_', '').replace('_', ' ')
        try:
            t()
            print('  ok    %s' % name)
            n_pass += 1
        except Exception as e:
            print('  FAIL  %s\n        %s' % (name, e))
            fails.append((name, traceback.format_exc()))
    print('\n%d/%d passed' % (n_pass, len(sel)))
    for name, tb in fails:
        print('\n--- %s ---\n%s' % (name, tb))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
