"""The prior each reported quantity is measured against, including the derived ones.

WHY THIS EXISTS

  `print_cls_sigma` compares a posterior to its prior to decide whether a parameter was
  measured at all.  For a parameter sampled from a uniform box that comparison is exact
  and needs nothing but the box.  Three classes of reported number are NOT of that kind,
  and every one of them appears in the manuscript:

    1. NON-UNIFORM PRIORS.  `r_d` under configuration A is sampled from
       N(147.05, 0.30) -- the Planck measurement, imported.  Table 1 then reports
       `r_d = 147.05 +/- 0.30` in the result column of all four config-A rows.  A box
       comparison cannot see that this is the prior returned unchanged.

    2. DERIVED QUANTITIES.  `q0` is `C_1 - 1` and `H(0)` under the exp model is
       `H0_LCDM + A`.  Neither is sampled, so neither has a prior box -- but each has an
       INDUCED prior, which is what its posterior has to beat to count as measured.

    3. THE POSITIVITY FACTOR.  `joint_model` adds `-H_PENALTY * area / DZ_REF` for H(z)
       below H_FLOOR.  That is a `numpyro.factor` with no data in it, so it is part of the
       PRIOR.  Under the envelope prior E >= 0 holds on [0, y_max] by construction, so
       the factor bites only where H0 E < H_FLOOR, a sliver along the lower envelope; it
       is kept so the induced prior stays the one the sampler actually used.

  So the induced prior is built the only way that is defensible: by drawing from the same
  distributions the model samples from, pushing them through the same deterministic maps,
  and weighting by the same positivity factor.  The draws are importance-weighted rather
  than accepted/rejected, which is exactly what the smooth penalty means.

  Nothing here re-implements a prior.  The boxes come from `BASELINE` and
  `dovekie_evidence`, the shape prior from `dc.sample_envelope_prior`, the penalty from
  `dc._h_positive`, and the redshift grid from `dc.build_blocks`, so a prior that is
  widened anywhere cannot leave a stale copy in this file.

USAGE

    import prior_reference as pr
    draws, w = pr.cosmo_prior('des_only', 'A', 'cosmo', P=4)
    draws, w = pr.evidence_prior('exp')
"""

import numpy as np

import dovekie_cosmography as dc

N_DRAW_DEFAULT = 60000
CHUNK = 4000        # draws per H(z)-grid block; the full array is ~n x NGRID doubles
SEED_DEFAULT = 12345

_ZGRID_CACHE = {}


def _zgrid(sample, use_cc, z_floor=0.0):
    """The model's own integration grid, so the positivity factor is the model's own.

    `build_blocks` reads the supernova catalogue, so the result is cached per
    configuration: the induced prior for eight ladder cells would otherwise re-read the
    same two catalogues eight times."""
    key = (sample, bool(use_cc), float(z_floor))
    if key not in _ZGRID_CACHE:
        _ZGRID_CACHE[key] = np.asarray(
            dc.build_blocks(sample, use_cc, z_floor=z_floor)['zgrid'], dtype=float)
    return _ZGRID_CACHE[key]


def _positivity_logw(Hg, zgrid):
    """log-weight of the model's smooth positivity factor, in numpy.

    Deliberately the same arithmetic as `dc._h_positive`, which is written in jax and
    returns the clipped grid alongside the penalty.  `test_repo`-style agreement with it
    is asserted in `check()` below rather than assumed."""
    viol = np.clip(dc.H_FLOOR - Hg, 0.0, None)
    v2 = viol ** 2
    area = np.sum(0.5 * (v2[..., 1:] + v2[..., :-1]) * np.diff(zgrid), axis=-1)
    return -dc.H_PENALTY * area / dc.H_PENALTY_DZ_REF


def cosmo_prior(sample, config, background, P=None, joint_delta=False, z_floor=0.0,
                n=N_DRAW_DEFAULT, seed=SEED_DEFAULT, y_max=None):
    """Importance-weighted draws from the prior of one ladder/legacy/floorsweep cell.

    Returns ({name: values}, weights).  The weights are exp(positivity factor), normalised
    to their maximum so they can be summed without underflow; they are NOT probabilities
    of anything else, because no data enter here.  `y_max` is where the envelope is
    enforced: the ladder's Y_MAX by default, `dc.shape_y_max('ap')` for the AP cells."""
    P = dc.BASELINE['P'] if P is None else P
    if config == 'A':
        use_cc, rd_mode = False, 'planck'
    elif config == 'B':
        use_cc, rd_mode = True, 'free'
    else:
        raise ValueError("config must be 'A' or 'B', got %r" % config)

    rng = np.random.default_rng(seed)
    B = dc.BASELINE
    out = {}

    out['M'] = rng.uniform(B['M_LO'], B['M_HI'], n)
    if rd_mode == 'planck':
        out['rd'] = rng.normal(B['RD_PLANCK'], B['SIG_RD'], n)
    else:
        out['rd'] = rng.uniform(B['RD_FREE_LO'], B['RD_FREE_HI'], n)
    if joint_delta:
        w = B['DELTA_HALFWIDTH']
        out['Delta'] = rng.uniform(-w, w, n)

    zgrid = _zgrid(sample, use_cc, z_floor)
    y = zgrid / (1.0 + zgrid)

    if background == 'cosmo':
        H0 = rng.uniform(B['H0_LO'], B['H0_HI'], n)
        C, _tried = dc.sample_envelope_prior(P, n, seed=seed, y_max=y_max)   # exact draws
        v = dc.v_of_shape(C, P, y_max)
        out['H0'] = H0
        for k in range(P):                                        # 1-based, as the physics
            out['v[%d]' % (k + 1)] = v[:, k]
            out['C[%d]' % (k + 1)] = C[:, k + 1]
        # q0 exactly as the model records it: C_1 - 1
        out['q0'] = C[:, 1] - 1.0
        c = H0[:, None] * C                                      # dimensional
        # n x NGRID doubles is ~0.4 GB at the default n, so H(z) is built a block at a
        # time: the penalty is a per-draw scalar and never needs the whole array at once.
        pw = y[:, None] ** np.arange(P + 1)[None, :]              # (ngrid, P+1)
        logw = np.concatenate([_positivity_logw(c[i:i + CHUNK] @ pw.T, zgrid)
                               for i in range(0, n, CHUNK)])
    else:
        H0 = rng.uniform(B['H0_LO'], B['H0_HI'], n)
        Om = rng.uniform(B['OM_LO'], B['OM_HI'], n)
        out['H0'] = H0
        out['Om'] = Om
        out['q0'] = 1.5 * Om - 1.0
        E = np.sqrt(np.clip(Om[:, None] * (1.0 + zgrid[None, :]) ** 3 + 1.0 - Om[:, None],
                            0.0, None))
        logw = _positivity_logw(H0[:, None] * E, zgrid)

    w = np.exp(logw - logw.max())
    return out, w


def evidence_prior(model, n=N_DRAW_DEFAULT, seed=SEED_DEFAULT):
    """Draws from the prior of one `dovekie_evidence` cell, through its own ptform.

    Uses `make_ptform` itself rather than re-deriving the boxes, so the induced prior on
    the derived `H0_at_0 = H0_LCDM + A` is the one the sampler actually explored.  The
    prior is proper and carries no factor, so the weights are unity."""
    import dovekie_evidence as de
    ndim = {'lcdm': 3, 'exp': 5, 'cpl': 5}[model]
    rng = np.random.default_rng(seed)
    ptform = de.make_ptform(model)
    u = rng.uniform(size=(n, ndim))
    X = np.array([ptform(row) for row in u])   # per-row: ptform is written for one draw
    names = {'lcdm': ('Om', 'H0_LCDM', 'rd'),
             'exp': ('Om', 'H0_LCDM', 'A', 'z1', 'rd'),
             'cpl': ('Om', 'H0_LCDM', 'w0', 'wa', 'rd')}[model]
    out = {nm: X[:, i] for i, nm in enumerate(names)}
    if model == 'exp':
        out['H0_at_0'] = X[:, 1] + X[:, 2]
    return out, np.ones(n)


def check():
    """Assert the numpy positivity factor equals the model's jax one on random draws."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    zgrid = np.linspace(0.0, 2.5, 400)
    for _ in range(20):
        Hg = rng.uniform(-40.0, 120.0, len(zgrid))
        _, pen = dc._h_positive(jnp.asarray(Hg), jnp.asarray(zgrid))
        mine = _positivity_logw(Hg[None, :], zgrid)[0]
        assert np.allclose(float(pen), mine, rtol=1e-6, atol=1e-6), (float(pen), mine)
    print('positivity factor agrees with dovekie_cosmography on 20 random grids')


if __name__ == '__main__':
    check()
