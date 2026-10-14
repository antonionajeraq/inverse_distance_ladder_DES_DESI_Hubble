"""Mode and 1-sigma confidence limits for every chain, without assuming a Gaussian.

WHY THIS EXISTS

  Every number in the manuscript is currently quoted as `mean +/- std` -- the first two
  moments of the posterior.  For a Gaussian that is the whole story.  Several of these
  posteriors are visibly not Gaussian, `z1` worst of all: its prior is U(0.01, 0.5), the
  likelihood only bounds it from below, and the posterior piles up against the lower edge
  with a long tail to the right.  A mean of such a distribution sits in the tail, and a
  symmetric std straddles the boundary, so `0.129 +/- 0.146` is not a statement about
  where the probability actually is.

  This module reports instead the MODE and the 1-sigma HIGHEST-POSTERIOR-DENSITY interval,
  which are well defined whatever the shape.  The algorithm is that of an
  earlier Print_CLs_sigma.py, reproduced here function-for-function; only the driver is new.  In
  outline:

    1. Build a robust window: median-centred, half-width Window_sigmas * sigma_rob, where
       sigma_rob is half the central-68% inter-quantile range.  This anchors the grid to
       the CORE, so a heavy tail cannot starve the peak of resolution.
    2. Bin inside that window at dx = sigma_rob/K, weighting each sample by its nested-
       sampling weight, and smooth the density by sigma_rob/5 so the arg-max does not
       jitter between neighbouring bins and manufacture a false asymmetry.
    3. Solve for the density contour enclosing erf(1/sqrt(2)) = 68.27% of the mass, and
       read off where it cuts the density either side of the peak.  That is the HPD
       interval, and it is asymmetric exactly when the posterior is.
    4. Escalate K through (30, 60, 120, 240) until the contour resolves.

  The interpolations in get_mode_CL are sub-pixel: the mode comes from a parabolic vertex
  through the peak bin and its neighbours, the limits from linear crossings of the contour
  level.  Pixel counts are reported so an under-resolved limit is visible.

ONE-SIDED LIMITS -- WHEN THE TWO-SIDED INTERVAL IS NOT DEFINED

  Step 3 always returns two crossings, because the smoothed density always comes back down
  on both sides of the peak.  That does not mean both crossings are BOUNDS.  If the lower
  crossing falls below the lowest sample, there is no posterior mass down there for it to
  exclude: the density is descending into empty space, and the number is set by where the
  samples ran out and by the smoothing kernel, not by the data.  Quoting `mode -a +b` then
  advertises a lower bound that was never measured.

  So every interval gets a second stage.  Let [x_min, x_max] be the region the samples
  actually cover (below).  Then:

    lower crossing < x_min  ->  all the mass below the upper crossing is inside the 68.27%
                                HPD region, so P(x < High_CL) = 68.27% exactly, and the
                                result is reported as `x < High_CL` at 1 sigma.
    upper crossing > x_max  ->  symmetrically, `x > Low_CL`.
    both                    ->  the contour clears the samples at both ends; there is no
                                limit of either sign, and the entry is flagged rather than
                                quoted.

  The limit value is the SAME contour crossing the two-sided branch would have reported --
  nothing is re-solved, only re-interpreted -- so a parameter that drifts across the test
  from one run to the next does not jump in value.

  `x_min`/`x_max` are weight-aware.  For the NUTS chains (unit weights) they are exactly
  min(x) and max(x), which is the literal reading.  For the dynesty cells they cannot be:
  nested sampling starts at the prior and its earliest samples span the whole box carrying
  weights of order 1e-300, so a raw min() would place the support edge at the prior wall
  and the test would never fire on precisely the posteriors it exists for.  The support
  edge is therefore the weighted quantile at 0.5/N_eff with N_eff the Kish effective
  sample size, which reduces to min/max identically when the weights are equal.

WEIGHTS -- THE POINT OF DOING THIS AT ALL FOR DYNESTY

  Nested sampling does not produce equally-weighted draws.  Its samples run from the prior
  inwards and the early ones carry weights of order 1e-300; treating them as equal draws
  is meaningless.  Every chain here is read with its own weights: the dynesty cells under
  chains/evidence/ use the stored `weights` array, and the NUTS chains use unit weights,
  which is correct for them.  The label in each output file records which.

USAGE

    python print_cls_sigma.py                 # every chain
    python print_cls_sigma.py --group evidence
    python print_cls_sigma.py --plot          # show the density + contour per parameter

  Writes `<chain>_CLs.txt` beside each chain, one aggregate table to
  results/mode_CLs_summary.txt, and results/mode_CLs.json for the table builders.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d  # Density smoothing (stabilises the mode).

import dovekie_cosmography as dc

# ---------------------------------------------------------------------------------------
# Tuning constants, from Print_CLs_sigma.py
# ---------------------------------------------------------------------------------------
sqrt_half = np.sqrt(0.5)
Threshold_sigmas = 1.0      # Desired number of sigmas at which to obtain the interval.
P_threshold = math.erf(Threshold_sigmas * sqrt_half)
Pixel_warning_threshold = 5

_one_sigma_mass = math.erf(sqrt_half)
Sigma_q_lo = 0.5 * (1.0 - _one_sigma_mass)   # 15.87 percentile, robust sigma lower edge.
Sigma_q_hi = 0.5 * (1.0 + _one_sigma_mass)   # 84.13 percentile, upper edge.
Window_sigmas = max(5.0, Threshold_sigmas + 2.0)   # Window half-width, in robust sigmas.
Bins_per_sigma_options = (30, 60, 120, 240)        # Escalating core resolution K.
Smooth_frac = 0.2           # Gaussian smoothing width = sigma/5; set to 0 to disable.


# ---------------------------------------------------------------------------------------
# 1.  The original algorithm, reproduced
# ---------------------------------------------------------------------------------------
def get_contour_level(P_values_initial, Normalisation, P_target):
    Target_u = (1.0 - P_target) * np.sum(P_values_initial) / Normalisation
    P_values = sorted(P_values_initial)
    n = len(P_values)
    P_right_values = np.empty(n)

    P_right_values[0] = 2.0 * P_values[0]
    for i in range(1, n):
        P_right_values[i] = 2.0 * P_values[i] - P_values[i - 1]

    u = 1.0 - Normalisation
    for i in np.arange(n):
        P = P_values[i]
        u = u + P
        if u > Target_u:
            break

    P_remaining = P - u + Target_u
    Slope = 2.0 * (P_right_values[i] - P_values[i])
    # A zero/non-finite slope means the density is flat where the contour falls, so the
    # crossing is undefined; raise so the caller refines the grid rather than returning a
    # NaN contour level.
    if Slope == 0.0 or not np.isfinite(Slope):
        raise ValueError('contour slope degenerate (flat density at contour); refine '
                         'binning.')
    Slope_inv = 1.0 / Slope
    u = P * Slope_inv - 0.5
    x = np.sqrt(2.0 * P_remaining * Slope_inv + u * u) - u
    return P + (x - 0.5) * Slope


def get_mode_CL(x_values, P_values, Contour_level):
    """[Low_CL, Mode, High_CL] plus any warnings raised, as a list of strings."""
    dx = x_values[1] - x_values[0]
    n = len(x_values)
    n_minus_1 = n - 1
    warn = []

    for i in np.arange(n):
        P = P_values[i]
        if i == 0 or P > P_max:
            P_max = P
            i_max = i

    if Contour_level >= P_max:
        raise ValueError('desired confidence interval unresolved, try using finer bins.')

    if i_max == 0 or i_max == n_minus_1:
        Mode = x_values[i_max]
        warn.append('mode at first or last pixel.')
    else:
        g_left = P_max - P_values[i_max - 1]
        g_right = P_values[i_max + 1] - P_max
        Mode = x_values[i_max] + dx * (g_left / (g_left - g_right) - 0.5)

    for i in np.arange(i_max + 1):
        P = P_values[i]
        if P > Contour_level:
            break
    if i == 0:
        Low_CL = x_values[i] - 0.5 * dx * (1.0 - Contour_level / P)
    else:
        Low_CL = x_values[i] - dx * (P - Contour_level) / (P - P_values[i - 1])

    for i in np.flip(np.arange(n)):
        P = P_values[i]
        if P > Contour_level:
            break
    if i == n_minus_1:
        High_CL = x_values[i] + 0.5 * dx * (1.0 - Contour_level / P)
    else:
        High_CL = x_values[i] + dx * (P - Contour_level) / (P - P_values[i + 1])

    if i_max == 0:
        warn.append('only trust the upper limit: mode at lower edge of allowed region.')
    elif i_max == n_minus_1:
        warn.append('only trust the lower limit: mode at upper edge of allowed region.')
    return [Low_CL, Mode, High_CL], warn


def weighted_quantile(Values, Weights, q):
    """Order-independent weighted q-quantile (cumulative-minus-half-weight positions)."""
    order = np.argsort(Values)
    sv = Values[order]
    sw = np.asarray(Weights, dtype=float)[order]
    cw = np.cumsum(sw) - 0.5 * sw
    cw /= sw.sum()
    return float(np.interp(q, cw, sv))


def get_robust_window(Values, Weights):
    """Median-centred [Min, Max] window of half-width Window_sigmas*sigma_rob, plus
    sigma_rob = half the central-68% inter-quantile range (tail-insensitive, = the true
    sigma for a Gaussian).  Returns None for an effectively constant sample."""
    m = weighted_quantile(Values, Weights, 0.5)
    q_lo = weighted_quantile(Values, Weights, Sigma_q_lo)
    q_hi = weighted_quantile(Values, Weights, Sigma_q_hi)
    sigma = 0.5 * (q_hi - q_lo)
    if not (sigma > 0.0):
        lo, hi = float(np.min(Values)), float(np.max(Values))
        if hi <= lo:
            return None
        # Degenerate core (spike/step): fall back to a fraction of the full span.
        sigma = (hi - lo) / 10.0
    half = Window_sigmas * sigma
    return [float(m - half), float(m + half), float(sigma)]


def resolve_mode_CL_in_window(Values, Weights, Min, Max, sigma):
    """Mode + 1-sigma HPD interval from the robust window, escalating the resolution on
    failure.  Samples outside the window are excluded, not clamped into the edge bins.
    Returns (Low_CL, Mode, High_CL, dx, x_centres, P_density, Contour_density, warnings)."""
    Weights = np.asarray(Weights, dtype=float)
    Wsum = np.sum(Weights)
    last_err = None
    for K in Bins_per_sigma_options:
        try:
            Steps = max(int(round((Max - Min) / (sigma / K))), 2 * K)
            dx = (Max - Min) / float(Steps)
            inside = (Values >= Min) & (Values <= Max)
            Index = np.floor((Values[inside] - Min) / dx).astype(np.int64)
            np.clip(Index, 0, Steps - 1, out=Index)
            P_values = np.bincount(Index, weights=Weights[inside],
                                   minlength=Steps)[:Steps].astype(float)
            P_values /= Wsum   # Mass per bin (sums to the retained fraction R <= 1).
            if Smooth_frac > 0.0:
                P_values = gaussian_filter1d(P_values, Smooth_frac * K)
            x_centre_values = Min + dx * (0.5 + np.arange(Steps))
            R = float(np.sum(P_values))
            Contour_level = get_contour_level(P_values, R, P_threshold)
            (Low_CL, Mode, High_CL), warn = get_mode_CL(x_centre_values, P_values,
                                                        Contour_level)
            if not (np.isfinite(Low_CL) and np.isfinite(Mode) and np.isfinite(High_CL)):
                raise ValueError('non-finite mode/CL; refine binning.')
            return (Low_CL, Mode, High_CL, dx, x_centre_values, P_values / dx,
                    Contour_level / dx, warn)
        except Exception as e:      # noqa: BLE001 -- any failure means "try a finer grid"
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError('interval did not resolve')


def effective_support(Values, Weights):
    """(x_min, x_max): the range the samples actually cover, weight-aware.

    For equally weighted draws this returns exactly (min, max) -- `weighted_quantile`
    places the lowest sample at cumulative weight 0.5/N by construction, which is the
    quantile asked for here -- so for the NUTS chains it IS the literal sample range.
    For weighted draws it is the same statement about the posterior rather than about
    the sampler's bookkeeping: N_eff is Kish's (sum w)^2 / sum w^2, so a sample carrying
    e^-700 of the weight moves the edge by nothing, as it should."""
    w = np.asarray(Weights, dtype=float)
    s = float(np.sum(w))
    ssq = float(np.sum(w * w))
    if not (s > 0.0 and ssq > 0.0):
        return float(np.min(Values)), float(np.max(Values))
    n_eff = max(s * s / ssq, 2.0)
    eps = 0.5 / n_eff
    return (weighted_quantile(Values, w, eps), weighted_quantile(Values, w, 1.0 - eps))


def classify_limit(Low_CL, High_CL, x_min, x_max):
    """'two-sided' | 'upper' | 'lower' | 'none' for a contour interval against the support.

    'upper' means the LOWER crossing is unbounded, so the parameter is quoted as
    x < High_CL; 'lower' is the mirror image; 'none' means neither crossing bounds
    anything and there is no limit to quote at all."""
    below = Low_CL < x_min
    above = High_CL > x_max
    if below and above:
        return 'none'
    if below:
        return 'upper'
    if above:
        return 'lower'
    return 'two-sided'


# ---------------------------------------------------------------------------------------
# 2.  Reading this project's two chain layouts
# ---------------------------------------------------------------------------------------
# `chains/evidence/` holds dynesty runs: a single (N, ndim) `samples` block, a `weights`
# array, and the parameter names inside summary_json.  Everywhere else holds NUTS runs:
# one named 1-D array per scalar parameter, equally weighted.  The multi-column `v` (the
# sampled node values E(y_j)) and `C` (the shape C_1..C_P) blocks are expanded per column.
VECTOR_KEYS = ('v', 'C')
SKIP_KEYS = ('manifest_json', 'summary_json', 'row_json', 'scalars_json', 'diag_json',
             'samples', 'weights')


def read_chain(path, d):
    """(params, weights, sampler_label) where params is an ordered list of (name, values).

    `d` comes from `load_validated`, never from a raw np.load: a chain whose manifest the
    main loader would refuse must not be readable here by a shorter route.

    `dc.cache_load` hands back a plain dict of arrays, not the NpzFile it opened, so the
    key list is `d` itself; `hasattr(d, 'files')` keeps a raw NpzFile working for anyone
    calling this directly."""
    keys = list(d.files) if hasattr(d, 'files') else list(d)
    if 'samples' in keys and 'weights' in keys:
        S = np.asarray(d['samples'], dtype=float)
        w = np.asarray(d['weights'], dtype=float)
        # chains/evidence/ carries summary_json; chains/evorder/ (evidence_order.py)
        # carries row_json, whose params are [H0, v1..vP, M, rd, (Delta)].
        summ = json.loads(str(d['summary_json'] if 'summary_json' in keys
                              else d['row_json']))
        names = list(summ['params'])
        params = [(nm, S[:, i]) for i, nm in enumerate(names)]
        if 'row_json' in keys:
            # Report the shape the way the NUTS chains do: q0 and 1-based C[k].
            P = int(summ['P'])
            C = dc.shape_of_v(S[:, 1:P + 1], P)
            params.append(('q0', C[:, 1] - 1.0))
            params += [('C[%d]' % k, C[:, k]) for k in range(1, P + 1)]
        # H(0) is the model's own present-day rate and is what Table 2 quotes.  For `exp`
        # it is H0_LCDM + A, NOT the LCDM background, so it is a derived column and has to
        # be reconstructed here rather than read off.
        if summ.get('model') == 'exp':
            params.append(('H0_at_0', S[:, 1] + S[:, 2]))
        return params, w, 'dynesty (nested-sampling weights applied)'

    params = []
    for k in keys:
        if k in SKIP_KEYS:
            continue
        a = np.asarray(d[k])
        if a.dtype.kind not in 'fiu':
            continue
        if a.ndim == 1:
            params.append((k, a.astype(float)))
        elif a.ndim == 2 and k in VECTOR_KEYS:
            # 1-BASED, like the physics: b[1]..b[P] and C[1]..C[P].  Zero-based labels
            # made `C[0]` mean C_1 = 1 + q0, a column a reader takes for C_0 = 1.
            for j in range(a.shape[1]):
                params.append(('%s[%d]' % (k, j + 1), a[:, j].astype(float)))
    if not params:
        return None, None, None
    n = len(params[0][1])
    return params, np.ones(n), 'NUTS (equally weighted draws)'


def find_chains(groups=None):
    # dc.CHAIN_DIR, not a relative 'chains' -- this script must give the same answer from
    # any working directory, and it must see a CHAINS_DIR override like everything else.
    root = dc.CHAIN_DIR
    out = []
    for grp in sorted(os.listdir(root)):
        gdir = os.path.join(root, grp)
        if not os.path.isdir(gdir) or (groups and grp not in groups):
            continue
        for fn in sorted(os.listdir(gdir)):
            if fn.endswith('.npz'):
                out.append((grp, os.path.join(gdir, fn)))
    return out


def load_validated(group, path):
    """A chain, through dc.cache_load, so its manifest is CHECKED.

    Raw np.load accepted caches the main loader would have refused -- a cell written
    under a superseded schema or a different BASELINE reads back as a perfectly ordinary
    array, and nothing about the numbers says which run produced them."""
    cell = os.path.basename(path)[:-4]
    c = dc.cache_load(group, cell, ignore_p=True)
    if c is None:
        return None
    return c


# ---------------------------------------------------------------------------------------
# 3.  Is the parameter MEASURED, or is the posterior just the prior?
# ---------------------------------------------------------------------------------------
# The mode/HPD summary above says how far a posterior is from a Gaussian.  It does NOT say
# whether the posterior carries any information, and the two failures look nothing alike
# while both producing a large `shift`/`asym`.  `z1` is the case that forced this section:
# under DES+legacy its posterior is FLAT across the whole of U(0.001, 0.50), so the mean
# and std of the samples reproduce the prior's own mean (0.2505) and std (0.1440) to two
# digits.  Quoting `0.284 +/- 0.140` then reports the prior, not the data -- and the
# Gaussian test flags it as the single most non-Gaussian parameter in the run, which
# invites the reader to quote a MODE that is only the tilt of a flat density.
#
# So each parameter with a known uniform prior box is also measured against that box:
#
#   kl          KL(posterior || prior) in nats, from a 50-bin histogram over the box.
#               0 means the posterior IS the prior.  This is the primary statistic.
#   width_ratio sigma_post / sigma_prior, with sigma_prior = (hi-lo)/sqrt(12).  1.0 means
#               the width was not reduced.  Note it is NOT sufficient on its own: a sharp
#               peak sitting on a flat plateau (des_only/lfree3 z1) scores ~1 while
#               genuinely carrying information, which is exactly why kl leads.
#   hpd_frac    fraction of the prior range covered by the 68% HPD.  0.68 means flat.
#   edge_lo/hi  posterior mass in the outer 5% of the box.  Above 5% the density there is
#               at or above the prior's own level, so the posterior is busy at the wall.
#   wall_lo/hi  density AT the wall divided by the density at the peak.  This is the one
#               that says whether the wall CUT the posterior off, and it is reported, not
#               thresholded, because the answer is a matter of degree: ~1 means the
#               posterior is still at full height where the box ends, so the quoted std is
#               a truncation; a small value means it rose, peaked and fell inside the box
#               and is merely close to the wall.
PRIOR_TOL_KL = 0.10         # below this the posterior is indistinguishable from the prior
PRIOR_TOL_WIDTH = 0.90      # ... and the width was not reduced either
PRIOR_EDGE_FRAC = 0.05      # outer fraction of the box counted as "at the edge"
PRIOR_EDGE_TOL = 0.05       # posterior mass there above which the box is doing the work
PRIOR_TOL_SHIFT = 6.0       # |mean_post - mean_prior|/sigma_prior above which
                            # the pairing itself is suspect, not the parameter
PRIOR_KL_BINS = 50


def prior_boxes():
    """{parameter: (lo, hi)} for every UNIFORM prior this repo samples under.

    Read from the modules that define them, never re-typed here, so a prior that is
    widened in `dovekie_evidence` or `BASELINE` cannot silently keep an old box in the
    diagnostic.  Parameters whose prior is NOT a box -- `rd` under the Planck Gaussian --
    and those that are not sampled at all -- `q0`, `H0_at_0` -- are served by
    `prior_reference` instead, not left untested."""
    import dovekie_evidence as de
    B = dc.BASELINE
    box = {'Om': (de.OM_LO, de.OM_HI),
           'A': (-de.A_MAX, de.A_MAX),
           'z1': (de.Z1_MIN, de.Z1_MAX),
           'w0': (de.W0_LO, de.W0_HI),
           'wa': (de.WA_LO, de.WA_HI),
           'M': (B['M_LO'], B['M_HI']),
           'Delta': (-B['DELTA_HALFWIDTH'], B['DELTA_HALFWIDTH'])}
    # H0 is sampled from a wider box in the evidence cells than in the cosmographic
    # ladder, so it is resolved per group rather than stored under one name.
    box['_H0_evidence'] = (de.H0_LO, de.H0_HI)
    box['_H0_ladder'] = (B['H0_LO'], B['H0_HI'])
    # Neither the node values v_j nor the C_k have a box prior: the shape prior is uniform
    # over the envelope region, whose marginals are not uniform, so both are served by
    # prior_reference's exact draws instead.
    return box


_PRIOR_BOX_CACHE = {}


def prior_box(name, group):
    if not _PRIOR_BOX_CACHE:
        try:
            _PRIOR_BOX_CACHE.update(prior_boxes())
        except Exception:      # noqa: BLE001 -- diagnostic only; never block the CLs
            _PRIOR_BOX_CACHE['_failed'] = True
    b = _PRIOR_BOX_CACHE
    if b.get('_failed'):
        return None
    if name in ('H0', 'H0_LCDM'):
        return b['_H0_evidence'] if group == 'evidence' else b['_H0_ladder']
    return b.get(name)


# --- Which prior does a given (group, cell, parameter) live under? ----------------------
# Three answers, and the point of the section is that all three are answered rather than
# the awkward ones being skipped:
#
#   ('box',   (lo, hi))     a uniform box; exact, no sampling
#   ('gauss', (mu, sigma))  the Planck r_d prior; exact, no sampling
#   ('draws', (values, w))  everything else, from `prior_reference`: the INDUCED prior of
#                           a derived quantity, obtained by pushing prior draws through
#                           the same map the model uses and weighting by the same
#                           positivity factor.  `q0` and `H(0)` are only testable this way.
_INDUCED_CACHE = {}


def _cell_config(group, cell):
    """(sample, config, background, joint_delta, z_floor) for a NUTS cell, from its name.

    The names are built by the tasks in `dovekie_cosmography`, so this parses rather than
    guesses; an unrecognised name returns None and the parameter simply goes untested
    instead of being tested against the wrong prior."""
    stem = cell[:-3] if cell.endswith('_P%d' % dc.BASELINE['P']) else cell
    if group == 'ladder':
        cfg, rest = stem[0], stem[2:]
        for bg in ('cosmo', 'lcdm'):
            if rest.endswith('_' + bg):
                return (rest[:-len(bg) - 1], cfg, bg, False, 0.0)
        return None
    if group == 'legacy':
        for bg in ('cosmo', 'lcdm'):
            if stem.startswith(bg + '_'):
                rest = stem[len(bg) + 1:]
                for suf, jd in (('_joint', True), ('_fixed', False)):
                    if rest.endswith(suf):
                        return (rest[:-len(suf)], 'A', bg, jd, 0.0)
        return None
    if group == 'floorsweep':
        smp, _, fl = stem.rpartition('_')
        try:
            return (smp, 'A', 'cosmo', False, float(fl))
        except ValueError:
            return None
    return None


def induced_prior(group, cell, summ):
    """{parameter: (values, weights)} of the prior this cell was actually sampled under."""
    key = (group, cell)
    if key in _INDUCED_CACHE:
        return _INDUCED_CACHE[key]
    out = None
    try:
        import prior_reference as pr
        if group == 'evidence':
            model = summ.get('model')
            if model:
                d, w = pr.evidence_prior(model)
                out = {k: (v, w) for k, v in d.items()}
        elif group == 'ap':
            # The AP cells sample `sn_only_model`: the same Bernstein prior, enforced to
            # the AP ceiling, and the same positivity factor, but no r_d, no M and no BAO.
            smp = cell.rpartition('_')[0]
            d, w = pr.cosmo_prior(smp, 'A', 'cosmo', y_max=dc.shape_y_max('ap'))
            out = {k: (v, w) for k, v in d.items()
                   if k == 'H0' or k == 'q0' or k.startswith('v[') or k.startswith('C[')}
        else:
            cfg = _cell_config(group, cell)
            if cfg is not None:
                smp, config, bg, jd, zf = cfg
                d, w = pr.cosmo_prior(smp, config, bg, joint_delta=jd, z_floor=zf)
                out = {k: (v, w) for k, v in d.items()}
    except Exception:      # noqa: BLE001 -- diagnostic only; never block the CLs
        out = None
    _INDUCED_CACHE[key] = out
    return out


def prior_ref(name, group, cell, summ):
    """('box'|'gauss'|'draws', payload) or None if this parameter has no known prior."""
    # r_d is the one sampled parameter with a non-uniform prior, and Table 1 quotes it in
    # a result column in every configuration-A row, so it must not go untested.
    if name == 'rd':
        rd_free = (group in ('ladder',) and cell.startswith('B_'))
        if rd_free:
            return ('box', (dc.BASELINE['RD_FREE_LO'], dc.BASELINE['RD_FREE_HI']))
        return ('gauss', (dc.BASELINE['RD_PLANCK'], dc.BASELINE['SIG_RD']))
    box = prior_box(name, group)
    if box is not None:
        return ('box', box)
    ind = induced_prior(group, cell, summ)
    if ind and name in ind:
        return ('draws', ind[name])
    return None


def _prior_bin_mass(ref, edges):
    """Prior mass in each bin of `edges`, plus (mean, std) of the prior."""
    kind, payload = ref
    if kind == 'box':
        lo_b, hi_b = payload
        m = np.clip(edges, lo_b, hi_b)
        q = np.diff(m) / (hi_b - lo_b)
        return q, 0.5 * (lo_b + hi_b), (hi_b - lo_b) / math.sqrt(12.0)
    if kind == 'gauss':
        mu, sig = payload
        from scipy.stats import norm
        c = norm.cdf(edges, mu, sig)
        return np.diff(c), mu, sig
    v, w = payload
    wn = np.asarray(w, dtype=float)
    wn = wn / wn.sum()
    idx = np.digitize(v, edges) - 1
    inside = (idx >= 0) & (idx < len(edges) - 1)
    q = np.bincount(idx[inside], weights=wn[inside], minlength=len(edges) - 1)
    mean = float(np.sum(wn * v))
    std = float(np.sqrt(np.sum(wn * (v - mean) ** 2)))
    return q, mean, std


def prior_diagnostics(values, weights, ref):
    """(kl, width_ratio, shift_prior, edge_lo, edge_hi, wall_lo, wall_hi) against `ref`.

    The grid is the prior's own central 99.9% range, so a posterior far narrower than its
    prior lands in a few bins and scores a large KL, which is the intended reading.  A
    posterior that has moved OUTSIDE the prior's range would be a bug, not a result, and
    shows up as `shift_prior` far above 1."""
    q_edges = None
    kind, payload = ref
    if kind == 'box':
        lo_b, hi_b = payload
    elif kind == 'gauss':
        mu, sig = payload
        lo_b, hi_b = mu - 4.0 * sig, mu + 4.0 * sig
    else:
        pv, pw = payload
        pwn = np.asarray(pw, dtype=float)
        pwn = pwn / pwn.sum()
        lo_b = weighted_quantile(np.asarray(pv, dtype=float), pwn, 0.0005)
        hi_b = weighted_quantile(np.asarray(pv, dtype=float), pwn, 0.9995)
    width = float(hi_b - lo_b)
    if not (width > 0):
        return None
    wn = np.asarray(weights, dtype=float)
    wn = wn / wn.sum()
    nb = PRIOR_KL_BINS
    edges = np.linspace(lo_b, hi_b, nb + 1)
    idx = np.clip(np.digitize(values, edges) - 1, 0, nb - 1)
    q = np.bincount(idx, weights=wn, minlength=nb)
    p, prior_mean, prior_std = _prior_bin_mass(ref, edges)
    # A prior bin the draws never reached is not evidence of infinite information, only of
    # a finite prior sample, so it is floored at one draw's worth before the log.
    p = np.maximum(np.asarray(p, dtype=float), 1e-9)
    p = p / p.sum()
    m = q > 0
    kl = float(np.sum(q[m] * np.log(q[m] / p[m])))
    mean = float(np.sum(wn * values))
    std = float(np.sqrt(np.sum(wn * (values - mean) ** 2)))
    width_ratio = std / prior_std if prior_std > 0 else float('nan')
    shift_prior = abs(mean - prior_mean) / prior_std if prior_std > 0 else float('nan')

    # The edge test only means anything for a hard box: a Gaussian or an induced prior has
    # no wall for the posterior to pile against.
    if kind != 'box':
        return kl, width_ratio, shift_prior, 0.0, 0.0, float('nan'), float('nan')
    e = PRIOR_EDGE_FRAC * width
    edge_lo = float(np.sum(wn[values < lo_b + e]))
    edge_hi = float(np.sum(wn[values > hi_b - e]))
    # Mass alone says the posterior is BUSY near the wall; it does not say whether the
    # wall cut it off.  The ratio of the density AT the wall to the density at the peak
    # is what separates the two, and it is reported rather than thresholded: ~1 means the
    # posterior is still at full height when the box ends (truncated), while a small
    # value means it rose, peaked and fell inside the box (measured, merely close).
    ds = gaussian_filter1d(q, 1.0)
    peak = float(ds.max())
    wall_lo = float(ds[0]) / peak if peak > 0 else float('nan')
    wall_hi = float(ds[-1]) / peak if peak > 0 else float('nan')
    return kl, width_ratio, shift_prior, edge_lo, edge_hi, wall_lo, wall_hi


# ---------------------------------------------------------------------------------------
# 4.  Driver
# ---------------------------------------------------------------------------------------
def _prior_label(r):
    """The prior a NOT MEASURED entry is reproducing, written out so the reader can see
    the quoted number IS it rather than having to look the box up."""
    kind, b = r.get('prior_kind'), r.get('prior_box')
    if kind == 'box' and b:
        return 'U(%.4g, %.4g) = %.4g +/- %.3g' % (
            b[0], b[1], 0.5 * (b[0] + b[1]), (b[1] - b[0]) / math.sqrt(12.0))
    if kind == 'gauss' and b:
        return 'N(%.5g, %.4g) -- the prior itself' % (b[0], b[1])
    return 'induced prior of a derived quantity'


def gaussian_gap(mode, lo, hi, mean, std):
    """How far the mode+HPD summary has moved from mean+/-std, in units of std.

    Two diagnostics, both dimensionless: the shift of the centre, and the asymmetry of the
    interval.  A posterior that really is Gaussian scores ~0 on both, so anything large is
    a parameter whose published `mean +/- std` misrepresents it."""
    if not (std > 0):
        return float('nan'), float('nan')
    return abs(mode - mean) / std, abs((hi - mode) - (mode - lo)) / std


def analyse_chain(path, group, plot=False):
    d = load_validated(group, path)
    if d is None:
        print('  REFUSED by cache_load (stale manifest): %s'
              % os.path.relpath(path, dc.HERE))
        return None
    params, w, sampler = read_chain(path, d)
    if params is None:
        return None
    stem = path[:-4] if path.endswith('.npz') else path
    txt_path = stem + '_CLs.txt'

    summ = (json.loads(str(d['summary_json'])) if 'summary_json' in d else {})

    lines = ['# Mode and 1-sigma highest-posterior-density limits',
             '# chain   : %s' % os.path.relpath(path, dc.HERE),
             '# sampler : %s' % sampler,
             '# samples : %d' % len(params[0][1]),
             '# method  : Print_CLs_sigma.py algorithm -- robust window, weighted binning',
             '#           at dx = sigma/K with K escalated over %s, density smoothed by'
             % (Bins_per_sigma_options,),
             '#           sigma/%g, %.2f%% HPD contour solved sub-pixel.'
             % (1.0 / Smooth_frac, 100.0 * P_threshold),
             '#',
             '# limits  : a row reading `< v` or `> v` is ONE-SIDED: the opposite contour',
             '#           crossing fell outside the range the samples cover, so it bounds',
             '#           nothing and the two-sided interval is not defined.  Read it as',
             '#           P(x < v) = %.2f%% (or P(x > v) = %.2f%%).  `v` is the same contour'
             % (100.0 * P_threshold, 100.0 * P_threshold),
             '#           crossing the two-sided row would have printed, not a re-fit.',
             '#',
             '# kl      : KL(posterior || prior) in nats; 0 means the posterior IS the',
             '#           prior.  s/sp is sigma_post/sigma_prior.  The prior is the box',
             '#           for a boxed parameter, N(147.05, 0.30) for r_d, and for a',
             '#           DERIVED quantity the induced prior from prior_reference.py --',
             '#           prior draws pushed through the same map, weighted by the same',
             '#           positivity factor.  Blank only where no prior is known.',
             '#',
             '# %-12s %14s %14s %14s   %10s %10s %7s %6s   %s'
             % ('parameter', 'mode', '-1sigma', '+1sigma', 'mean', 'std', 'kl', 's/sp',
                'notes')]
    records = []

    for name, values in params:
        values = np.asarray(values, dtype=float)
        ok = np.isfinite(values)
        v, wv = values[ok], np.asarray(w, dtype=float)[ok]
        window = get_robust_window(v, wv)
        if window is None:
            lines.append('  %-12s %14.8f  (constant)' % (name, float(np.min(v))))
            continue
        try:
            (lo, mode, hi, dx, xc, dens, cdens, warn) = resolve_mode_CL_in_window(
                v, wv, *window)
        except Exception as e:      # noqa: BLE001
            lines.append('  %-12s  FAILED: %s' % (name, e))
            continue

        # The moments, for the side-by-side comparison.  Weighted, so they reproduce the
        # `*_mean`/`*_std` already stored in summary_json for the dynesty cells.
        wn = wv / wv.sum()
        mean = float(np.sum(wn * v))
        std = float(np.sqrt(np.sum(wn * (v - mean) ** 2)))

        notes = list(warn)
        px_lo, px_hi = (mode - lo) / dx, (hi - mode) / dx
        if px_lo < Pixel_warning_threshold or px_hi < Pixel_warning_threshold:
            notes.append('under-resolved: %.1f/%.1f pixels to the 1-sigma limits'
                         % (px_lo, px_hi))
        shift, asym = gaussian_gap(mode, lo, hi, mean, std)

        # Second stage: is either crossing actually a BOUND?  A crossing that falls
        # outside the sampled range excludes no posterior mass, so the interval collapses
        # to a one-sided limit on the other side -- at the same contour, hence the same
        # number.  See ONE-SIDED LIMITS in the module docstring.
        x_min, x_max = effective_support(v, wv)
        limit = classify_limit(lo, hi, x_min, x_max)
        limit_value = {'upper': hi, 'lower': lo}.get(limit)
        # The weighted mass the limit actually encloses, measured on the SAMPLES rather
        # than on the smoothed grid the contour was solved on.  These differ, and for a
        # posterior against a hard wall they differ visibly: the sigma/5 smoothing spreads
        # density across the wall into bins holding no samples, that leaked mass counts
        # towards the 68.27% inside the contour, and the limit therefore sits further out
        # than a plain quantile would put it.  It is reported, not corrected: the crossing
        # is what was asked for, and the same leak is in the two-sided intervals, so
        # silently switching this one branch to a quantile would make the two disagree.
        enclosed = float('nan')
        if limit in ('upper', 'lower'):
            wn_all = wv / wv.sum()
            enclosed = float(np.sum(wn_all[v < hi] if limit == 'upper'
                                    else wn_all[v > lo]))
        if limit == 'upper':
            notes.append('ONE-SIDED: the -1sigma crossing (%.6g) falls below the lowest '
                         'sample (%.6g) and bounds nothing -- quote %s < %.6g '
                         '(nominal %.1f%%, encloses %.1f%% of the samples)'
                         % (lo, x_min, name, hi, 100.0 * P_threshold, 100.0 * enclosed))
        elif limit == 'lower':
            notes.append('ONE-SIDED: the +1sigma crossing (%.6g) falls above the highest '
                         'sample (%.6g) and bounds nothing -- quote %s > %.6g '
                         '(nominal %.1f%%, encloses %.1f%% of the samples)'
                         % (hi, x_max, name, lo, 100.0 * P_threshold, 100.0 * enclosed))
        elif limit == 'none':
            notes.append('NO LIMIT: the %.1f%% contour clears the sampled range at BOTH '
                         'ends ([%.6g, %.6g]) -- neither crossing bounds anything'
                         % (100.0 * P_threshold, x_min, x_max))

        # The prior comparison comes FIRST, because it changes what the Gaussian test
        # means.  For a flat posterior the mode is the tilt of a flat density and the
        # asymmetry is the distance to the two box walls; reporting those as
        # `NON-GAUSSIAN` would invite exactly the mis-quote this is meant to prevent.
        ref = prior_ref(name, group, os.path.basename(stem), summ)
        pd = prior_diagnostics(v, wv, ref) if ref else None
        kl = width_ratio = shift_prior = float('nan')
        edge_lo = edge_hi = wall_lo = wall_hi = float('nan')
        prior_dominated = False
        if pd is not None:
            kl, width_ratio, shift_prior, edge_lo, edge_hi, wall_lo, wall_hi = pd
            prior_dominated = (kl < PRIOR_TOL_KL and width_ratio > PRIOR_TOL_WIDTH)
            if shift_prior > PRIOR_TOL_SHIFT:
                # The posterior sits outside the prior it was drawn from -- that is a bug
                # in the pairing, not a result, so say so rather than reporting a number.
                notes.append('PRIOR MISMATCH: posterior mean is %.1f prior sigma from the '
                             'prior mean; check the prior this cell is paired with'
                             % shift_prior)
            if prior_dominated:
                # The remedy differs by prior kind, so the note says which one applies:
                # an imported measurement handed back unchanged is an INPUT being
                # reported in a result column, whereas a flat box is a parameter the data
                # never constrained and that has no summary to quote at all.
                fix = ('this is the imported prior, not a result'
                       if ref[0] == 'gauss' else
                       'quote a limit or say so, not mean+/-std')
                notes.append('NOT MEASURED: posterior = prior (KL %.3f nat, sigma is '
                             '%.2f of the prior width) -- %s'
                             % (kl, width_ratio, fix))
            elif max(edge_lo, edge_hi) > PRIOR_EDGE_TOL:
                notes.append('PRIOR EDGE: %.0f%% of the mass in the outer %.0f%% of the '
                             'box -- the limit is the box, not the data'
                             % (100 * max(edge_lo, edge_hi), 100 * PRIOR_EDGE_FRAC))
        # A one-sided entry is already being told to quote a limit; adding NON-GAUSSIAN on
        # top of that would point at the mode + interval it has just been told not to use.
        if (shift > 0.3 or asym > 0.3) and not prior_dominated and limit == 'two-sided':
            notes.append('NON-GAUSSIAN: mode-mean %.2f sigma, asymmetry %.2f sigma'
                         % (shift, asym))

        # Same 44-character field either way, so the columns line up whichever branch a
        # row takes: 14 + 1 + 14 + 1 + 14 for mode/-1sigma/+1sigma, or the limit and its
        # reading for a one-sided row.
        if limit in ('upper', 'lower'):
            cols = '%14s %-29s' % (('%s %.6f' % ('<' if limit == 'upper' else '>',
                                                 limit_value)),
                                   '  (%.1f%% %s limit)' % (100.0 * P_threshold, limit))
        else:
            cols = '%14.6f %14.6f %14.6f' % (mode, mode - lo, hi - mode)
        lines.append('  %-12s %s   %10.6f %10.6f %7s %6s   %s'
                     % (name, cols, mean, std,
                        '%.3f' % kl if pd else '', '%.2f' % width_ratio if pd else '',
                        '; '.join(notes)))
        records.append(dict(group=group, chain=os.path.basename(stem), param=name,
                            mode=mode, minus=mode - lo, plus=hi - mode,
                            limit=(limit if limit != 'two-sided' else None),
                            limit_value=limit_value, limit_enclosed=enclosed,
                            x_min=x_min, x_max=x_max,
                            mean=mean, std=std, shift=shift, asym=asym,
                            kl=kl, width_ratio=width_ratio,
                            shift_prior=shift_prior,
                            prior_kind=ref[0] if ref else None,
                            edge_lo=edge_lo, edge_hi=edge_hi,
                            wall_lo=wall_lo, wall_hi=wall_hi,
                            prior_dominated=prior_dominated,
                            prior_box=(list(ref[1]) if ref and ref[0] in ('box', 'gauss')
                                       else None),
                            pixels_lo=px_lo, pixels_hi=px_hi, notes=notes,
                            sampler=sampler.split(' ')[0],
                            sample=summ.get('sample'), anchor=summ.get('anchor'),
                            model=summ.get('model')))

        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(xc, dens, color='royalblue', linewidth=2)
            plt.axhline(y=cdens, color='red', linestyle='--',
                        label='68.27%% contour = %.5g' % cdens)
            plt.axvline(x=mode, color='k', linestyle=':', label='mode = %.5g' % mode)
            if limit in ('upper', 'lower'):
                # Draw the crossing that IS a bound, and the sample edge that killed the
                # other one, so the reason for the one-sided reading is visible.
                plt.axvline(x=limit_value, color='darkorange', linewidth=2,
                            label='%s %s %.5g (%.1f%%)'
                            % (name, '<' if limit == 'upper' else '>', limit_value,
                               100.0 * P_threshold))
                plt.axvline(x=(x_min if limit == 'upper' else x_max), color='grey',
                            linestyle='-.', label='sample edge')
            plt.xlabel('%s' % name)
            plt.ylabel('Probability density')
            plt.title('%s: %s' % (os.path.basename(stem), name))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.show()

    with open(txt_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return txt_path, records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('USAGE')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', nargs='*', help='chain subdirectories to process')
    ap.add_argument('--plot', action='store_true', help='show density + contour per param')
    a = ap.parse_args(argv)

    chains = find_chains(a.group)
    if not chains:
        sys.exit('no chains found under %s' % os.path.join(dc.HERE, 'chains'))

    all_records = []
    for group, path in chains:
        res = analyse_chain(path, group, plot=a.plot)
        if res is None:
            print('  skipped (no numeric parameters): %s' % path)
            continue
        txt_path, records = res
        all_records.extend(records)
        print('  %-58s -> %s' % (os.path.relpath(path, dc.HERE),
                                 os.path.basename(txt_path)))

    os.makedirs(dc.RESULT_DIR, exist_ok=True)
    json_path = os.path.join(dc.RESULT_DIR, 'mode_CLs.json')
    with open(json_path, 'w') as fh:
        json.dump(all_records, fh, indent=2)

    # The aggregate file leads with the parameters that are NOT MEASURED, because those
    # need a different fix from the merely-skewed ones: a skewed posterior should be
    # re-quoted as mode + HPD, but a prior-dominated one has no summary to re-quote and
    # has to be reported as a limit or dropped.
    unmeasured = [r for r in all_records if r.get('prior_dominated')]
    unmeasured.sort(key=lambda r: r['kl'])
    edge = [r for r in all_records
            if not r.get('prior_dominated')
            and max(r.get('edge_lo') or 0.0, r.get('edge_hi') or 0.0) > PRIOR_EDGE_TOL]
    edge.sort(key=lambda r: -max(r['edge_lo'], r['edge_hi']))
    onesided = [r for r in all_records if r.get('limit') in ('upper', 'lower', 'none')]
    onesided.sort(key=lambda r: (r['limit'], r['group'], r['chain']))
    flagged = [r for r in all_records
               if (r['shift'] > 0.3 or r['asym'] > 0.3) and not r.get('prior_dominated')
               and r.get('limit') is None]
    flagged.sort(key=lambda r: -max(r['shift'], r['asym']))

    sum_path = os.path.join(dc.RESULT_DIR, 'mode_CLs_summary.txt')
    with open(sum_path, 'w') as fh:
        fh.write('Parameters whose published `mean +/- std` misrepresents the posterior.\n'
                 'Four separate failures, in decreasing order of how much they matter.\n')

        fh.write('\n\n(1) NOT MEASURED -- the posterior is the prior.\n'
                 '    KL(post || prior) < %.2f nat AND sigma_post > %.2f sigma_prior, so\n'
                 '    the quoted mean and std are the PRIOR\'s own.  For a box that means\n'
                 '    the data never constrained the parameter and there is nothing to\n'
                 '    re-quote; for r_d it means an imported measurement is being reported\n'
                 '    back in a result column.\n\n'
                 % (PRIOR_TOL_KL, PRIOR_TOL_WIDTH))
        fh.write('%-10s %-30s %-10s %16s %7s %6s %-22s\n'
                 % ('group', 'chain', 'param', 'mean +/- std', 'KL', 's/sp',
                    'the prior it reproduces'))
        for r in unmeasured:
            fh.write('%-10s %-30s %-10s %16s %7.3f %6.2f %-30s\n'
                     % (r['group'], r['chain'][:30], r['param'],
                        '%.4g +/- %.3g' % (r['mean'], r['std']), r['kl'],
                        r['width_ratio'], _prior_label(r)))
        fh.write('\n%d of %d parameter-chain pairs are not measured.\n'
                 % (len(unmeasured), len(all_records)))

        fh.write('\n\n(2) PRIOR EDGE -- constrained, but the box is doing part of the work.\n'
                 '    More than %.0f%% of the posterior mass lies in the outer %.0f%% of the\n'
                 '    prior box.  Read wall/pk, the density at the wall over the density at\n'
                 '    the peak: near 1 the box has TRUNCATED the posterior and the quoted std\n'
                 '    is not the data\'s; small, and the posterior merely peaks near the wall.\n\n'
                 % (100 * PRIOR_EDGE_TOL, 100 * PRIOR_EDGE_FRAC))
        fh.write('%-10s %-30s %-10s %16s %8s %8s %9s %9s\n'
                 % ('group', 'chain', 'param', 'mean +/- std', 'P(low)', 'P(high)',
                    'wall/pk_lo', 'wall/pk_hi'))
        for r in edge:
            fh.write('%-10s %-30s %-10s %16s %8.3f %8.3f %9.2f %9.2f\n'
                     % (r['group'], r['chain'][:30], r['param'],
                        '%.4g +/- %.3g' % (r['mean'], r['std']),
                        r['edge_lo'], r['edge_hi'], r['wall_lo'], r['wall_hi']))
        fh.write('\n%d of %d parameter-chain pairs sit against a prior edge.\n'
                 % (len(edge), len(all_records)))

        fh.write('\n\n(3) ONE-SIDED -- quote a limit, not a mode and an interval.\n'
                 '    One of the two %.1f%% contour crossings fell outside the range the\n'
                 '    samples cover, so it excludes no posterior mass and is not a bound:\n'
                 '    it is where the smoothed density ran out of samples.  The surviving\n'
                 '    crossing IS a %.1f%% limit, because all the mass on the open side is\n'
                 '    inside the contour.  `none` means both crossings cleared the samples\n'
                 '    and there is no limit of either sign to quote.\n'
                 '    `encl` is the weighted sample mass the limit actually encloses.  It\n'
                 '    exceeds the nominal %.1f%% when the sigma/%g smoothing spreads density\n'
                 '    across a hard prior wall into bins holding no samples: that leaked\n'
                 '    mass counts inside the contour, so the crossing sits further out.\n'
                 '    A row marked (1) is ALSO prior-dominated, from section (1): a flat\n'
                 '    posterior has no interior contour, so its "limit" is the far wall of\n'
                 '    the box arriving on schedule -- an `encl` far above %.1f%% is the tell.\n'
                 '    Those are not results and must not be quoted as limits either.\n\n'
                 % (100 * P_threshold, 100 * P_threshold, 100 * P_threshold,
                    1.0 / Smooth_frac, 100 * P_threshold))
        fh.write('%-10s %-30s %-10s %18s %7s %16s %24s\n'
                 % ('group', 'chain', 'param', 'limit', 'encl', 'mean +/- std',
                    'sampled range'))
        for r in onesided:
            lab = ('no limit (both ends open)' if r['limit'] == 'none' else
                   '%s %.5g' % ('<' if r['limit'] == 'upper' else '>', r['limit_value']))
            fh.write('%-10s %-30s %-10s %18s %7s %16s %24s %s\n'
                     % (r['group'], r['chain'][:30], r['param'], lab,
                        ('' if r['limit'] == 'none'
                         else '%.1f%%' % (100 * r['limit_enclosed'])),
                        '%.4g +/- %.3g' % (r['mean'], r['std']),
                        '[%.4g, %.4g]' % (r['x_min'], r['x_max']),
                        '(1)' if r.get('prior_dominated') else ''))
        fh.write('\n%d of %d parameter-chain pairs are one-sided.\n'
                 % (len(onesided), len(all_records)))

        fh.write('\n\n(4) NON-GAUSSIAN -- measured, two-sided, but skewed.  Re-quote as\n'
                 '    mode + HPD.\n'
                 '    Ranked by max(|mode-mean|/std, |asymmetry|/std); both are 0 for a\n'
                 '    Gaussian.  Full per-chain output is in the *_CLs.txt beside each chain.\n\n')
        fh.write('%-10s %-30s %-10s %16s %16s %8s %8s\n'
                 % ('group', 'chain', 'param', 'mode -1s +1s', 'mean +/- std',
                    'shift', 'asym'))
        for r in flagged:
            fh.write('%-10s %-30s %-10s %16s %16s %8.2f %8.2f\n'
                     % (r['group'], r['chain'][:30], r['param'],
                        '%.4g -%.3g +%.3g' % (r['mode'], r['minus'], r['plus']),
                        '%.4g +/- %.3g' % (r['mean'], r['std']),
                        r['shift'], r['asym']))
        fh.write('\n%d of %d parameter-chain pairs are skewed.\n'
                 % (len(flagged), len(all_records)))
    print('\nwrote %s\nwrote %s' % (json_path, sum_path))
    print('of %d parameter-chain pairs: %d NOT MEASURED (posterior = prior), '
          '%d at a prior edge, %d one-sided (quote a limit), %d skewed.'
          % (len(all_records), len(unmeasured), len(edge), len(onesided), len(flagged)))
    return all_records


if __name__ == '__main__':
    main()
