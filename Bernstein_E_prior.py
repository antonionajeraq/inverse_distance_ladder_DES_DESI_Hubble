#!/usr/bin/env python3
"""
Bernstein_E_prior.py
====================

Bounded-polynomial prior for the normalised expansion rate

    E(y) = H(y)/H0 = 1 + sum_{k=1}^{K} C_k y^k,      y = 1 - a = z/(1+z),

which keeps  L(y) <= E(y) <= U(y)  everywhere on [0, y_max], exactly and with
no rejection step.  Default envelope: L = 0, U = 2 + 7y (twice 1 + 3.5y).
Requires only numpy (dynesty only for the optional --demo).

Why the priors are not put on C_k directly
------------------------------------------
The set of C_k satisfying the envelope is convex but strongly tilted, so no
axis-aligned box on the C_k matches it: a box large enough to cover it lets E
leave the envelope, a box small enough to stay inside wastes most of the
allowed region, and box + rejection has an acceptance that collapses with K
(and needs a -ln(acceptance) correction to lnZ).

Bernstein reparametrisation
---------------------------
With t = y / y_max in [0, 1] write

    E = sum_{j=0}^{K} b_j B_{j,K}(t),      B_{j,K}(t) = C(K,j) t^j (1-t)^(K-j).

The B_{j,K} are non-negative and sum to 1, so E(t) is a weighted average of
the b_j.  An envelope polynomial of degree <= K has exact Bernstein
coefficients l_j, u_j (degree elevation), and E - U = sum_j (b_j - u_j) B_{j,K},
so

    l_j <= b_j <= u_j  for every j   ==>   L(y) <= E(y) <= U(y) on [0, y_max].

Prior: b_0 = 1 pinned (E(0) = 1); b_j ~ U(l_j, u_j) for j = 1..K, i.e. exactly
K free parameters; H0 ~ U(50, 90).  Sampled vector: theta = [H0, b_1..b_K].

C = M b with M fixed and invertible, so a flat prior on b IS a flat prior on
the C_k (constant Jacobian; the evidence is identical), supported on a tilted
parallelepiped instead of an axis-aligned box.  Sample b, convert to C_k for
the likelihood and for plots.

The condition is sufficient, not necessary: the box covers a fraction of the
full allowed set that falls with K.  What it leaves out is dominated by rapidly
oscillating polynomials with steep endpoint slopes.

Exact identities
----------------
    E(0)     = b_0 = 1
    E(y_max) = b_K
    C_1      = K (b_1 - 1) / y_max,        q0 = C_1 - 1

so q0 depends on b_1 alone and has a FLAT prior, but its width grows with K:
q0 prior widths are not comparable across orders.

Reporting basis
---------------
The paper's triangle plots show {H0, q0, C_2, ..., C_K} (columns of
`physical()`), with prior draws from `prior_samples(basis="physical")`
overlaid.  The joint prior on the C_k is uniform inside a parallelepiped, so
only q0 = C_1 - 1 has a uniform marginal; C_2..C_K have broad humped
marginals (uniform conditionals: the humps come from the shape of the allowed
region) and strong prior correlations (|corr| up to 0.92 at K = 4).
`hybrid()` gives the alternative {H0, q0, b_2, ..., b_K}, whose prior is an
exact box (uniform, independent marginals), but whose interior b_j do not
read as anything physical.

Caveats for reporting
---------------------
* Interior b_j are control values, NOT values of E: E(y) is a weighted
  average of the b_j with weight B_{j,K}(y/y_max) peaking at
  y_j = j y_max / K (`y_nodes`).  For an LCDM-shaped E at K = 4,
  y_max = 0.70, b_1..b_3 differ from E(y_j) by -11%, +13%, -46%, and are
  not even monotonic though E is.  Only b_K = E(y_max) is exact.
* The C_k are strongly correlated at high K (cond(M) ~ 1e3 at K = 6 for
  y_max ~ 0.7); derived quantities (q0, E at the data redshifts) are far
  better constrained than individual C_k.
* y_max is fixed before sampling (largest y in the data) and must be the same
  in every run whose evidences are compared: it sets where the envelope is
  enforced.  Beyond y_max, E is unconstrained and must not be used.

Usage
-----
    from Bernstein_E_prior import BernsteinEPrior, y_from_z, y_max_from_z

    prior = BernsteinEPrior(k_max=4, y_max=y_max_from_z(z_all_data))
    y_cc  = y_from_z(z_cc)

    def loglike(theta):
        H0 = theta[0]
        c  = prior.coeffs(theta)            # c[0] = 1, c[k] = C_k
        H  = H0 * prior.E(y_cc, c)
        return -0.5 * np.sum(((H - H_cc) / sig_cc) ** 2)

    sampler = dynesty.NestedSampler(loglike, prior.prior_transform, prior.ndim)
    ...
    phys = prior.physical(res.samples)      # columns: H0, C_1..C_K, q0
    pri  = prior.prior_samples(1_000_000, basis="physical")
    # getdist: MCSamples(samples=phys, weights=w,
    #                    names=prior.phys_names, labels=prior.phys_labels),
    #          same for pri; plot ['H0', 'q0', 'C2', ..., f'C{K}'],
    #          prior overlaid in grey

    python3 Bernstein_E_prior.py            # self-tests for K = 2..6
    python3 Bernstein_E_prior.py --demo 4   # dynesty fit to mock H(z) data
"""
import sys
from math import comb

import numpy as np


def y_from_z(z):
    z = np.asarray(z, float)
    return z / (1.0 + z)


def y_max_from_z(z_data):
    """y_max = largest y probed by the data; compute once, before sampling."""
    return float(np.max(y_from_z(z_data)))


def bernstein_to_monomial(n, y_max):
    """(n+1)x(n+1) matrix M with C = M @ b.

    b_j: degree-n Bernstein coefficients on t = y / y_max.
    C_k: coefficient of y^k.
    """
    M = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        for j in range(k + 1):
            M[k, j] = comb(n, k) * comb(k, j) * (-1) ** (k - j) / y_max ** k
    return M


def monomial_to_bernstein(coeffs, n, y_max):
    """Degree-n Bernstein coefficients (on t = y / y_max) of sum_k coeffs[k] y^k.

    Exact for any polynomial of degree <= n (degree elevation); this is how a
    polynomial envelope becomes a box on the b_j.
    """
    coeffs = np.atleast_1d(np.asarray(coeffs, float))
    if len(coeffs) > n + 1 and np.any(coeffs[n + 1:] != 0):
        raise ValueError(f"polynomial of degree {len(coeffs) - 1} cannot be "
                         f"represented at Bernstein degree {n}")
    m = min(len(coeffs), n + 1)
    a = np.zeros(n + 1)
    a[:m] = coeffs[:m] * y_max ** np.arange(m)          # monomial coeffs in t
    return np.array([sum(comb(j, k) / comb(n, k) * a[k] for k in range(j + 1))
                     for j in range(n + 1)])


def poly_eval(y, c):
    """Horner evaluation of sum_k c[..., k] y^k; c (..., K+1), y (m,) -> (..., m)."""
    y = np.asarray(y, float)
    c = np.asarray(c, float)
    out = np.broadcast_to(c[..., -1, None], c.shape[:-1] + y.shape).copy()
    for k in range(c.shape[-1] - 2, -1, -1):
        out = out * y + c[..., k, None]
    return out


class BernsteinEPrior:
    """Prior on [H0, b_1..b_K] guaranteeing L(y) <= E(y) <= U(y) on [0, y_max].

    lower, upper: monomial coefficients in y of the envelope polynomials
                  (degree <= k_max), e.g. upper=(2.0, 7.0) for U = 2 + 7y.
    """

    def __init__(self, k_max, y_max, lower=(0.0,), upper=(2.0, 7.0),
                 H0_range=(50.0, 90.0)):
        if int(k_max) != k_max or k_max < 1:
            raise ValueError("k_max must be a positive integer")
        if not 0.0 < y_max < 1.0:
            raise ValueError("y_max must lie in (0, 1)")
        self.k_max = K = int(k_max)
        self.y_max = float(y_max)
        self.lower = np.atleast_1d(np.asarray(lower, float))
        self.upper = np.atleast_1d(np.asarray(upper, float))
        self.H0_lo, self.H0_hi = map(float, H0_range)

        self.M = bernstein_to_monomial(K, self.y_max)
        l = monomial_to_bernstein(self.lower, K, self.y_max)
        u = monomial_to_bernstein(self.upper, K, self.y_max)
        if not l[0] <= 1.0 <= u[0]:
            raise ValueError("envelope must contain E(0) = 1")
        if np.any(u[1:] <= l[1:]):
            raise ValueError("empty Bernstein interval: envelopes too close for this k_max")
        self.b_lo, self.b_hi = l[1:], u[1:]
        self.ndim = 1 + K

        self.y_nodes = np.arange(K + 1) * self.y_max / K     # where B_{j,K} peaks

        self.param_names = ["H0"] + [f"b{j}" for j in range(1, K + 1)]
        self.hybrid_names = ["H0", "q0"] + [f"b{j}" for j in range(2, K + 1)]
        self.hybrid_labels = ["H_0", "q_0"] + [f"b_{j}" for j in range(2, K + 1)]
        self.phys_names = ["H0"] + [f"C{k}" for k in range(1, K + 1)] + ["q0"]
        self.phys_labels = ["H_0"] + [f"C_{k}" for k in range(1, K + 1)] + ["q_0"]

    # --- sampling ---------------------------------------------------------
    def prior_transform(self, u):
        """Unit cube -> theta = [H0, b_1..b_K] (dynesty convention)."""
        u = np.asarray(u, float)
        theta = np.empty(u.shape)
        theta[..., 0] = self.H0_lo + (self.H0_hi - self.H0_lo) * u[..., 0]
        theta[..., 1:] = self.b_lo + (self.b_hi - self.b_lo) * u[..., 1:]
        return theta

    def in_prior(self, theta):
        """Support test, for samplers that use a log-prior instead (e.g. MCMC)."""
        theta = np.asarray(theta, float)
        return bool(self.H0_lo <= theta[0] <= self.H0_hi
                    and np.all((self.b_lo <= theta[1:]) & (theta[1:] <= self.b_hi)))

    # --- conversions ------------------------------------------------------
    def coeffs(self, theta):
        """(1, C_1..C_K) from theta = [H0, b_1..b_K]; theta (ndim,) or (N, ndim)."""
        theta = np.asarray(theta, float)
        b = np.concatenate([np.ones(theta.shape[:-1] + (1,)), theta[..., 1:]], axis=-1)
        return b @ self.M.T

    E = staticmethod(poly_eval)

    def E_bernstein(self, y, theta):
        """E evaluated directly in the Bernstein basis (cross-check of coeffs + E)."""
        t = np.asarray(y, float) / self.y_max
        b = np.concatenate([[1.0], np.asarray(theta, float)[1:]])
        K = self.k_max
        return sum(b[j] * comb(K, j) * t ** j * (1 - t) ** (K - j) for j in range(K + 1))

    def physical(self, samples):
        """Chain in the reporting basis: columns H0, C_1..C_K, q0."""
        samples = np.atleast_2d(samples)
        c = self.coeffs(samples)
        return np.column_stack([samples[:, 0], c[:, 1:], c[:, 1] - 1.0])

    def hybrid(self, samples):
        """Chain in the alternative basis: columns H0, q0, b_2..b_K.

        Complete and non-redundant (q0 is affine in b_1 alone), with an exact
        box prior: uniform, independent marginals.
        """
        samples = np.atleast_2d(samples)
        q0 = self.k_max * (samples[:, 1] - 1.0) / self.y_max - 1.0
        return np.column_stack([samples[:, 0], q0, samples[:, 2:]])

    def hybrid_ranges(self):
        """Prior limits in the hybrid basis, as a getdist `ranges` dict."""
        q_lo, q_hi = self.q0_range()
        r = {"H0": (self.H0_lo, self.H0_hi), "q0": (float(q_lo), float(q_hi))}
        r.update({f"b{j}": (float(self.b_lo[j - 1]), float(self.b_hi[j - 1]))
                  for j in range(2, self.k_max + 1)})
        return r

    def prior_samples(self, n, seed=None, basis="hybrid"):
        """Prior draws in the 'hybrid' or 'physical' basis, for plot overlays."""
        rng = np.random.default_rng(seed)
        theta = self.prior_transform(rng.random((n, self.ndim)))
        return self.hybrid(theta) if basis == "hybrid" else self.physical(theta)

    # --- diagnostics ------------------------------------------------------
    def C_ranges(self):
        """Exact min/max of each C_k over the prior (linear map of a box)."""
        Mf, m0 = self.M[1:, 1:], self.M[1:, 0]
        lo = m0 + np.where(Mf > 0, Mf * self.b_lo, Mf * self.b_hi).sum(1)
        hi = m0 + np.where(Mf > 0, Mf * self.b_hi, Mf * self.b_lo).sum(1)
        return lo, hi

    def q0_range(self):
        K, ym = self.k_max, self.y_max
        return K * (self.b_lo[0] - 1) / ym - 1, K * (self.b_hi[0] - 1) / ym - 1

    def summary(self):
        lo, hi = self.C_ranges()
        q_lo, q_hi = self.q0_range()
        lines = [f"k_max = {self.k_max}, y_max = {self.y_max:.4f} "
                 f"(z_max = {self.y_max / (1 - self.y_max):.3f}), "
                 f"cond(M) = {np.linalg.cond(self.M[1:, 1:]):.3g}",
                 f"  H0 ~ U({self.H0_lo:g}, {self.H0_hi:g})"]
        lines += [f"  b_{j} ~ U({self.b_lo[j-1]:.4f}, {self.b_hi[j-1]:.4f})"
                  f"   peaks at y = {self.y_nodes[j]:.4f} (z = {self.y_nodes[j] / (1 - self.y_nodes[j]):.3f})"
                  + ("   = E(y_max) exactly" if j == self.k_max else "")
                  for j in range(1, self.k_max + 1)]
        lines += [f"  C_{k} in [{lo[k-1]:10.3f}, {hi[k-1]:10.3f}]"
                  + ("   (flat)" if k == 1 else "   (marginal prior not flat)")
                  for k in range(1, self.k_max + 1)]
        lines.append(f"  q0 ~ U({q_lo:.3f}, {q_hi:.3f})")
        return "\n".join(lines)


# -------------------------------------------------------------------------
def self_test(k_max, y_max=0.7, n_draw=100_000, seed=1):
    p = BernsteinEPrior(k_max, y_max)
    rng = np.random.default_rng(seed)
    K = p.k_max
    theta = p.prior_transform(rng.random((n_draw, p.ndim)))
    c = p.coeffs(theta)

    # 1. M inverts monomial_to_bernstein
    Minv = np.column_stack([monomial_to_bernstein(np.eye(K + 1)[k], K, y_max)
                            for k in range(K + 1)])
    err_inv = np.abs(p.M @ Minv - np.eye(K + 1)).max()

    # 2. round trip b -> C -> b
    b_back = np.array([monomial_to_bernstein(ci, K, y_max) for ci in c[:2000]])
    err_rt = np.abs(b_back[:, 1:] - theta[:2000, 1:]).max() + np.abs(b_back[:, 0] - 1).max()

    # 3. envelope holds for every prior draw, on a fine grid
    y = np.linspace(0.0, y_max, 1001)
    E = np.concatenate([p.E(y, ci) for ci in np.array_split(c, 20)])
    L, U = poly_eval(y, p.lower), poly_eval(y, p.upper)
    margin = min((E - L).min(), (U - E).min())

    # 4. exact identities and Horner vs direct Bernstein
    err_E0 = np.abs(c[:, 0] - 1).max()
    err_Ey = np.abs(p.E(np.array([y_max]), c)[:, 0] - theta[:, -1]).max()
    err_q0 = np.abs((c[:, 1] - 1) - (K * (theta[:, 1] - 1) / y_max - 1)).max()
    err_Bz = max(np.abs(p.E(y, c[i]) - p.E_bernstein(y, theta[i])).max() for i in range(200))

    # 5. hybrid basis: same q0 as the C basis, and draws fill its box
    hyb = p.hybrid(theta)
    err_hq = np.abs(hyb[:, 1] - p.physical(theta)[:, -1]).max()
    rng_h = p.hybrid_ranges()
    in_box = all(np.all((hyb[:, i] >= rng_h[nm][0] - 1e-12) & (hyb[:, i] <= rng_h[nm][1] + 1e-12))
                 for i, nm in enumerate(p.hybrid_names))

    ok = (err_inv < 1e-9 and err_rt < 1e-9 and margin >= -1e-12 and
          max(err_E0, err_Ey, err_q0, err_Bz, err_hq) < 1e-9 and in_box)
    print(p.summary())
    print(f"  checks: M.Minv-I {err_inv:.1e} | b->C->b {err_rt:.1e} | "
          f"min envelope margin {margin:.2e} over {n_draw} draws | "
          f"E(0) {err_E0:.1e}, E(y_max)=b_K {err_Ey:.1e}, q0 {err_q0:.1e}, "
          f"Horner-vs-Bernstein {err_Bz:.1e} | hybrid q0 {err_hq:.1e}, in box {in_box}"
          f"  ->  {'PASS' if ok else 'FAIL'}\n")
    return ok


def demo(k_max=4, seed=42):
    """Fit mock cosmic-chronometer-like H(z) data (flat LCDM truth) with dynesty."""
    import dynesty

    H0_true, Om = 68.0, 0.31
    z = np.array([0.07, 0.09, 0.12, 0.17, 0.179, 0.199, 0.2, 0.27, 0.28, 0.352,
                  0.3802, 0.4, 0.4004, 0.4247, 0.4497, 0.47, 0.4783, 0.48, 0.593,
                  0.68, 0.75, 0.781, 0.875, 0.88, 0.9, 1.037, 1.3, 1.363, 1.43,
                  1.53, 1.75, 1.965])
    H_true = H0_true * np.sqrt(Om * (1 + z) ** 3 + 1 - Om)
    rng = np.random.default_rng(seed)
    sig = 0.03 * H_true
    H_obs = H_true + sig * rng.standard_normal(z.size)

    prior = BernsteinEPrior(k_max, y_max_from_z(z))
    y = y_from_z(z)
    print(prior.summary(), "\n")

    def loglike(theta):
        H = theta[0] * prior.E(y, prior.coeffs(theta))
        return -0.5 * np.sum(((H - H_obs) / sig) ** 2)

    sampler = dynesty.NestedSampler(loglike, prior.prior_transform, prior.ndim,
                                    nlive=500, sample="rwalk", rstate=rng)
    sampler.run_nested(dlogz=0.1, print_progress=False)
    res = sampler.results
    w = res.importance_weights()

    print(f"lnZ = {res.logz[-1]:.3f} +/- {res.logzerr[-1]:.3f}")
    truth = {"H0": H0_true, "q0": 1.5 * Om - 1}
    # correlations exclude H0, and q0 in the physical basis (it duplicates C1)
    for basis, names, chain, cols in (
            ("hybrid", prior.hybrid_names, prior.hybrid(res.samples), slice(1, None)),
            ("physical", prior.phys_names, prior.physical(res.samples), slice(1, -1))):
        cov = np.cov(chain[:, cols].T, aweights=w)
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        print(f"  {basis} basis (max posterior |corr| among shape params = "
              f"{np.abs(corr - np.eye(len(corr))).max():.3f}):")
        for i, name in enumerate(names):
            m = np.sum(w * chain[:, i])
            s = np.sqrt(np.sum(w * (chain[:, i] - m) ** 2))
            tr = f"   (truth {truth[name]:.3f})" if name in truth else ""
            print(f"    {name:>3s} = {m:9.3f} +/- {s:7.3f}{tr}")
    E_all = prior.E(np.linspace(0, prior.y_max, 501), prior.coeffs(res.samples))
    print(f"  min E, max E/U over all {len(w)} samples: {E_all.min():.3f}, "
          f"{(E_all / poly_eval(np.linspace(0, prior.y_max, 501), prior.upper)).max():.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        results = [self_test(K) for K in range(2, 7)]
        print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
        sys.exit(0 if all(results) else 1)
