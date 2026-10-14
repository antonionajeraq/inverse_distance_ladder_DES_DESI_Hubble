# A model-independent inverse distance ladder with DES-Dovekie and DESI DR2

Code, chains and mock ensembles for

> **Does the inverse distance ladder rule out late-time solutions to the Hubble tension? A
> model-independent test using DES-Dovekie supernovae and DESI DR2 BAO**
> J. A. Nájera, I. Banik and H. Desmond (MNRAS, in preparation)

The figures exactly as they appear in the paper are already in `results/` and
`results_mock/`. Every figure and table can be rebuilt from this repository in about seven
minutes on one core, with no sampler or cluster:

```bash
pip install -r requirements.txt
bash reproduce.sh
```

The posterior chains and the two 10⁴-realisation mock ensembles are included, because
regenerating them takes days on a cluster. The scripts that produce
them are here too (see [Rerunning the samplers](#rerunning-the-samplers)).

## The analysis in brief

The supernovae are **uncalibrated**: their absolute magnitude `M_DES` is fitted jointly
with everything else. The sound horizon sets the absolute scale. In configuration A it
comes from a *Planck* prior on r_d. In configuration B it is derived from cosmic
chronometers (CC), with no CMB input. The BAO then calibrate the supernovae inside the
redshift range they share.

The expansion history is a polynomial in ã ≡ 1 − a = z/(1+z),

    H(ã) = H0 E(ã),    E(ã) = 1 + Σ_{k=1}^{k_max} C_k ã^k,    q0 = C_1 − 1,

with k_max = 4 as the baseline and k_max = 2–6 as robustness tests. The C_k are uniform
over every polynomial with 0 ≤ E(ã) ≤ 2 + 7ã on 0 ≤ ã ≤ 0.70. One expansion feeds all three
likelihoods: the SNe through D_L, the BAO through D_M, D_H and D_V, and the CC through H(z).
Flat ΛCDM replaces the polynomial as a fixed-shape cross-check. A rigid late-time `exp`
model and CPL are compared to ΛCDM by nested sampling, with the compressed SPA CMB in the
likelihood.

The independent variable is called `y` in the code (`Y_MAX`, `y_of_z`) and ã in the paper.
The polynomial order is `P` in the code and k_max in the paper.

## Layout

```
dovekie_cosmography.py    the inverse distance ladder: data, likelihood, prior, NUTS, cache.
                          Every sampling task of the paper is a --task of this file.
dovekie_evidence.py       rigid exp / CPL / ΛCDM models and the forward-modelled H0 anchors
dovekie_evidence_cmb.py   the Bayesian model comparison WITH the CMB (Tables 3 and 4)
evidence_order.py         nested-sampling evidence for k_max = 2..6 (Appendix B)
mock_idl.py               the ΛCDM mocks: swaps the data and nothing else
dh0_ensemble.py           the ΔH0 null test: runs it (sampling) and plots it (--analyse)

  renderers: read the cache, never sample
make_tables.py            Tables 1, 2, 5 and A1
make_figures.py           Figs 4-8 and A1
make_E_prior_graphic.py   Fig. 1
make_dz_ratio_figure.py   Fig. 2
make_adot.py              Fig. 3
make_evidence_order.py    Fig. 9, Tables B1 and B2
make_triangles.py         Figs C1-C4
make_mock_traces_figure.py  Fig. D2
analyse_mock.py           floor-sweep null statistics (Appendix D)
print_cls_sigma.py        mode and 68% HPD of every chain -> results/mode_CLs.json
check_Hk_prior.py         the induced prior on H_1..H_4 shown in Fig. C2

  modules imported by the above
ls_errors.py, ls_delta_order.py, make_bic_order.py   least-squares fits for Tables B1/B2
prior_reference.py        the prior each reported quantity is compared against
mock_paths.py             where the mock json lives
strokecheck.py            vector-stroke budget check on every saved figure
Bernstein_E_prior.py      used only by test_repo.py, as an independent check of the
                          envelope-prior volume

test_repo.py              46 acceptance tests, ~20 s
reproduce.sh              renders the whole paper from the shipped cache
matplotlibrc              figure style (Type 42 fonts)

data/                     fetch_data.sh (downloads the SN and CC inputs) and PROVENANCE.md
chains/                   the posteriors behind every number in the paper (143 npz, 112 MB)
results/                  the paper's figures (Figs 1-9, A1, C1-C4) as PDF; output directory
results_mock/             the paper's Appendix D figures; the mock ensembles (two archives)
                          and the noiseless/real-data runs
```

## Reproducing the paper

`bash reproduce.sh` runs the steps below in order. Set `PYTHON=` to choose an interpreter.

| step | command | time |
|---|---|---|
| download the inputs (6 MB, checksummed) | `bash data/fetch_data.sh` | seconds |
| unpack the mock ensembles | `tar xzf results_mock/{dh0,lsweep}_ensemble.tar.gz -C results_mock` | 10 s |
| acceptance tests | `python test_repo.py` | 20 s |
| AP and legacy aggregates from the cache | `python dovekie_cosmography.py --task ap` and `--task legacy` | 5 s |
| modes and 68% HPDs | `python print_cls_sigma.py --group ap bic deltaorder evorder floorsweep ladder legacy` | 1 min |
| Tables 3 and 4 | `python dovekie_evidence_cmb.py --task table`, `--task cls`, `--task tex` | 1.5 min |
| everything else | the renderers listed in the table below | 3 min |

Pass `--group` to `print_cls_sigma.py` explicitly. The CMB evidence chains get their own
pass (`dovekie_evidence_cmb.py --task cls`, which writes `results/mode_CLs_cmb.json`).

`make_tables.py` prints `Table 2 needs the evidence grid` and carries on. That message is
about an older evidence table *without* the CMB, which the paper no longer uses; its chains
are not shipped. The paper's Table 2 is `table1_baseline.tex` (see below).

### Which command makes which figure and table

Outputs land in `results/`, except Appendix D, which lands in `results_mock/`. The PDFs
shipped there are the paper's, with their white margins cropped (pdfCropMargins); a rerun
overwrites them with uncropped copies of the same content. The
manuscript types its tables by hand, so the generated `.tex` files are the reference the
printed numbers were checked against, not files it `\input`s. Their names are historical and
do not follow the paper's table numbers.

| paper | label | output | command |
|---|---|---|---|
| Fig. 1 | `fig:E_prior` | `E_prior_graphic` | `make_E_prior_graphic.py` |
| Fig. 2 | `fig:dz_ratio` | `dz_ratio` | `make_dz_ratio_figure.py` |
| Fig. 3 | `fig:adot-reconstruction` | `adot_reconstruction` | `make_adot.py` |
| Fig. 4 | `fig:h0_q0` | `h0_q0_P4_spa` | `make_figures.py` |
| Fig. 5 | `fig:floorsweep-real-new` | `floorsweep_real_H0`, `floorsweep_real_q0` | `make_figures.py` |
| Fig. 6 | `fig:offset_result` | `ladder_rungs` | `make_figures.py` |
| Fig. 7 | `fig:offset_split` | `ladder_splits` | `make_figures.py` |
| Fig. 8 | `fig:legacy-h0` | `h0_rows`, `h0_delta` | `make_figures.py` |
| Fig. 9 | `fig:polynomial-order` | `evidence_order` | `make_evidence_order.py` |
| Fig. A1 | `fig:AP-new` | `FAP_ratio_LCDM_paper` | `make_figures.py` |
| Fig. C1 | `fig:tri-legacy` | `triangle_legacy` | `make_triangles.py` |
| Fig. C2 | `fig:tri-main` | `triangle_{cosmo,lcdm}_{A,B}` | `make_triangles.py` |
| Fig. C3 | `fig:tri-evidence-exp` | `triangle_evidence_exp` | `make_triangles.py` |
| Fig. C4 | `fig:tri-evidence-exp-anchored` | `triangle_evidence_exp_{lfree3,breuval}` | `make_triangles.py` |
| Fig. D1 | `fig:dh0-hist` | `dh0_scatter`, `dh0_hist` | `dh0_ensemble.py --analyse` |
| Fig. D2 | `fig:floorsweep-mock` | `floorsweep_mock_H0`, `floorsweep_mock_q0` | `make_mock_traces_figure.py` |
| Table 1 | `tab:priors` | `table5_priors.tex` | `make_tables.py` |
| Table 2 | `tab:baseline` | `table1_baseline.tex` | `make_tables.py` |
| Table 3 | `tab:evidence` | `table_evidence_cmb.tex` | `dovekie_evidence_cmb.py --task tex` |
| Table 4 | `tab:logz1` | `table_logz1_cmb.tex` | `dovekie_evidence_cmb.py --task tex` |
| Table 5 | `tab:legacy-h0` | `table3_legacy.tex` | `make_tables.py` |
| Table A1 | `tab:AP-new` | `table4_ap.tex` | `make_tables.py` |
| Tables B1, B2 | `tab:ordersweep-ls-{A,B}` | `table_ordersweep_evidence.tex` | `make_evidence_order.py` |

Other numbers in the text:
- Posterior summaries are in `results/mode_CLs.json` and `results/mode_CLs_cmb.json`.
- The evidence-weighted order average is in `results/evidence_order_numbers.txt`.
- The ΔH0 null count (124 of 10⁴ mocks) is in `results_mock/dh0_hist.txt`.
- The floor-sweep null statistics are in `results_mock/mock_summary.txt`.

Run on a clean checkout, `reproduce.sh` gives the paper's figures pixel for pixel (PNG) and
its generated tables byte for byte, checked against the authors' working copy.

## Rerunning the samplers

Each task skips a cell that is already cached and whose manifest matches the current
settings. A cache written under different settings is refused with a message, never
silently reused. `--recompute` forces a rerun.
`python dovekie_cosmography.py --list-cache` shows what is cached and whether it would be
accepted.

| chains | command | cells | time per cell (upper bound) |
|---|---|---|---|
| `chains/ladder` | `python dovekie_cosmography.py --task ladder` | 8 | 12 h (NUTS, 4 cores) |
| `chains/legacy` | `python dovekie_cosmography.py --task legacy` | 6 | 12 h |
| `chains/floorsweep` | `python dovekie_cosmography.py --task floorsweep` | 48 | 12 h |
| `chains/ap` | `python dovekie_cosmography.py --task ap` | 2 | 8 h |
| `chains/deltaorder` | `python dovekie_cosmography.py --task deltaorder` (see below) | 9 | 16 h |
| `chains/bic` | `python dovekie_cosmography.py --task bic --check-box`, then `python ls_delta_order.py` and `python make_bic_order.py` | 9 | 4 h in total (least squares, no sampler) |
| `chains/evorder` | `python evidence_order.py` | 25 | 24 h (dynesty, 1 core) |
| `chains/evidence_cmb` | `SAMPLER=rslice python dovekie_evidence_cmb.py --task grid`, then the same with `Z1_PRIOR=log` | 27 + 9 | 12 h (dynesty, 1 core) |

The second `evidence_cmb` pass samples only the nine `exp` cells with a log-uniform z_1
prior (Table 4). Every other cell is already cached under the same name.

**Running on a cluster.** Every grid can be split into independent jobs, so it maps onto
any batch system's job array. Set one array task per cell:

| grid | how to run shard `i` of `n` |
|---|---|
| `dovekie_cosmography.py --task ...` | `--shard i --ntasks n` (defaults to `$SLURM_ARRAY_TASK_ID`) |
| `evidence_order.py` | environment `SHARD=i NTASKS=n` (25 cells) |
| `dovekie_evidence_cmb.py --task grid` | one cell per job with `--only <sample>/<anchor>/<model>`, e.g. `--only full/none/cpl` |
| `dh0_ensemble.py`, `mock_idl.py --task lsweep` | `--reals 1-10000 --shard i --ntasks n` |

- A sharded run writes only its own cells. Aggregate files such as `results/ap_table.json`
  are written by a final unsharded pass, which finds every cell cached and finishes in seconds.
- **`deltaorder`.** The paper uses only its nine k_max = 4 cells: shards 3–5, 16–18 and 22–24
  with `--ntasks 25`. An unsharded `--task deltaorder` also samples the 16 other orders.
  That is legitimate, but costly.
- **Threads.** Pin BLAS to one thread (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`).
  The likelihoods are small dense products, and threading them makes them slower.

### The mock ensembles

Both null tests replace the **data** with a noisy realisation of flat ΛCDM at the SPA
parameters (`mock_idl.py` monkeypatches the SN, BAO and CC providers). They then call the
unmodified least-squares fitters of `dovekie_cosmography`.

```bash
# ΔH0 = H0(DES) − H0(DES+legacy): 10^4 realisations, plus the noiseless mock (0) and real data (-1)
python dh0_ensemble.py --reals 1-10000
python dh0_ensemble.py --real 0 ; python dh0_ensemble.py --real -1

# the redshift-floor sweep: 10^4 realisations, plus the noiseless mock and real data
python mock_idl.py --task lsweep --reals 1-10000
python mock_idl.py --task lsweep --real 0 ; python mock_idl.py --task lsweep --real -1
```

## Settings

All settings that change a result live in the `BASELINE` dict at the top of
`dovekie_cosmography.py`. They are written into the manifest of every cached chain, and a
mismatch makes the cache refuse to load. Only the keys listed in `_ENV_OVERRIDABLE` can be
set from the environment, and every override is printed and recorded.

**Data.**
- The full, unbinned DES-Dovekie catalogue with its 1820 × 1820 STAT+SYS covariance.
- 82 BEAMS-deweighted SNe with `MUERR > 1` are dropped. That leaves 1541 DES SNe
  (`des_only`, z = 0.073–1.14) and 1738 with the legacy low-z SNe (`full`, down to
  z = 0.025).
- The complete DESI DR2 BAO vector (13 observables, z = 0.295–2.330).
- The 15 Moresco BC03 chronometers that have a published, full covariance.

**Priors.**

| parameter | prior |
|---|---|
| H0 | U(50, 90) km/s/Mpc |
| shape C_1..C_k | uniform over 0 ≤ E(ã) ≤ 2 + 7ã on [0, 0.70] (AP test: [0, 0.534]) |
| M_DES | U(−21, −18) |
| r_d | N(147.05, 0.30) Mpc in configuration A; U(120, 175) Mpc in configuration B |
| ΔM_legacy (when free) | U(−1, 1) mag |
| Ω_m (flat ΛCDM) | U(0.05, 0.70) |
| `exp`, CPL, CMB parameters | see `dovekie_evidence.py` and `dovekie_evidence_cmb.py` (Table 1 of the paper) |

The envelope prior is sampled in the node values v_j = E(ã_j), which are linear in the C_k.
In dynesty the region outside the envelope has log L = −∞, and the evidence is corrected by
ln acc(k), the deterministic fraction of the sampling box that lies inside the envelope.
NUTS uses a steep smooth wall instead.

**Sampling.**
- NUTS (numpyro): 4 chains of 1500 warm-up + 2000 draws each, dense mass matrix,
  target acceptance 0.8. Chains start at flat ΛCDM (Ω_m = 0.315, H0 = 68).
- A chain reaches a table only if R̂ ≤ 1.02 and n_eff ≥ 400. Divergences are recorded but
  do not fail a chain. `ALLOW_BAD_CHAINS=1` overrides the gate.
- dynesty: Δln Z = 0.01, with 800 live points for the model comparison (Tables 3 and 4)
  and 1000 for the order sweep (`evidence_order.py`).

JAX runs in float32, as for the published numbers. The covariance factorisations are done in
float64 by numpy.

## Reading a chain

Every cached cell carries its own manifest: the full `BASELINE`, any environment overrides,
the package versions, the git commit and a SHA-256 of the likelihood source.

```python
import json, numpy as np
d = np.load('chains/ladder/A_des_only_cosmo_P4.npz', allow_pickle=True)
man = json.loads(str(d['manifest_json']))
print(man['baseline']['P'], man['packages'])
print(d['H0'].mean(), d['H0'].std())
```

The ladder cells are named `<config>_<sample>_<model>_P<k_max>`: config `A` or `B`, sample
`des_only` or `full` (DES + legacy), and model `cosmo` (polynomial) or `lcdm`.
- NUTS cells hold the draws (`H0`, `M`, `rd`, `q0`, `C` = C_1..C_k, and the sampling
  coordinates `v`), plus the diagnostics in `scalars_json`.
- Nested-sampling cells hold `samples` and `weights`, with ln Z in `summary_json`
  (`evidence_cmb`) or `row_json` (`evorder`).

## Environment

The chains were sampled with Python 3.13, numpy 2.1, scipy 1.15, jax 0.8.0, numpyro 0.19.0
and dynesty 3.0.0. The figures and tables were rendered, and this repository verified,
with Python 3.12, jax 0.7.0, numpyro 0.21.0, getdist 1.7.6 and matplotlib 3.10.8.
`requirements.txt` sets minimum versions only. The exact versions behind each chain are in
its manifest.

## Data

All inputs are public. See `data/PROVENANCE.md` for sources, commits and checksums.

- **Downloaded, not included:** the DES-Dovekie Hubble diagram and covariance (Popovic et al.
  2026, from `des-science/DES-SN5YR`) and the Moresco cosmic-chronometer data and covariance
  (from `mmoresco/CCcovariance`). Neither upstream repository grants redistribution, so
  `bash data/fetch_data.sh` downloads them from the exact commits used here and checks their
  SHA-256 (`reproduce.sh` runs it first).
- **Inlined in the code:** DESI DR2 BAO, the *Planck* r_d, the SPA compressed CMB, the BBN
  ω_b and the local-H0 anchors. These are short published vectors.

## Citing and licence

If you use this code, please cite the paper and the data releases listed in
`data/PROVENANCE.md`. The code is released under the MIT licence (`LICENSE`). The
downloaded input data remain under their providers' terms.
