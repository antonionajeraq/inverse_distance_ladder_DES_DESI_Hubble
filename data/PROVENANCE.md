# Data provenance

The four input files are **downloaded, not shipped**: neither upstream repository carries a
licence that grants redistribution. Run

```bash
bash data/fetch_data.sh
```

(`reproduce.sh` does this for you). It fetches each file from the exact upstream commit the
paper used and checks its SHA-256, so the download is exactly as reproducible as a vendored
copy. `dovekie_cosmography.data_path` checks the same checksums again at read time, so a
swapped or truncated input is caught immediately rather than as a moved number in a table.

The DESI DR2 BAO vectors, the *Planck* sound horizon, the compressed CMB, the BBN prior and
the local-H0 anchors are **not** files: they are short published tables, inlined in the code
(see the end of this file).

## SHA256

```
2f57019d783eaa976df80a41b0054171a2d994ee9808d715ce850c2df5720aaf  DES-Dovekie_HD.csv
ffd3124b32148b1372bd95fda9299269f0352a9f8eee02d416c610e38495463b  STAT+SYS.npz
577ac2f346e346fe7cf94daa7b7000c05d04ebc8a029cda31e0d8643b956a485  CC_mm20/data_MM20.dat
32ce92caf251cb60a7a837c71f1856bea2b44fa5c1041f85410d11cb8164da98  CC_mm20/HzTable_MM_BC03.dat
```

## `DES-Dovekie_HD.csv` (148 KB)

The DES-Dovekie supernova Hubble diagram, 1820 spectroscopically and photometrically
classified SNe Ia.

- Source: <https://github.com/des-science/DES-SN5YR>, `4_DISTANCES_COVMAT/`, at commit
  `442c24841b55c3d629ed7177c09763dc237ba8e1` (2025-11-14).
- Reference: Popovic et al. (2026, MNRAS 548, 1; arXiv:2511.07517), the DES-Dovekie
  reanalysis of DES-SN5YR. The original DES-SN5YR release is
  [10.5281/zenodo.12720778](https://doi.org/10.5281/zenodo.12720778); the Dovekie files
  postdate it and are not part of that record.

SNANA format despite the `.csv` extension: comment lines, a `VARNAMES:` header, then rows
prefixed `SN:`. Columns used here are `CID`, `IDSURVEY`, `zHD`, `zHEL`, `MU`, `MUERR`.

`IDSURVEY == 10` is DES. The rest is the low-redshift legacy anchor: `150` is Foundation
and `61`–`66` are the CfA and CSP samples.

`MU` assumes H0 = 70 and a fiducial absolute magnitude M = −19.349, which is
`BASELINE['M_B_FID']`. The analysis works in apparent magnitude `m = MU + M_B_FID` so that
the sampled `M_des` is a genuine absolute magnitude, directly comparable to the SH0ES
value rather than an offset against Dovekie's fiducial.

**Row order matters** and is preserved: it is the order of the covariance matrix. The
separate metadata file in the upstream release has a different order and must not be used
with the covariance; `fetch_data.sh` does not download it.

## `STAT+SYS.npz` (6.2 MB)

The full statistical + systematic covariance, 1820 × 1820.

- Same source, DOI and reference as above.
- **It is the packed upper triangle of the INVERSE covariance**, as the upstream README
  states. `dovekie_cosmography.read_covariance` unpacks it, symmetrises and inverts,
  mirroring the release's own `DES-Dovekie-SN_Likelihood.py`.

## `CC_mm20/` (4 KB)

The cosmic-chronometer measurements that have a published covariance.

- From <https://gitlab.com/mmoresco/CCcovariance>, `data/`, at commit
  `881413330a7f1e1e5203607d6964db49b4c6c461` (2021-03-19).
- `HzTable_MM_BC03.dat` — 15 BC03 D4000 measurements, columns `z, Hz, errHz, stat_contr,
  met_contr, reference`. **`errHz` is stat+met only**: it does not include the model
  systematic, which is supplied separately.
- `data_MM20.dat` — model systematic contributions as a percentage of H(z), columns
  `z, IMF, stlib, mod, mod_ooo`.

Prescribed combination, per the authors' own `examples/CC_covariance.ipynb`:

```
Cov = diag(errHz^2) + Cov_IMF + Cov_SPSooo,      Cov^X_ij = H_i x_i H_j x_j
```

with the IMF and `mod_ooo` terms fully correlated across all redshifts. The notebook uses
`mod_ooo`, **not** `mod`, and does not include `stlib`; we follow it exactly.

Which measurements are covered — 15 of the 33 in the usual compilation:

| source | n | z |
|---|---|---|
| Moresco et al. (2012) | 8 | 0.1791, 0.1993, 0.3519, 0.5929, 0.6797, 0.7812, 0.8754, 1.037 |
| Moresco et al. (2016) | 5 | 0.3802, 0.4004, 0.4247, 0.4497, 0.4783 |
| Moresco (2015) | 2 | 1.363, 1.965 |

The other 18 points have **no published covariance** — including every point below
z = 0.179 and the z = 1.3/1.43/1.53/1.75 group. They are dropped rather than left
diagonal, which would down-weight the best-characterised data and let the fit lean on the
least-characterised. The `data_MM20.dat` percentages are properties of the BC03 D4000
modelling, so extending them to other groups, SPS models and methods would be an
assumption, not a lookup.

Cite Moresco et al. (2012, 2015, 2016) for the measurements and Moresco et al. (2020),
ApJ 898, 82 for the covariance.

## Inlined, not shipped as files

**DESI DR2 BAO** — means and covariance taken verbatim from the official likelihood files
(`CobayaSampler/bao_data`, `desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt`),
arXiv:2503.14738. Thirteen observables: `D_V/r_d` at z = 0.295, and the pair
`[D_M/r_d, D_H/r_d]` with its 2 × 2 covariance at z = 0.510, 0.706, 0.934, 1.321, 1.484
and 2.330. Note the DR2 file lists `D_H` before `D_M` at z = 2.330; the pair is reordered
in `DESI_DMDH_COV` so every block is (D_M, D_H).

**Planck 2018 sound horizon** — r_d = 147.05 ± 0.30 Mpc (TT,TE,EE+lowE), Planck
Collaboration (2020).

**Compressed CMB (Tables 3 and 4 only)**: `MU_CMB` and `COV_CMB` in
`dovekie_evidence_cmb.py`. A Gaussian on (100θ*, ω_b, ω_bc) with the SPA (SPT-3G + ACT
DR6 + *Planck*) mean and covariance (SPT-3G Collaboration 2026, arXiv:2506.20707). The
z* and z_drag fits follow appendix A of Banik et al. (2026). This likelihood replaces the
*Planck* r_d prior in those two tables.

**BBN**: ω_b = 0.021915 ± 0.000215 (`WB_BBN`, `SIG_WB_BBN` in `dovekie_evidence_cmb.py`),
as cited in the paper (Giovanetti et al. 2025, arXiv:2408.14531).

**Local-H0 anchors and their redshift windows** — inlined in `dovekie_evidence.py` with
per-anchor provenance notes: Jensen et al. (2025) SBF with the Blakeslee et al. (2021)
Table 3 `v_grp` window; Alfradique et al. (2026) Table 2 bright siren; Vogl et al. (2025)
Table 1; Breuval et al. (2024) with the 277 Pantheon+SH0ES `USED_IN_SH0ES_HF` redshifts.
