#!/bin/bash
# Render every figure and table of the paper from the shipped chains and mock ensembles.
# Nothing here samples.  About 7-15 minutes of CPU on one core, plus a ~6 MB download of the
# inputs; wall time can be much longer on a networked filesystem.
#
#   bash reproduce.sh                # uses `python` on your PATH
#   PYTHON=/path/to/python bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

# One thread per process: the likelihoods are small dense products, and threading them is
# slower, not faster.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

# The four input files are downloaded from upstream (pinned commits, checksummed).
bash data/fetch_data.sh

# The two 10^4-realisation mock ensembles ship as archives.
for t in dh0 lsweep; do
    if ! ls results_mock/json/${t}_real1.json >/dev/null 2>&1; then
        echo "=== extracting results_mock/${t}_ensemble.tar.gz"
        tar xzf results_mock/${t}_ensemble.tar.gz -C results_mock
    fi
done

run() { echo; echo "=== $*"; "$PY" "$@"; }

run test_repo.py

# aggregates the tables read, rebuilt from the cached chains (seconds, no sampling)
run dovekie_cosmography.py --task ap
run dovekie_cosmography.py --task legacy

# modes and 68% HPDs of every chain.  The CMB evidence group has its own pass below.
run print_cls_sigma.py --group ap bic deltaorder evorder floorsweep ladder legacy
run dovekie_evidence_cmb.py --task table
run dovekie_evidence_cmb.py --task cls
run dovekie_evidence_cmb.py --task tex

# tables and figures
run make_tables.py
run make_figures.py
run make_E_prior_graphic.py
run make_dz_ratio_figure.py
run make_adot.py
run make_evidence_order.py
run make_triangles.py

# the mock nulls (Appendix D)
run dh0_ensemble.py --analyse
run analyse_mock.py
run make_mock_traces_figure.py

# the H_k prior check behind Fig. C2
run check_Hk_prior.py

echo; echo "=== done: figures and tables in results/ and results_mock/"
