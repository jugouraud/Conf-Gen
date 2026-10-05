"""Pytest bootstrap: pin the OpenMP thread pools before numpy/sklearn/POT load.

On macOS, scikit-learn (KMeans, used by the low-cardinality estimators) and
POT's network-simplex EMD (used by the task-space Reweighter) each pull in an
OpenMP runtime. Calling KMeans and then emd2 in the same process segfaults
unless the thread pools are pinned first. The experiment scripts already do
this at the top of each module; conftest applies the same guard to the test
session, where the import order is pytest's to choose.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
