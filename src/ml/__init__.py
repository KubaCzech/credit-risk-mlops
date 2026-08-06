import os

# Must be set before numpy/scikit-learn/torch load their native BLAS/OpenMP runtimes - this
# is the first thing executed for any `ml.*` import (see README: mixing joblib's n_jobs=-1
# parallelism, e.g. in RandomForest/XGBoost, with PyTorch in the same process segfaults on
# macOS otherwise - each joblib worker spawning its own internal BLAS thread pool on top of
# PyTorch's own threading oversubscribes and crashes).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
