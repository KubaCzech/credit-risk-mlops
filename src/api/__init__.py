import os

# Same fix as ml/__init__.py: the API loads both joblib-parallel models (RandomForest,
# n_jobs=-1) and the PyTorch MLP into the same process - without this, predicting with one
# after the other segfaults on macOS (see README, Neural Network section).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
