# Credit Risk Model Comparison Platform

ML/MLOps portfolio project: comparing classical ML algorithms (Logistic Regression, SVM,
Random Forest, and eventually a PyTorch MLP) on a real credit-scoring dataset, wrapped in
a full engineering stack (experiment tracking, an API, a database, containers, CI/CD).

Goal of the project is twofold: build solid intuition for how the classical algorithms
actually work (not just call `.fit()`), and demonstrate the engineering practices around
an ML project — reproducible data pipelines, tracked experiments, tested code.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 2011) — credit
scoring. ~150,000 anonymized customers, 10 features (payment history, debt, income,
demographics), binary target `SeriousDlqin2yrs` (1 = serious delinquency, 90+ days late,
within 2 years). Task: estimate probability of default — the classic bank credit-decision
problem.

The target is strongly imbalanced (~6.7% positive, ratio ≈ 1:14), which drives most of the
methodology below (metric choice, `class_weight`, stratified splitting).

## Project structure

```
data/                       raw CSV (cs-training.csv)
notebooks/
  01-dataset-analysis.ipynb EDA: distributions, missingness, sentinel values, correlations
src/ml/
  config.py                 central constants (paths, random_state, CV folds, MLflow/Optuna URIs)
  data/
    loader.py                load_raw_data / split_data (stratified) / split_X_y
    preprocessing.py          sklearn-compatible cleaning transformers (see below)
  features/
    engineering.py             hand-picked derived features (delinquency aggregate, interactions)
  models/
    baseline.py               pipeline factories: LogReg, SVM (linear+RBF), Random Forest, Decision Tree, XGBoost, Naive Bayes
    mlp.py                     PyTorch MLP: hand-written training loop wrapped in a sklearn-compatible estimator
  training/
    train_baseline.py         runs CV for all baseline models, logs to MLflow
    tune.py                    per-model Optuna studies (search spaces, pruning), logs best trial to MLflow
  evaluation/
    metrics.py                StratifiedKFold CV + precision/recall/F1/PR-AUC
    final_evaluation.py        the one-time test-set evaluation; saves artifacts, logs test metrics
  tracking/
    mlflow_utils.py            MLflow config + run logging
  artifacts/                  gitignored *.joblib pipelines, regenerated from MLflow tuned runs
  api/                       FastAPI serving layer — not built yet
tests/                       not built yet
docker/, k8s/                containerization/orchestration — not built yet
requirements.txt
```

Being upfront about it: `api/`, `tests/`, `docker/`, `k8s/` are scaffolded directories with
nothing in them yet — see [Roadmap](#roadmap) for what's actually done.

## Setup

Python **3.12** in an isolated `.venv` (the system Python on this machine is 3.9, which is
EOL — deliberately not used for this project).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Jupyter: a dedicated kernel ("Give Me Some Credit (.venv)") is registered and set as the
notebook's default, so `notebooks/01-dataset-analysis.ipynb` should pick up `.venv`
automatically. If your editor opens it with a different kernel, switch manually — editors
sometimes remember the last-used kernel per notebook independently of what's stored in the
file.

## Data pipeline (`src/ml/data/`)

`loader.py` reads the raw CSV and produces a **stratified** train/test split (`stratify` on
the target) so both splits keep the same ~6.7% positive rate — with this much imbalance a
plain random split can easily shift class balance by a percentage point or more between
folds, which would make metrics across runs not comparable.

`preprocessing.py` is a `sklearn.Pipeline` of small, DataFrame-in/DataFrame-out
transformers (`BaseEstimator` + `TransformerMixin`), all fit **only on train** and applied
unchanged to test — no statistic (median, percentile) is ever computed on data the model
will later be evaluated on.

| Transformer | What it does | Why |
|---|---|---|
| `DelinquencySentinelHandler` | Replaces 96/98 error codes in the 3 delinquency columns with the train median; adds `has_delinquency_sentinel` flag | These aren't real counts — they're a data-entry error code. The 269 affected rows have a 54.6% default rate vs 6.7% overall, so the flag turned out to be one of the strongest signals in the dataset |
| `MonthlyIncomeImputer` | Median-imputes `MonthlyIncome` (~20% missing); adds `MonthlyIncome_was_missing` flag | Missingness itself carries signal (MNAR) — rows with missing income also have `DebtRatio > 1` in 93.9% of cases vs 6% otherwise, suggesting `DebtRatio` is unreliable for these rows specifically |
| `DependentsImputer` | Median-imputes `NumberOfDependents` (~2.6% missing), no flag | Missing rate too low for a flag to carry useful signal |
| `AgeCleaner` | Clips `age` to a floor of 18 | One row had `age == 0` — a domain rule (adult borrower), not a statistical outlier |
| `OutlierCapper` | Winsorizes `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines`, and `MonthlyIncome` at the 99th percentile (fit on train) | All three have extreme right tails (`MonthlyIncome` max is 557x its median) that would otherwise dominate distance/margin-based models (SVM) and inflate variance for linear models. See [Feature Engineering](#feature-engineering) for why this beat the alternatives, measured rather than assumed |

## Exploratory Data Analysis

`notebooks/01-dataset-analysis.ipynb` — dataset description and column dictionary,
descriptive statistics, per-feature histograms, class imbalance visualization, missing
values, the sentinel-value investigation, outlier distributions, before/after view of the
cleaning pipeline, and a full feature correlation heatmap. Executed end-to-end, no errors.

Key findings baked into the design decisions above:
- Class imbalance ratio is **1:14** (6.7% positive) — accuracy is not used as a metric anywhere in this project because of it (see below).
- The delinquency sentinel flag and the income-missing flag both turned out to be genuine signal, not just missing-data bookkeeping.
- `DebtRatio` is likely meaningless for the ~20% of rows with missing income (see table above) — a non-obvious data quality issue only visible after cross-referencing two columns.

## Feature Engineering

`src/ml/features/engineering.py` adds two hand-picked transformers (not automatic
`PolynomialFeatures` expansion — that would produce dozens of mostly-meaningless terms and
defeat the point of understanding every feature that goes into the model):

- **`DelinquencyAggregator`** — `total_delinquency` (sum of the 3 delinquency columns) and `has_any_delinquency`, on top of the individual columns (which only correlate with each other at 0.22–0.31, so they're not redundant).
- **`InteractionFeatures`** — `utilization_x_delinquency` (an explicit product term: linear models can't form this combination on their own, unlike tree-based models which can approximate it through successive splits) and `income_per_dependent`.

**The capping-vs-log-transform decision was tested empirically, not decided from theory** —
the original plan was to replace `OutlierCapper` with `log1p` (avoids the artificial spike
of identical values at the winsorization threshold, which looked like a legitimate concern
for SVM margins). Measuring it on Logistic Regression PR-AUC (5-fold CV) said otherwise:

| variant | PR-AUC |
|---|---|
| **capping only** | **0.3834** |
| capping + log (both, as extra columns) + new features | 0.3805 |
| capping + new features (aggregate + interaction) | 0.3801 |
| quantile binning (5 bins) + one-hot, instead of capping | 0.3750 |
| no treatment at all (raw) | 0.3560 |
| log1p instead of capping | 0.3515 |
| log1p + new features | 0.3497 |

Plain capping, with nothing else added, won outright. The likely reason: capping only
touches the extreme top 1% of each distribution, leaving the other 99% on its original
scale — which already had a roughly linear relationship with the log-odds of the target.
`log1p` and binning reshape *that whole distribution*, not just the outlier tail, and lose
whatever made the untouched version work. The theoretical argument for `log1p` wasn't
wrong, exactly — it just didn't survive contact with this specific dataset and model.

Final call: keep `OutlierCapper`, keep the two new feature transformers (their effect is
inside the fold-to-fold noise band — neither clearly helps nor hurts on an untuned model),
and let L1/ElasticNet regularization during hyperparameter tuning decide computationally
whether they earn their place, instead of a human guessing from one untuned comparison.

## Baseline models

`src/ml/models/baseline.py` builds one `Pipeline` per model: the cleaning steps above, a
`StandardScaler` for the scale-sensitive models (Logistic Regression, SVM — L2
regularization and margins are both scale-dependent), and no scaler for the tree models
(Random Forest, Decision Tree, XGBoost — splits threshold one feature at a time, so
they're scale-invariant) or for `GaussianNB` (fits per-class, per-feature mean/variance
independently, so it's invariant to per-feature affine rescaling for a different reason
than the trees).

Eight models total: **Logistic Regression, Linear SVM, RBF SVM, Random Forest, Decision
Tree, XGBoost, Gaussian Naive Bayes, and a PyTorch MLP** (own section below — different
enough infrastructure to warrant one). Decision Tree is included specifically to contrast
against Random Forest (same base learner, no ensembling) rather than as a competing model
in its own right.

**Methodology**, decided deliberately before running anything:
- **Imbalance correction on every model** — `class_weight="balanced"` for the sklearn models, `scale_pos_weight` (XGBoost's equivalent, computed from the actual train split, not hardcoded) for XGBoost, `priors=[0.5, 0.5]` (the closest Naive Bayes analogue) for GaussianNB. Pretending we don't already know about the imbalance would be artificial.
- **5-fold `StratifiedKFold` CV on train only** — the test set is not touched until a final model is chosen; evaluating every iteration against the test set would slowly turn it into a de facto validation set and invalidate it as an independent check.
- **Precision / Recall / F1 / PR-AUC**, not accuracy or ROC-AUC — PR-AUC (`average_precision`) is threshold-independent and is what's used to rank models overall.
- **SVM at scale**: `SVC` (especially with an RBF kernel) has roughly O(n²)–O(n³) training cost, intractable on 120k rows. `LinearSVC` (`dual=False`, appropriate since n_samples ≫ n_features) runs on the full training set; RBF `SVC` is fit and cross-validated on a separate stratified 12,000-row subsample — the only way it finishes in reasonable time.
- **MLflow**, backed by SQLite (`mlflow.db`) — MLflow 3.x deprecated the plain filesystem tracking store, and SQLite needs no server. Models are logged with `serialization_format="cloudpickle"`; MLflow's default `skops` serializer refuses to load our custom transformer classes as "untrusted types" (a real security feature — cloudpickle is the pragmatic choice for a local, trusted-source project).

### Results (5-fold CV on train, ~150k rows total, with engineered features)

| model | precision | recall | F1 | **PR-AUC** |
|---|---|---|---|---|
| **mlp (untuned)** | 0.212 | 0.779 | 0.333 | **0.389** |
| logistic_regression | 0.206 | 0.770 | 0.325 | 0.380 |
| linear_svm | 0.199 | 0.780 | 0.317 | 0.371 |
| naive_bayes | 0.332 | 0.555 | **0.415** | 0.351 |
| random_forest | 0.424 | 0.376 | 0.399 | 0.344 |
| xgboost (untuned) | 0.265 | 0.560 | 0.359 | 0.325 |
| rbf_svm (12k subsample, untuned) | 0.216 | 0.702 | 0.330 | 0.229 |
| decision_tree (single, unpruned) | 0.257 | 0.248 | 0.252 | 0.114 |

The MLP — with a fixed, unremarkable architecture and no tuning at all — is the best
baseline by PR-AUC. See [Neural Network](#neural-network-pytorch-mlp) below for why: it
gets the hand-engineered interaction term (`utilization_x_delinquency`) for free, since a
hidden layer can approximate feature interactions on its own.

Two results here contradicted predictions made *before* running anything — worth stating
plainly rather than only reporting the numbers that confirmed expectations:

- **Naive Bayes was expected to underperform** — its independence assumption is measurably violated (the correlation heatmap shows several feature pairs at 0.22–0.43). It didn't: PR-AUC 0.351 beats Random Forest and untuned XGBoost, and its F1 (0.415) is the best of all seven models. Violating an assumption doesn't automatically translate to proportionally worse ranking performance — a real example of the gap between "the theory is technically wrong" and "the model is practically bad."
- **XGBoost, untuned, loses to plain Random Forest** (0.325 vs 0.344) — counterintuitive given gradient boosting's reputation as the tabular-data default, but well-documented: boosting is far more hyperparameter-sensitive than bagging (RF's default settings are close to reasonable; XGBoost's default `learning_rate=0.3` often isn't). This is the expected shape of things to change most once tuning is added.

**Decision Tree vs. Random Forest is the cleanest result in the table**: PR-AUC 0.114 vs
0.344 — a 3x gap between one unpruned tree (which overfits: it grows until every leaf is
pure, memorizing noise) and 300 of them averaged together. About as direct a demonstration
of why ensembling reduces variance as this dataset can produce.

**Why not accuracy** — illustrated, not just asserted: a model that always predicts "no
default" scores **93.3%** accuracy on this dataset. Random Forest scores 92.5% (worse than
the do-nothing baseline). Logistic Regression — the best model by PR-AUC — scores only
**80.3%** accuracy, the lowest of the three actually measured. Accuracy rewards predicting
the majority class and would have picked the wrong "best" model here.

### Running it

```bash
source .venv/bin/activate
PYTHONPATH=src python -m ml.training.train_baseline

# inspect runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Note for macOS: port 5000 is often claimed by the AirPlay Receiver, which returns HTTP 403
and looks like the server "isn't doing anything." Use a different port (as above) or turn
off AirPlay Receiver in System Settings → General → AirDrop & Handoff.

## Neural Network (PyTorch MLP)

`src/ml/models/mlp.py`. Three infrastructure options were on the table for wiring PyTorch
into a project otherwise built entirely on sklearn `Pipeline`s:

1. **`skorch`** — wraps a PyTorch module in an sklearn-compatible API. Drops straight into
   the existing `Pipeline`/`cross_validate_pipeline`/`tune.py` with zero new code, but hides
   exactly the mechanics (forward pass, backward pass, optimizer step, epochs) that were the
   actual point of including a neural network in this project.
2. **Raw PyTorch, fully separate evaluation path** — full visibility into the training loop,
   but duplicates the CV/metrics/MLflow infrastructure already built for the other 7 models
   instead of reusing it.
3. **Raw PyTorch training loop, wrapped in a minimal custom sklearn estimator** (chosen) —
   the model and training loop are hand-written (visible below), but wrapped in a thin
   `ClassifierMixin`/`BaseEstimator` subclass so the *same* `Pipeline`, `cross_validate_pipeline`,
   and `tune.py` machinery works unmodified. No framework hiding the training mechanics, no
   duplicated evaluation infrastructure.

**Architecture** (fixed, not searched — see Hyperparameter Tuning below):

```
Input(16) → Linear(64) → ReLU → Dropout(0.3)
          → Linear(32) → ReLU → Dropout(0.3)
          → Linear(1)  → (raw logit)
```

`BCEWithLogitsLoss` fuses sigmoid + binary cross-entropy in one numerically stable op
(avoids overflow at extreme logits that computing them separately can hit) and takes a
`pos_weight` argument — the same neg/pos ratio idea as `scale_pos_weight` in XGBoost and
`class_weight="balanced"` everywhere else, computed from each fold's actual training labels.
Training uses early stopping on a held-out validation slice of the training data (patience
15 epochs) instead of a fixed epoch count, and `StandardScaler` — tested against
`MinMaxScaler` empirically (0.389 vs 0.391 PR-AUC, within the noise band); ReLU doesn't
saturate for unbounded positive inputs the way sigmoid/tanh do, so the usual
bounded-input argument for MinMax is weaker here than for the classical models. CPU only:
MPS is available on this Apple Silicon machine, but a network this size trains in seconds
on CPU — GPU transfer overhead isn't worth it at this scale.

### A real infrastructure bug, not just theory

Wiring the MLP into `train_baseline.py` alongside the other 7 models **segfaulted**
(exit code 139) partway through the run — reproducible down to "`RandomForestClassifier`
(`n_jobs=-1`) followed by the MLP in the same process." Root cause: joblib's worker
processes for RF's parallel tree building each spin up their own internal BLAS/OpenMP
thread pool; PyTorch brings its own. Too many competing thread pools fighting over the
same cores crashes on macOS. `KMP_DUPLICATE_LIB_OK=TRUE` (the common but blunter
workaround suggested by most Stack Overflow threads for this exact symptom) did **not**
fix it - the real fix was `OMP_NUM_THREADS=1` + `MKL_NUM_THREADS=1`, set in `src/ml/__init__.py`
(the first code that executes for any `ml.*` import, before numpy/sklearn/torch load their
native libraries). This isn't just a crash workaround — it's the technically correct
threading configuration once something is already parallel at the joblib level: each worker
shouldn't *also* be multi-threaded internally.

### Why the untuned MLP already wins

It gets `RevolvingUtilizationOfUnsecuredLines × total_delinquency` — the interaction term we
had to hand-engineer for the linear models — for free. A hidden layer with a non-linear
activation can approximate feature interactions on its own; Logistic Regression and Linear
SVM cannot, which is exactly why `InteractionFeatures` exists in the first place. This is a
concrete, measured illustration of the classical trade-off between linear models and neural
networks: linear models need you to hand-supply non-linearity, networks can learn some of it.

## Hyperparameter Tuning

`src/ml/training/tune.py` — one Optuna study per model (TPE sampler, `MedianPruner` with a
2-fold warm-up), optimizing the same 5-fold CV PR-AUC used everywhere else. Each fold's
running score is reported back to Optuna mid-CV so a clearly bad trial gets pruned after
2-3 folds instead of wasting the remaining ones. Studies persist to `optuna.db` (SQLite),
so a run can be resumed instead of starting over.

**Search spaces**: `C` (log-scale) for Logistic Regression/SVM; `penalty` (l1/l2/elasticnet
for LogReg, l1/l2 for LinearSVC) — the feature-selection mechanism the Feature Engineering
section deferred to this step; `max_depth`/`ccp_alpha` for the trees (direct antidote to
the Decision Tree overfitting seen in the baseline); the usual boosting knobs
(`learning_rate`, `subsample`, `colsample_bytree`, `max_depth`, L1/L2 reg) for XGBoost;
`lr`/`dropout`/`weight_decay`/`batch_size` for the MLP — its architecture stays fixed (see
Neural Network above), only training hyperparameters are searched, since a full
architecture search is a different-scale project than tuning the other 7 models.
`scale_pos_weight` for XGBoost is fixed, not tuned — the imbalance correction stays
constant, only the model's own hyperparameters are searched. Trial budget is 30 for most
models; `linear_svm`, `rbf_svm`, `naive_bayes`, and `mlp` get 15 (LinearSVC's `l1` penalty
at high `C` converges slowly — capped `max_iter` at 2000 instead of letting a few hard
trials dominate wall-clock time; `rbf_svm` is already on the 12k subsample; `naive_bayes`
has one real hyperparameter, so a 1D search doesn't need many trials; `mlp` is ~43s/trial,
comparable to `linear_svm`). Full run: 8 models, ~30 minutes.

### Results (best trial per model, same 5-fold CV, full metrics recomputed for comparability)

| model | precision | recall | F1 | **PR-AUC** | Δ vs. baseline |
|---|---|---|---|---|---|
| **xgboost** | 0.215 | 0.777 | 0.337 | **0.402** | +0.077 |
| random_forest | 0.232 | 0.743 | 0.354 | 0.395 | +0.051 |
| mlp | 0.212 | 0.778 | 0.333 | 0.392 | +0.003 |
| logistic_regression | 0.212 | 0.756 | 0.331 | 0.384 | +0.004 |
| linear_svm | 0.199 | 0.778 | 0.317 | 0.373 | +0.002 |
| decision_tree | 0.205 | 0.766 | 0.323 | 0.366 | **+0.252** |
| rbf_svm | 0.219 | 0.663 | 0.329 | 0.363 | +0.134 |
| naive_bayes | 0.340 | 0.542 | 0.418 | 0.352 | +0.000 |

**XGBoost takes over first place**, exactly as flagged in the baseline write-up — its tuned
`max_depth=4`, `learning_rate≈0.019` is far more conservative than the untuned defaults
(`max_depth=6`, `learning_rate=0.3`), confirming boosting needed the tuning far more than
bagging did.

**Decision Tree's +0.252 is the largest jump on the board, and it's the same lesson as the
baseline's ensembling story told from the other side**: `ccp_alpha` (cost-complexity
pruning) and a `max_depth` of 7 (down from unbounded) fix almost the entire gap to Random
Forest by directly attacking the overfitting that caused it. A tuned single tree gets most
of the way to an ensemble; an unpruned one doesn't get close.

**Logistic Regression, Linear SVM, Naive Bayes, and the MLP barely move** — consistent with
being models whose untuned defaults were already close to reasonable for this data. The
MLP's case is the clearest why: its baseline already included early stopping and a
sensible fixed architecture, so tuning only had training hyperparameters left to improve
(best trial: `lr≈0.00019`, `dropout≈0.22`, `batch_size=128`) — it still finishes 3rd
overall, ahead of every classical model except the two tree ensembles.

**L1 regularization did perform real feature selection**, resolving the question the
Feature Engineering section deferred to this step. Both Logistic Regression and Linear SVM
independently landed on `penalty='l1'` with a small `C` (strong regularization). Inspecting
the tuned Logistic Regression's coefficients: 3 of 16 features get zeroed out exactly
(`NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfDependents`, `MonthlyIncome_was_missing`)
— but **none of the four engineered features are among them**. `has_any_delinquency` and
`total_delinquency` rank 2nd and 3rd by coefficient magnitude, right behind
`RevolvingUtilizationOfUnsecuredLines`. An unbiased, automatic selection process — not a
human's guess from one untuned comparison — confirms the engineered features earn their
place.

### Running it

```bash
PYTHONPATH=src python -m ml.training.tune

# inspect studies (per-trial history, parameter importance) in Optuna's own dashboard
optuna-dashboard sqlite:///optuna.db
```

## Final Evaluation

`src/ml/evaluation/final_evaluation.py` — the only place in this project that touches the
test set. Every step before this one (baseline, feature engineering ablations, tuning) was
validated exclusively through 5-fold CV on train, specifically so this evaluation would be
a single, honest, unbiased check rather than another data point to iterate against.

**`save_tuned_artifacts()`** pulls each model's already-fitted pipeline out of MLflow (fit
once during tuning, on train only) and writes it to `src/ml/artifacts/*.joblib` — not a
retrain, so there's no discrepancy between what was CV-scored and what gets evaluated (a
retrain would re-roll Random Forest's bootstrap sampling, XGBoost's row/column subsampling,
and the MLP's weight init and batch order). `evaluate_on_test()` loads each artifact,
scores it once on the 30k held-out rows, and appends the result as new metrics on the
*same* MLflow run that already holds that model's CV score — so a model's CV and test
numbers live together, not in disconnected places.

### Results

| model | CV PR-AUC | test PR-AUC | gap | test precision | test recall | test accuracy |
|---|---|---|---|---|---|---|
| **xgboost** | 0.4025 | **0.4038** | -0.0014 | 0.2170 | 0.7870 | 0.7960 |
| random_forest | 0.3952 | 0.4005 | -0.0053 | 0.2325 | 0.7536 | 0.8173 |
| mlp | 0.3924 | 0.3975 | -0.0051 | 0.2100 | 0.7880 | 0.7877 |
| logistic_regression | 0.3842 | 0.3933 | -0.0091 | 0.2147 | 0.7701 | 0.7964 |
| linear_svm | 0.3729 | 0.3877 | -0.0148 | 0.2016 | 0.7880 | 0.7773 |
| decision_tree | 0.3665 | 0.3705 | -0.0040 | 0.2060 | 0.7810 | 0.7842 |
| rbf_svm | 0.3631 | 0.3693 | -0.0062 | 0.2254 | 0.6813 | 0.8223 |
| naive_bayes | 0.3520 | 0.3554 | -0.0034 | 0.3441 | 0.5436 | 0.9002 |

(`always predict 0` baseline: 0.9332 accuracy — every real model scores below it; see below.)

**Every single model scores at least as well on test as its CV estimate predicted** (gap is
negative throughout — CV, if anything, was slightly conservative). This is the check for
the "optimizer's curse": running 30 Optuna trials per model against the same 5 CV folds
creates some risk of quietly overfitting to those specific folds, which would show up here
as test scores *below* CV estimates. It didn't happen. The model ranking on test is also
identical to the CV ranking, top to bottom — another sign the whole comparison pipeline
(not just the winning model) is measuring something real rather than fold-specific noise.

**Accuracy is tracked too now** (`SCORING` in `evaluation/metrics.py`, `test_accuracy` here)
— not to rank models, only to close the loop on the "why not accuracy" argument with
tracked numbers across all 8 *tuned, final* models instead of the 3-model ad-hoc
illustration in the baseline section above. The "always predict 0" baseline scores 93.3%
accuracy on test; every real model scores below that, from 90.0% (Naive Bayes) down to
77.7% (Linear SVM). The clearest inversions: **Naive Bayes is worst by PR-AUC (rank 8/8)
but best by accuracy (rank 1/8)**, and **RBF SVM is 2nd-worst by PR-AUC but 2nd-best by
accuracy** — both are the most conservative models (closest in behavior to always
predicting the majority class), which accuracy rewards and PR-AUC does not. XGBoost, the
actual best model, ranks only 5th of 8 on accuracy — solidly below-median, not the extreme
"worst" case, but still a clear illustration that these two metrics are not just
differently scaled, they disagree on ranking outright.

### Running it

```bash
PYTHONPATH=src python -m ml.evaluation.final_evaluation
```

`src/ml/artifacts/*.joblib` is gitignored — regenerated from the MLflow-tracked tuned runs,
not committed (Random Forest's alone is ~34MB).

## Roadmap

- [x] Data loading + stratified split
- [x] Cleaning pipeline (missing values, sentinel values, outliers) as sklearn transformers
- [x] EDA notebook
- [x] `.venv` on a supported Python version, pinned `requirements.txt`
- [x] Baseline models: LogReg, Linear SVM, RBF SVM, Random Forest, Decision Tree, XGBoost, Gaussian Naive Bayes — with CV + MLflow tracking
- [x] Feature engineering: delinquency aggregate + interaction terms, validated empirically against log-transform and binning alternatives (capping won)
- [x] Baseline + tuned PyTorch MLP (hand-written training loop, sklearn-compatible wrapper): best untuned baseline of all 8 models (PR-AUC 0.389 — gets feature interactions for free); tuned to 0.392, 3rd overall behind the two tree ensembles. Uncovered and fixed a real segfault from mixing joblib (`n_jobs=-1`) and PyTorch threading on macOS (see Neural Network section)
- [x] Hyperparameter tuning (Optuna, 8 studies, pruning): XGBoost took over first place (PR-AUC 0.402), Decision Tree closed most of the gap to Random Forest via pruning, L1 confirmed the engineered features are worth keeping. Feature *extraction* (PCA) stays out of scope: ~15 features is low-dimensional, and credit scoring specifically benefits from features staying interpretable
- [x] SMOTE ablation (separate experiment, not adopted): SMOTENC beats `class_weight="balanced"` on F1 but loses on PR-AUC for both models tested — oversampling shifts the training distribution away from the true one, hurting score ranking even though one fixed threshold (0.5) looks better. Kept `class_weight`/`scale_pos_weight`/`priors` as the primary imbalance strategy
- [x] Final test-set evaluation: artifacts saved to `src/ml/artifacts/`, every model scores at or above its CV estimate on the untouched test set (no "optimizer's curse" from 30 Optuna trials per model against the same folds), ranking identical to CV top to bottom
- [ ] FastAPI serving layer
- [ ] PostgreSQL + SQLAlchemy + Alembic
- [ ] Docker/Podman, Kubernetes manifests
- [ ] GitHub Actions CI/CD
- [ ] Tests
