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
src/api/                    FastAPI serving layer (sibling of src/ml/, not nested under it)
  main.py                    app + /health, /models, /predict endpoints; logs every prediction to the DB
  schemas.py                 Pydantic request/response models (aliased to the raw dataset's column names)
  model_registry.py          loads all 8 artifacts + their DB metadata once at startup
src/db/                     SQLAlchemy layer (also a sibling of src/ml/)
  models.py                  ORM models: Model (lookup table), Prediction (audit log, FK to Model)
  session.py                 engine + session factory + get_db() FastAPI dependency
  seed_models.py              populates the models table from the static snapshot below
  seed_data.json               committed snapshot of tuned metrics - no MLflow dependency at seed time
  export_seed_data.py          local-dev only: regenerates seed_data.json from MLflow after retuning
  config.py                   DATABASE_URL (env var, local-dev default included)
alembic/                    migrations (env.py reads db/models.py's metadata + db/config.py's URL)
alembic.ini
docker/
  Dockerfile                 builds the API image (CPU-only torch, package layer cached separately)
  entrypoint.sh                migrate + seed + serve, every container start
docker-compose.yml          API + Postgres, one Compose-managed network (see Docker below)
.dockerignore
tests/                       not built yet
k8s/                        orchestration - not built yet
requirements.txt
pyproject.toml               src-layout packaging (`pip install -e .` - see Setup)
```

Being upfront about it: `tests/` and `k8s/` are scaffolded directories with nothing in them
yet — see [Roadmap](#roadmap) for what's actually done.

## Setup

Python **3.12** in an isolated `.venv` (the system Python on this machine is 3.9, which is
EOL — deliberately not used for this project).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

The last line installs this repo itself in editable mode (`pyproject.toml`, src-layout:
`ml` and `api` live under `src/`) so `import ml` / `import api` work from anywhere, in any
shell, without a `PYTHONPATH=src` prefix on every command — the standard fix, not a
workaround. `--no-deps` because dependencies are already pinned and installed via
`requirements.txt` above; `pyproject.toml` intentionally lists none, so there's one source
of truth for versions, not two files that can drift apart.

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
python -m ml.training.train_baseline

# inspect runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Note for macOS: port 5000 is often claimed by the AirPlay Receiver, which returns HTTP 403
and looks like the server "isn't doing anything." Use a different port (as above) or turn
off AirPlay Receiver in System Settings → General → AirDrop & Handoff.

## Resampling Ablations (SMOTE, Undersampling)

Two separate, limited experiments — not adopted, but run to check the imbalance-handling
choice empirically rather than assume `class_weight="balanced"` is best just because it's
simplest. Both use `imbalanced-learn`'s `Pipeline` (a plain sklearn `Pipeline` can't contain
a resampling step - it doesn't touch `y` or change row counts) on Logistic Regression and
Random Forest, with `class_weight=None` on the classifier itself (resampling and cost-
sensitive weighting both correct the same imbalance - stacking them would over-correct).

| model | variant | PR-AUC | precision | recall | F1 |
|---|---|---|---|---|---|
| logistic_regression | `class_weight="balanced"` (current) | 0.3801 | 0.206 | 0.769 | 0.325 |
| logistic_regression | SMOTENC | 0.3327 | 0.280 | 0.544 | 0.370 |
| logistic_regression | random undersampling (16k of 120k rows) | 0.3793 | 0.204 | 0.773 | 0.323 |
| random_forest | `class_weight="balanced"` (current) | 0.3440 | 0.424 | 0.376 | 0.399 |
| random_forest | SMOTENC | 0.3260 | 0.391 | 0.384 | 0.388 |
| random_forest | random undersampling | 0.3482 | 0.206 | 0.769 | 0.324 |

**SMOTE loses on PR-AUC for both models**, despite winning on F1 for Logistic Regression.
Oversampling trains the model on an artificially rebalanced 50/50 distribution, which
shifts where its predicted probabilities are calibrated - that helps the specific operating
point F1 measures (threshold 0.5) but distorts the overall score ranking PR-AUC measures.

**Undersampling is a near-wash on PR-AUC for both models** (Logistic Regression:
effectively identical; Random Forest: marginally better, likely noise) — despite
discarding **87% of the training data** (120k rows down to ~16k). That the model barely
changes after losing most of the majority class is more evidence for the "information
ceiling" read of this dataset from the Final Evaluation section: most of the discarded
majority-class rows were redundant for this decision boundary, not load-bearing. Random
Forest's precision/recall *profile* does shift heavily toward Logistic Regression's shape
(0.206/0.769 vs. the balanced-weight version's 0.424/0.376) even though its ranking quality
barely moves - undersampling changes the natural decision threshold, not just the data size.

**Kept `class_weight`/`scale_pos_weight`/`priors` as the primary imbalance strategy** for
all 8 models - neither resampling technique beat it on PR-AUC, and both add a source of
randomness (which rows get kept/synthesized) that cost-sensitive weighting simply doesn't have.

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
python -m ml.training.tune

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
python -m ml.evaluation.final_evaluation
```

`src/ml/artifacts/*.joblib` is gitignored — regenerated from the MLflow-tracked tuned runs,
not committed (Random Forest's alone is ~34MB).

## API (FastAPI)

`src/api/` — a thin serving layer over the same 8 joblib artifacts from Final Evaluation.
On theme for a *model comparison* project: rather than hardcoding a single "the" model, the
caller picks one per request (defaulting to XGBoost, the best by test PR-AUC).

- **`GET /health`** — liveness check, lists which models are loaded.
- **`GET /models`** — all 8 models with their test-set precision/recall/F1/PR-AUC/accuracy (the same numbers as the Final Evaluation table), read from the `models` table (see Database below) so a caller can pick a model deliberately instead of blindly trusting the default.
- **`POST /predict`** — body is the 10 raw applicant fields (aliased to the exact Kaggle column names, e.g. `"NumberOfTime30-59DaysPastDueNotWorse"`) plus an optional `model_name`. `MonthlyIncome` and `NumberOfDependents` are optional — omitting them exercises the same imputers built for exactly this case back in the Data pipeline section, not a special code path. Each loaded object is the *entire* fitted pipeline (cleaning → features → scaling → model), so the endpoint itself does no feature engineering — it hands the raw request straight to `.predict_proba()`. Every call is logged to the `predictions` table before the response is returned.

Startup loads all 8 artifacts into memory once (~40MB total, dominated by Random Forest's
300 trees) rather than lazily per-request, trading a slightly slower cold start for
predictable request latency. Model metadata (the numbers in `/models`) is loaded once at
startup too, from the database, not re-queried per request.

### Running it

```bash
python -m api.main

# interactive docs
open http://127.0.0.1:8000/docs
```

Matches the `python -m ml.xxx.yyy` convention every other entry point in this project uses,
instead of a separately-remembered `uvicorn app:module --reload` invocation. `main.py`'s
`if __name__ == "__main__":` block calls `uvicorn.run("api.main:app", ..., reload=True)`
itself - the direct `uvicorn api.main:app --reload` CLI form still works identically if
you prefer it.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "application": {
          "RevolvingUtilizationOfUnsecuredLines": 0.766,
          "age": 45,
          "NumberOfTime30-59DaysPastDueNotWorse": 2,
          "DebtRatio": 0.803,
          "MonthlyIncome": 9120,
          "NumberOfOpenCreditLinesAndLoans": 13,
          "NumberOfTimes90DaysLate": 0,
          "NumberRealEstateLoansOrLines": 6,
          "NumberOfTime60-89DaysPastDueNotWorse": 0,
          "NumberOfDependents": 2
        },
        "model_name": "xgboost"
      }'
# {"model_used":"xgboost","probability_of_default":0.8846,"prediction":1}
```

That request is the first row of `cs-training.csv` verbatim, which really did default
(`SeriousDlqin2yrs=1`) — the model agrees.

## Database

PostgreSQL + SQLAlchemy 2.0 (typed `Mapped[...]` declarative style) + Alembic. Two tables,
deliberately not more — see `TODO.md` for the extension (an `outcomes` table for drift
monitoring) that was considered and left out because there's no real feedback source to
populate it yet.

**`models`** — one row per tuned model: the same precision/recall/F1/PR-AUC/accuracy as the
Final Evaluation table, plus the `mlflow_run_id` that produced them. A lookup table, not
written to by the API - seeded by `db/seed_models.py` from a committed static snapshot
(`db/seed_data.json`), not a live MLflow query (that was the original design; it broke
inside a container, which has no access to the host's `mlflow.db` - see Docker below).
`db/export_seed_data.py` regenerates the snapshot from MLflow after retuning, locally. This
replaced a `TEST_METRICS` dict that used to be hardcoded in `api/model_registry.py`.

**`predictions`** — one row per `POST /predict` call: `model_id` (FK → `models.id`), every
input feature as its own column (not a JSON blob — real SQL aggregation over the audit log
is the point, e.g. `AVG(probability_of_default) GROUP BY ...`, which a JSON column would
make much more awkward), the returned probability, and the prediction. `monthly_income` and
`number_of_dependents` are nullable, matching the optional API fields.

```
models                          predictions
├─ id (PK)                      ├─ id (PK)
├─ name (unique)          ┌────>├─ model_id (FK)
├─ test_precision         │     ├─ created_at (indexed)
├─ test_recall            │     ├─ <10 raw feature columns>
├─ test_f1                │     ├─ probability_of_default
├─ test_pr_auc            │     └─ prediction
├─ test_accuracy          │
├─ mlflow_run_id          │
└─ created_at ─────────────┘
```

### Running it

```bash
# local dev Postgres (one container, not the full docker-compose stack yet - see Roadmap)
docker volume create credit-risk-pgdata   # named volume - survives `docker rm`, unlike the
                                            # container's own writable layer
docker run -d --name credit-risk-postgres \
  -e POSTGRES_USER=credit_risk -e POSTGRES_PASSWORD=credit_risk -e POSTGRES_DB=credit_risk \
  -p 5432:5432 -v credit-risk-pgdata:/var/lib/postgresql/data postgres:16

alembic upgrade head        # create the tables (idempotent - tracked in the alembic_version table)
python -m db.seed_models    # populate `models` from MLflow's tuned runs
```

`DATABASE_URL` (env var, default `postgresql+psycopg://credit_risk:credit_risk@localhost:5432/credit_risk`
in `db/config.py`) is the one place the connection string lives — `alembic/env.py` reads it
from there instead of duplicating it in `alembic.ini`.

**Persistence, precisely**: restarting the FastAPI app never touches the database - it just
opens a new connection to whatever's already there. Stopping/starting the *same* container
(`docker stop` / `docker start`) also keeps the data - only `docker rm` destroys a
container's own filesystem layer. The named volume above is what survives that: verified by
deleting the container entirely and recreating it against the same volume, data intact.
Without `-v`, `docker rm` would silently wipe everything.

(The manual `docker run` above is what to use for local dev without the API, e.g. to run
`db/seed_models.py` from your own shell. The full stack - API included - now runs through
`docker compose`, below, which manages its own Postgres container and volume instead.)

## Docker

`docker-compose.yml` (project root) + `docker/Dockerfile` + `docker/entrypoint.sh` - two
services, API and Postgres, in one Compose-managed network.

**Why the `DATABASE_URL` changes here and nowhere else**: `db/config.py`'s default
(`@localhost:5432`) only ever worked because the API process ran directly on the host,
where Docker had port-forwarded Postgres to `localhost`. Inside Compose, the API is *also*
a container - `localhost` from its point of view is itself, not the `db` container next to
it. Compose gives every service a DNS name matching its key in the YAML, so the fix is
one environment variable override (`DATABASE_URL: ...@db:5432/...`) in `docker-compose.yml`,
not a code change - exactly what parameterizing it as an env var was for.

**`docker/entrypoint.sh`** runs on every container start, not just the first: `alembic
upgrade head` (idempotent - tracked by the `alembic_version` table) then `python -m
db.seed_models` (idempotent - upserts by name) then `exec uvicorn ... --host 0.0.0.0`. The
whole stack self-heals its schema and reference data on every `docker compose up`; no
manual migration step. `--host 0.0.0.0` (not the dev default `127.0.0.1`) matters inside a
container - `127.0.0.1` there means "only reachable from inside this container," which
would make the port mapping to the host a no-op.

**Image contents**: built from the full `requirements.txt`, including packages the API
never imports at runtime (Jupyter, matplotlib, torch's training-only paths) - simpler than
maintaining a second, leaner requirements file, at the cost of a larger image. Noted as a
possible future optimization, not fixed now (see `TODO.md`).

**Layer caching**: `requirements.txt` is copied and installed *before* the application code
(`COPY src/`). Docker caches each layer keyed on its inputs - editing `src/api/main.py`
therefore doesn't invalidate the multi-minute torch/xgboost/mlflow install step, only the
fast steps after it.

### Three real bugs the build surfaced, not just theory

Building an image from a clean `requirements.txt` is a genuine clean-room test - it caught
three things `pip install`-ing packages one at a time into a long-lived local `.venv` had
silently let slide.

1. **`pandas==3.0.5` in `requirements.txt` didn't match what was actually installed
   locally (`2.3.3`).** Some earlier `pip install` in this project's history quietly
   resolved pandas down to satisfy mlflow's `pandas<3` constraint, and `requirements.txt`
   was never updated to match - a `.venv` accumulates state silently across many separate
   installs; a fresh `pip install -r requirements.txt` in a container forces one
   simultaneous resolution across everything, and immediately refused a combination that
   had been quietly working (`ResolutionImpossible`). Fixed by re-pinning to what was
   actually running.

2. **PyPI's default Linux `torch` wheel pulls ~1.6GB of unconditional `nvidia-*`/`triton`
   CUDA runtime packages** - irrelevant here (CPU-only by design, see Neural Network
   above), and large enough that it looked like a flaky network timeout the first two times
   the build failed, rather than what it actually was: a genuinely enormous, pointless
   download. Never showed up locally because Apple Silicon has no CUDA build to begin with,
   so the macOS wheel never declares those dependencies. Fixed with
   `--extra-index-url https://download.pytorch.org/whl/cpu` in the Dockerfile.

3. **`db/seed_models.py` originally queried MLflow live** - fine locally, where
   `mlflow.db` sits right there on disk, but inside a fresh container there's no access to
   the host's tracking store at all, so it created an empty one and found zero tuned runs.
   Fixed by decoupling seeding from a live MLflow connection entirely: `db/seed_data.json`
   is a static, committed snapshot of the tuned metrics (regenerated locally via
   `python -m db.export_seed_data` after retuning), and `seed_models.py` just reads that
   file - works identically locally and in a container, no MLflow dependency at request
   time. Surfaced a second, smaller version of the same class of bug right behind it: a
   non-editable `pip install .` only copies `.py` files into `site-packages` by default,
   silently dropping `seed_data.json` and the `*.joblib` model artifacts until both were
   declared under `[tool.setuptools.package-data]` in `pyproject.toml`. Editable installs
   (`pip install -e .`, used everywhere else in this project) never hit this, since they
   just point back at the source tree instead of copying files - another gap invisible
   until the exact packaging mode used in the container was actually exercised.

### Running it

```bash
docker compose up --build
# API:      http://127.0.0.1:8000/docs
# Postgres: localhost:5432 (same credentials as the manual setup above)
```

Verified end-to-end against the real containers, not just assumed from the config: both
services healthy, `/health` and `/models` responding, a real `/predict` call scored and
its row visible in `predictions` via `docker compose exec db psql ...` with the `models`
join resolving correctly.

## Roadmap

- [x] Data loading + stratified split
- [x] Cleaning pipeline (missing values, sentinel values, outliers) as sklearn transformers
- [x] EDA notebook
- [x] `.venv` on a supported Python version, pinned `requirements.txt`
- [x] Baseline models: LogReg, Linear SVM, RBF SVM, Random Forest, Decision Tree, XGBoost, Gaussian Naive Bayes — with CV + MLflow tracking
- [x] Feature engineering: delinquency aggregate + interaction terms, validated empirically against log-transform and binning alternatives (capping won)
- [x] Baseline + tuned PyTorch MLP (hand-written training loop, sklearn-compatible wrapper): best untuned baseline of all 8 models (PR-AUC 0.389 — gets feature interactions for free); tuned to 0.392, 3rd overall behind the two tree ensembles. Uncovered and fixed a real segfault from mixing joblib (`n_jobs=-1`) and PyTorch threading on macOS (see Neural Network section)
- [x] Hyperparameter tuning (Optuna, 8 studies, pruning): XGBoost took over first place (PR-AUC 0.402), Decision Tree closed most of the gap to Random Forest via pruning, L1 confirmed the engineered features are worth keeping. Feature *extraction* (PCA) stays out of scope: ~15 features is low-dimensional, and credit scoring specifically benefits from features staying interpretable
- [x] Resampling ablations (SMOTE + undersampling, separate experiments, neither adopted): SMOTENC beats `class_weight="balanced"` on F1 but loses on PR-AUC; undersampling ties on PR-AUC despite discarding 87% of training data (more evidence for the "information ceiling" read of this dataset). Kept `class_weight`/`scale_pos_weight`/`priors` as the primary imbalance strategy — see Resampling Ablations section
- [x] Final test-set evaluation: artifacts saved to `src/ml/artifacts/`, every model scores at or above its CV estimate on the untouched test set (no "optimizer's curse" from 30 Optuna trials per model against the same folds), ranking identical to CV top to bottom
- [x] FastAPI serving layer: `/health`, `/models`, `/predict` over the 8 saved artifacts, caller picks the model per request; verified end-to-end with a live server (including validation errors and the missing-income/dependents imputation path)
- [x] PostgreSQL + SQLAlchemy + Alembic: `models` (lookup, seeded from MLflow) + `predictions` (audit log, FK to `models`); `/predict` writes to it, `/models` reads from it; verified end-to-end against a live container (migration, seed, insert, join query)
- [x] Docker: `docker-compose.yml` (API + Postgres), verified end-to-end against real containers. Caught 3 real bugs a long-lived local `.venv` had let slide (stale `pandas` pin, ~1.6GB of unneeded CUDA deps on Linux, MLflow-dependent seeding breaking with no host tracking store) - see Docker section. Podman: not tested, but the Dockerfile/Compose file are standard OCI, expected to work unchanged
- [ ] Kubernetes manifests
- [ ] GitHub Actions CI/CD
- [ ] Tests
