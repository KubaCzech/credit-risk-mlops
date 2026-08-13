import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..config import RANDOM_STATE
from .baseline import cleaning_and_feature_steps


class MLPModule(nn.Module):
    """Input -> Linear(64) -> ReLU -> Dropout -> Linear(32) -> ReLU -> Dropout -> Linear(1).

    Outputs a raw logit, not a probability - paired with BCEWithLogitsLoss, which fuses
    sigmoid + binary cross-entropy into one numerically stable op (avoids overflow for
    extreme logits that a separate sigmoid-then-BCE would hit)."""

    def __init__(self, n_features: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Hand-written training loop (forward, loss, backward, optimizer step, epochs) wrapped
    in a minimal sklearn-compatible estimator, so it slots into the same Pipeline /
    cross_validate_pipeline / Optuna tuning code as every other model - no separate
    evaluation path, no framework (skorch) hiding what the training loop actually does.

    CPU only: MPS is available on this machine, but a network this small (64+32 hidden
    units) on 100k-ish rows trains in seconds on CPU - GPU transfer overhead isn't worth it
    at this scale.
    """

    def __init__(
        self,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 15,
        validation_fraction: float = 0.1,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, y) -> "TorchMLPClassifier":
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, stratify=y, random_state=self.random_state
        )

        # BCEWithLogitsLoss's pos_weight is the same idea as scale_pos_weight in XGBoost /
        # class_weight="balanced" elsewhere - weight the rare class's loss contribution by
        # the neg/pos ratio, computed from this fold's actual training labels.
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model_ = MLPModule(X.shape[1], self.hidden_dim1, self.hidden_dim2, self.dropout)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        n_train = X_train_t.shape[0]
        generator = torch.Generator().manual_seed(self.random_state)

        for epoch in range(self.max_epochs):  # noqa: B007 - used after the loop, below
            self.model_.train()
            permutation = torch.randperm(n_train, generator=generator)
            for start in range(0, n_train, self.batch_size):
                batch_idx = permutation[start : start + self.batch_size]
                xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        assert best_state is not None, "max_epochs must be >= 1 - at least one epoch has to run to have a best_state"
        self.model_.load_state_dict(best_state)
        self.n_epochs_trained_ = epoch + 1
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
            positive_proba = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - positive_proba, positive_proba])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_mlp_pipeline(scaler=None, **mlp_kwargs) -> Pipeline:
    return Pipeline(
        [
            *cleaning_and_feature_steps(),
            ("scaler", scaler if scaler is not None else StandardScaler()),
            ("clf", TorchMLPClassifier(**mlp_kwargs)),
        ]
    )
