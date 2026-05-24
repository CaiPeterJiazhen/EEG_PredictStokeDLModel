from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class FrozenHeadCandidate:
    name: str
    estimator: BaseEstimator


def build_pair_difference_embedding(eo_embedding: np.ndarray, ec_embedding: np.ndarray) -> np.ndarray:
    eo_embedding = np.asarray(eo_embedding, dtype=float)
    ec_embedding = np.asarray(ec_embedding, dtype=float)
    if eo_embedding.shape != ec_embedding.shape:
        raise ValueError("EO and EC embeddings must have the same shape.")
    return np.concatenate(
        [
            eo_embedding,
            ec_embedding,
            ec_embedding - eo_embedding,
            np.abs(ec_embedding - eo_embedding),
        ],
        axis=1,
    )


def build_frozen_head_candidates(seed: int = 42) -> dict[str, FrozenHeadCandidate]:
    candidates: dict[str, FrozenHeadCandidate] = {}
    for c_value, token in ((0.03, "0_03"), (0.1, "0_1"), (0.3, "0_3"), (1.0, "1"), (3.0, "3"), (10.0, "10")):
        candidates[f"logistic_l2_C{token}"] = FrozenHeadCandidate(
            f"logistic_l2_C{token}",
            _scaled(
                LogisticRegression(
                    C=c_value,
                    class_weight=None,
                    max_iter=2000,
                    random_state=seed,
                    solver="liblinear",
                )
            ),
        )
        candidates[f"logistic_balanced_C{token}"] = FrozenHeadCandidate(
            f"logistic_balanced_C{token}",
            _scaled(
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="liblinear",
                )
            ),
        )
        candidates[f"linear_svm_C{token}"] = FrozenHeadCandidate(
            f"linear_svm_C{token}",
            _scaled(SVC(C=c_value, kernel="linear", probability=True, random_state=seed)),
        )
        candidates[f"linear_svm_balanced_C{token}"] = FrozenHeadCandidate(
            f"linear_svm_balanced_C{token}",
            _scaled(SVC(C=c_value, class_weight="balanced", kernel="linear", probability=True, random_state=seed)),
        )
    for c_value, token in ((0.1, "0_1"), (0.3, "0_3"), (1.0, "1"), (3.0, "3"), (10.0, "10")):
        candidates[f"rbf_svm_C{token}_scale"] = FrozenHeadCandidate(
            f"rbf_svm_C{token}_scale",
            _scaled(SVC(C=c_value, gamma="scale", kernel="rbf", probability=True, random_state=seed)),
        )
        candidates[f"rbf_svm_balanced_C{token}_scale"] = FrozenHeadCandidate(
            f"rbf_svm_balanced_C{token}_scale",
            _scaled(SVC(C=c_value, class_weight="balanced", gamma="scale", kernel="rbf", probability=True, random_state=seed)),
        )
    candidates["lda_shrinkage_auto"] = FrozenHeadCandidate(
        "lda_shrinkage_auto",
        _scaled(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    )
    candidates["gaussian_nb"] = FrozenHeadCandidate("gaussian_nb", _scaled(GaussianNB()))
    for n_neighbors in (1, 3, 5):
        candidates[f"knn_k{n_neighbors}"] = FrozenHeadCandidate(
            f"knn_k{n_neighbors}",
            _scaled(KNeighborsClassifier(n_neighbors=n_neighbors)),
        )
    for hidden_dim, hidden_token in ((0, "linear"), (8, "mlp_h8"), (16, "mlp_h16")):
        for lr, lr_token in ((0.003, "0_003"), (0.01, "0_01")):
            for weight_decay, wd_token in ((0.001, "0_001"), (0.01, "0_01"), (0.1, "0_1")):
                name = f"neural_{hidden_token}_lr{lr_token}_wd{wd_token}"
                candidates[name] = FrozenHeadCandidate(
                    name,
                    _scaled(
                        TorchBinaryHeadEstimator(
                            hidden_dim=hidden_dim,
                            lr=lr,
                            weight_decay=weight_decay,
                            epochs=400,
                            dropout=0.1 if hidden_dim > 0 else 0.0,
                            seed=seed,
                            class_weight="balanced",
                        )
                    ),
                )
    for n_features, feature_token in ((64, "d64"), (128, "d128")):
        for gamma, gamma_token in ((0.004, "0_004"), (0.01, "0_01"), (0.03, "0_03"), (0.1, "0_1")):
            for weight_decay, wd_token in ((0.001, "0_001"), (0.01, "0_01")):
                name = f"neural_rff_{feature_token}_g{gamma_token}_lr0_01_wd{wd_token}"
                candidates[name] = FrozenHeadCandidate(
                    name,
                    _scaled(
                        TorchRFFBinaryHeadEstimator(
                            n_features=n_features,
                            gamma=gamma,
                            lr=0.01,
                            weight_decay=weight_decay,
                            epochs=500,
                            seed=seed,
                            class_weight="balanced",
                        )
                    ),
                )
    for seed_offset in (17, 101, 1009, 2027, 4099):
        for weight_decay, wd_token in ((0.001, "0_001"), (0.01, "0_01")):
            name = f"neural_rff_d64_g0_01_lr0_01_wd{wd_token}_rs{seed_offset}"
            candidates[name] = FrozenHeadCandidate(
                name,
                _scaled(
                    TorchRFFBinaryHeadEstimator(
                        n_features=64,
                        gamma=0.01,
                        lr=0.01,
                        weight_decay=weight_decay,
                        epochs=500,
                        seed=seed + seed_offset,
                        class_weight="balanced",
                    )
                ),
            )
    for gamma, gamma_token in ((0.004, "0_004"), (0.01, "0_01"), (0.03, "0_03"), (0.1, "0_1")):
        for weight_decay, wd_token in ((0.001, "0_001"), (0.01, "0_01"), (0.1, "0_1")):
            name = f"neural_rbf_centers_g{gamma_token}_lr0_01_wd{wd_token}"
            candidates[name] = FrozenHeadCandidate(
                name,
                _scaled(
                    TorchRBFCenterBinaryHeadEstimator(
                        gamma=gamma,
                        lr=0.01,
                        weight_decay=weight_decay,
                        epochs=500,
                        seed=seed,
                        class_weight="balanced",
                    )
                ),
            )
            margin_name = f"neural_rbf_margin_g{gamma_token}_lr0_01_wd{wd_token}"
            candidates[margin_name] = FrozenHeadCandidate(
                margin_name,
                _scaled(
                    TorchRBFMarginHeadEstimator(
                        gamma=gamma,
                        lr=0.01,
                        weight_decay=weight_decay,
                        epochs=800,
                        seed=seed,
                        class_weight="balanced",
                    )
                ),
            )
    return candidates


class TorchBinaryHeadEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        hidden_dim: int = 8,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        epochs: int = 400,
        dropout: float = 0.1,
        seed: int = 42,
        class_weight: str | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout = dropout
        self.seed = seed
        self.class_weight = class_weight

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchBinaryHeadEstimator":
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if x_array.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        if self.hidden_dim < 0:
            raise ValueError("hidden_dim must be non-negative.")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")

        torch.manual_seed(int(self.seed))
        model = self._build_model(x_array.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        pos_weight = self._pos_weight(y_array)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        x_tensor = torch.as_tensor(x_array)
        y_tensor = torch.as_tensor(y_array)
        model.train()
        for _ in range(int(self.epochs)):
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.model_ = model.eval()
        self.n_features_in_ = x_array.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("This TorchBinaryHeadEstimator instance is not fitted yet.")
        x_array = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(x_array))
            positive = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _build_model(self, input_dim: int) -> nn.Module:
        if self.hidden_dim == 0:
            return nn.Linear(input_dim, 1)
        return nn.Sequential(
            nn.Linear(input_dim, int(self.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(self.dropout)),
            nn.Linear(int(self.hidden_dim), 1),
        )

    def _pos_weight(self, y_array: np.ndarray) -> torch.Tensor:
        if self.class_weight != "balanced":
            return torch.ones(1, dtype=torch.float32)
        positives = float(np.sum(y_array >= 0.5))
        negatives = float(np.sum(y_array < 0.5))
        if positives <= 0 or negatives <= 0:
            return torch.ones(1, dtype=torch.float32)
        return torch.as_tensor([negatives / positives], dtype=torch.float32)


class TorchRFFBinaryHeadEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        n_features: int = 64,
        gamma: float = 0.004,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        epochs: int = 500,
        seed: int = 42,
        class_weight: str | None = None,
    ) -> None:
        self.n_features = n_features
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.class_weight = class_weight

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchRFFBinaryHeadEstimator":
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if x_array.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        if self.n_features < 1:
            raise ValueError("n_features must be positive.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")

        torch.manual_seed(int(self.seed))
        model = _RandomFourierBinaryHead(
            input_dim=x_array.shape[1],
            n_features=int(self.n_features),
            gamma=float(self.gamma),
            seed=int(self.seed),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        pos_weight = self._pos_weight(y_array)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        x_tensor = torch.as_tensor(x_array)
        y_tensor = torch.as_tensor(y_array)
        model.train()
        for _ in range(int(self.epochs)):
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.model_ = model.eval()
        self.n_features_in_ = x_array.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("This TorchRFFBinaryHeadEstimator instance is not fitted yet.")
        x_array = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(x_array))
            positive = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _pos_weight(self, y_array: np.ndarray) -> torch.Tensor:
        if self.class_weight != "balanced":
            return torch.ones(1, dtype=torch.float32)
        positives = float(np.sum(y_array >= 0.5))
        negatives = float(np.sum(y_array < 0.5))
        if positives <= 0 or negatives <= 0:
            return torch.ones(1, dtype=torch.float32)
        return torch.as_tensor([negatives / positives], dtype=torch.float32)


class _RandomFourierBinaryHead(nn.Module):
    def __init__(self, *, input_dim: int, n_features: int, gamma: float, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(input_dim, n_features, generator=generator) * float(np.sqrt(2.0 * gamma))
        bias = torch.rand(n_features, generator=generator) * float(2.0 * np.pi)
        self.register_buffer("projection", projection)
        self.register_buffer("bias", bias)
        self.output = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cos(x @ self.projection + self.bias)
        features = features * float(np.sqrt(2.0 / self.projection.shape[1]))
        return self.output(features)


class TorchRBFCenterBinaryHeadEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        gamma: float = 0.004,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        epochs: int = 500,
        seed: int = 42,
        class_weight: str | None = None,
    ) -> None:
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.class_weight = class_weight

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchRBFCenterBinaryHeadEstimator":
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if x_array.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")

        torch.manual_seed(int(self.seed))
        model = _RBFCenterBinaryHead(centers=torch.as_tensor(x_array), gamma=float(self.gamma))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        pos_weight = self._pos_weight(y_array)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        x_tensor = torch.as_tensor(x_array)
        y_tensor = torch.as_tensor(y_array)
        model.train()
        for _ in range(int(self.epochs)):
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.model_ = model.eval()
        self.n_features_in_ = x_array.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("This TorchRBFCenterBinaryHeadEstimator instance is not fitted yet.")
        x_array = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(x_array))
            positive = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _pos_weight(self, y_array: np.ndarray) -> torch.Tensor:
        if self.class_weight != "balanced":
            return torch.ones(1, dtype=torch.float32)
        positives = float(np.sum(y_array >= 0.5))
        negatives = float(np.sum(y_array < 0.5))
        if positives <= 0 or negatives <= 0:
            return torch.ones(1, dtype=torch.float32)
        return torch.as_tensor([negatives / positives], dtype=torch.float32)


class _RBFCenterBinaryHead(nn.Module):
    def __init__(self, *, centers: torch.Tensor, gamma: float) -> None:
        super().__init__()
        self.register_buffer("centers", centers.clone().detach())
        self.gamma = float(gamma)
        self.output = nn.Linear(centers.shape[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distances = torch.sum((x[:, None, :] - self.centers[None, :, :]) ** 2, dim=2)
        features = torch.exp(-self.gamma * distances)
        return self.output(features)


class TorchRBFMarginHeadEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        gamma: float = 0.004,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        epochs: int = 800,
        seed: int = 42,
        class_weight: str | None = None,
    ) -> None:
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.class_weight = class_weight

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchRBFMarginHeadEstimator":
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if x_array.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")

        torch.manual_seed(int(self.seed))
        model = _RBFCenterBinaryHead(centers=torch.as_tensor(x_array), gamma=float(self.gamma))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        x_tensor = torch.as_tensor(x_array)
        signed_y = torch.where(torch.as_tensor(y_array) >= 0.5, 1.0, -1.0)
        sample_weight = torch.as_tensor(self._sample_weights(y_array), dtype=torch.float32)
        model.train()
        for _ in range(int(self.epochs)):
            optimizer.zero_grad()
            logits = model(x_tensor)
            margin_loss = torch.clamp(1.0 - signed_y * logits, min=0.0) ** 2
            loss = (sample_weight * margin_loss).mean()
            loss.backward()
            optimizer.step()
        self.model_ = model.eval()
        self.n_features_in_ = x_array.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("This TorchRBFMarginHeadEstimator instance is not fitted yet.")
        x_array = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(x_array))
            positive = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _sample_weights(self, y_array: np.ndarray) -> np.ndarray:
        if self.class_weight != "balanced":
            return np.ones_like(y_array, dtype=np.float32)
        positives = float(np.sum(y_array >= 0.5))
        negatives = float(np.sum(y_array < 0.5))
        total = positives + negatives
        if positives <= 0 or negatives <= 0:
            return np.ones_like(y_array, dtype=np.float32)
        positive_weight = total / (2.0 * positives)
        negative_weight = total / (2.0 * negatives)
        return np.where(y_array >= 0.5, positive_weight, negative_weight).astype(np.float32)


def select_score_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if y_true.size == 0:
        raise ValueError("threshold selection requires at least one sample.")
    if metric not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("metric must be 'accuracy' or 'balanced_accuracy'.")

    best_threshold = float(np.min(y_score))
    best_value = -np.inf
    best_accuracy = -np.inf
    for threshold in sorted(float(value) for value in np.unique(y_score)):
        y_pred = (y_score >= threshold).astype(int)
        accuracy = float(accuracy_score(y_true, y_pred))
        value = accuracy if metric == "accuracy" else float(balanced_accuracy_score(y_true, y_pred))
        if value > best_value or (value == best_value and accuracy > best_accuracy):
            best_value = value
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def fit_predict_selected_frozen_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int = 42,
    candidate_names: Iterable[str] | None = None,
    selection_metric: str = "balanced_accuracy",
    threshold_strategy: str = "inner-loso",
) -> dict[str, object]:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    x_test = np.asarray(x_test, dtype=float)
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("x_train and x_test must be two-dimensional arrays.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must contain the same number of samples.")
    if np.unique(y_train).size < 2:
        raise ValueError("frozen head training requires both classes in y_train.")
    if threshold_strategy not in {"inner-loso", "fixed-0.5"}:
        raise ValueError("threshold_strategy must be 'inner-loso' or 'fixed-0.5'.")

    candidates = build_frozen_head_candidates(seed)
    if candidate_names is not None:
        selected_names = tuple(candidate_names)
        missing = [name for name in selected_names if name not in candidates]
        if missing:
            raise ValueError(f"Unknown frozen head candidate(s): {', '.join(missing)}")
        candidates = {name: candidates[name] for name in selected_names}

    best: dict[str, object] | None = None
    for candidate in candidates.values():
        oof_scores = _inner_loso_scores(candidate.estimator, x_train, y_train)
        threshold = 0.5 if threshold_strategy == "fixed-0.5" else select_score_threshold(y_train, oof_scores, metric=selection_metric)
        adjusted_oof_scores = oof_scores - threshold + 0.5
        y_oof_pred = (adjusted_oof_scores >= 0.5).astype(int)
        accuracy = float(accuracy_score(y_train, y_oof_pred))
        balanced = float(balanced_accuracy_score(y_train, y_oof_pred))
        value = accuracy if selection_metric == "accuracy" else balanced
        summary = {
            "selected_head": candidate.name,
            "inner_accuracy": accuracy,
            "inner_balanced_accuracy": balanced,
            "threshold": float(threshold),
            "selection_value": value,
        }
        if best is None or _is_better_head(summary, best, selection_metric):
            best = summary

    if best is None:
        raise RuntimeError("No frozen head candidates were evaluated.")
    candidate = candidates[str(best["selected_head"])]
    estimator = clone(candidate.estimator)
    estimator.fit(x_train, y_train)
    raw_score = float(_estimator_scores(estimator, x_test)[0])
    calibrated_score = raw_score if threshold_strategy == "fixed-0.5" else raw_score - float(best["threshold"]) + 0.5
    return {
        **best,
        "raw_score": raw_score,
        "y_score": float(calibrated_score),
        "y_pred": int(calibrated_score >= 0.5),
    }


def _scaled(estimator: BaseEstimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _inner_loso_scores(estimator: BaseEstimator, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    scores = np.zeros(y_train.shape[0], dtype=float)
    for inner_train, inner_test in LeaveOneOut().split(x_train):
        fitted = clone(estimator)
        fitted.fit(x_train[inner_train], y_train[inner_train])
        scores[inner_test[0]] = _estimator_scores(fitted, x_train[inner_test])[0]
    return scores


def _estimator_scores(estimator: BaseEstimator, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x)
        return np.asarray(probabilities[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(x)
        return np.asarray(decision, dtype=float)
    prediction = estimator.predict(x)
    return np.asarray(prediction, dtype=float)


def _is_better_head(candidate: dict[str, object], incumbent: dict[str, object], metric: str) -> bool:
    metric_key = "inner_accuracy" if metric == "accuracy" else "inner_balanced_accuracy"
    candidate_value = float(candidate[metric_key])
    incumbent_value = float(incumbent[metric_key])
    if candidate_value != incumbent_value:
        return candidate_value > incumbent_value
    candidate_accuracy = float(candidate["inner_accuracy"])
    incumbent_accuracy = float(incumbent["inner_accuracy"])
    if candidate_accuracy != incumbent_accuracy:
        return candidate_accuracy > incumbent_accuracy
    return str(candidate["selected_head"]) < str(incumbent["selected_head"])
