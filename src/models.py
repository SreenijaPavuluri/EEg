"""
Classifier models for EEG Motor Imagery.
All CPU-safe. Includes classical ML, EEGNet, ShallowConvNet.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


# ──────────────────────────────────────────────────────────────
# Classical ML baselines
# ──────────────────────────────────────────────────────────────

def make_fbcsp_lda(sfreq=128.0, n_components=4):
    """FBCSP + LDA pipeline."""
    from src.features import FBCSPFeatures
    return Pipeline([
        ('fbcsp', FBCSPFeatures(sfreq=sfreq, n_components=n_components)),
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis()),
    ])


def make_riemannian_mdm():
    """Riemannian MDM (Minimum Distance to Mean) classifier."""
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    return Pipeline([
        ('cov', Covariances(estimator='lwf')),
        ('clf', MDM(metric='riemann')),
    ])


def make_riemannian_svm():
    """Riemannian tangent-space SVM."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    return Pipeline([
        ('cov', Covariances(estimator='lwf')),
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=SEED)),
    ])


def make_logvar_lda(sfreq=128.0):
    """Log-variance + LDA (very simple baseline)."""
    from src.features import PowerSpectralFeatures
    return Pipeline([
        ('feats', PowerSpectralFeatures(sfreq=sfreq)),
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis()),
    ])


# ──────────────────────────────────────────────────────────────
# EEGNet (Lawhern et al. 2018) – compact neural baseline
# ──────────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet: A compact convolutional network for EEG classification.
    Lawhern et al. 2018, J. Neural Eng.
    """

    def __init__(self, n_classes=2, n_channels=64, n_times=256,
                 F1=8, D=2, F2=16, dropout=0.5, kernel_size=64):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        F2_actual = F1 * D

        # Block 1: temporal + depthwise spatial
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_size), padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2_actual, (1, 1), bias=False),
            nn.BatchNorm2d(F2_actual),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.block1(dummy)
            out = self.block2(out)
            self.fc_in = out.view(1, -1).shape[1]

        self.fc = nn.Linear(self.fc_in, n_classes)

    def forward(self, x):
        # x: (batch, n_ch, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_ch, n_times)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet: Schirrmeister et al. 2017, Human Brain Mapping.
    Shallow temporal + spatial convolution.
    """

    def __init__(self, n_classes=2, n_channels=64, n_times=256,
                 n_filters_time=40, n_filters_spat=40,
                 filter_time_length=25, dropout=0.5):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=False)
        self.spatial_conv = nn.Conv2d(n_filters_time, n_filters_spat, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.temporal_conv(dummy)
            out = self.spatial_conv(out)
            out = self.pool(out)
            self.fc_in = out.view(1, -1).shape[1]

        self.fc = nn.Linear(self.fc_in, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x ** 2  # square activation
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # log activation
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper for PyTorch neural models.
    CPU-only training with early stopping.
    """

    def __init__(self, model_cls, model_kwargs=None, lr=1e-3, n_epochs=150,
                 batch_size=32, patience=20, random_state=SEED):
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([label_map[c] for c in y])

        n_ch, n_times = X.shape[1], X.shape[2]
        kwargs = dict(self.model_kwargs)
        kwargs.update({'n_classes': n_classes, 'n_channels': n_ch, 'n_times': n_times})

        self.model_ = self.model_cls(**kwargs)
        self.model_.train()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y_int)

        # Train/val split for early stopping (80/20)
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        val_size = max(1, int(0.2 * n))
        train_idx = idx[val_size:]
        val_idx = idx[:val_size]

        X_train, y_train = X_t[train_idx], y_t[train_idx]
        X_val, y_val = X_t[val_idx], y_t[val_idx]

        best_val_loss = float('inf')
        patience_cnt = 0
        best_state = None

        for epoch in range(self.n_epochs):
            self.model_.train()
            perm = torch.randperm(len(X_train))
            total_loss = 0.0
            for start in range(0, len(X_train), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb, yb = X_train[batch_idx], y_train[batch_idx]
                optimizer.zero_grad()
                out = self.model_(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_out = self.model_(X_val)
                val_loss = criterion(val_out, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.FloatTensor(X))
            probs = F.softmax(out, dim=1).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
