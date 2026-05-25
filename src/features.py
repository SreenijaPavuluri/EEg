"""
Feature extraction modules for EEG Motor Imagery.
All CPU-safe. No GPU required.
"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf


def bandpass_filter(X, low, high, sfreq, order=4):
    """Bandpass filter applied to EEG epochs (n_epochs, n_ch, n_times)."""
    nyq = sfreq / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, X, axis=-1)


class CovarianceFeatures(BaseEstimator, TransformerMixin):
    """Compute regularized covariance matrices from EEG epochs."""

    def __init__(self, estimator='lwf'):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X: (n_epochs, n_ch, n_times)
        n_epochs, n_ch, n_times = X.shape
        covs = np.zeros((n_epochs, n_ch, n_ch))
        for i in range(n_epochs):
            lw = LedoitWolf()
            lw.fit(X[i].T)
            covs[i] = lw.covariance_
        return covs


class FBCSPFeatures(BaseEstimator, TransformerMixin):
    """
    Filter-Bank Common Spatial Pattern (FBCSP) feature extractor.
    Applies CSP in multiple frequency bands and concatenates log-variance features.
    CPU-safe. No neural network involved.
    """

    def __init__(self, bands=None, sfreq=128.0, n_components=4):
        # Default: theta, alpha, beta, low-gamma
        self.bands = bands or [
            (4, 8), (8, 12), (12, 16), (16, 20),
            (20, 24), (24, 28), (28, 32)
        ]
        self.sfreq = sfreq
        self.n_components = n_components
        self.csps_ = None

    def fit(self, X, y):
        from sklearn.pipeline import Pipeline
        from mne.decoding import CSP
        self.csps_ = []
        for (low, high) in self.bands:
            Xf = bandpass_filter(X, low, high, self.sfreq)
            csp = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
            csp.fit(Xf, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        feats = []
        for csp, (low, high) in zip(self.csps_, self.bands):
            Xf = bandpass_filter(X, low, high, self.sfreq)
            feats.append(csp.transform(Xf))
        return np.concatenate(feats, axis=1)


class TangentSpaceFeatures(BaseEstimator, TransformerMixin):
    """
    Project covariance matrices to the Riemannian tangent space.
    Uses pyriemann.
    """

    def __init__(self, metric='riemann'):
        self.metric = metric
        self.ts_ = None

    def fit(self, X, y=None):
        from pyriemann.tangentspace import TangentSpace
        covs = CovarianceFeatures().transform(X)
        self.ts_ = TangentSpace(metric=self.metric)
        self.ts_.fit(covs)
        return self

    def transform(self, X):
        covs = CovarianceFeatures().transform(X)
        return self.ts_.transform(covs)


class PowerSpectralFeatures(BaseEstimator, TransformerMixin):
    """Log band-power features: mean log-power in theta, alpha, beta bands."""

    def __init__(self, sfreq=128.0, bands=None):
        self.sfreq = sfreq
        self.bands = bands or [(4, 8), (8, 13), (13, 30)]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for (low, high) in self.bands:
            Xf = bandpass_filter(X, low, high, self.sfreq)
            power = np.log(np.mean(Xf ** 2, axis=-1) + 1e-8)  # (n_epochs, n_ch)
            feats.append(power)
        return np.concatenate(feats, axis=1)
