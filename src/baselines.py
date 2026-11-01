"""Simple baselines of plan section 4.3. Inputs are the scaled windows X (B, T, F) with rho in column 0;
every prediction is an affine map of rho values, so it inverse-transforms like the deep models' output."""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

RIDGE_ALPHAS = np.logspace(-3, 3, 13)  # nhóm tự chọn


class Persistence:
    """rho_hat(t+1) = rho(t)."""
    n_params = 0

    def fit(self, *_):
        return self

    def predict(self, X):
        return X[:, -1, 0]


class MovingAverage:
    """Mean of rho over the last n days."""
    n_params = 0

    def __init__(self, n=5):
        self.n = n

    def fit(self, *_):
        return self

    def predict(self, X):
        return X[:, -self.n:, 0].mean(1)


class AR1:
    """rho_hat(t+1) = a + b * rho(t), fit on train."""
    n_params = 2

    def fit(self, X, y, *_):
        self.model = LinearRegression().fit(X[:, -1, :1], y)
        return self

    def predict(self, X):
        return self.model.predict(X[:, -1, :1])


class RidgeWindow:
    """Ridge on the flattened T x F window; alpha picked by validation MSE, weights fit on train only."""

    def fit(self, X, y, X_val, y_val):
        fits = [Ridge(alpha=a).fit(X.reshape(len(X), -1), y) for a in RIDGE_ALPHAS]
        mse = [np.mean((m.predict(X_val.reshape(len(X_val), -1)) - y_val) ** 2) for m in fits]
        self.model = fits[int(np.argmin(mse))]
        self.alpha = self.model.alpha
        self.n_params = X[0].size + 1
        return self

    def predict(self, X):
        return self.model.predict(X.reshape(len(X), -1))


BASELINES = {"Persistence": Persistence, "MA5": MovingAverage, "AR(1)": AR1, "Ridge": RidgeWindow}
