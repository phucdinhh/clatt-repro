"""Plan section 6.1. Every function takes original-scale rho: y_true, y_pred, and rho_prev = rho(t)."""
import numpy as np

NEAR_ZERO = 0.05


def mse(y_true, y_pred):  # Eq. 13
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):  # Eq. 14
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true, y_pred):  # Eq. 16
    return float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))


def mape_terms(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs((y_pred - y_true) / y_true)


def mape(y_true, y_pred):  # Eq. 15 as a fraction, like Table 5
    """Rows with y_true == 0 give inf/nan terms; they are dropped (counted by n_mape_dropped) so the mean
    and metrics.json stay finite."""
    terms = mape_terms(y_true, y_pred)
    return float(np.mean(terms[np.isfinite(terms)]))


def n_mape_dropped(y_true, y_pred):
    return int(np.sum(~np.isfinite(mape_terms(y_true, y_pred))))


def mape_excl(y_true, y_pred):
    keep = np.abs(y_true) >= NEAR_ZERO
    return mape(y_true[keep], y_pred[keep])


def smape(y_true, y_pred):
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))


def n_near_zero(y_true):
    return int(np.sum(np.abs(y_true) < NEAR_ZERO))


def skill(y_true, y_pred, rho_prev):
    return 1 - mse(y_true, y_pred) / mse(y_true, rho_prev)


def r2_delta(y_true, y_pred, rho_prev):
    return r2(y_true - rho_prev, y_pred - rho_prev)


def dir_acc(y_true, y_pred, rho_prev):
    """Share of days where the predicted change has the true change's sign; days with no change are dropped.
    A prediction equal to rho_prev has sign 0 and never counts as a hit, so persistence scores 0."""
    moved = y_true != rho_prev
    return float(np.mean(np.sign(y_pred - rho_prev)[moved] == np.sign(y_true - rho_prev)[moved]))


def compute_all(y_true, y_pred, rho_prev):
    return {
        "mse": mse(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred), "mape": mape(y_true, y_pred), "mape_excl": mape_excl(y_true, y_pred),
        "n_mape_dropped": n_mape_dropped(y_true, y_pred),
        "smape": smape(y_true, y_pred), "n_near_zero": n_near_zero(y_true),
        "skill": skill(y_true, y_pred, rho_prev), "r2_delta": r2_delta(y_true, y_pred, rho_prev),
        "dir_acc": dir_acc(y_true, y_pred, rho_prev),
    }
