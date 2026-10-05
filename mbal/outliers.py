"""Screening of pressure surveys before and after regression.

Nothing here deletes data: the functions only score surveys so the engineer can
decide which to switch off.
"""
import numpy as np
from scipy import stats


def robust_scale(r):
    """Standard deviation estimated from the median absolute deviation about zero.
    Unlike the ordinary standard deviation it is not inflated by the outliers."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    return float(1.4826 * np.median(np.abs(r))) if len(r) else float("nan")


def _trend_dev(x, p, nb_ok, half_window):
    n = len(p)
    dev = np.full(n, np.nan)
    idx = np.flatnonzero(nb_ok)
    for i in range(1, n):
        nb = np.concatenate([idx[idx < i][-half_window:], idx[idx > i][:half_window]])
        if len(nb) < 3:
            continue
        xn, pn = x[nb], p[nb]
        if np.ptp(xn) <= 0:
            pred = np.median(pn)
        else:
            slope, icpt, *_ = stats.theilslopes(pn, xn)
            pred = icpt + slope * x[i]
        dev[i] = p[i] - pred
    return dev


def trend_screen(x, p, use=None, nsig=3.0, half_window=3, floor=None, max_frac=0.25):
    """Model-free screen: how far does each pressure sit from its neighbours' trend?

    For every survey a robust straight line (Theil-Sen, the median of pairwise slopes)
    is drawn through up to `half_window` neighbours on each side - not the point itself -
    on the pressure versus cumulative production trend, and the survey is compared with
    that line. The worst survey beyond `nsig` is then set aside so it cannot distort the
    trend seen by its neighbours, and the screen is repeated until nothing new is found
    (at most `max_frac` of the surveys).

    Returns (deviation, z). z = deviation / robust scatter; |z| > nsig is suspect.
    Index 0 (initial conditions) is used as a neighbour but never scored. Surveys that
    are switched off are scored but never used as neighbours.
    """
    x, p = np.asarray(x, float), np.asarray(p, float)
    n = len(p)
    use = np.ones(n, bool) if use is None else np.asarray(use, bool)
    floor = floor if floor is not None else 1e-3 * float(p[0])
    ok = use.copy()
    for _ in range(max(int(max_frac * n), 1) + 1):
        dev = _trend_dev(x, p, ok, half_window)
        s = max(robust_scale(dev[ok]), floor)
        z = dev / s
        cand = np.where(ok & np.isfinite(z), np.abs(z), 0.0)
        cand[0] = 0.0
        worst = int(np.argmax(cand))
        if cand[worst] <= nsig or _ == max(int(max_frac * n), 1):
            break
        ok[worst] = False
    return dev, z


def residual_screen(p_sim, p_obs, use=None, floor=None):
    """Model-based screen: standardised residuals of the current match.

    Returns (residual = history - model, z). The scatter comes from the surveys that are
    switched on, by the median absolute deviation, so the suspect points themselves do
    not hide behind an inflated standard deviation."""
    r = np.asarray(p_obs, float) - np.asarray(p_sim, float)
    r[0] = np.nan
    use = np.ones(len(r), bool) if use is None else np.asarray(use, bool)
    s = robust_scale(r[use])
    s = max(s, floor if floor is not None else 1e-3 * float(p_obs[0]))
    return r, r / s


def runs_test(r):
    """Wald-Wolfowitz runs test on the signs of the residuals (in time order).

    Random scatter changes sign often. Few, long runs of one sign mean the model is
    systematically off - a model problem (aquifer, compressibility, PVT), which
    switching surveys off must not be used to hide.
    Returns dict(runs, expected, z, pvalue, longest, systematic)."""
    r = np.asarray(r, float)
    s = np.sign(r[np.isfinite(r) & (r != 0)])
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    n = n1 + n2
    out = {"runs": 0, "expected": float("nan"), "z": float("nan"), "pvalue": float("nan"),
           "longest": 0, "systematic": False, "n": n}
    if n < 8 or n1 == 0 or n2 == 0:
        out["systematic"] = n >= 8          # every residual on one side
        out["longest"] = n
        out["runs"] = 1 if n else 0
        return out
    change = np.flatnonzero(np.diff(s) != 0)
    runs = len(change) + 1
    edges = np.concatenate([[-1], change, [n - 1]])
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n ** 2 * (n - 1.0))
    z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    pval = float(stats.norm.cdf(z))         # one-sided: too few runs
    out.update(runs=int(runs), expected=float(mu), z=float(z), pvalue=pval,
               longest=int(np.diff(edges).max()), systematic=bool(pval < 0.01))
    return out
