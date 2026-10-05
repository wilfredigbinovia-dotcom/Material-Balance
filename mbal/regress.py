"""Non-linear regression of tank and aquifer parameters on measured pressure."""
import numpy as np
from scipy.optimize import least_squares

from .outliers import robust_scale


def pressure_rms(p_sim, p_obs, w=None):
    """Weighted RMS pressure mismatch over the surveys that are switched on (w > 0)."""
    w = np.ones(len(p_obs)) if w is None else np.asarray(w, float).copy()
    w[0] = 0.0
    m = np.isfinite(p_sim) & np.isfinite(p_obs) & (w > 0)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.sum(w[m] * (p_sim[m] - p_obs[m]) ** 2) / np.sum(w[m])))


def regress(simulate, p_obs, x0, lo, hi, w=None, loss="linear", max_nfev=200):
    """Fit positive parameters in log space.

    simulate(values) -> simulated pressure array (NaN where the model fails).
    w    : weight of each survey (0 = switched off). Residuals are scaled by sqrt(w).
    loss : 'linear' is ordinary least squares. 'soft_l1' is a robust loss: residuals
           larger than the typical scatter pull on the fit linearly instead of
           quadratically, so an isolated bad survey cannot drag the match. The scatter
           is estimated from the data (median absolute deviation) in two passes.
    Returns dict(x, rms0, rms, se_rel (relative 1-sigma, may be NaN), ...).
    """
    x0, lo, hi = (np.asarray(v, float) for v in (x0, lo, hi))
    x0 = np.clip(x0, lo * (1 + 1e-9), hi * (1 - 1e-9))
    p_obs = np.asarray(p_obs, float)
    w = np.ones(len(p_obs)) if w is None else np.asarray(w, float).copy()
    w[0] = 0.0
    on = w > 0
    sw = np.sqrt(w[on])
    scale = float(p_obs[0])

    def resid(lx):
        r = (simulate(np.exp(lx)) - p_obs)[on]
        return sw * np.where(np.isfinite(r), r, scale)   # heavy penalty where the tank runs dry

    kw = dict(bounds=(np.log(lo), np.log(hi)), method="trf", x_scale=1.0, diff_step=1e-3,
              max_nfev=max_nfev)
    sol = least_squares(resid, np.log(x0), **kw)
    f_scale = None
    if loss != "linear":
        for _ in range(2):
            f_scale = max(robust_scale(sol.fun), 1e-4 * scale)
            sol = least_squares(resid, sol.x, loss=loss, f_scale=2.0 * f_scale, **kw)
    x = np.exp(sol.x)
    p_fit = simulate(x)
    n, k = int(on.sum()), len(x)
    se, corr = np.full(k, np.nan), None
    try:
        s2 = float(sol.fun @ sol.fun) / max(n - k, 1)
        if f_scale is not None:
            s2 = f_scale ** 2
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        se = np.sqrt(np.diag(cov))        # std of ln(x) ~ relative std of x
        corr = cov / np.outer(se, se)
    except (np.linalg.LinAlgError, FloatingPointError):
        pass
    return {"x": x, "p": p_fit, "rms0": pressure_rms(simulate(x0), p_obs, w),
            "rms": pressure_rms(p_fit, p_obs, w), "se_rel": se, "corr": corr,
            "success": bool(sol.success), "nfev": int(sol.nfev), "f_scale": f_scale,
            "at_bound": (np.abs(sol.x - np.log(lo)) < 1e-6) | (np.abs(sol.x - np.log(hi)) < 1e-6)}


def leave_one_out(simulate, p_obs, x, lo, hi, w, loss="linear"):
    """Influence of each survey: relative change in every matched parameter when that
    survey alone is switched off and the fit is repeated from the matched solution.
    Returns an array (n_points, n_params); rows of NaN for surveys already off."""
    w = np.asarray(w, float)
    out = np.full((len(p_obs), len(x)), np.nan)
    for i in range(1, len(p_obs)):
        if w[i] <= 0:
            continue
        wi = w.copy()
        wi[i] = 0.0
        r = regress(simulate, p_obs, x, lo, hi, wi, loss, max_nfev=40)
        out[i] = r["x"] / np.asarray(x, float) - 1.0
    return out
