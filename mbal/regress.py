"""Non-linear regression of tank and aquifer parameters on measured pressure."""
import numpy as np
from scipy.optimize import least_squares


def pressure_rms(p_sim, p_obs):
    m = np.isfinite(p_sim) & np.isfinite(p_obs)
    m[0] = False
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((p_sim[m] - p_obs[m]) ** 2)))


def regress(simulate, p_obs, x0, lo, hi):
    """Fit positive parameters in log space.

    simulate(values) -> simulated pressure array (NaN where the model fails).
    Returns dict(x, rms0, rms, se_rel (relative 1-sigma, may be NaN), success, nfev).
    """
    x0, lo, hi = (np.asarray(v, float) for v in (x0, lo, hi))
    x0 = np.clip(x0, lo * (1 + 1e-9), hi * (1 - 1e-9))
    scale = float(p_obs[0])

    def resid(lx):
        ps = simulate(np.exp(lx))
        r = (ps - p_obs)[1:]
        return np.where(np.isfinite(r), r, scale)   # heavy penalty where the tank runs dry

    r0 = resid(np.log(x0))
    sol = least_squares(resid, np.log(x0), bounds=(np.log(lo), np.log(hi)),
                        method="trf", x_scale=1.0, diff_step=1e-3, max_nfev=200)
    x = np.exp(sol.x)
    n, k = len(sol.fun), len(x)
    se = np.full(k, np.nan)
    try:
        s2 = float(sol.fun @ sol.fun) / max(n - k, 1)
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        se = np.sqrt(np.diag(cov))        # std of ln(x) ~ relative std of x
    except np.linalg.LinAlgError:
        pass
    corr = None
    try:
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    except Exception:
        pass
    return {"x": x, "rms0": float(np.sqrt(np.mean(r0 ** 2))),
            "rms": float(np.sqrt(np.mean(sol.fun ** 2))), "se_rel": se, "corr": corr,
            "success": bool(sol.success), "nfev": int(sol.nfev),
            "at_bound": (np.abs(sol.x - np.log(lo)) < 1e-6) | (np.abs(sol.x - np.log(hi)) < 1e-6)}
