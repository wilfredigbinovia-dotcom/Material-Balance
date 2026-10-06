"""Gas well deliverability: tubing pressure drop and back-pressure inflow fitted to well tests.

Inflow (back-pressure equation, corrected for changing reservoir conditions)
    q = C * [ M * (pr^2 - pwf^2) ]^n
    M = gas mobility relative to initial conditions (relative permeability, viscosity, Z)

Tubing (average temperature and Z method, single-phase gas with the well-stream gravity)
    pwf^2 = e^s pth^2 + 6.67e-4 q^2 f T^2 z^2 (e^s - 1) (MD/TVD) / d^5
    s = 0.0375 * gravity * TVD / (T z)          q in Mscf/d, T in degR, d in inches
Liquid loading is not modelled: at low rates the true bottomhole pressure is higher.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

ROUGHNESS = 0.0006   # in, commercial steel tubing


@dataclass
class Well:
    name: str
    C: float = np.nan          # MMscf/d / psi^(2n)
    n: float = 1.0
    tvd: float = np.nan        # ft
    md: float = np.nan         # ft
    tid: float = np.nan        # in
    wht: float = np.nan        # degF, flowing wellhead temperature
    min_thp: float = np.nan    # psia
    min_bhp: float = np.nan    # psia
    qmax: float = np.nan       # MMscf/d separator gas
    n_tests: int = 0
    note: str = ""

    @property
    def has_tubing(self):
        return all(np.isfinite(v) and v > 0 for v in (self.tvd, self.tid, self.wht))

    @property
    def control(self):
        """'thp', 'bhp' or None when no usable limit is set."""
        if self.has_tubing and np.isfinite(self.min_thp):
            return "thp"
        if np.isfinite(self.min_bhp):
            return "bhp"
        return None


def tubing_bhp(pth, q_mmscfd, gz, t_res, well, wet=1.0):
    """Flowing bottomhole pressure from tubing-head pressure. q is separator gas; `wet`
    scales it to well-stream gas so that the produced condensate is carried too."""
    tvd = well.tvd
    md = well.md if np.isfinite(well.md) and well.md >= tvd else tvd
    tbar = 0.5 * (well.wht + t_res)
    tr = tbar + 459.67
    q = max(q_mmscfd, 0.0) * wet * 1000.0       # Mscf/d
    pwf = pth * 1.2
    for _ in range(30):
        pbar = 0.5 * (pth + pwf)
        z = float(gz(pbar, tbar))
        s = 0.0375 * gz.sg * tvd / (tr * z)
        fric = 0.0
        if q > 0:
            re = max(20.09 * gz.sg * q / (float(gz.visc(pbar, tbar)) * well.tid), 2100.0)
            f = 0.25 / np.log10(ROUGHNESS / (3.7 * well.tid) + 5.74 / re ** 0.9) ** 2
            fric = 6.67e-4 * q * q * f * tr * tr * z * z * (np.exp(s) - 1.0) * (md / tvd) / well.tid ** 5
        new = float(np.sqrt(np.exp(s) * pth * pth + fric))
        if abs(new - pwf) < 0.05:
            return new
        pwf = new
    return pwf


def fit_ipr(q, x):
    """Fit q = C x^n with x = M (pr^2 - pwf^2). Returns (C, n, note).

    One usable test: n = 1 is assumed. Several: n is fitted and kept within 0.5 to 1.
    """
    q, x = np.asarray(q, float), np.asarray(x, float)
    ok = (q > 0) & (x > 0) & np.isfinite(q) & np.isfinite(x)
    q, x = q[ok], x[ok]
    if len(q) == 0:
        return np.nan, 1.0, "no usable test (flowing pressure must be below reservoir pressure)"
    note = ""
    n = 1.0
    if len(q) >= 2 and np.ptp(np.log(x)) > 0.05:
        n = float(np.polyfit(np.log(x), np.log(q), 1)[0])
        if n > 1.0 or n < 0.5:
            note = f"fitted exponent {n:.2f} is outside 0.5 to 1 and was limited"
            n = float(np.clip(n, 0.5, 1.0))
    elif len(q) >= 2:
        note = "tests are at nearly the same drawdown; n = 1 assumed"
    else:
        note = "single test; n = 1 assumed"
    C = float(np.exp(np.mean(np.log(q) - n * np.log(x))))
    return C, n, note


def well_rate(well, pr, M, gz, t_res, wet=1.0):
    """Separator gas rate (MMscf/d) and flowing bottomhole pressure at the well's limit."""
    if not np.isfinite(well.C) or well.control is None or M <= 0 or not np.isfinite(pr):
        return 0.0, np.nan

    def ipr(pwf):
        return well.C * max(M * (pr * pr - pwf * pwf), 0.0) ** well.n

    if well.control == "bhp":
        pwf = well.min_bhp
        q = ipr(pwf)
    else:
        if tubing_bhp(well.min_thp, 0.0, gz, t_res, well, wet) >= pr:
            return 0.0, np.nan
        f = lambda qq: ipr(tubing_bhp(well.min_thp, qq, gz, t_res, well, wet)) - qq   # noqa: E731
        hi = ipr(0.0)
        q = brentq(f, 0.0, hi, xtol=1e-4) if f(hi) < 0 else hi
        pwf = tubing_bhp(well.min_thp, q, gz, t_res, well, wet)
        if np.isfinite(well.min_bhp) and pwf < well.min_bhp:
            pwf, q = well.min_bhp, ipr(well.min_bhp)
    if np.isfinite(well.qmax) and q > well.qmax:
        q = well.qmax
        # choked back: bottomhole pressure follows from the inflow equation
        pwf = float(np.sqrt(max(pr * pr - (q / well.C) ** (1.0 / well.n) / M, 0.0)))
    return float(q), float(pwf)
