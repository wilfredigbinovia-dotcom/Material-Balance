"""
resmb - reservoir material balance in field units.

Contents
--------
BlackOilPVT        PVT correlations (Standing / Vasquez-Beggs / Glaso, Beggs-Robinson,
                   Dranchuk-Abou-Kassem z, Lee-Gonzalez-Eakin, McCain Bw)
WD                 Van Everdingen-Hurst dimensionless water influx (Stehfest inversion)
Aquifer            aquifer description for the single-tank analyses
oil_mbe, gas_mbe   Havlena-Odeh straight-line analysis (plus p/z for gas)
simulate_tank,
analytical_match   run one tank forward in time and regress chosen parameters on measured pressure
Tank, Connection,
MultiTank          connected-tank simulator with pressure history matching
aggregate_wells,
build_history      turn per-well production and pressure surveys into a reservoir history
reservoir_table    pick one reservoir out of a multi-reservoir history file

Units
-----
pressure psia | time days | Np, Wp, Winj MMSTB | Gp, Ginj MMscf (oil) | Gp Bscf (gas)
Bo, Bw rb/STB | Rs scf/STB | Bg rb/scf | compressibility 1/psi | F, We MMrb
aquifer constants rb/psi | J rb/(day.psi) | Wei bbl (single tank) or MMbbl (Tank)

Requires numpy, scipy and pandas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ive, kve

__all__ = ["BlackOilPVT", "z_factor", "WD", "Aquifer", "veh_influx_function",
           "fetkovich_influx", "oil_mbe", "gas_mbe", "MBResult",
           "simulate_tank", "analytical_match", "MatchResult", "FIT_PARAMETERS",
           "Tank", "Connection", "MultiTank", "MTResult",
           "standardise_columns", "aggregate_wells", "build_history", "common_origin", "reservoir_table"]


# =============================================================================
# PVT correlations
# =============================================================================
def _dak_z(ppr: float, tpr: float) -> float:
    """Dranchuk-Abou-Kassem z-factor from pseudo-reduced pressure and temperature."""
    A1, A2, A3, A4, A5 = 0.3265, -1.07, -0.5339, 0.01569, -0.05165
    A6, A7, A8, A9, A10, A11 = 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.721
    c1 = A1 + A2 / tpr + A3 / tpr**3 + A4 / tpr**4 + A5 / tpr**5
    c2 = A6 + A7 / tpr + A8 / tpr**2
    c3 = A9 * (A7 / tpr + A8 / tpr**2)

    def zf(r):
        return (1 + c1 * r + c2 * r * r - c3 * r**5
                + A10 * (1 + A11 * r * r) * (r * r / tpr**3) * math.exp(-A11 * r * r))

    target = 0.27 * ppr / tpr          # = rho_r * z
    lo, hi = 0.0, 3.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mid * zf(mid) < target:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    return target / r if r > 0 else 1.0


def _pseudo_criticals(gas_gravity: float):
    """Sutton pseudo-critical temperature (R) and pressure (psia)."""
    g = gas_gravity
    return 169.2 + 349.5 * g - 74.0 * g * g, 756.8 - 131.07 * g - 3.6 * g * g


def z_factor(p: float, temp_f: float, gas_gravity: float) -> float:
    """Gas z-factor (Dranchuk-Abou-Kassem with Sutton pseudo-criticals)."""
    tpc, ppc = _pseudo_criticals(gas_gravity)
    return _dak_z(p / ppc, (temp_f + 459.67) / tpc)


class BlackOilPVT:
    """Black-oil and gas properties from correlations.

    Give either ``rsb`` (solution GOR at the bubble point, scf/STB) or ``pb`` (psia).
    ``correlation`` is 'standing', 'vasquez-beggs' or 'glaso' and sets pb, Rs and Bo.
    """

    def __init__(self, api: float, gas_gravity: float, temp_f: float,
                 rsb: Optional[float] = None, pb: Optional[float] = None,
                 correlation: str = "standing"):
        if (rsb is None) == (pb is None):
            raise ValueError("Give exactly one of rsb or pb.")
        self.api, self.gg, self.T = api, gas_gravity, temp_f
        self.TR = temp_f + 459.67
        self.go = 141.5 / (131.5 + api)
        self.correlation = correlation.lower()
        api_, gg, T, TR, go = api, gas_gravity, temp_f, self.TR, self.go

        if self.correlation in ("vasquez-beggs", "vb"):
            c = ([0.0362, 1.0937, 25.724, 4.677e-4, 1.751e-5, -1.811e-8] if api_ <= 30
                 else [0.0178, 1.187, 23.931, 4.67e-4, 1.1e-5, 1.337e-9])
            e = c[0] * gg * math.exp(c[2] * api_ / TR)
            self._rs_p = lambda p: e * p ** c[1]
            self._pb_rs = lambda rs: (rs / e) ** (1 / c[1])
            self._bo_rs = lambda rs: 1 + c[3] * rs + (T - 60) * (api_ / gg) * (c[4] + c[5] * rs)
        elif self.correlation == "glaso":
            def pb_rs(rs):
                x = math.log10((rs / gg) ** 0.816 * T ** 0.172 / api_ ** 0.989)
                return 10 ** (1.7669 + 1.7447 * x - 0.30218 * x * x)

            def rs_p(p):
                d = 3.04398 - 1.20872 * (math.log10(p) - 1.7669)
                x = (1.7447 - math.sqrt(d)) / 0.60436 if d > 0 else 1.7447 / 0.60436
                return gg * (api_ ** 0.989 / T ** 0.172 * 10 ** x) ** (1 / 0.816)

            def bo_rs(rs):
                lb = math.log10(rs * (gg / go) ** 0.526 + 0.968 * T)
                return 1 + 10 ** (-6.58511 + 2.91329 * lb - 0.27683 * lb * lb)
            self._pb_rs, self._rs_p, self._bo_rs = pb_rs, rs_p, bo_rs
        elif self.correlation == "standing":
            a = 0.00091 * T - 0.0125 * api_
            self._pb_rs = lambda rs: 18.2 * ((rs / gg) ** 0.83 * 10 ** a - 1.4)
            self._rs_p = lambda p: gg * ((p / 18.2 + 1.4) * 10 ** (-a)) ** (1 / 0.83)
            self._bo_rs = lambda rs: 0.9759 + 0.00012 * (rs * math.sqrt(gg / go) + 1.25 * T) ** 1.2
        else:
            raise ValueError(f"Unknown correlation {correlation!r}")

        if pb is not None:
            self.pb, self.rsb = float(pb), self._rs_p(pb)
        else:
            self.rsb, self.pb = float(rsb), self._pb_rs(rsb)
        if not self.pb > 14.7:
            raise ValueError("These inputs give no physical bubble point.")
        self.bob = self._bo_rs(self.rsb)
        # Vasquez-Beggs undersaturated compressibility: co = A / p
        self._A = 1e-5 * (-1433 + 5 * self.rsb + 17.2 * T - 1180 * gg + 12.61 * api_)
        # Beggs-Robinson dead and live oil viscosity
        self.mu_dead = 10 ** (10 ** (3.0324 - 0.02023 * api_) * T ** -1.163) - 1
        self.mu_ob = self._mu_live(self.rsb)
        self.tpc, self.ppc = _pseudo_criticals(gg)

    def _mu_live(self, rs):
        return 10.715 * (rs + 100) ** -0.515 * self.mu_dead ** (5.44 * (rs + 150) ** -0.338)

    def at(self, p: float) -> dict:
        """All properties at one pressure (psia)."""
        T, TR = self.T, self.TR
        if p >= self.pb:
            rs = self.rsb
            bo = self.bob * (self.pb / p) ** self._A
            co = self._A / p
            muo = self.mu_ob * (p / self.pb) ** (2.6 * p ** 1.187 * math.exp(-11.513 - 8.98e-5 * p))
        else:
            rs = max(0.0, min(self.rsb, self._rs_p(p)))
            bo, co, muo = self._bo_rs(rs), float("nan"), self._mu_live(rs)
        z = _dak_z(p / self.ppc, TR / self.tpc)
        bg = 0.005035 * z * TR / p                               # rb/scf
        Ma = 28.97 * self.gg                                     # Lee-Gonzalez-Eakin
        K = (9.4 + 0.02 * Ma) * TR ** 1.5 / (209 + 19 * Ma + TR)
        X = 3.5 + 986 / TR + 0.01 * Ma
        rho = 1.4935e-3 * p * Ma / (z * TR)
        mug = 1e-4 * K * math.exp(X * rho ** (2.4 - 0.2 * X))
        dvt = -1.0001e-2 + 1.33391e-4 * T + 5.50654e-7 * T * T   # McCain Bw
        dvp = -1.95301e-9 * p * T - 1.72834e-13 * p * p * T - 3.58922e-7 * p - 2.25341e-10 * p * p
        return dict(p=p, Rs=rs, Bo=bo, co=co, mu_o=muo, z=z, Bg=bg, mu_g=mug,
                    Bw=(1 + dvp) * (1 + dvt))

    def table(self, pressures: Sequence[float]) -> pd.DataFrame:
        """PVT table at the given pressures."""
        return pd.DataFrame([self.at(float(p)) for p in pressures])

    def fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``data`` with Bo, Rs, Bg, Bw and z set from its 'p' column."""
        out = data.copy()
        tab = self.table(out["p"])
        for c in ("Bo", "Rs", "Bg", "Bw", "z"):
            out[c] = tab[c].values
        return out


# =============================================================================
# Van Everdingen-Hurst dimensionless influx
# =============================================================================
def _stehfest_coefficients(n=10):
    h = n // 2
    f = math.factorial
    V = []
    for i in range(1, n + 1):
        s = 0.0
        for k in range((i + 1) // 2, min(i, h) + 1):
            s += k ** h * f(2 * k) / (f(h - k) * f(k) * f(k - 1) * f(i - k) * f(2 * k - i))
        V.append((-1) ** (h + i) * s)
    return np.array(V)


_STEH = _stehfest_coefficients(10)


def _laplace_wd(s, geometry, reD):
    """Laplace-space cumulative influx for a constant terminal pressure."""
    a = np.sqrt(s)
    if geometry == "linear":                       # closed outer boundary, tD = kt/(phi mu ct L^2)
        return np.tanh(a) / s ** 1.5
    if not np.isfinite(reD):
        return kve(1, a) / (s ** 1.5 * kve(0, a))
    R = reD * a
    e = np.exp(2 * a - 2 * R)                      # exponentially scaled Bessel functions
    nu = ive(1, R) * kve(1, a) - kve(1, R) * ive(1, a) * e
    de = ive(0, a) * kve(1, R) * e + kve(0, a) * ive(1, R)
    return nu / (s ** 1.5 * de)


def WD(tD, geometry: str = "radial", reD: float = np.inf):
    """Dimensionless cumulative water influx W_D(t_D).

    geometry 'radial' (reD = ra/ro, inf for an infinite aquifer) or 'linear' (closed end).
    Accepts a scalar or an array.
    """
    scalar = np.isscalar(tD)
    tD = np.atleast_1d(np.asarray(tD, dtype=float))
    out = np.zeros_like(tD)
    reD = np.inf if reD is None or not reD > 1 else float(reD)
    pos = tD > 0
    if geometry == "linear":
        early, late = pos & (tD < 0.05), pos & (tD > 30)
        out[late] = 1.0
    else:
        early = pos & (tD < 1e-3)
        late = pos & (tD > 50 * reD * reD) if np.isfinite(reD) else np.zeros_like(pos)
        out[late] = (reD * reD - 1) / 2 if np.isfinite(reD) else 0.0
    out[early] = 2 * np.sqrt(tD[early] / np.pi) + (0 if geometry == "linear" else tD[early] / 2)
    mid = pos & ~early & ~late
    if mid.any():
        t = tD[mid]
        s = np.outer(np.arange(1, len(_STEH) + 1), math.log(2) / t)
        out[mid] = (_STEH[:, None] * _laplace_wd(s, geometry, reD)).sum(axis=0) * math.log(2) / t
    return float(out[0]) if scalar else out


def veh_influx_function(t, p, td_per_day, geometry="radial", reD=np.inf):
    """Superposition sum  S_i = sum_j dP_j * W_D(tD_i - tD_j)  (psi).  We = B * S."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    n = len(t)
    dP = np.empty(max(n - 1, 0))
    for j in range(n - 1):
        dP[j] = (p[0] - p[1]) / 2 if j == 0 else (p[j - 1] - p[j + 1]) / 2
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = np.dot(dP[:i], WD((t[i] - t[:i]) * td_per_day, geometry, reD))
    return out


def fetkovich_influx(t, p, Wei, J, pi):
    """Fetkovich pseudo-steady-state cumulative influx in bbl. Wei in bbl, J in rb/(day.psi)."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    We, pa, out = 0.0, pi, [0.0]
    for i in range(1, len(t)):
        pr = 0.5 * (p[i - 1] + p[i])
        We += (Wei / pi) * (pa - pr) * (1 - math.exp(-J * pi * (t[i] - t[i - 1]) / Wei))
        pa = pi * (1 - We / Wei)
        out.append(We)
    return np.array(out)


# =============================================================================
# Aquifer description for the single-tank analyses
# =============================================================================
@dataclass
class Aquifer:
    """Aquifer for ``oil_mbe`` / ``gas_mbe``.

    model     'pot', 'steady' (Schilthuis), 'veh' (Van Everdingen-Hurst) or 'fetkovich'
    geometry  'radial' (ro, reD, theta) or 'linear' (L, w)
    tune      veh: also fit the aquifer time constant; fetkovich: fit Wei and J
    The aquifer strength (C or B) is always fitted for pot, steady and veh.
    """
    model: str = "pot"
    geometry: str = "radial"
    k: float = 0.0          # md
    phi: float = 0.0
    mu_w: float = 0.0       # cp
    ct: float = 0.0         # 1/psi
    h: float = 0.0          # ft
    ro: float = 0.0         # ft, reservoir radius
    reD: float = np.inf     # ra/ro
    theta: float = 360.0    # encroachment angle, degrees
    L: float = 0.0          # ft, linear aquifer length
    w: float = 0.0          # ft, linear aquifer width
    tune: bool = False
    C: float = 0.0          # pot: rb/psi, steady: rb/(psi.day). Used by simulate_tank / analytical_match

    def constants(self, pi: float) -> dict:
        """Time constant, influx constant, water volume and productivity from rock properties."""
        k, phi, mu, ct, h = self.k, self.phi, self.mu_w, self.ct, self.h
        if self.geometry == "linear":
            L, w = self.L, self.w
            if not all(v > 0 for v in (k, phi, mu, ct, h, L, w)):
                raise ValueError("Linear aquifer needs k, phi, mu_w, ct, h, L and w.")
            Wi = w * h * L * phi / 5.615
            return dict(td=0.006328 * k / (phi * mu * ct * L * L), B=0.1781 * w * h * L * phi * ct,
                        Wi=Wi, Wei=ct * Wi * pi, J=0.003381 * k * w * h / (mu * L), reD=np.inf)
        ro, f = self.ro, self.theta / 360.0
        if not all(v > 0 for v in (k, phi, mu, ct, h, ro)):
            raise ValueError("Radial aquifer needs k, phi, mu_w, ct, h and ro.")
        finite = np.isfinite(self.reD) and self.reD > 1
        Wi = math.pi * (self.reD ** 2 - 1) * ro * ro * h * phi * f / 5.615 if finite else np.inf
        J = 0.00708 * k * h * f / (mu * (math.log(self.reD) - 0.75)) if finite else np.nan
        return dict(td=0.006328 * k / (phi * mu * ct * ro * ro), B=1.119 * phi * ct * h * ro * ro * f,
                    Wi=Wi, Wei=ct * Wi * pi, J=J, reD=self.reD if finite else np.inf)


def _aquifer_candidates(t, p, aq: Optional[Aquifer]):
    """Return (kind, K, candidates, build, refine). 'lin' columns are fitted, 'fixed' subtracted."""
    pi = p[0]
    logs = lambda a, b, n: 10 ** np.linspace(a, b, n)
    if aq is None or aq.model == "none":
        return "none", None, [None], lambda m: None, None
    if aq.model == "pot":
        return "lin", None, [None], lambda m: pi - p, None
    if not np.any(t > 0):
        raise ValueError("This aquifer model needs survey times in column 't'.")
    if aq.model == "steady":
        dp = pi - p
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (dp[1:] + dp[:-1]) * np.diff(t))])
        return "lin", None, [None], lambda m: cum, None
    K = aq.constants(pi)
    if aq.model == "veh":
        cands = [dict(k=v) for v in logs(-2, 2, 41)] if aq.tune else [dict(k=1.0)]
        refine = lambda m: [dict(k=v) for v in logs(math.log10(m["k"]) - .1, math.log10(m["k"]) + .1, 21)]
        build = lambda m: veh_influx_function(t, p, K["td"] * m["k"], aq.geometry, K["reD"])
        return "lin", K, cands, build, refine
    if aq.model == "fetkovich":
        if not np.isfinite(K["Wi"]):
            raise ValueError("Fetkovich needs a finite aquifer (reD > 1).")
        cands = ([dict(w=w, j=j) for w in logs(-1.3, 1.3, 27) for j in logs(-2, 2, 41)]
                 if aq.tune else [dict(w=1.0, j=1.0)])
        refine = lambda m: [dict(w=w, j=j)
                            for w in logs(math.log10(m["w"]) - .05, math.log10(m["w"]) + .05, 11)
                            for j in logs(math.log10(m["j"]) - .05, math.log10(m["j"]) + .05, 11)]
        build = lambda m: fetkovich_influx(t, p, K["Wei"] * m["w"], K["J"] * m["j"], pi) / 1e6
        return "fixed", K, cands, build, refine
    raise ValueError(f"Unknown aquifer model {aq.model!r}")


def _fit(F, base, t, p, aq):
    """Least squares through the origin on rows 1.., searching aquifer candidates if tuning."""
    kind, K, cands, build, refine = _aquifer_candidates(t, p, aq)
    npar = base.shape[1] + (kind == "lin") + (len(cands) > 1) * (2 if kind == "fixed" else 1)
    if len(F) - 1 < npar + 1:
        raise ValueError(f"Need at least {npar + 1} rows after the initial row for {npar} unknowns.")
    y0, X0 = F[1:], base[1:]

    def solve(m):
        col = build(m)
        X = np.column_stack([X0, col[1:]]) if kind == "lin" else X0
        y = y0 - col[1:] if kind == "fixed" else y0
        s = np.linalg.lstsq(X, y, rcond=None)[0]
        yh = X @ s
        return dict(s=s, sse=float(((y - yh) ** 2).sum()), y=y, yh=yh, mult=m, col=col)

    best = None

    def search(cs):
        nonlocal best
        for m in cs:
            sol = solve(m)
            if not sol["s"][0] > 0:
                continue
            if kind == "lin" and len(cands) > 1 and sol["s"][-1] < 0:
                continue
            if best is None or sol["sse"] < best["sse"]:
                best = sol

    search(cands)
    if best is not None and refine is not None and len(cands) > 1:
        search(refine(best["mult"]))
    if best is None:
        best = solve(cands[0])
    y, yh = best["y"], best["yh"]
    st = float(((y - y.mean()) ** 2).sum())
    best["r2"] = 1 - best["sse"] / st if st > 0 else np.nan

    info, We = {}, np.zeros(len(F))
    if kind == "lin":
        C = best["s"][-1]                                   # MMrb per unit of the influx function
        We = C * best["col"]
        info = dict(model=aq.model)
        if aq.model == "veh":
            info.update(B=C * 1e6, B_from_inputs=K["B"], td_per_day=K["td"] * best["mult"]["k"],
                        time_constant_multiplier=best["mult"]["k"], implied_k=aq.k * best["mult"]["k"])
        else:
            info.update(C=C * 1e6)                          # rb/psi (pot) or rb/(psi.day) (steady)
    elif kind == "fixed":
        We = best["col"]
        info = dict(model=aq.model, Wei=K["Wei"] * best["mult"]["w"], J=K["J"] * best["mult"]["j"],
                    Wei_from_inputs=K["Wei"], J_from_inputs=K["J"])
    return best, info, We


# =============================================================================
# Single-tank material balance
# =============================================================================
@dataclass
class MBResult:
    """Result of oil_mbe / gas_mbe. ``in_place`` is N (MMSTB) or G (Bscf)."""
    in_place: float
    r2: float
    table: pd.DataFrame
    aquifer: dict = field(default_factory=dict)
    m: float = 0.0                 # oil: gas cap ratio
    gas_cap: float = 0.0           # oil: gas cap gas, Bscf
    G_pz: float = np.nan           # gas: GIIP from the p/z line, Bscf
    pz_fit: tuple = ()             # gas: (intercept, slope) of p/z vs Gp
    Gp_abandon: float = np.nan     # gas: recoverable at the abandonment p/z, Bscf

    def __repr__(self):
        bits = [f"in_place={self.in_place:.2f}", f"R2={self.r2:.4f}"]
        if self.m:
            bits.append(f"m={self.m:.3f}")
        if np.isfinite(self.G_pz):
            bits.append(f"G_pz={self.G_pz:.2f}")
        if self.aquifer:
            bits.append("aquifer={" + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in self.aquifer.items()) + "}")
        return "MBResult(" + ", ".join(bits) + ")"


def _cols(data: pd.DataFrame, required, optional):
    d = pd.DataFrame(data).reset_index(drop=True)
    missing = [c for c in required if c not in d]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = {c: d[c].astype(float).to_numpy() for c in required}
    for c, default in optional.items():
        out[c] = d[c].astype(float).fillna(default).to_numpy() if c in d else np.full(len(d), default)
    return out


def oil_mbe(data: pd.DataFrame, Swi: float, cw: float = 3e-6, cf: float = 4e-6,
            m: float = 0.0, fit_m: bool = False, aquifer: Optional[Aquifer] = None) -> MBResult:
    """Havlena-Odeh analysis of an oil reservoir.

    F = N [Eo + m Eg + Efw] + We.   Row 0 of ``data`` is initial conditions.
    Columns: p, Np, Gp, Bo, Rs, Bg  (+ optional t, Wp, Winj, Ginj, Bw).
    Solves for N, optionally m (``fit_m``) and the aquifer.
    """
    d = _cols(data, ["p", "Np", "Gp", "Bo", "Rs", "Bg"],
              dict(t=0.0, Wp=0.0, Winj=0.0, Ginj=0.0, Bw=1.0))
    p, Np, Gp, Bo, Rs, Bg, Bw = (d[c] for c in ("p", "Np", "Gp", "Bo", "Rs", "Bg", "Bw"))
    dp = p[0] - p
    A = Np * Bo + (Gp - Np * Rs) * Bg                       # oil + free gas voidage, MMrb
    F = A + d["Wp"] * Bw - d["Winj"] * Bw - d["Ginj"] * Bg
    Eo = (Bo - Bo[0]) + (Rs[0] - Rs) * Bg
    Eg = Bo[0] * (Bg / Bg[0] - 1)
    Efw0 = Bo[0] * (cw * Swi + cf) / (1 - Swi) * dp
    base = (np.column_stack([Eo + Efw0, Eg + Efw0]) if fit_m
            else (Eo + m * Eg + (1 + m) * Efw0)[:, None])
    best, aq_info, We = _fit(F, base, d["t"], p, aquifer)
    N = best["s"][0]
    if fit_m:
        m = best["s"][1] / N
    Efw = (1 + m) * Efw0
    Et = Eo + m * Eg + Efw
    with np.errstate(divide="ignore", invalid="ignore"):
        tab = pd.DataFrame(dict(
            t=d["t"], p=p, Rp=np.where(Np > 0, Gp / Np, np.nan), F=F, Eo=Eo, Eg=Eg, Efw=Efw, Et=Et,
            We=We, F_over_Et=F / Et,
            DDI=N * Eo / A, SDI=(N * m * Eg + d["Ginj"] * Bg) / A, CDI=N * Efw / A,
            WDI=(We - d["Wp"] * Bw + d["Winj"] * Bw) / A))
    tab.iloc[0, 2:] = np.nan
    return MBResult(in_place=N, r2=best["r2"], table=tab, aquifer=aq_info, m=m,
                    gas_cap=N * m * Bo[0] / Bg[0] / 1000)


def gas_mbe(data: pd.DataFrame, temp_f: float, Swi: float, cw: float = 3e-6, cf: float = 0.0,
            aquifer: Optional[Aquifer] = None, pz_abandon: Optional[float] = None) -> MBResult:
    """p/z and Havlena-Odeh analysis of a dry gas reservoir.

    F = G [Eg + Efw] + We.   Columns: p, z, Gp (Bscf)  (+ optional t, Wp, Bw).
    """
    d = _cols(data, ["p", "z", "Gp"], dict(t=0.0, Wp=0.0, Bw=1.0))
    p, z, Gp = d["p"], d["z"], d["Gp"]
    Bg = 0.005035 * z * (temp_f + 459.67) / p
    dp = p[0] - p
    F = Gp * 1000 * Bg + d["Wp"] * d["Bw"]                   # MMrb
    Eg = Bg - Bg[0]
    Efw = Bg[0] * (cw * Swi + cf) / (1 - Swi) * dp
    Et = Eg + Efw
    slope, intercept = np.polyfit(Gp, p / z, 1)
    best, aq_info, We = _fit(F, Et[:, None], d["t"], p, aquifer)
    Gmm = best["s"][0]                                       # MMscf
    with np.errstate(divide="ignore", invalid="ignore"):
        tab = pd.DataFrame(dict(t=d["t"], p=p, p_over_z=p / z, Bg=Bg, F=F, Eg=Eg, Efw=Efw, Et=Et,
                                We=We, F_over_Et=F / Et / 1000, GDI=Gmm * Eg / F,
                                CDI=Gmm * Efw / F, WDI=(We - d["Wp"] * d["Bw"]) / F))
    tab.iloc[0, 4:] = np.nan
    return MBResult(in_place=Gmm / 1000, r2=best["r2"], table=tab, aquifer=aq_info,
                    G_pz=-intercept / slope, pz_fit=(intercept, slope),
                    Gp_abandon=(pz_abandon - intercept) / slope if pz_abandon else np.nan)


# =============================================================================
# Analytical method: simulate one tank and regress on measured pressure
# =============================================================================
FIT_PARAMETERS = {
    "in_place": "Oil or gas in place",
    "m": "Gas cap ratio m",
    "C": "Aquifer constant C (pot, steady)",
    "reD": "Outer/inner radius ratio",
    "ro": "Reservoir radius",
    "theta": "Encroachment angle",
    "h": "Aquifer thickness",
    "phi": "Aquifer porosity",
    "k": "Aquifer permeability",
    "L": "Linear aquifer length",
    "w": "Linear aquifer width",
}


def _extrap(x, xp, fp):
    """Linear interpolation with linear extrapolation beyond both ends."""
    if x < xp[0]:
        return fp[0] + (fp[1] - fp[0]) * (x - xp[0]) / (xp[1] - xp[0])
    if x > xp[-1]:
        return fp[-1] + (fp[-1] - fp[-2]) * (x - xp[-1]) / (xp[-1] - xp[-2])
    return float(np.interp(x, xp, fp))


def _pvt_functions(d, fluid, temp_f, pvt):
    """Return f(p) -> (Bo, Rs, Bg, Bw) and the lowest pressure it can be trusted at."""
    if pvt is not None:
        def f(p):
            a = pvt.at(p)
            return a["Bo"], a["Rs"], a["Bg"], a["Bw"]
        return f, 50.0
    p = d["p"]
    order = np.argsort(p)
    xp, idx = np.unique(p[order], return_index=True)
    if len(xp) < 2:
        raise ValueError("The table needs at least two different pressures to interpolate PVT.")
    col = lambda c: d[c][order][idx]
    if fluid == "gas":
        if temp_f is None:
            raise ValueError("temp_f is required for a gas tank.")
        z, bw, TR = col("z"), col("Bw"), temp_f + 459.67

        def f(p_):
            return 0.0, 0.0, 0.005035 * max(_extrap(p_, xp, z), 0.05) * TR / p_, _extrap(p_, xp, bw)
    else:
        bo, rs, inv_bg, bw = col("Bo"), col("Rs"), 1.0 / col("Bg"), col("Bw")   # 1/Bg is near-linear in p

        def f(p_):
            return (_extrap(p_, xp, bo), max(_extrap(p_, xp, rs), 0.0),
                    1.0 / max(_extrap(p_, xp, inv_bg), 1e-9), _extrap(p_, xp, bw))
    return f, max(50.0, 0.3 * xp[0])


def simulate_tank(data: pd.DataFrame, in_place: float, fluid: str = "oil", Swi: float = 0.2,
                  cw: float = 3e-6, cf: float = 4e-6, m: float = 0.0, temp_f: Optional[float] = None,
                  aquifer: Optional[Aquifer] = None, pvt: Optional[BlackOilPVT] = None) -> pd.DataFrame:
    """Predict tank pressure at every row of the history from the production alone.

    ``data`` is the same table as for ``oil_mbe`` / ``gas_mbe`` (row 0 = initial conditions). PVT is
    interpolated from the table's own columns unless a ``BlackOilPVT`` is given. The aquifer is
    described by its physical properties (or ``Aquifer.C`` for pot and steady-state models).

    Returns t, p_obs, p_sim and We (MMrb). p_sim is NaN from the first row that cannot be solved.
    """
    from scipy.optimize import brentq
    gas = fluid == "gas"
    d = (_cols(data, ["p", "z", "Gp"], dict(t=0.0, Wp=0.0, Bw=1.0)) if gas else
         _cols(data, ["p", "Np", "Gp", "Bo", "Rs", "Bg"], dict(t=0.0, Wp=0.0, Winj=0.0, Ginj=0.0, Bw=1.0)))
    n, pi = len(d["p"]), float(d["p"][0])
    t = d["t"] if np.all(np.diff(d["t"]) > 0) else np.arange(n, dtype=float)
    props, p_low = _pvt_functions(d, fluid, temp_f, pvt)
    Boi, Rsi, Bgi, _ = props(pi)
    cfw = (cw * Swi + cf) / (1 - Swi)
    model = aquifer.model if aquifer is not None else "none"
    K = aquifer.constants(pi) if model in ("veh", "fetkovich") else None
    if model == "fetkovich" and not np.isfinite(K["Wi"]):
        raise ValueError("Fetkovich needs a finite aquifer (reD > 1).")
    geo = aquifer.geometry if aquifer is not None else "radial"

    P, WE, dPs = [pi], [0.0], []
    S, We_prev, pa = 0.0, 0.0, pi                       # steady-state integral, Fetkovich state
    for k in range(1, n):
        dt, p_prev = t[k] - t[k - 1], P[-1]
        if model == "veh":
            s_known = float(np.dot(dPs, WD((t[k] - t[:k - 1]) * K["td"], geo, K["reD"]))) if k > 1 else 0.0
            w_last, pm = WD(dt * K["td"], geo, K["reD"]), (P[0] if k == 1 else P[k - 2])

        def influx(x):
            if model == "pot":
                return aquifer.C * (pi - x) / 1e6
            if model == "steady":
                return aquifer.C * (S + 0.5 * ((pi - p_prev) + (pi - x)) * dt) / 1e6
            if model == "veh":
                return K["B"] * (s_known + (pm - x) / 2 * w_last) / 1e6
            if model == "fetkovich":
                return We_prev + (K["Wei"] / pi) * (pa - (p_prev + x) / 2) * (
                    1 - math.exp(-K["J"] * pi * dt / K["Wei"])) / 1e6
            return 0.0

        def resid(x):
            Bo, Rs, Bg, Bw = props(x)
            dp = pi - x
            if gas:
                F = d["Gp"][k] * 1000 * Bg + d["Wp"][k] * Bw
                E = in_place * 1000 * ((Bg - Bgi) + Bgi * cfw * dp)
            else:
                F = (d["Np"][k] * Bo + (d["Gp"][k] - d["Np"][k] * Rs) * Bg
                     + (d["Wp"][k] - d["Winj"][k]) * Bw - d["Ginj"][k] * Bg)
                E = in_place * ((Bo - Boi) + (Rsi - Rs) * Bg + m * Boi * (Bg / Bgi - 1)
                                + (1 + m) * Boi * cfw * dp)
            return F - E - influx(x)

        try:
            lo, hi = p_low, 2.0 * pi
            if resid(lo) * resid(hi) > 0:
                break
            x = brentq(resid, lo, hi, xtol=1e-3)
        except (ValueError, ZeroDivisionError, OverflowError):
            break
        we = influx(x)
        if model == "steady":
            S += 0.5 * ((pi - p_prev) + (pi - x)) * dt
        if model == "fetkovich":
            We_prev, pa = we, pi * (1 - we * 1e6 / K["Wei"])
        if model == "veh":
            dPs.append((P[0] - x) / 2 if k == 1 else (P[k - 2] - x) / 2)
        P.append(x)
        WE.append(we)
    pad = [np.nan] * (n - len(P))
    return pd.DataFrame(dict(t=t, p_obs=d["p"], p_sim=P + pad, We=WE + pad))


@dataclass
class MatchResult:
    """Result of analytical_match."""
    start: dict
    fitted: dict
    rms_before: float
    rms_after: float
    table: pd.DataFrame            # t, p_obs, p_start, p_sim, We
    in_place: float
    m: float
    aquifer: Optional[Aquifer]
    warnings: list = field(default_factory=list)
    success: bool = True
    runs: int = 0

    def __repr__(self):
        fit = ", ".join(f"{k}={v:.5g}" for k, v in self.fitted.items())
        return f"MatchResult(rms {self.rms_before:.1f} -> {self.rms_after:.1f} psi, {fit})"


def _lumped(aq: Aquifer, pi: float):
    """The lumped constants the aquifer response actually depends on."""
    if aq.model in ("pot", "steady"):
        return [aq.C]
    K = aq.constants(pi)
    if aq.model == "veh":
        return [K["B"], K["td"]] + ([K["reD"]] if aq.geometry == "radial" and np.isfinite(K["reD"]) else [])
    return [K["Wei"], K["J"]]


def _redundancy(aq, names, pi):
    """Number of independent aquifer parameters among ``names`` (rank of d ln(lumped) / d ln(param))."""
    from dataclasses import replace
    base = np.log(np.abs(_lumped(aq, pi)))
    J = []
    for nme in names:
        up = replace(aq, **{nme: getattr(aq, nme) * 1.01})
        J.append((np.log(np.abs(_lumped(up, pi))) - base) / math.log(1.01))
    return int(np.linalg.matrix_rank(np.array(J), tol=1e-6)) if J else 0


def analytical_match(data: pd.DataFrame, in_place: float, fit: Sequence[str] = ("in_place",),
                     fluid: str = "oil", Swi: float = 0.2, cw: float = 3e-6, cf: float = 4e-6,
                     m: float = 0.0, temp_f: Optional[float] = None, aquifer: Optional[Aquifer] = None,
                     pvt: Optional[BlackOilPVT] = None, max_runs: int = 300) -> MatchResult:
    """Regress the chosen parameters so the simulated tank pressure matches the measured pressure.

    ``fit`` lists what to regress on, from ``FIT_PARAMETERS``: 'in_place', 'm', and the aquifer
    properties 'C', 'reD', 'ro', 'theta', 'h', 'phi', 'k', 'L', 'w'. ``in_place``, ``m`` and the
    values in ``aquifer`` are the starting point; everything not listed stays fixed.

    Several aquifer properties scale the response the same way (thickness and encroachment angle,
    for example), so ticking them together gives a non-unique answer. That is reported in
    ``.warnings`` rather than refused.
    """
    from dataclasses import replace
    from scipy.optimize import least_squares
    fit = list(dict.fromkeys(fit))
    unknown = [f for f in fit if f not in FIT_PARAMETERS]
    if unknown:
        raise ValueError(f"Unknown fit parameters: {unknown}")
    model = aquifer.model if aquifer is not None else "none"
    allowed = {"none": [], "pot": ["C"], "steady": ["C"]}.get(
        model, ["L", "w", "h", "phi", "k"] if aquifer is not None and aquifer.geometry == "linear"
        else ["reD", "ro", "theta", "h", "phi", "k"])
    bad = [f for f in fit if f not in ("in_place", "m") and f not in allowed]
    if bad:
        raise ValueError(f"These parameters do not apply to the chosen aquifer model: {bad}")
    if "m" in fit and fluid == "gas":
        raise ValueError("m applies to oil tanks only.")
    if not fit:
        raise ValueError("Choose at least one parameter to regress on.")
    aq0 = replace(aquifer) if aquifer is not None else None
    if aq0 is not None and "reD" in fit and not np.isfinite(aq0.reD):
        aq0.reD = 10.0                                   # an infinite aquifer cannot be a starting point
    start = {f: (in_place if f == "in_place" else m if f == "m" else float(getattr(aq0, f))) for f in fit}
    for f, v in start.items():
        if f != "m" and not v > 0:
            raise ValueError(f"The starting value for {FIT_PARAMETERS[f]} must be positive.")

    # transforms: log for positive quantities, log(reD - 1) keeps the ratio above 1, m stays linear
    enc = {f: (lambda v: v) if f == "m" else (lambda v: math.log(v - 1)) if f == "reD" else math.log for f in fit}
    dec = {f: (lambda x: x) if f == "m" else (lambda x: 1 + math.exp(x)) if f == "reD" else math.exp for f in fit}
    lo = [0.0 if f == "m" else math.log(0.05) if f == "reD" else -np.inf for f in fit]
    hi = [20.0 if f == "m" else math.log(360.0) if f == "theta" else math.log(0.6) if f == "phi"
          else math.log(500.0) if f == "reD" else np.inf for f in fit]
    x0 = np.clip([enc[f](start[f]) for f in fit], lo, hi)
    p_obs = _cols(data, ["p"], {})["p"]

    def unpack(x):
        vals = {f: dec[f](xi) for f, xi in zip(fit, x)}
        aq = replace(aq0, **{f: v for f, v in vals.items() if f not in ("in_place", "m")}) if aq0 else None
        return vals, vals.get("in_place", in_place), vals.get("m", m), aq

    def run(x):
        _, N_, m_, aq = unpack(x)
        return simulate_tank(data, N_, fluid, Swi, cw, cf, m_, temp_f, aq, pvt)

    def resid(x):
        try:
            sim = run(x)["p_sim"].to_numpy()[1:]
        except (ValueError, ZeroDivisionError, OverflowError):
            return np.full(len(p_obs) - 1, 1e4)
        return np.where(np.isfinite(sim), sim - p_obs[1:], 1e4)

    rms = lambda r: float(np.sqrt(np.mean(r ** 2)))
    r0 = resid(x0)
    if len(r0) < len(fit):
        raise ValueError(f"{len(fit)} parameters need at least {len(fit)} rows after the initial row.")
    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", diff_step=1e-3, max_nfev=max_runs)
    x = sol.x if rms(sol.fun) <= rms(r0) else x0
    fitted, N_, m_, aq = unpack(x)
    first, final = run(x0), run(x)
    table = final.rename(columns={"p_sim": "p_sim"}).assign(p_start=first["p_sim"])[
        ["t", "p_obs", "p_start", "p_sim", "We"]]

    notes = []
    aq_names = [f for f in fit if f not in ("in_place", "m")]
    if aq is not None and len(aq_names) > 1:
        rank = _redundancy(aq, aq_names, float(p_obs[0]))
        if rank < len(aq_names):
            notes.append(f"{len(aq_names)} aquifer parameters were regressed but they only change the aquifer "
                         f"response in {rank} independent way(s), so their individual values are not unique. "
                         "Fix the ones you know.")
    if not np.all(np.isfinite(final["p_sim"])):
        notes.append("The tank could not be solved at every row with the matched values.")
    for f, b_lo, b_hi, xi in zip(fit, lo, hi, x):
        if (np.isfinite(b_hi) and xi >= b_hi - 1e-6) or (np.isfinite(b_lo) and xi <= b_lo + 1e-6):
            notes.append(f"{FIT_PARAMETERS[f]} stopped at its limit ({fitted[f]:.4g}).")
    return MatchResult(start=start, fitted=fitted, rms_before=rms(r0), rms_after=rms(resid(x)), table=table,
                       in_place=N_, m=m_, aquifer=aq, warnings=notes, success=bool(sol.success), runs=int(sol.nfev))


# =============================================================================
# Multi-tank simulator
# =============================================================================
@dataclass
class Tank:
    """One tank. ``production`` has cumulative columns t, Np, Gp, Wp and optional p_obs.

    in_place   MMSTB (oil) or Bscf (gas);  Gp is MMscf for both fluids
    aquifer    'none', 'pot' (C), 'fetkovich' (Wei MMbbl, J), 'veh_radial' / 'veh_linear'
               (C = B in rb/psi, td_per_day, reD)
    """
    name: str
    in_place: float
    pi: float
    fluid: str = "oil"
    m: float = 0.0
    Swi: float = 0.2
    cf: float = 4e-6
    aquifer: str = "none"
    C: float = 0.0
    Wei: float = 0.0
    J: float = 0.0
    td_per_day: float = 0.0
    reD: float = np.inf
    fit_in_place: bool = False
    fit_aquifer: bool = False
    production: Optional[pd.DataFrame] = None


@dataclass
class Connection:
    """Flow between two tanks: q = T (p_a - p_b), reservoir bbl/day. T in rb/(day.psi)."""
    a: str
    b: str
    T: float
    fit: bool = False


@dataclass
class MTResult:
    t: np.ndarray                 # days
    pressure: pd.DataFrame        # one column per tank
    influx: pd.DataFrame          # cumulative We per tank, MMrb
    crossflow: pd.DataFrame       # cumulative a -> b per connection, MMrb
    converged: bool = True
    message: str = ""


class MultiTank:
    """Connected tanks solved implicitly in time.

    For each tank  F_i = N_i Et_i + We_i + sum_j T_ij * integral(p_j - p_i) dt.
    Fluid properties come from a ``BlackOilPVT``.
    """

    def __init__(self, tanks: Sequence[Tank], connections: Sequence[Connection] = (),
                 pvt: Optional[BlackOilPVT] = None, cw: float = 3e-6, max_step: float = 30.0):
        if pvt is None:
            raise ValueError("A BlackOilPVT is required.")
        self.tanks, self.connections = list(tanks), list(connections)
        self.pvt, self.cw, self.max_step = pvt, cw, max_step
        names = [t.name for t in self.tanks]
        if len(set(names)) != len(names):
            raise ValueError("Tank names must be unique.")
        self._idx = {n: i for i, n in enumerate(names)}
        for c in self.connections:
            if c.a not in self._idx or c.b not in self._idx or c.a == c.b:
                raise ValueError(f"Bad connection {c.a} -> {c.b}")
        # PVT lookup table: uniform in pressure, linear interpolation
        self._p0 = 14.7
        pmax = max(t.pi for t in self.tanks) + 500
        self._n = 1200
        self._dp = (pmax - self._p0) / self._n
        self._tab = []
        for i in range(self._n + 1):
            a = pvt.at(self._p0 + i * self._dp)
            self._tab.append((a["Bo"], a["Rs"], a["Bg"], a["Bw"]))
        self._build_schedule()

    # ---- schedule ---------------------------------------------------------------
    def _build_schedule(self):
        times, self._prod, self.observations = {0.0}, [], []
        for i, tk in enumerate(self.tanks):
            df = tk.production
            if df is None or len(df) == 0:
                self._prod.append((np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1)))
                continue
            df = df.sort_values("t")
            t = df["t"].astype(float).to_numpy()
            get = lambda c: (df[c].astype(float).fillna(0).to_numpy() if c in df else np.zeros(len(df)))
            z0 = [0.0]
            self._prod.append((np.concatenate([z0, t]), np.concatenate([z0, get("Np")]),
                               np.concatenate([z0, get("Gp")]), np.concatenate([z0, get("Wp")])))
            times.update(t.tolist())
            if "p_obs" in df:
                for tt, po in zip(t, df["p_obs"].astype(float)):
                    if np.isfinite(po):
                        self.observations.append((i, float(tt), float(po)))
        base, grid = sorted(times), [0.0]
        for k in range(1, len(base)):
            n = math.ceil((base[k] - base[k - 1]) / self.max_step)
            for j in range(1, n + 1):
                grid.append(base[k] if j == n else base[k - 1] + (base[k] - base[k - 1]) * j / n)
        self.grid = np.array(grid)
        self._cum = [tuple(np.interp(self.grid, pr[0], pr[c]) for c in (1, 2, 3)) for pr in self._prod]
        self._gi = {t: k for k, t in enumerate(grid)}

    def _look(self, p):
        x = (p - self._p0) / self._dp
        x = min(max(x, 0.0), self._n - 1e-9)
        i = int(x)
        f = x - i
        a, b = self._tab[i], self._tab[i + 1]
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f)

    # ---- simulation ---------------------------------------------------------------
    def simulate(self) -> MTResult:
        T, n, grid = self.tanks, len(self.tanks), self.grid
        cons = [(self._idx[c.a], self._idx[c.b], c.T) for c in self.connections]
        init = [self._look(t.pi) for t in T]                       # Boi, Rsi, Bgi
        cfw = [(self.cw * t.Swi + t.cf) / (1 - t.Swi) for t in T]
        veh = [t.aquifer in ("veh_radial", "veh_linear") for t in T]
        geo = ["linear" if t.aquifer == "veh_linear" else "radial" for t in T]
        p = [t.pi for t in T]
        We, pa, Q = [0.0] * n, [t.pi for t in T], [0.0] * len(cons)
        dPs = [[] for _ in T]                                      # VEH pressure steps
        P, WE, QQ = [p[:]], [We[:]], [Q[:]]
        ok, msg = True, ""

        for k in range(1, len(grid)):
            t, dt, p_prev = grid[k], grid[k] - grid[k - 1], p[:]
            vb = [None] * n
            for i, tk in enumerate(T):
                if veh[i]:
                    s = (float(np.dot(dPs[i], WD((t - grid[:k - 1]) * tk.td_per_day, geo[i], tk.reD)))
                         if k > 1 else 0.0)
                    vb[i] = (s, WD(dt * tk.td_per_day, geo[i], tk.reD), P[0][i] if k == 1 else P[k - 2][i])

            def influx(i, x):
                tk = T[i]
                if vb[i] is not None:
                    s, w, pm = vb[i]
                    return tk.C * (s + (pm - x) / 2 * w) / 1e6
                if tk.aquifer == "pot":
                    return tk.C * (tk.pi - x) / 1e6
                if tk.aquifer == "fetkovich":
                    Wei = tk.Wei * 1e6
                    return We[i] + (Wei / tk.pi) * (pa[i] - (p_prev[i] + x) / 2) * (
                        1 - math.exp(-tk.J * tk.pi * dt / Wei)) / 1e6
                return 0.0

            def resid(x):
                r = [0.0] * n
                for i, tk in enumerate(T):
                    Bo, Rs, Bg, Bw = self._look(x[i])
                    Boi, Rsi, Bgi, _ = init[i]
                    Np, Gp, Wp = (self._cum[i][c][k] for c in (0, 1, 2))
                    dp = tk.pi - x[i]
                    if tk.fluid == "gas":
                        F = Gp * Bg + Wp * Bw
                        E = tk.in_place * 1000 * ((Bg - Bgi) + Bgi * cfw[i] * dp)
                    else:
                        F = Np * Bo + (Gp - Np * Rs) * Bg + Wp * Bw
                        E = tk.in_place * ((Bo - Boi) + (Rsi - Rs) * Bg + tk.m * Boi * (Bg / Bgi - 1)
                                           + (1 + tk.m) * Boi * cfw[i] * dp)
                    r[i] = F - E - influx(i, x[i])
                for j, (a, b, Tr) in enumerate(cons):
                    q = Q[j] + dt * Tr * (x[a] - x[b]) / 1e6
                    r[a] += q
                    r[b] -= q
                return r

            x, done = p[:], False
            for _ in range(40):                                      # Newton, numerical Jacobian
                r0 = resid(x)
                Jm = np.empty((n, n))
                for j in range(n):
                    hstep = max(0.05, x[j] * 1e-5)
                    xx = x[:]
                    xx[j] += hstep
                    r1 = resid(xx)
                    Jm[:, j] = [(r1[i] - r0[i]) / hstep for i in range(n)]
                try:
                    d = np.linalg.solve(Jm, -np.array(r0))
                except np.linalg.LinAlgError:
                    break
                d = np.clip(d, -400, 400)
                x = [max(15.0, x[i] + d[i]) for i in range(n)]
                if np.abs(d).max() < 0.01:
                    done = True
                    break
            if not done or not all(math.isfinite(v) for v in x):
                ok, msg = False, (f"Pressure solve did not converge at day {t:.0f}. "
                                  "A tank may be too small for its production.")
                break
            for i, tk in enumerate(T):
                We[i] = influx(i, x[i])
                if tk.aquifer == "fetkovich":
                    pa[i] = tk.pi * (1 - We[i] / tk.Wei)
                if veh[i]:
                    dPs[i].append((P[0][i] - x[i]) / 2 if k == 1 else (P[k - 2][i] - x[i]) / 2)
            for j, (a, b, Tr) in enumerate(cons):
                Q[j] += dt * Tr * (x[a] - x[b]) / 1e6
            p = x
            P.append(p[:]); WE.append(We[:]); QQ.append(Q[:])

        names = [t.name for t in T]
        tt = grid[:len(P)]
        return MTResult(t=tt, pressure=pd.DataFrame(P, columns=names, index=pd.Index(tt, name="t")),
                        influx=pd.DataFrame(WE, columns=names, index=pd.Index(tt, name="t")),
                        crossflow=pd.DataFrame(QQ, columns=[f"{c.a} -> {c.b}" for c in self.connections],
                                               index=pd.Index(tt, name="t")),
                        converged=ok, message=msg)

    # ---- matching ---------------------------------------------------------------
    def _sse(self, res: MTResult):
        if not res.converged:
            return 1e12
        P = res.pressure.to_numpy()
        return sum((P[self._gi[t], i] - po) ** 2 for i, t, po in self.observations)

    def rms_error(self, res: Optional[MTResult] = None) -> float:
        """RMS difference between simulated and observed pressures, psi."""
        res = res or self.simulate()
        return math.sqrt(self._sse(res) / max(1, len(self.observations)))

    def _fit_params(self):
        L = []
        for tk in self.tanks:
            if tk.fit_in_place:
                L.append((tk, "in_place"))
            if tk.fit_aquifer:
                if tk.aquifer == "pot" and tk.C > 0:
                    L.append((tk, "C"))
                elif tk.aquifer == "fetkovich":
                    L += [(tk, "Wei"), (tk, "J")]
                elif tk.aquifer in ("veh_radial", "veh_linear"):
                    if tk.C > 0:
                        L.append((tk, "C"))
                    L.append((tk, "td_per_day"))
        L += [(c, "T") for c in self.connections if c.fit and c.T > 0]
        return L

    def history_match(self, verbose: bool = False) -> dict:
        """Adjust every parameter flagged fit_* to minimise squared pressure error.

        Nelder-Mead in log space. Tanks and connections are updated in place.
        Returns the RMS error before and after and the fitted values.
        """
        if not self.observations:
            raise ValueError("No observed pressures: add a 'p_obs' column to the production tables.")
        L = self._fit_params()
        if not L:
            raise ValueError("Nothing to fit: set fit_in_place / fit_aquifer / fit on tanks or connections.")
        x0 = np.log([getattr(o, a) for o, a in L])

        def apply(x):
            for (o, a), v in zip(L, x):
                setattr(o, a, float(math.exp(v)))

        def f(x):
            apply(x)
            return self._sse(self.simulate())

        before = math.sqrt(f(x0) / len(self.observations))
        simplex = np.vstack([x0] + [x0 + 0.4 * np.eye(len(x0))[i] for i in range(len(x0))])
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, xatol=1e-4, fatol=1e-6,
                                    maxfev=80 * len(x0) + 200, disp=verbose))
        apply(res.x)
        after = math.sqrt(self._sse(self.simulate()) / len(self.observations))
        return dict(rms_before=before, rms_after=after, runs=res.nfev,
                    fitted={f"{getattr(o, 'name', None) or o.a + ' -> ' + o.b}.{a}": getattr(o, a)
                            for o, a in L})


# =============================================================================
# Bringing in data by reservoir and by well
# =============================================================================
_ALIASES = {
    "well": ("well", "well_name", "wellname", "well name", "uwi", "string"),
    "reservoir": ("reservoir", "res", "zone", "sand", "tank"),
    "date": ("date", "month", "period", "survey_date", "survey date"),
    "t": ("t", "time", "days", "day"),
    "oil": ("oil", "oil_volume", "oil volume", "oil_prod", "qo"),
    "gas": ("gas", "gas_volume", "gas volume", "gas_prod", "qg"),
    "water": ("water", "water_volume", "water volume", "water_prod", "qw"),
    "winj": ("winj", "water_inj", "water injection", "water_injection"),
    "ginj": ("ginj", "gas_inj", "gas injection", "gas_injection"),
    "p": ("p", "pressure", "static_pressure", "static pressure", "reservoir_pressure",
          "reservoir pressure", "pres"),
}
_VOLS = ["oil", "gas", "water", "winj", "ginj"]
_CUMS = ["Np", "Gp", "Wp", "Winj", "Ginj"]


def standardise_columns(df: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Rename columns to the names this module expects, ignoring case and common variants."""
    lookup = {a: k for k, names in _ALIASES.items() for a in names}
    lookup.update({str(c).lower(): c for c in extra})
    rename, used = {}, set()
    for c in df.columns:
        target = lookup.get(str(c).strip().lower())
        if target is not None and target not in used:
            rename[c] = target
            used.add(target)
    return df.rename(columns=rename)


def _time_key(df, what):
    if "date" in df:
        return "date"
    if "t" in df:
        return "t"
    raise ValueError(f"The {what} table needs a 'date' column or a 't' column (days).")


def _to_dates(x, dayfirst):
    """Parse dates: ISO (yyyy-mm-dd) first, otherwise day-first or month-first as told."""
    x = pd.Series(x)
    if pd.api.types.is_datetime64_any_dtype(x):
        return x
    try:
        return pd.to_datetime(x, format="ISO8601")
    except (ValueError, TypeError):
        return pd.to_datetime(x.astype(str), format="mixed", dayfirst=dayfirst, errors="coerce")


def _parse_time(df, key, dayfirst):
    df = df.copy()
    df[key] = (_to_dates(df[key], dayfirst) if key == "date"
               else pd.to_numeric(df[key], errors="coerce"))
    return df.dropna(subset=[key])


def _days(x, base):
    """Days from ``base`` for a datetime or numeric series."""
    x = pd.Series(x)
    if pd.api.types.is_datetime64_any_dtype(x):
        return ((x - base) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    return (x - base).to_numpy(dtype=float)


def _step(index):
    d = pd.Series(index).diff().dropna()
    if len(d) == 0:
        return pd.Timedelta(days=30) if pd.api.types.is_datetime64_any_dtype(pd.Series(index)) else 30.0
    return d.median()


def aggregate_wells(production: pd.DataFrame, volumes: str = "period", period_dates: str = "end",
                    reservoir=None, wells: Optional[Sequence] = None, dayfirst: bool = False) -> pd.DataFrame:
    """Sum well production into reservoir cumulatives.

    production   columns well, date (or t in days), oil, gas, water; optional reservoir, winj, ginj
    volumes      'period' (volume produced in each period) or 'cumulative' (running total per well)
    period_dates 'end' or 'start': which end of the period each date marks ('period' volumes only)
    reservoir    keep only this reservoir (needs a 'reservoir' column)
    wells        keep only these wells

    Returns date (or t) and cumulative Np, Gp, Wp, Winj, Ginj in the units of the input.
    """
    df = standardise_columns(pd.DataFrame(production))
    if reservoir is not None and "reservoir" in df:
        df = df[df["reservoir"].astype(str) == str(reservoir)]
    if "well" not in df:
        df = df.assign(well="all")
    if wells is not None:
        df = df[df["well"].astype(str).isin([str(w) for w in wells])]
    key = _time_key(df, "production")
    df = _parse_time(df, key, dayfirst)
    if df.empty:
        raise ValueError("No production rows left after filtering by reservoir and wells.")
    for c in _VOLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df else 0.0
    how = "sum" if volumes == "period" else "max"
    wide = df.groupby(["well", key])[_VOLS].agg(how).unstack("well").sort_index()
    wide = wide.fillna(0.0).cumsum() if volumes == "period" else wide.ffill().fillna(0.0)
    cum = wide.T.groupby(level=0).sum().T[_VOLS]
    if volumes == "period":
        step, zero = _step(cum.index), pd.DataFrame([[0.0] * len(_VOLS)], columns=_VOLS)
        if period_dates == "start":      # cumulative at a date excludes that period's volume
            idx = list(cum.index) + [cum.index[-1] + step]
        else:                            # production starts one period before the first date
            idx = [cum.index[0] - step] + list(cum.index)
        cum = pd.concat([zero, cum.reset_index(drop=True)], ignore_index=True)
        cum.index = idx
    out = cum.rename(columns=dict(zip(_VOLS, _CUMS))).rename_axis(key).reset_index()
    return out


def _surveys(pressures, reservoir, group, dayfirst, key):
    ps = standardise_columns(pd.DataFrame(pressures))
    if reservoir is not None and "reservoir" in ps:
        ps = ps[ps["reservoir"].astype(str) == str(reservoir)]
    if key not in ps or "p" not in ps:
        raise ValueError(f"The pressure table needs '{key}' and 'pressure' columns, "
                         "using the same kind of time column as the production table.")
    ps = _parse_time(ps, key, dayfirst)
    ps["p"] = pd.to_numeric(ps["p"], errors="coerce")
    ps = ps.dropna(subset=["p"])
    if ps.empty:
        raise ValueError("No pressure surveys left after filtering.")
    if group and key == "date":
        ps = ps.groupby(ps["date"].dt.to_period(group)).agg(date=("date", "mean"), p=("p", "mean"))
    else:
        ps = ps.groupby(key, as_index=False)["p"].mean()
    return ps.sort_values(key).reset_index(drop=True)


def common_origin(production, pressures, volumes="period", period_dates="end", dayfirst=False):
    """Earliest time across a production and a pressure table: the shared day 0 for several reservoirs."""
    pr = standardise_columns(pd.DataFrame(production))
    key = _time_key(pr, "production")
    pr = _parse_time(pr, key, dayfirst)
    ps = _parse_time(standardise_columns(pd.DataFrame(pressures)), key, dayfirst)
    first = pr[key].min()
    if volumes == "period" and period_dates == "end":
        first = first - _step(np.sort(pr[key].unique()))
    return min(first, ps[key].min())


def build_history(production: pd.DataFrame, pressures: pd.DataFrame, volumes: str = "period",
                  period_dates: str = "end", reservoir=None, wells: Optional[Sequence] = None,
                  initial_pressure: Optional[float] = None, group: Optional[str] = None,
                  dayfirst: bool = False, origin=None) -> pd.DataFrame:
    """Reservoir history for material balance from per-well data.

    Sums the wells (see ``aggregate_wells``), averages the pressure surveys taken on the same date
    (or in the same month 'M', quarter 'Q' or year 'Y' if ``group`` is given) and interpolates the
    cumulative volumes to each survey date.

    pressures         columns date (or t), pressure; optional well, reservoir
    initial_pressure  adds a first row at the start of production with zero volumes. Without it the
                      first survey is taken as initial conditions.
    origin            fixed day 0 (date or number). By default day 0 is the first row.

    Returns t (days), p, Np, Gp, Wp, Winj, Ginj (+ date). Volumes keep the input units; add the PVT
    columns before passing the table to ``oil_mbe`` or ``gas_mbe``. Any warnings are in ``.attrs['notes']``.
    """
    cum = aggregate_wells(production, volumes, period_dates, reservoir, wells, dayfirst)
    key = "date" if "date" in cum else "t"
    ps = _surveys(pressures, reservoir, group, dayfirst, key)
    base = origin if origin is not None else min(cum[key].iloc[0], ps[key].iloc[0])
    if key == "date":
        base = pd.Timestamp(base)
    tc, tp = _days(cum[key], base), _days(ps[key], base)
    out = pd.DataFrame({"t": tp, "p": ps["p"].to_numpy(dtype=float)})
    for c in _CUMS:
        out[c] = np.interp(tp, tc, cum[c].to_numpy(dtype=float))
    notes = []
    if volumes == "period" and len(out) and len(tc) > 1 and out["t"].iloc[0] <= tc[0] + 0.1 * (tc[1] - tc[0]):
        out.loc[out.index[0], _CUMS] = 0.0          # a survey at the very start of production is initial
    if initial_pressure:
        out = out[out["t"] > tc[0]]
        first = {"t": tc[0], "p": float(initial_pressure), **{c: 0.0 for c in _CUMS}}
        out = pd.concat([pd.DataFrame([first]), out], ignore_index=True)
        if volumes == "cumulative" and cum[_CUMS].iloc[0].sum() > 0:
            notes.append("The cumulative data does not start at zero, so the date of initial conditions "
                         "is unknown; the initial row was placed at the first production record.")
    elif out[["Np", "Gp", "Wp"]].iloc[0].sum() > 0:
        notes.append("The first pressure survey was taken after production started, so the first row is "
                     "not initial conditions. Give the initial reservoir pressure.")
    if origin is None:
        shift = out["t"].iloc[0]
        out["t"] -= shift
        if key == "date":
            base = base + pd.Timedelta(days=float(shift))
    if key == "date":
        out["date"] = base + pd.to_timedelta(out["t"], unit="D")
    out = out.reset_index(drop=True)
    out.attrs["notes"] = notes
    return out


def reservoir_table(df: pd.DataFrame, columns: Sequence[str], reservoir=None, dayfirst: bool = False,
                    origin=None) -> pd.DataFrame:
    """One reservoir's history from a file that may hold several.

    Filters on the 'reservoir' column, turns a 'date' column into t (days from the first row or from
    ``origin``), sorts by time and returns ``columns`` (missing ones are left empty).
    """
    d = standardise_columns(pd.DataFrame(df), extra=columns)
    if reservoir is not None and "reservoir" in d:
        d = d[d["reservoir"].astype(str) == str(reservoir)]
    d = d.copy()
    if "t" not in d and "date" in d:
        dates = _to_dates(d["date"], dayfirst)
        d["t"] = _days(dates, pd.Timestamp(origin) if origin is not None else dates.min())
    for c in columns:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    if d["t"].notna().any():
        d = d.sort_values("t")
    return d[list(columns)].reset_index(drop=True)
