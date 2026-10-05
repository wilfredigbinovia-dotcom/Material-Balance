"""PVT for retrograde gas condensate: lab table or correlations."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

BG_CONST = 0.00503676  # rb/scf = BG_CONST * z * T(degR) / p(psia)   (14.696 psia, 60 degF)


# ----------------------------------------------------------------- correlations
def condensate_props(api):
    """Specific gravity and molecular weight (Cragoe) of stock-tank condensate."""
    sg_o = 141.5 / (api + 131.5)
    mw_o = 5954.0 / max(api - 8.811, 1.0)
    return sg_o, mw_o


def gas_equivalent(api):
    """scf of gas equivalent to 1 STB of condensate."""
    sg_o, mw_o = condensate_props(api)
    return 133316.0 * sg_o / mw_o


def wet_gas_gravity(sg_sep, cgr, api):
    """Reservoir (well-stream) gas gravity from separator gas gravity and CGR [STB/MMscf]."""
    if cgr <= 0:
        return sg_sep
    sg_o, mw_o = condensate_props(api)
    R = 1.0e6 / cgr  # scf/STB
    return (sg_sep + 4584.0 * sg_o / R) / (1.0 + 132800.0 * sg_o / (R * mw_o))


def pseudo_criticals(sg, co2=0.0, h2s=0.0, n2=0.0):
    """Sutton (1985) pseudo-criticals with Wichert-Aziz sour gas correction. Returns (Tpc degR, Ppc psia)."""
    inert = co2 + h2s + n2
    sg_hc = (sg - 1.5195 * co2 - 1.1765 * h2s - 0.9672 * n2) / max(1.0 - inert, 1e-6)
    sg_hc = max(sg_hc, 0.55)
    tpc_hc = 169.2 + 349.5 * sg_hc - 74.0 * sg_hc ** 2
    ppc_hc = 756.8 - 131.0 * sg_hc - 3.6 * sg_hc ** 2
    tpc = (1 - inert) * tpc_hc + 547.6 * co2 + 672.4 * h2s + 227.2 * n2
    ppc = (1 - inert) * ppc_hc + 1071.0 * co2 + 1306.0 * h2s + 493.1 * n2
    a = co2 + h2s
    if a > 0:
        eps = 120.0 * (a ** 0.9 - a ** 1.6) + 15.0 * (h2s ** 0.5 - h2s ** 4)
        tpc_c = tpc - eps
        ppc = ppc * tpc_c / (tpc + h2s * (1 - h2s) * eps)
        tpc = tpc_c
    return tpc, ppc


def z_dak(ppr, tpr):
    """Dranchuk & Abou-Kassem (1975) single-phase z-factor."""
    A = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)
    c1 = A[0] + A[1] / tpr + A[2] / tpr ** 3 + A[3] / tpr ** 4 + A[4] / tpr ** 5
    c2 = A[5] + A[6] / tpr + A[7] / tpr ** 2
    c3 = A[8] * (A[6] / tpr + A[7] / tpr ** 2)

    def f(z):
        r = 0.27 * ppr / (z * tpr)
        zc = (1 + c1 * r + c2 * r ** 2 - c3 * r ** 5
              + A[9] * (1 + A[10] * r ** 2) * (r ** 2 / tpr ** 3) * np.exp(-A[10] * r ** 2))
        return z - zc

    if ppr <= 1e-6:
        return 1.0
    try:
        return brentq(f, 0.15, 4.0, xtol=1e-10)
    except ValueError:
        return float("nan")


def z_rayes(ppr, tpr):
    """Rayes, Piper, McCain & Poston (1992) two-phase z-factor for rich gas condensates."""
    return (2.24353 - 0.0375281 * ppr - 3.56539 / tpr + 0.000829231 * ppr ** 2
            + 1.53428 / tpr ** 2 + 0.131987 * ppr / tpr)


def gas_viscosity(p, T, z, sg):
    """Lee, Gonzalez & Eakin gas viscosity, cp."""
    Tr = T + 459.67
    M = 28.97 * sg
    rho = 1.4935e-3 * p * M / (z * Tr)
    K = (9.4 + 0.02 * M) * Tr ** 1.5 / (209.0 + 19.0 * M + Tr)
    X = 3.5 + 986.0 / Tr + 0.01 * M
    Y = 2.4 - 0.2 * X
    return 1e-4 * K * np.exp(X * rho ** Y)


def water_props(p, T, salinity_ppm=0.0):
    """Bw (McCain), cw (Osif), mu_w (Beggs-Brill). Returns (rb/STB, 1/psi, cp)."""
    dvt = -1.0001e-2 + 1.33391e-4 * T + 5.50654e-7 * T ** 2
    dvp = -1.95301e-9 * p * T - 1.72834e-13 * p ** 2 * T - 3.58922e-7 * p - 2.25341e-10 * p ** 2
    bw = (1 + dvp) * (1 + dvt)
    s_gl = salinity_ppm / 1000.0
    cw = 1.0 / max(7.033 * p + 541.5 * s_gl - 537.0 * T + 403300.0, 1e4)
    mu = float(np.exp(1.003 - 1.479e-2 * T + 1.982e-5 * T ** 2))
    return bw, cw, mu


def hall_cf(phi):
    """Hall (1953) formation compressibility, 1/psi."""
    return 1.782e-6 / max(phi, 0.01) ** 0.438


def correlation_table(sg_sep, cgr_i, api, T, p_dew, p_max, co2=0.0, h2s=0.0, n2=0.0,
                      two_phase="rayes", n=30):
    """Generate p, z (two-phase below the dew point), producing CGR table.

    two_phase: 'rayes' uses the Rayes et al. two-phase z below the dew point (rich gas,
    C7+ >= 4 mol%), scaled to be continuous with the single-phase z at the dew point.
    'single' keeps the single-phase z (lean gas approximation).
    """
    sg_w = wet_gas_gravity(sg_sep, cgr_i, api)
    tpc, ppc = pseudo_criticals(sg_w, co2, h2s, n2)
    tpr = (T + 459.67) / tpc
    p = np.unique(np.round(np.concatenate([np.linspace(200.0, p_max, n), [p_dew]]), 1))
    z = np.empty_like(p)
    z_dew = z_dak(p_dew / ppc, tpr)
    scale = z_dew / z_rayes(p_dew / ppc, tpr)
    for i, pp in enumerate(p):
        if pp >= p_dew or two_phase != "rayes":
            z[i] = z_dak(pp / ppc, tpr)
        else:
            z[i] = scale * z_rayes(pp / ppc, tpr)
    # Indicative producing CGR below dew point (liquid drops out in the reservoir).
    ratio = np.clip(p / p_dew, 0.0, 1.0)
    cgr = cgr_i * (0.30 + 0.70 * ratio ** 2.0)
    return pd.DataFrame({"p": p, "z": z, "cgr": cgr})


# ----------------------------------------------------------------- PVT object
@dataclass
class PVT:
    p_tab: np.ndarray
    z_tab: np.ndarray
    cgr_tab: np.ndarray
    T: float        # degF
    api: float
    sg_wet: float = 0.8

    def __post_init__(self):
        o = np.argsort(self.p_tab)
        self.p_tab = np.asarray(self.p_tab, float)[o]
        self.z_tab = np.asarray(self.z_tab, float)[o]
        self.cgr_tab = np.asarray(self.cgr_tab, float)[o]
        self._z = (PchipInterpolator(self.p_tab, self.z_tab, extrapolate=False)
                   if len(self.p_tab) >= 3 else None)
        self.ge = gas_equivalent(self.api)

    @classmethod
    def from_frame(cls, df, T, api, sg_wet=0.8):
        d = df.dropna(subset=["p", "z"]).drop_duplicates("p").sort_values("p")
        if len(d) < 2:
            raise ValueError("The PVT table needs at least two rows with pressure and Z-factor.")
        cgr = d["cgr"].to_numpy(float) if "cgr" in d else np.zeros(len(d))
        cgr = np.where(np.isnan(cgr), 0.0, cgr)
        return cls(d["p"].to_numpy(float), d["z"].to_numpy(float), cgr, T, api, sg_wet)

    @property
    def p_range(self):
        return float(self.p_tab[0]), float(self.p_tab[-1])

    def z(self, p):
        pc = np.clip(np.asarray(p, float), self.p_tab[0], self.p_tab[-1])
        if self._z is None:
            return np.interp(pc, self.p_tab, self.z_tab)
        return self._z(pc)

    def bg(self, p):
        """Two-phase gas FVF, rb/scf."""
        p = np.asarray(p, float)
        return BG_CONST * self.z(p) * (self.T + 459.67) / p

    def cgr(self, p):
        return np.interp(np.asarray(p, float), self.p_tab, self.cgr_tab)

    def visc(self, p):
        return gas_viscosity(np.asarray(p, float), self.T, self.z(p), self.sg_wet)
