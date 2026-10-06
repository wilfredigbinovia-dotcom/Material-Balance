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


def cvd_two_phase_z(df, p_dew, z_dew=None):
    """Two-phase Z from a constant-volume-depletion report that does not list it.

    df columns: p [psia], gp = cumulative well-stream produced [mol % of the fluid at the dew
    point], and optionally zg = equilibrium (single-phase) gas Z and sl = retrograde liquid
    [% of the cell volume at the dew point].

    The cell volume is constant, so a mole balance on what is left in it gives
        Z2 = p / [ (pd / Zd) * (1 - Gp) ]
    At and above the dew point the fluid is single phase and Z2 = Zg.

    The liquid volume and the gas Z are not needed for Z2 itself; they are used to check the
    report: the moles left as liquid,
        nL = (1 - Gp) - (p / Zg) * (1 - SL) / (pd / Zd),
    must be positive (returned as liq_mol, mol % of the initial fluid).

    z_dew: gas Z at the dew point. If None it is read from the table (row at the dew point,
    otherwise interpolated between the rows either side of it).
    Returns (DataFrame[p, z, zg, sl, gp, liq_mol], z_dew, list of warnings).
    """
    d = df.copy()
    for col in ("p", "gp", "zg", "sl"):
        d[col] = pd.to_numeric(d[col], errors="coerce") if col in d else np.nan
    d = d.dropna(subset=["p"]).drop_duplicates("p").sort_values("p", ascending=False)
    d = d[d["p"] > 0].reset_index(drop=True)
    if d.empty:
        raise ValueError("Enter the CVD table first.")
    tol = 0.003 * p_dew
    above = d["p"].to_numpy() >= p_dew - tol
    notes = []

    if z_dew is None or not z_dew > 0:
        zrow = d[(d["p"] - p_dew).abs() <= tol]["zg"].dropna()
        known = d.dropna(subset=["zg"]).sort_values("p")
        if len(zrow):
            z_dew = float(zrow.iloc[0])
        elif len(known) >= 2 and known["p"].iloc[0] < p_dew < known["p"].iloc[-1]:
            z_dew = float(np.interp(p_dew, known["p"], known["zg"]))
            notes.append("No row at the dew point: Z at the dew point was interpolated from the "
                         "gas Z column. Enter it directly if the report gives it.")
        else:
            raise ValueError("The gas Z-factor at the dew point is needed: add a row at the dew "
                             "point pressure or enter the value.")

    gp = d["gp"].to_numpy(float) / 100.0
    gp = np.where(above & np.isnan(gp), 0.0, gp)
    if np.nanmax(np.where(above, gp, 0.0)) > 1e-6:
        notes.append("Fluid is reported as produced at or above the dew point. Check the dew "
                     "point pressure entered under Fluid description.")
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = d["p"].to_numpy() / ((p_dew / z_dew) * (1.0 - gp))
    zg = d["zg"].to_numpy(float)
    z = np.where(above, np.where(np.isnan(zg), np.nan, zg), z2)
    z = np.where(np.abs(d["p"].to_numpy() - p_dew) <= tol, z_dew, z)
    sl = np.where(above & d["sl"].isna().to_numpy(), 0.0, d["sl"].to_numpy(float)) / 100.0
    liq = (1.0 - gp) - (d["p"].to_numpy() / zg) * (1.0 - sl) / (p_dew / z_dew)
    liq = np.where(above, 0.0, liq)

    below = ~above
    if np.any(below & np.isnan(d["gp"].to_numpy(float))):
        notes.append("Rows below the dew point without a cumulative produced value cannot be "
                     "converted and are left out.")
    g = gp[below & ~np.isnan(gp)]
    if len(g) and (np.any(np.diff(g) <= 0) or g.min() <= 0 or g.max() >= 1):
        notes.append("Cumulative produced should rise steadily from 0 towards 100 mol % as "
                     "pressure falls. Check the column (it must be in percent, not a fraction).")
    if np.any(liq[below & ~np.isnan(liq)] < -0.005):
        notes.append("The gas Z and retrograde liquid columns leave less than zero moles in the "
                     "liquid phase at some pressures: the report columns are not consistent "
                     "(check units of the liquid volume and which Z column was copied).")
    bad = below & ~np.isnan(z) & ((z < 0.3) | (z > 2.5))
    if np.any(bad):
        notes.append("Some calculated two-phase Z values are outside 0.3 to 2.5.")
    out = pd.DataFrame({"p": d["p"], "z": z, "zg": zg, "sl": sl * 100.0, "gp": gp * 100.0,
                        "liq_mol": liq * 100.0})
    return out, z_dew, notes


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


# ----------------------------------------------------------------- flowing gas / water vapour
class GasZ:
    """Single-phase Z of the flowing well-stream gas at any pressure and temperature
    (DAK), tabulated per temperature so repeated tubing calculations stay fast."""

    def __init__(self, sg, co2=0.0, h2s=0.0, n2=0.0):
        self.sg = sg
        self.tpc, self.ppc = pseudo_criticals(sg, co2, h2s, n2)
        self._tab = {}

    def __call__(self, p, T):
        key = round(float(T))
        if key not in self._tab:
            pp = np.concatenate([[14.7], np.linspace(100.0, 15000.0, 75)])
            zz = np.array([z_dak(x / self.ppc, (key + 459.67) / self.tpc) for x in pp])
            ok = np.isfinite(zz)
            self._tab[key] = (pp[ok], zz[ok])
        pp, zz = self._tab[key]
        return np.interp(p, pp, zz)

    def visc(self, p, T):
        return gas_viscosity(p, T, self(p, T), self.sg)


def water_vapour(p, T):
    """Water carried as vapour by the reservoir gas, STB/MMscf (Bukacek, 1955).
    It condenses at surface, so every gas well makes this much water with no aquifer at all."""
    tc = (np.asarray(T, float) - 32.0) / 1.8
    a, b, c = np.where(tc < 100.0, (8.07131, 1730.63, 233.426), (8.14019, 1810.94, 244.485)).T \
        if np.ndim(tc) else ((8.07131, 1730.63, 233.426) if tc < 100.0 else (8.14019, 1810.94, 244.485))
    psat = 10.0 ** (a - b / (c + tc)) * 14.696 / 760.0          # psia
    B = 10.0 ** (-3083.87 / (np.asarray(T, float) + 459.6) + 6.69449)
    return (47484.0 * psat / np.asarray(p, float) + B) / 350.0   # lb/MMscf -> STB/MMscf
