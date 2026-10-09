"""Tank-scale (pseudo) relative permeability: Corey curves, tank saturations, water-gas ratio.

Gas-water (aquifer water moving into the gas zone)
    Swn = (Sw - Swc) / (1 - Swc - Sgrw)
    krw = krw_max * Swn^nw          krg = krg_max * (1 - Swn)^ng
Gas-condensate (retrograde liquid dropping out below the dew point)
    krg multiplier = ((1 - Swc - So) / (1 - Swc))^ngo
    kro = ((So - Soc) / (1 - Swc - Soc))^no        (condensate flows only above Soc)

Tank saturations come from the material balance:
    PV  = PVi * (1 - cf (pi - p))
    Sw  = [PVi Swi (1 + cw (pi - p)) + We - Wp Bw] / PV
    So  = retrograde liquid from the CVD report (fraction of hydrocarbon pore volume) * (1 - Swi)
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .pvt import water_vapour


@dataclass
class RelPerm:
    swc: float
    sgrw: float = 0.25
    krw_max: float = 0.3
    krg_max: float = 1.0
    nw: float = 3.0
    ng: float = 2.0
    soc: float = 0.2
    no: float = 3.0
    ngo: float = 2.0

    def swn(self, sw):
        return np.clip((np.asarray(sw, float) - self.swc) / max(1.0 - self.swc - self.sgrw, 1e-6), 0.0, 1.0)

    def gas_water(self, sw):
        """(krw, krg) at water saturation sw."""
        s = self.swn(sw)
        return self.krw_max * s ** self.nw, self.krg_max * (1.0 - s) ** self.ng

    def cond_mult(self, so):
        """Reduction of gas permeability by retrograde condensate, 0..1."""
        so = np.clip(np.asarray(so, float), 0.0, 1.0 - self.swc)
        return ((1.0 - self.swc - so) / (1.0 - self.swc)) ** self.ngo

    def kro(self, so):
        s = np.clip((np.asarray(so, float) - self.soc) / max(1.0 - self.swc - self.soc, 1e-6), 0.0, 1.0)
        return s ** self.no

    def krg_rel(self, sw, so=0.0):
        """Gas relative permeability as a fraction of its end point."""
        return self.gas_water(sw)[1] / self.krg_max * self.cond_mult(so)


class TankState:
    """Saturations, mobility and water-gas ratio of a matched tank at given conditions."""

    def __init__(self, tank, muw, sl_table=None, p_dew=None):
        self.tank, self.muw = tank, muw
        self.pvi = tank.G * tank.bgi / (1.0 - tank.swi)       # rb
        self.sl, self.p_dew = None, p_dew
        if sl_table is not None and p_dew:
            d = sl_table.dropna(subset=["p", "sl"]).sort_values("p")
            d = d[d["p"] < p_dew]
            if len(d) >= 2:
                self.sl = (np.append(d["p"].to_numpy(float), p_dew),
                           np.append(d["sl"].to_numpy(float), 0.0) / 100.0)
        self._mz_i = float(tank.pvt.visc(tank.pi) * tank.zi)

    @property
    def has_condensate(self):
        return self.sl is not None

    def sw(self, p, we, wp):
        t = self.tank
        dp = t.pi - np.asarray(p, float)
        pv = self.pvi * (1.0 - t.cf * dp)
        return (self.pvi * t.swi * (1.0 + t.cw * dp) + we - wp * t.bw) / pv

    def so(self, p):
        if self.sl is None:
            return np.zeros_like(np.asarray(p, float))
        return np.interp(p, *self.sl, left=self.sl[1][0], right=0.0) * (1.0 - self.tank.swi)

    def mobility(self, rp, p, sw):
        """Gas mobility relative to initial conditions: krg/(mu z), scaled to 1 at pi, Swi, So=0."""
        pv = self.tank.pvt
        return rp.krg_rel(sw, self.so(p)) * self._mz_i / (pv.visc(p) * pv.z(p))

    def wgr_free(self, rp, p, sw):
        """Free (mobile) water per unit of well-stream gas, STB/MMscf."""
        pv = self.tank.pvt
        krw, krg = rp.gas_water(sw)
        krg = krg * rp.cond_mult(self.so(p))
        lam_w = krw / (self.muw * self.tank.bw)                # STB basis
        lam_g = np.maximum(krg, 1e-12) / (pv.visc(p) * pv.bg(p))   # scf basis
        return np.minimum(lam_w / lam_g * 1e6, 1e7)

    def wgr_vapour(self, p):
        return water_vapour(p, self.tank.pvt.T)


def water_history(state, rp, p, we, vapour=True):
    """Model cumulative water (STB) at the history points, for comparison with measured Wp."""
    h = state.tank.h
    gpw = state.tank.gpw
    sw = state.sw(p, we, h.wp)
    wgr = state.wgr_free(rp, p, sw)
    vap = state.wgr_vapour(p) if vapour else np.zeros_like(p)
    wp_free, wp_vap = np.zeros(len(p)), np.zeros(len(p))
    for i in range(1, len(p)):
        wp_free[i] = wp_free[i - 1] + 0.5 * (wgr[i] + wgr[i - 1]) * (gpw[i] - gpw[i - 1]) / 1e6
        wp_vap[i] = wp_vap[i - 1] + 0.5 * (vap[i] + vap[i - 1]) * (h.gp[i] - h.gp[i - 1]) / 1e6
    return dict(sw=sw, swn=rp.swn(sw), wgr=wgr, vap=vap, wp_free=wp_free, wp_vap=wp_vap,
                wp=wp_free + wp_vap)


def fit_water(state, rp, p, we, vapour=True):
    """Fit krw end point and water exponent to the measured cumulative water.

    Returns (RelPerm, info). info['ok'] is False, with a reason, when the history cannot
    constrain the curves (no free water produced, or the tank water saturation never moved).
    """
    h = state.tank.h
    base = water_history(state, rp, p, we, vapour)
    free_obs = h.wp - base["wp_vap"]
    if base["swn"].max() < 0.03:
        return rp, dict(ok=False, reason="The tank water saturation hardly changes over the history "
                        "(little or no aquifer influx), so the history says nothing about the "
                        "water curve.")
    if free_obs[-1] <= 0.05 * max(h.wp[-1], 1.0):
        return rp, dict(ok=False, reason="All of the produced water is explained by water vapour "
                        "condensing from the gas. No free water has reached the wells yet, so the "
                        "history cannot fix the water curve; it only says water is not yet mobile.")
    scale = max(free_obs[-1], 1.0)

    def make(x):
        return RelPerm(**{**rp.__dict__, "krw_max": float(np.exp(x[0])), "nw": float(x[1])})

    def resid(x):
        return (water_history(state, make(x), p, we, vapour)["wp_free"] - free_obs)[1:] / scale

    best = None
    for nw0 in (1.5, 3.0, 5.0):
        r = least_squares(resid, [np.log(min(max(rp.krw_max, 1e-3), 1.0)), nw0],
                          bounds=([np.log(1e-4), 1.0], [0.0, 8.0]))
        if best is None or r.cost < best.cost:
            best = r
    new = make(best.x)
    rms = float(np.sqrt(np.mean((best.fun * scale) ** 2)))
    return new, dict(ok=True, rms=rms, at_limit=bool(new.nw > 7.9 or new.nw < 1.05 or new.krw_max > 0.99))


# ----------------------------------------------------------------- fractional flow matching
FIT_PARAMS = {   # key -> (RelPerm attribute, lower, upper, log-scale)
    "rp_sgrw": ("sgrw", 0.02, 0.6, False),
    "rp_krw": ("krw_max", 1e-4, 1.0, True),
    "rp_nw": ("nw", 1.0, 8.0, False),
    "rp_ng": ("ng", 1.0, 8.0, False),
    "rp_krg": ("krg_max", 0.05, 1.0, True),
}


def fw_model(state, rp, p, sw):
    """Reservoir water cut  fw = (krw/mu_w) / (krw/mu_w + krg/mu_g)."""
    p, sw = np.asarray(p, float), np.asarray(sw, float)
    krw, krg = rp.gas_water(sw)
    krg = krg * rp.cond_mult(state.so(p))
    lw = krw / state.muw
    lg = krg / state.tank.pvt.visc(p)
    return lw / np.maximum(lw + lg, 1e-30)


def fw_measured(state, p, wgr_free_wet):
    """Free water-gas ratio (STB per MMscf of well-stream gas) -> reservoir water cut."""
    qw = np.asarray(wgr_free_wet, float) * state.tank.bw              # rb per MMscf
    qg = 1e6 * state.tank.pvt.bg(np.asarray(p, float))                 # rb per MMscf
    return qw / (qw + qg)


def field_points(state, p, we, vapour=True):
    """One point per interval between pressure surveys (tank level)."""
    t = state.tank
    h = t.h
    pm = 0.5 * (p[1:] + p[:-1])
    swm = 0.5 * (state.sw(p[1:], we[1:], h.wp[1:]) + state.sw(p[:-1], we[:-1], h.wp[:-1]))
    dgw = np.diff(t.gpw)
    dgs = np.diff(h.gp)
    dw = np.diff(h.wp)
    dn = np.diff(h.np_)
    ok = dgw > 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        wgr = dw / dgw * 1e6                                           # STB/MMscf wet gas
        vap = (state.wgr_vapour(pm) if vapour else np.zeros_like(pm)) * dgs / dgw
        cgr = dn / dgs * 1e6                                          # STB/MMscf separator gas
    free = np.maximum(wgr - vap, 0.0)
    d = pd.DataFrame({"t": 0.5 * (h.t[1:] + h.t[:-1]), "p": pm, "sw": swm, "wgr": wgr, "vap": vap,
                      "free": free, "gas": dgw, "cgr": cgr, "well": "Field"})[ok]
    d["fw"] = fw_measured(state, d["p"], d["free"])
    return d.reset_index(drop=True)


def well_points(state, p, we, wells, start_date, vapour=True):
    """One point per well per production record (monthly in most files). The tank pressure and
    water saturation are interpolated in time from the history match. wells: well, date, gp [MMscf],
    np [MSTB], wp [MSTB] cumulative."""
    t = state.tank
    h = t.h
    sw_h = state.sw(p, we, h.wp)
    out = []
    w = wells.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date", "well"])
    for name, g in w.groupby("well"):
        g = g.sort_values("date")
        day = (g["date"] - pd.Timestamp(start_date)).dt.days.to_numpy(float)
        gp = pd.to_numeric(g["gp"], errors="coerce").fillna(0.0).to_numpy() * 1e6
        npc = pd.to_numeric(g["np"], errors="coerce").fillna(0.0).to_numpy() * 1e3
        wp = pd.to_numeric(g["wp"], errors="coerce").fillna(0.0).to_numpy() * 1e3
        day = np.concatenate([[day[0] - 30.4], day])
        gp, npc, wp = (np.concatenate([[0.0], a]) for a in (gp, npc, wp))
        dgs, dn, dw = np.diff(gp), np.diff(npc), np.diff(wp)
        dgw = dgs + t.pvt.ge * dn
        tm = 0.5 * (day[1:] + day[:-1])
        ok = (dgw > 1e5) & (tm >= h.t[0]) & (tm <= h.t[-1]) & (dw >= 0)
        if not ok.any():
            continue
        pm = np.interp(tm, h.t, p)
        with np.errstate(divide="ignore", invalid="ignore"):
            wgr = dw / dgw * 1e6
            vap = (state.wgr_vapour(pm) if vapour else np.zeros_like(pm)) * dgs / dgw
            cgr = dn / dgs * 1e6
        d = pd.DataFrame({"t": tm, "p": pm, "sw": np.interp(tm, h.t, sw_h), "wgr": wgr, "vap": vap,
                          "free": np.maximum(wgr - vap, 0.0), "gas": dgw, "cgr": cgr,
                          "well": str(name)})[ok]
        out.append(d)
    if not out:
        return pd.DataFrame(columns=["t", "p", "sw", "wgr", "vap", "free", "gas", "cgr", "well", "fw"])
    d = pd.concat(out, ignore_index=True)
    d["fw"] = fw_measured(state, d["p"], d["free"])
    return d


def fit_fw(state, rp, pts, keys):
    """Regress the ticked Corey parameters so that the model water cut matches the measured
    points. Residuals are in water cut, weighted by the gas each point represents.
    Returns (RelPerm, info)."""
    if not keys:
        raise ValueError("Tick at least one parameter.")
    d = pts.dropna(subset=["fw", "sw", "p"])
    if len(d) <= len(keys):
        raise ValueError("More parameters than measured points.")
    if d["fw"].max() <= 0 or (d["free"] * d["gas"]).sum() < 0.05 * (d["wgr"] * d["gas"]).sum():
        return rp, dict(ok=False, reason="No free water: every point is explained by water vapour, "
                                         "so the history cannot fix the curves.")
    w = np.sqrt(d["gas"].to_numpy() / d["gas"].mean())
    lo = np.array([FIT_PARAMS[k][1] for k in keys])
    hi = np.array([FIT_PARAMS[k][2] for k in keys])
    logs = np.array([FIT_PARAMS[k][3] for k in keys])
    hi = np.where([k == "rp_sgrw" for k in keys], np.minimum(hi, 0.94 - rp.swc), hi)

    def make(x):
        vals = np.where(logs, np.exp(x), x)
        return RelPerm(**{**rp.__dict__, **{FIT_PARAMS[k][0]: float(v) for k, v in zip(keys, vals)}})

    def resid(x):
        r = make(x)
        return w * (fw_model(state, r, d["p"].to_numpy(), d["sw"].to_numpy()) - d["fw"].to_numpy())

    x0 = np.array([getattr(rp, FIT_PARAMS[k][0]) for k in keys], float)
    x0 = np.clip(x0, lo * 1.0001, hi * 0.9999)
    tlo, thi = np.where(logs, np.log(lo), lo), np.where(logs, np.log(hi), hi)
    tx0 = np.where(logs, np.log(x0), x0)
    r0 = resid(tx0)
    best = None
    starts = [tx0] + [tlo + f * (thi - tlo) for f in (0.25, 0.5, 0.75)]
    for s in starts:
        r = least_squares(resid, s, bounds=(tlo, thi))
        if best is None or r.cost < best.cost:
            best = r
    new = make(best.x)
    at_lim = [k for k, v, a, b in zip(keys, best.x, tlo, thi) if min(v - a, b - v) < 1e-3 * (b - a)]
    rms = lambda r: float(np.sqrt(np.mean((r / w) ** 2)))     # noqa: E731
    return new, dict(ok=True, rms0=rms(r0), rms=rms(best.fun), at_limit=at_lim, n=len(d))


def wgr_from_fw(state, p, fw):
    """Reservoir water cut -> free water per MMscf of well-stream gas (STB/MMscf)."""
    fw = np.clip(np.asarray(fw, float), 0.0, 0.999999)
    return fw / (1.0 - fw) * 1e6 * state.tank.pvt.bg(np.asarray(p, float)) / state.tank.bw


def fit_yield(pts):
    """Straight line of produced condensate yield against pressure, weighted by gas.
    Returns (slope, intercept, p_min, p_max) or None."""
    d = pts.dropna(subset=["cgr", "p"])
    d = d[np.isfinite(d["cgr"]) & (d["cgr"] >= 0)]
    if len(d) < 3 or np.ptp(d["p"]) < 50.0:
        return None
    a, b = np.polyfit(d["p"], d["cgr"], 1, w=np.sqrt(d["gas"] / d["gas"].mean()))
    return float(a), float(b), float(d["p"].min()), float(d["p"].max())
