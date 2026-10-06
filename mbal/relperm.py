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
