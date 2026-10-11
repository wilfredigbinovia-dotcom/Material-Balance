"""Black-oil material balance: PVT, tank, saturations, relative permeability, wells and forecast.

Material balance (Havlena-Odeh)
    F   = Np [Bo + (Rp - Rs) Bg] + Wp Bw          = Np Bo + (Gp - Np Rs) Bg + Wp Bw
    Eo  = (Bo - Boi) + (Rsi - Rs) Bg              oil and solution gas expansion
    Eg  = Boi (Bg / Bgi - 1)                       gas cap expansion (per unit m)
    Efw = (1 + m) Boi (cw Swi + cf) / (1 - Swi) (pi - p)
    F   = N (Eo + m Eg + Efw) + We
N is the stock-tank oil initially in place, m the gas cap to oil zone volume ratio at pi.
Bg is in rb/scf throughout.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from .aquifer import Aquifer
from .pvt import BG_CONST, gas_viscosity, pseudo_criticals, z_dak


# ================================================================= PVT
def standing_rs(p, api, sg, T):
    return sg * ((np.asarray(p, float) / 18.2 + 1.4) * 10 ** (0.0125 * api - 0.00091 * T)) ** 1.2048


def standing_pb(rsi, api, sg, T):
    return 18.2 * ((rsi / sg) ** (1 / 1.2048) * 10 ** (0.00091 * T - 0.0125 * api) - 1.4)


def standing_bo(rs, api, sg, T):
    sgo = 141.5 / (api + 131.5)
    return 0.9759 + 0.00012 * (np.asarray(rs, float) * (sg / sgo) ** 0.5 + 1.25 * T) ** 1.2


def beggs_robinson_dead(api, T):
    x = T ** -1.163 * np.exp(6.9824 - 0.04658 * api)
    return 10 ** x - 1.0


def beggs_robinson_live(rs, api, T):
    mod = beggs_robinson_dead(api, T)
    rs = np.asarray(rs, float)
    return 10.715 * (rs + 100) ** -0.515 * mod ** (5.44 * (rs + 150) ** -0.338)


def gas_bg_z(p, T, sg, co2=0.0, h2s=0.0, n2=0.0):
    """Gas FVF (rb/scf) and Z from DAK with Sutton pseudo-criticals."""
    tpc, ppc = pseudo_criticals(sg, co2, h2s, n2)
    p = np.atleast_1d(np.asarray(p, float))
    z = np.array([z_dak(x / ppc, (T + 459.67) / tpc) for x in p])
    return BG_CONST * z * (T + 459.67) / p, z


def correlation_table(api, sg, rsi, T, pmax, pb=None, co2=0.0, h2s=0.0, n2=0.0, n=30):
    """Black-oil table from correlations: Standing Rs, Bo and bubble point, Vasquez-Beggs
    compressibility and viscosity above the bubble point, Beggs-Robinson viscosity, DAK gas.
    If a bubble point is given, Rs is scaled so that it reaches Rsi there."""
    pb_st = standing_pb(rsi, api, sg, T)
    pb = pb_st if not pb else pb
    p = np.unique(np.round(np.concatenate([np.linspace(100.0, max(pmax, pb + 100), n), [pb]]), 1))
    scale = rsi / float(standing_rs(pb, api, sg, T))
    rs = np.where(p < pb, np.minimum(standing_rs(p, api, sg, T) * scale, rsi), rsi)
    bob = float(standing_bo(rsi, api, sg, T))
    A = (-1433.0 + 5.0 * rsi + 17.2 * T - 1180.0 * sg + 12.61 * api) / 1e5
    bo = np.where(p < pb, standing_bo(rs, api, sg, T), bob * (pb / p) ** A)
    muob = float(beggs_robinson_live(rsi, api, T))
    mexp = 2.6 * p ** 1.187 * np.exp(-11.513 - 8.98e-5 * p)
    muo = np.where(p < pb, beggs_robinson_live(rs, api, T), muob * (p / pb) ** mexp)
    bg, z = gas_bg_z(p, T, sg, co2, h2s, n2)
    mug = gas_viscosity(p, T, z, sg)
    return pd.DataFrame({"p": p, "bo": bo, "rs": rs, "bg": bg * 1e3, "muo": muo, "mug": mug}), float(pb)


def lab_table(dl, pb, bofb, rsfb, cce=None, T=200.0, sg=0.75, api=35.0, co2=0.0, h2s=0.0, n2=0.0):
    """Black-oil table from a differential liberation report corrected to separator conditions.

    dl : p, bod, rsd and optionally bg (rb/Mscf), muo, mug, z (rows at and below the bubble point)
    cce: p, vr (relative volume V/Vb) above the bubble point, optional
    Below pb:  Bo = Bod Bofb / Bodb          Rs = Rsfb - (Rsdb - Rsd) Bofb / Bodb
    Above pb:  Bo = Bofb Vr                  Rs = Rsfb
    Missing gas properties come from DAK / Lee-Gonzalez-Eakin, missing oil viscosity from
    Beggs-Robinson. Returns (table, notes)."""
    notes = []
    d = dl.copy()
    for c in ("p", "bod", "rsd", "bg", "muo", "mug", "z"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    d = d.dropna(subset=["p", "bod", "rsd"]).sort_values("p")
    d = d[d["p"] <= pb * 1.001]
    if len(d) < 2:
        raise ValueError("The differential liberation table needs at least two rows at or below "
                         "the bubble point with pressure, Bod and Rsd.")
    if abs(d["p"].iloc[-1] - pb) > 0.003 * pb:
        notes.append("No differential liberation row at the bubble point: Bodb and Rsdb were "
                     "extrapolated from the two highest rows.")
        x = d["p"].iloc[-2:].to_numpy()
        bodb = float(np.polyval(np.polyfit(x, d["bod"].iloc[-2:], 1), pb))
        rsdb = float(np.polyval(np.polyfit(x, d["rsd"].iloc[-2:], 1), pb))
    else:
        bodb, rsdb = float(d["bod"].iloc[-1]), float(d["rsd"].iloc[-1])
    f = bofb / bodb
    p = d["p"].to_numpy(float)
    bo = d["bod"].to_numpy(float) * f
    rs = rsfb - (rsdb - d["rsd"].to_numpy(float)) * f
    if np.any(rs < -1):
        notes.append("The corrected Rs falls below zero at low pressure; it is set to zero there.")
    rs = np.maximum(rs, 0.0)
    rows = pd.DataFrame({"p": p, "bo": bo, "rs": rs, "bg": d["bg"].to_numpy(float),
                         "muo": d["muo"].to_numpy(float), "mug": d["mug"].to_numpy(float)})
    if cce is not None and len(cce):
        c = cce.copy()
        c["p"], c["vr"] = pd.to_numeric(c["p"], errors="coerce"), pd.to_numeric(c["vr"], errors="coerce")
        c = c.dropna().sort_values("p")
        c = c[c["p"] > pb * 1.001]
        if len(c):
            rows = pd.concat([rows, pd.DataFrame({"p": c["p"], "bo": bofb * c["vr"], "rs": rsfb,
                                                  "bg": np.nan, "muo": np.nan, "mug": np.nan})])
    rows = rows.sort_values("p").reset_index(drop=True)
    if not (rows["p"] > pb * 1.001).any():
        notes.append("No CCE data above the bubble point: Bo is held at Bofb above it unless you "
                     "add rows.")
    # fill gas properties and viscosities
    miss = rows["bg"].isna()
    if miss.any():
        bg, z = gas_bg_z(rows.loc[miss, "p"], T, sg, co2, h2s, n2)
        rows.loc[miss, "bg"] = bg * 1e3
    if rows["mug"].isna().any():
        bgv = rows["bg"].to_numpy() / 1e3
        z = bgv * rows["p"].to_numpy() / (BG_CONST * (T + 459.67))
        rows.loc[rows["mug"].isna(), "mug"] = gas_viscosity(rows["p"], T, z, sg)[rows["mug"].isna().to_numpy()]
    if rows["muo"].isna().any():
        rows.loc[rows["muo"].isna(), "muo"] = beggs_robinson_live(rows.loc[rows["muo"].isna(), "rs"], api, T)
    return rows, notes


@dataclass
class OilPVT:
    """Black-oil properties from a table (p, bo, rs, bg [rb/Mscf], muo, mug) and the bubble point."""
    table: pd.DataFrame
    pb: float
    T: float

    def __post_init__(self):
        d = self.table.copy()
        for c in ("p", "bo", "rs", "bg", "muo", "mug"):
            d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        d = d.dropna(subset=["p", "bo", "rs"]).drop_duplicates("p").sort_values("p")
        if len(d) < 2:
            raise ValueError("The oil PVT table needs at least two rows with pressure, Bo and Rs.")
        if d["bg"].isna().all() or d["muo"].isna().all() or d["mug"].isna().all():
            raise ValueError("The oil PVT table needs Bg, oil viscosity and gas viscosity (fill "
                             "the table from the lab report or correlations).")
        self.p = d["p"].to_numpy(float)
        self._c = {}
        for c in ("bo", "rs", "bg", "muo", "mug"):
            s = d[c].to_numpy(float)
            ok = np.isfinite(s)
            self._c[c] = (self.p[ok], s[ok])

    @property
    def p_range(self):
        return float(self.p[0]), float(self.p[-1])

    def _i(self, c, p):
        x, y = self._c[c]
        return np.interp(np.asarray(p, float), x, y)

    def bo(self, p):
        return self._i("bo", p)

    def rs(self, p):
        return self._i("rs", p)

    def bg(self, p):
        """rb/scf"""
        return self._i("bg", p) / 1e3

    def muo(self, p):
        return self._i("muo", p)

    def mug(self, p):
        return self._i("mug", p)


# ================================================================= tank
class OilTank:
    """neighbour: (t_days, p_psia, T) - a connected reservoir with a known pressure history.
    Reservoir fluid crosses at q = T (p_neighbour - p) rb/day; it is treated as water (the usual
    case of reservoirs sharing an aquifer), so it adds to the support like aquifer influx."""

    def __init__(self, pvt: OilPVT, hist, N, m, pi, swi, cf, cw, bw=1.0, aquifer=None, neighbour=None):
        self.pvt, self.h = pvt, hist
        self.N, self.m = N, m
        self.pi, self.swi, self.cf, self.cw, self.bw = pi, swi, cf, cw, bw
        self.aq = aquifer or Aquifer()
        self.boi, self.rsi, self.bgi = float(pvt.bo(pi)), float(pvt.rs(pi)), float(pvt.bg(pi))
        self.ce = (cw * swi + cf) / (1.0 - swi)
        self.use, self.w = hist.use, hist.w
        self.p_drive = np.interp(hist.t, hist.t[self.use], hist.p[self.use])
        self.gpw = hist.np_            # x-axis for the drive diagnostics: cumulative oil
        self.nb_T, self.nb_t, self.nb_p = 0.0, None, None
        if neighbour is not None and neighbour[2] and len(neighbour[0]):
            o_ = np.argsort(neighbour[0])
            self.nb_t = np.asarray(neighbour[0], float)[o_]
            self.nb_p = np.asarray(neighbour[1], float)[o_]
            self.nb_T = float(neighbour[2])
        self.pn = self.neighbour_p(hist.t)
        self.qx_sim = np.zeros(len(hist.t))

    @property
    def has_neighbour(self):
        return self.nb_T > 0

    def neighbour_p(self, t):
        t = np.asarray(t, float)
        if self.nb_t is None:
            return np.full(t.shape, np.nan)
        return np.interp(t, self.nb_t, self.nb_p)

    def qx_step(self, i, p, qx, t=None, pn=None):
        """Cumulative reservoir volume received from the neighbour (rb), implicit in p[i]."""
        if not self.has_neighbour:
            return 0.0
        t = self.h.t if t is None else t
        pn = self.pn if pn is None else pn
        return qx[i - 1] + self.nb_T * (pn[i] - p[i]) * (t[i] - t[i - 1])

    def qx_history(self):
        qx = np.zeros(len(self.h.t))
        for i in range(1, len(qx)):
            qx[i] = self.qx_step(i, self.p_drive, qx)
        return qx

    # ---- expansion terms
    def Eo(self, p):
        pv = self.pvt
        return (pv.bo(p) - self.boi) + (self.rsi - pv.rs(p)) * pv.bg(p)

    def Eg(self, p):
        return self.boi * (self.pvt.bg(p) / self.bgi - 1.0)

    def Efw(self, p):
        return (1.0 + self.m) * self.boi * self.ce * (self.pi - np.asarray(p, float))

    def Et(self, p):
        return self.Eo(p) + self.m * self.Eg(p) + self.Efw(p)

    def F(self, p=None, np_=None, gp=None, wp=None, wi=None, gi=None):
        """Net underground withdrawal (rb): production less injection."""
        h, pv = self.h, self.pvt
        p = h.p if p is None else p
        np_ = h.np_ if np_ is None else np_
        gp = h.gp if gp is None else gp
        wp = h.wp if wp is None else wp
        wi = h.wi if wi is None else wi
        gi = h.gi if gi is None else gi
        bg = pv.bg(p)
        return np_ * pv.bo(p) + (gp - np_ * pv.rs(p)) * bg + wp * self.bw - wi * self.bw - gi * bg

    def We_history(self):
        return self.aq.influx(self.h.t, self.p_drive)

    def support_history(self):
        s = self.We_history()
        return s + self.qx_history() if self.has_neighbour else s

    def balance(self, p, we, np_, gp, wp, wi=0.0, gi=0.0, qx=0.0):
        return self.N * self.Et(p) + we + qx - self.F(p, np_, gp, wp, wi, gi)

    def simulate(self):
        """Pressure from the production history. Returns (p_sim, We_sim)."""
        h, aq = self.h, self.aq
        n = len(h.t)
        aq.prepare(h.t)
        p, We, qx = np.full(n, np.nan), np.zeros(n), np.zeros(n)
        p[0] = self.pi
        plo = max(self.pvt.p_range[0], 14.7)
        for i in range(1, n):
            def res(pp):
                p[i] = pp
                return self.balance(pp, aq.we_at(i, p, We), h.np_[i], h.gp[i], h.wp[i], h.wi[i], h.gi[i],
                                    self.qx_step(i, p, qx))
            try:
                r_hi, r_lo = res(self.pi), res(plo)
                if not (np.isfinite(r_hi) and np.isfinite(r_lo)):
                    raise ValueError
                if r_hi >= 0:
                    sol = self.pi
                elif r_lo < 0:
                    raise ValueError
                else:
                    sol = brentq(res, plo, self.pi, xtol=1e-4)
            except ValueError:
                p[i:] = np.nan
                We[i:] = np.nan
                break
            p[i] = sol
            We[i] = aq.we_at(i, p, We)
            qx[i] = self.qx_step(i, p, qx)
        self.qx_sim = qx
        return p, We

    def energy(self):
        p = self.p_drive
        we = np.maximum(self.We_history(), 0.0)
        nb = np.maximum(self.qx_history(), 0.0) if self.has_neighbour else 0.0
        inj = (self.h.wi * self.bw + self.h.gi * self.pvt.bg(p))
        a = self.N * self.Eo(p)
        b = self.N * self.m * self.Eg(p)
        c = self.N * self.Efw(p)
        tot = a + b + c + we + nb + inj
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {"Oil and solution gas expansion": a / tot}
            if self.m > 0:
                out["Gas cap expansion"] = b / tot
            out["Rock and connate water compressibility"] = c / tot
            out["Water influx"] = we / tot
            if self.has_neighbour:
                out["Neighbouring reservoir"] = nb / tot
            if np.any(inj > 0):
                out["Injection"] = inj / tot
            return out


OIL_METHODS = [
    "F against Et (volumetric)",
    "Campbell: F/Et against Np",
    "Havlena-Odeh: F/Et against We/Et",
    "Gas cap: (F - We)/Eo against Eg/Eo",
    "(F - We)/Et against Np",
]


def oil_graphical(tank: OilTank, method, sel=None):
    h = tank.h
    p = h.p
    n = len(p)
    sel = (np.ones(n, bool) if sel is None else np.asarray(sel, bool)) & tank.use
    F, Eo, Eg, Et = tank.F(), tank.Eo(p), tank.Eg(p), tank.Et(p)
    We = tank.We_history() if tank.aq.active else np.zeros(n)
    nz = np.arange(n) > 0
    out = {"method": method, "N": None, "extra": {}, "mask": nz}
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "F against Et (volumetric)":
            x, y = Et, F - We
            m_ = sel & nz & np.isfinite(x) & np.isfinite(y)
            fit = None
            if m_.sum() >= 2 and np.sum(x[m_] ** 2) > 0:
                N = float(np.sum(x[m_] * y[m_]) / np.sum(x[m_] ** 2))      # through the origin
                r2 = 1 - np.sum((y[m_] - N * x[m_]) ** 2) / max(np.sum((y[m_] - y[m_].mean()) ** 2), 1e-30)
                fit = {"slope": N, "intercept": 0.0, "r2": r2}
                out["N"] = N
            out.update(kind="line", xl="Et (rb/STB)", yl="F − We (rb)", xq="ratio", yq="res")
        elif method in ("Campbell: F/Et against Np", "(F - We)/Et against Np"):
            w = We if method.startswith("(F") else 0.0
            x, y = h.np_, (F - w) / Et
            m_ = sel & nz & np.isfinite(y)
            fit = None
            if m_.any():
                out["N"] = float(np.mean(y[m_]))
            out.update(kind="flat", xl="Cumulative oil (STB)", yl=("(F − We)/Et" if np.ndim(w) else "F/Et") + " (STB)")
        elif method == "Havlena-Odeh: F/Et against We/Et":
            x, y = We / Et, F / Et
            m_ = sel & nz & np.isfinite(x) & np.isfinite(y)
            fit = None
            if m_.sum() >= 2 and np.ptp(x[m_]) > 0:
                a, b = np.polyfit(x[m_], y[m_], 1)
                fit = {"slope": a, "intercept": b, "r2": np.corrcoef(x[m_], y[m_])[0, 1] ** 2}
                out["N"] = float(b)
                out["extra"]["slope"] = float(a)
            elif m_.any():
                out["N"] = float(np.mean(y[m_]))
            out.update(kind="line", xl="We/Et (STB)", yl="F/Et (STB)")
        else:   # gas cap
            x, y = Eg / Eo, (F - We) / Eo
            m_ = sel & nz & np.isfinite(x) & np.isfinite(y)
            fit = None
            if m_.sum() >= 2 and np.ptp(x[m_]) > 0:
                a, b = np.polyfit(x[m_], y[m_], 1)
                fit = {"slope": a, "intercept": b, "r2": np.corrcoef(x[m_], y[m_])[0, 1] ** 2}
                out["N"] = float(b)
                if b > 0:
                    out["extra"]["m"] = float(a / b)
            out.update(kind="line", xl="Eg/Eo", yl="(F − We)/Eo (STB)")
    out.update(x=x, y=y, fit=fit, sel=sel)
    return out


# ================================================================= saturations and rel perm
@dataclass
class OilRelPerm:
    swc: float
    sorw: float = 0.25
    sorg: float = 0.15
    sgc: float = 0.05
    krw_max: float = 0.3
    kro_max: float = 1.0
    krg_max: float = 0.8
    nw: float = 3.0
    now: float = 2.0
    ng: float = 2.0
    nog: float = 3.0

    def ow(self, sw):
        s = np.clip((np.asarray(sw, float) - self.swc) / max(1 - self.swc - self.sorw, 1e-6), 0, 1)
        return self.krw_max * s ** self.nw, self.kro_max * (1 - s) ** self.now

    def go(self, sg):
        span = max(1 - self.swc - self.sorg, 1e-6)
        sg = np.clip(np.asarray(sg, float), 0, span)
        sgn = np.clip((sg - self.sgc) / max(span - self.sgc, 1e-6), 0, 1)
        return self.krg_max * sgn ** self.ng, self.kro_max * (1 - sg / span) ** self.nog

    def kr(self, sw, sg):
        """(krw, kro, krg): kro is the normalised product of the oil-water and gas-oil values."""
        krw, krow = self.ow(sw)
        krg, krog = self.go(sg)
        return krw, krow * krog / self.kro_max, krg


class OilState:
    """Tank saturations and fluid mobilities of a matched oil tank."""

    def __init__(self, tank: OilTank, muw):
        self.t, self.muw = tank, muw
        self.pvi = (1 + tank.m) * tank.N * tank.boi / (1 - tank.swi)
        pv = tank.pvt
        self._lo_i = 1.0 / (float(pv.muo(tank.pi)) * tank.boi)

    def sats(self, p, we, np_, wp, wi=None, qx=None):
        """Tank saturations. wi: water injected (STB); qx: volume received from a neighbour (rb,
        counted as water). Both default to the history values when arrays of history length are
        passed, and to zero otherwise."""
        t = self.t
        n_h = len(t.h.t)
        if wi is None:
            wi = t.h.wi if np.ndim(p) and len(np.atleast_1d(p)) == n_h else 0.0
        if qx is None:
            qx = t.qx_sim if (t.has_neighbour and np.ndim(p) and len(np.atleast_1d(p)) == n_h) else 0.0
        dp = t.pi - np.asarray(p, float)
        pvol = self.pvi * (1 - t.cf * dp)
        vw = self.pvi * t.swi * (1 + t.cw * dp) + we - (wp - wi) * t.bw + qx
        vo = (t.N - np_) * t.pvt.bo(p)
        sw = vw / pvol
        so = vo / pvol
        sg = np.clip(1 - sw - so, 0.0, 1.0)
        return sw, so, sg

    def ratios(self, rp, p, sw, sg):
        """Producing GOR (scf/STB), water-oil ratio (STB/STB) and oil mobility relative to initial."""
        pv = self.t.pvt
        krw, kro, krg = rp.kr(sw, sg)
        lo = kro / (pv.muo(p) * pv.bo(p))
        lg = krg / (pv.mug(p) * pv.bg(p))
        lw = krw / (self.muw * self.t.bw)
        lo_safe = np.maximum(lo, 1e-12)
        gor = pv.rs(p) + lg / lo_safe
        wor = lw / lo_safe
        return gor, wor, lo / self._lo_i


def oil_points(state, rp, p, we):
    """Measured producing GOR and water-oil ratio per interval between pressure surveys, with
    the tank saturations at the interval midpoints."""
    t = state.t
    h = t.h
    sw, so, sg = state.sats(p, we, h.np_, h.wp)
    dn, dg, dw = np.diff(h.np_), np.diff(h.gp), np.diff(h.wp)
    ok = dn > 1.0
    pm = 0.5 * (p[1:] + p[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        d = pd.DataFrame({"t": 0.5 * (h.t[1:] + h.t[:-1]), "p": pm,
                          "sw": 0.5 * (sw[1:] + sw[:-1]), "sg": 0.5 * (sg[1:] + sg[:-1]),
                          "gor": dg / dn, "wor": dw / dn, "oil": dn, "well": "Reservoir"})
    d = d[ok].reset_index(drop=True)
    d["rs"] = t.pvt.rs(d["p"])
    d["fw"] = d["wor"] / (1 + d["wor"])
    return d


def oil_well_points(state, p, we, wells, start_date):
    """Per-well, per-record GOR and WOR. wells: well, date, gp [MMscf], np [MSTB], wp [MSTB]."""
    t = state.t
    h = t.h
    sw_h, so_h, sg_h = state.sats(p, we, h.np_, h.wp)
    out = []
    w = wells.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    for name, g in w.dropna(subset=["date", "well"]).groupby("well"):
        g = g.sort_values("date")
        day = (g["date"] - pd.Timestamp(start_date)).dt.days.to_numpy(float)
        gp = np.concatenate([[0.0], pd.to_numeric(g["gp"], errors="coerce").fillna(0).to_numpy() * 1e6])
        npc = np.concatenate([[0.0], pd.to_numeric(g["np"], errors="coerce").fillna(0).to_numpy() * 1e3])
        wp = np.concatenate([[0.0], pd.to_numeric(g["wp"], errors="coerce").fillna(0).to_numpy() * 1e3])
        day = np.concatenate([[day[0] - 30.4], day])
        dn, dg, dw = np.diff(npc), np.diff(gp), np.diff(wp)
        tm = 0.5 * (day[1:] + day[:-1])
        ok = (dn > 10.0) & (tm >= h.t[0]) & (tm <= h.t[-1]) & (dw >= 0) & (dg >= 0)
        if not ok.any():
            continue
        pm = np.interp(tm, h.t, p)
        out.append(pd.DataFrame({"t": tm, "p": pm, "sw": np.interp(tm, h.t, sw_h),
                                 "sg": np.interp(tm, h.t, sg_h), "gor": dg / np.maximum(dn, 1e-9),
                                 "wor": dw / np.maximum(dn, 1e-9), "oil": dn, "well": str(name)})[ok])
    if not out:
        return pd.DataFrame(columns=["t", "p", "sw", "sg", "gor", "wor", "oil", "well", "rs", "fw"])
    d = pd.concat(out, ignore_index=True)
    d["rs"] = t.pvt.rs(d["p"])
    d["fw"] = d["wor"] / (1 + d["wor"])
    return d


OIL_FIT = {   # key -> (attribute, lower, upper, log)
    "o_krw": ("krw_max", 1e-4, 1.0, True), "o_nw": ("nw", 1.0, 8.0, False),
    "o_sorw": ("sorw", 0.02, 0.6, False), "o_now": ("now", 1.0, 8.0, False),
    "o_krg": ("krg_max", 1e-3, 1.0, True), "o_ng": ("ng", 1.0, 8.0, False),
    "o_sgc": ("sgc", 0.0, 0.3, False), "o_nog": ("nog", 1.0, 8.0, False),
}


def fit_oil_ratios(state, rp, pts, keys, phase):
    """Regress Corey parameters to the measured water cut (phase='water') or producing GOR
    (phase='gas'). Residuals are weighted by the oil each point represents. Returns (rp, info)."""
    d = pts.dropna(subset=["p", "sw", "sg"])
    if not keys:
        raise ValueError("Tick at least one parameter.")
    if len(d) <= len(keys):
        raise ValueError("More parameters than measured points.")
    obs = d["fw"].to_numpy() if phase == "water" else d["gor"].to_numpy()
    if phase == "water" and obs.max() <= 1e-4:
        return rp, dict(ok=False, reason="No water has been produced, so the water curves cannot be fitted.")
    if phase == "gas" and np.all(obs <= d["rs"].to_numpy() * 1.02):
        return rp, dict(ok=False, reason="The producing GOR has stayed at the solution GOR: no free "
                                         "gas has been produced, so the gas curves cannot be fitted.")
    scale = 1.0 if phase == "water" else max(float(np.median(np.abs(obs))), 1.0)
    w = np.sqrt(d["oil"].to_numpy() / d["oil"].mean())
    lo = np.array([OIL_FIT[k][1] for k in keys])
    hi = np.array([OIL_FIT[k][2] for k in keys])
    logs = np.array([OIL_FIT[k][3] for k in keys])

    def make(x):
        v = np.where(logs, np.exp(x), x)
        return replace(rp, **{OIL_FIT[k][0]: float(a) for k, a in zip(keys, v)})

    def model(r):
        gor, wor, _ = state.ratios(r, d["p"].to_numpy(), d["sw"].to_numpy(), d["sg"].to_numpy())
        return wor / (1 + wor) if phase == "water" else gor

    def resid(x):
        return w * (model(make(x)) - obs) / scale

    tlo, thi = np.where(logs, np.log(lo), lo), np.where(logs, np.log(hi), hi)
    x0 = np.clip([getattr(rp, OIL_FIT[k][0]) for k in keys], lo * 1.0001 + 1e-9, hi * 0.9999)
    tx0 = np.where(logs, np.log(x0), x0)
    r0 = resid(tx0)
    best = None
    for s in [tx0] + [tlo + f * (thi - tlo) for f in (0.25, 0.5, 0.75)]:
        r = least_squares(resid, s, bounds=(tlo, thi))
        if best is None or r.cost < best.cost:
            best = r
    rms = lambda r: float(np.sqrt(np.mean((r / w * scale) ** 2)))    # noqa: E731
    at = [k for k, v, a, b in zip(keys, best.x, tlo, thi) if min(v - a, b - v) < 1e-3 * (b - a)]
    return make(best.x), dict(ok=True, rms0=rms(r0), rms=rms(best.fun), at_limit=at, n=len(d))


# ================================================================= wells
@dataclass
class OilWell:
    name: str
    J: float = np.nan           # STB/d/psi liquid, at initial mobility
    min_thp: float = np.nan
    min_bhp: float = np.nan
    qmax: float = np.nan        # STB/d liquid
    lift: object = None         # OilLiftTable
    n_tests: int = 0
    note: str = ""

    @property
    def control(self):
        if self.lift is not None and np.isfinite(self.min_thp):
            return "thp"
        if np.isfinite(self.min_bhp):
            return "bhp"
        return None


def ipr_shape(pr, pwf, pb):
    """Liquid rate per unit productivity index: straight line above the bubble point, Vogel below
    (composite inflow when pr > pb)."""
    pwf = np.maximum(np.asarray(pwf, float), 0.0)
    if pr <= pb:
        x = pwf / pr
        return pr / 1.8 * (1 - 0.2 * x - 0.8 * x * x)
    above = np.maximum(pr - np.maximum(pwf, pb), 0.0)
    x = np.minimum(pwf, pb) / pb
    below = np.where(pwf < pb, pb / 1.8 * (1 - 0.2 * x - 0.8 * x * x), 0.0)
    return above + below


def fit_pi(q, pr, pwf, pb, mob):
    """Productivity index at initial mobility from tests. Returns (J, note)."""
    q, pr, pwf, mob = (np.asarray(a, float) for a in (q, pr, pwf, mob))
    shape = np.array([float(ipr_shape(a, b, pb)) for a, b in zip(pr, pwf)])
    ok = (q > 0) & (shape > 0) & (mob > 0)
    if not ok.any():
        return np.nan, "no usable test (flowing pressure must be below reservoir pressure)"
    j = q[ok] / (shape[ok] * mob[ok])
    note = "" if ok.sum() == 1 or np.ptp(j) / np.mean(j) < 0.25 else \
        f"tests give productivity indices from {j.min():.3g} to {j.max():.3g}; the mean is used"
    return float(np.mean(j)), note


def oil_well_rate(well, pr, M, pb, wc, gor, stop_loading=False):
    """Liquid rate (STB/d) and flowing bottomhole pressure at the well's limit."""
    if not np.isfinite(well.J) or well.control is None or M <= 0 or not np.isfinite(pr):
        return 0.0, np.nan, False
    J = well.J * M

    def ipr(pwf):
        return J * float(ipr_shape(pr, pwf, pb))

    loading = False
    if well.control == "bhp":
        pwf = well.min_bhp
        q = ipr(pwf)
    else:
        hi = ipr(0.0)
        if hi <= 0:
            return 0.0, np.nan, False
        vlp = lambda qq: well.lift.bhp_at(well.min_thp, qq, wc, gor)          # noqa: E731
        f = lambda qq: ipr(vlp(qq)) - qq                                      # noqa: E731
        grid = np.geomspace(min(float(well.lift.q[0]), hi * 0.5), hi, 40)
        fv = np.array([f(x) for x in grid])
        cross = np.where((fv[:-1] > 0) & (fv[1:] <= 0))[0]
        if not len(cross):
            return 0.0, np.nan, False
        k = cross[-1]
        q = brentq(f, grid[k], grid[k + 1], xtol=1e-3)
        pwf = vlp(q)
        if np.isfinite(well.min_bhp) and pwf < well.min_bhp:
            pwf, q = well.min_bhp, ipr(well.min_bhp)
    if np.isfinite(well.qmax) and q > well.qmax:
        q = well.qmax
        lo_, hi_ = 0.0, pr
        for _ in range(60):                  # pwf at which the inflow gives qmax
            mid = 0.5 * (lo_ + hi_)
            lo_, hi_ = (mid, hi_) if ipr(mid) > q else (lo_, mid)
        pwf = 0.5 * (lo_ + hi_)
    return float(q), float(pwf), loading


# ================================================================= forecast
def oil_forecast(tank, p_sim, we_sim, wells, rp, state, days, dt=30.4375, q_target=None,
                 q_min=None, wc_max=None, gor_max=None, well_rp=None, qwi=0.0, vrr=None, qgi=0.0,
                 nb_mode="cutoff", nb_vol=None):
    """Step the matched oil tank forward.

    q_target / q_min   field oil rates (STB/d)       wc_max (fraction), gor_max (scf/STB): stop limits
    well_rp            {well name: OilRelPerm} curves fitted to that well's own GOR and water cut;
                       other wells use rp. Each well's GOR, water cut and productivity follow its
                       curves at the tank saturations.
    qwi, qgi           water (STB/d) and gas (scf/d) injection; vrr replaces qwi with water
                       injection equal to vrr times the reservoir voidage of the previous step
    nb_mode            neighbouring reservoir: 'cutoff', 'constant' (held at its last pressure) or
                       'deplete' (a closed volume nb_vol rb with compressibility cw + cf)
    Returns (DataFrame, summary)."""
    h, pv = tank.h, tank.pvt
    well_rp = well_rp or {}
    if not np.all(np.isfinite(p_sim)):
        raise ValueError("The model does not reproduce the whole history. Fix the history match first.")
    n0 = len(h.t)
    nf = int(np.ceil(days / dt))
    t = np.concatenate([h.t, h.t[-1] + dt * np.arange(1, nf + 1)])
    N_ = len(t)
    aq = replace(tank.aq).prepare(t)
    p, We, qx = np.full(N_, np.nan), np.zeros(N_), np.zeros(N_)
    p[:n0], We[:n0] = p_sim, we_sim
    qx[:n0] = tank.qx_sim if tank.has_neighbour else 0.0
    pn = tank.neighbour_p(t)
    nb_on = tank.has_neighbour and nb_mode != "cutoff"
    if tank.has_neighbour and nb_mode == "deplete":
        if not nb_vol or nb_vol <= 0:
            raise ValueError("Enter the neighbouring reservoir's volume, or choose how it behaves in the forecast.")
        ct = tank.cw + tank.cf
        qx_last = float(np.interp(tank.nb_t[-1], h.t, qx[:n0])) if tank.nb_t[-1] > h.t[0] else 0.0
        pn_last = float(tank.nb_p[-1])
    npc, gp, wp, wi, gi = (np.concatenate([a, np.zeros(nf)]) for a in (h.np_, h.gp, h.wp, h.wi, h.gi))
    cols = {k: np.full(N_, np.nan) for k in ("qo", "qg", "qw", "qwi", "qgi", "gor", "wc", "sw", "sg")}
    qwell = {w.name: np.full(N_, np.nan) for w in wells}
    qowell = {w.name: np.full(N_, np.nan) for w in wells}
    pwfw = {w.name: np.full(N_, np.nan) for w in wells}
    plo = max(pv.p_range[0], 14.7)
    reason, last = "end of forecast period", N_ - 1
    void_prev = 0.0
    for i in range(n0, N_):
        pr = p[i - 1]
        sw, so, sg = (float(x) for x in state.sats(pr, We[i - 1], npc[i - 1], wp[i - 1], wi[i - 1], qx[i - 1]))
        ql, qo_w, qg_w, qw_w, res = [], [], [], [], []
        for w in wells:
            r_ = well_rp.get(w.name, rp)
            gor, wor, M = (float(x) for x in state.ratios(r_, pr, sw, sg))
            wc = wor / (1 + wor)
            rr = oil_well_rate(w, pr, M, pv.pb, wc * 100.0, gor)
            res.append(rr)
            ql.append(rr[0])
            qo_w.append(rr[0] * (1 - wc))
            qw_w.append(rr[0] * wc)
            qg_w.append(rr[0] * (1 - wc) * gor)
        ql, qo_w, qg_w, qw_w = (np.array(a) for a in (ql, qo_w, qg_w, qw_w))
        qo_tot = qo_w.sum()
        f_ = 1.0
        if q_target and qo_tot > q_target:
            f_ = q_target / qo_tot
        qo, qg, qw = qo_tot * f_, qg_w.sum() * f_, qw_w.sum() * f_
        if qo <= max(q_min or 0.0, 1e-3):
            reason = ("field oil rate fell below the minimum rate" if qo > 1e-3
                      else "wells can no longer flow against their pressure limits")
            last = i - 1
            break
        wc_f, gor_f = qw / (qo + qw), qg / qo
        if wc_max and wc_f > wc_max:
            reason, last = f"water cut reached {wc_max:.0%}", i - 1
            break
        if gor_max and gor_f > gor_max:
            reason, last = f"producing GOR reached {gor_max:,.0f} scf/STB", i - 1
            break
        step = t[i] - t[i - 1]
        q_wi = (vrr * void_prev / tank.bw) if vrr else (qwi or 0.0)
        q_gi = qgi or 0.0
        npc[i], gp[i], wp[i] = npc[i - 1] + qo * step, gp[i - 1] + qg * step, wp[i - 1] + qw * step
        wi[i], gi[i] = wi[i - 1] + q_wi * step, gi[i - 1] + q_gi * step
        def qx_at(pp_arr):
            if not nb_on:
                return qx[i - 1]
            if nb_mode == "deplete":
                # neighbour pressure and transfer solved together (implicit, stable for strong links)
                C = ct * nb_vol
                k_ = tank.nb_T * step / C
                pn[i] = (pn_last - (qx[i - 1] - qx_last) / C + k_ * pp_arr[i]) / (1 + k_)
            return tank.qx_step(i, pp_arr, qx, t, pn)

        def bal(pp):
            p[i] = pp
            return tank.balance(pp, aq.we_at(i, p, We), npc[i], gp[i], wp[i], wi[i], gi[i], qx_at(p))

        r_hi, r_lo = bal(tank.pi), bal(plo)
        if not (np.isfinite(r_hi) and np.isfinite(r_lo)) or r_lo < 0:
            reason, last = "reservoir pressure fell below the PVT table", i - 1
            p[i] = np.nan
            break
        p_up = tank.pi if r_hi < 0 else max(tank.pi, pr) * 1.5
        if r_hi >= 0 and bal(p_up) >= 0:
            p[i] = p_up
        else:
            p[i] = brentq(bal, plo, p_up if r_hi >= 0 else tank.pi, xtol=1e-3)
        We[i] = aq.we_at(i, p, We)
        qx[i] = qx_at(p)
        void_prev = qo * pv.bo(p[i]) + max(qg - qo * pv.rs(p[i]), 0.0) * pv.bg(p[i]) + qw * tank.bw
        for k, v in (("qo", qo), ("qg", qg), ("qw", qw), ("qwi", q_wi), ("qgi", q_gi), ("gor", gor_f),
                     ("wc", wc_f), ("sw", sw), ("sg", sg)):
            cols[k][i] = v
        for w, a_, b_, r in zip(wells, ql, qo_w, res):
            qwell[w.name][i] = a_ * f_
            qowell[w.name][i] = b_ * f_
            pwfw[w.name][i] = r[1]
    sl = slice(0, last + 1)
    df = pd.DataFrame({"t": t[sl], "p": p[sl], "np": npc[sl], "gp": gp[sl], "wp": wp[sl], "wi": wi[sl],
                       "gi": gi[sl], "we": We[sl], "qx": qx[sl], "pn": pn[sl],
                       **{k: v[sl] for k, v in cols.items()},
                       **{f"q_{k}": v[sl] for k, v in qwell.items()},
                       **{f"qo_{k}": v[sl] for k, v in qowell.items()},
                       **{f"pwf_{k}": v[sl] for k, v in pwfw.items()}})
    df["forecast"] = np.arange(len(df)) >= n0
    summary = dict(reason=reason, n_hist=n0, p_end=float(p[last]), np_end=float(npc[last]),
                   gp_end=float(gp[last]), wp_end=float(wp[last]), wi_end=float(wi[last]),
                   gi_end=float(gi[last]), qx_end=float(qx[last]), rf=float(npc[last] / tank.N),
                   rf_hist=float(npc[n0 - 1] / tank.N), stopped=last < N_ - 1)
    return df, summary
