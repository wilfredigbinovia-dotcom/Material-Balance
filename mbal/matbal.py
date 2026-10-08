"""Gas-condensate tank material balance.

    F = G (Eg + Efw) + We
    F   = Gp_wet * Bg + Wp * Bw           underground withdrawal, rb
    Eg  = Bg - Bgi                        gas expansion, rb/scf (two-phase Z below dew point)
    Efw = Bgi (cw Swi + cf)/(1 - Swi) (pi - p)
    Gp_wet = Gp_separator_gas + GE * Np   (condensate converted to its gas equivalent)

G is the wet (well-stream) gas initially in place.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy import stats

from .aquifer import Aquifer
from .pvt import PVT


@dataclass
class History:
    t: np.ndarray    # days from start, t[0] = 0
    p: np.ndarray    # psia, p[0] = pi
    gp: np.ndarray   # scf separator gas
    np_: np.ndarray  # STB condensate
    wp: np.ndarray   # STB water
    use: np.ndarray = None   # bool, pressure survey switched on
    w: np.ndarray = None     # relative weight of each pressure survey

    def __post_init__(self):
        n = len(self.t)
        self.use = np.ones(n, bool) if self.use is None else np.asarray(self.use, bool)
        self.w = np.ones(n) if self.w is None else np.asarray(self.w, float)
        self.w_in = np.maximum(np.nan_to_num(self.w, nan=1.0), 0.0)   # as entered
        self.w = np.where(self.use, np.maximum(self.w, 0.0), 0.0)
        self.use = self.use & (self.w > 0)
        self.use[0], self.w[0] = True, 1.0


def build_history(df, start_date, pi):
    """df columns: date, p, gp [MMscf], np [MSTB], wp [MSTB] (cumulative), optional use, w.

    Prepends initial conditions. A survey that is switched off keeps its production
    record (the tank still needs it) but its pressure is ignored in every fit.
    """
    import pandas as pd
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("gp", "np", "wp"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["p"] = pd.to_numeric(d["p"], errors="coerce")
    d["use"] = _flag(d["use"]) if "use" in d else True
    d["w"] = pd.to_numeric(d["w"], errors="coerce").fillna(1.0) if "w" in d else 1.0
    d = d.dropna(subset=["date", "p"]).sort_values("date")
    t = (d["date"] - pd.Timestamp(start_date)).dt.total_seconds().to_numpy() / 86400.0
    keep = (t > 0) & (d["gp"].to_numpy() > 0)
    d, t = d[keep], t[keep]
    if len(d) < 3:
        raise ValueError("At least three production history rows (after the start date, with "
                         "pressure and cumulative gas) are needed.")
    z = np.zeros(1)
    return History(
        t=np.concatenate([z, t]),
        p=np.concatenate([[pi], d["p"].to_numpy(float)]),
        gp=np.concatenate([z, d["gp"].to_numpy(float) * 1e6]),
        np_=np.concatenate([z, d["np"].to_numpy(float) * 1e3]),
        wp=np.concatenate([z, d["wp"].to_numpy(float) * 1e3]),
        use=np.concatenate([[True], d["use"].to_numpy(bool)]),
        w=np.concatenate([[1.0], d["w"].to_numpy(float)]),
    )


def _flag(s):
    """Tick-box column -> bool, treating blanks as switched on."""
    return s.map(lambda v: True if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v))


def aggregate_wells(wells, pressures):
    """Sum well cumulatives at the reservoir pressure survey dates.

    wells: well, date, gp, np, wp (cumulative per well); pressures: date, p.
    Each well's cumulative is interpolated linearly in time (zero before its first
    record, flat after its last).
    """
    import pandas as pd
    w = wells.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date", "well"])
    pr = pressures.copy()
    pr["date"] = pd.to_datetime(pr["date"], errors="coerce")
    pr["p"] = pd.to_numeric(pr["p"], errors="coerce")
    pr = pr.dropna(subset=["date", "p"]).sort_values("date")
    out = pd.DataFrame({"date": pr["date"].to_numpy(), "p": pr["p"].to_numpy(),
                        "gp": 0.0, "np": 0.0, "wp": 0.0})
    out["use"] = _flag(pr["use"]).to_numpy() if "use" in pr else True
    out["w"] = (pd.to_numeric(pr["w"], errors="coerce").fillna(1.0).to_numpy()
                if "w" in pr else 1.0)
    x = pr["date"].astype("int64").to_numpy(float)
    for _, g in w.groupby("well"):
        g = g.sort_values("date")
        xg = g["date"].astype("int64").to_numpy(float)
        for c in ("gp", "np", "wp"):
            y = pd.to_numeric(g[c], errors="coerce").fillna(0.0).to_numpy(float)
            first = y[0] if len(xg) == 1 else 0.0
            out[c] += np.interp(x, xg, y, left=first if len(xg) == 1 else 0.0, right=y[-1])
    return out


class Tank:
    def __init__(self, pvt: PVT, hist: History, G, pi, swi, cf, cw, bw=1.0, aquifer=None,
                 neighbour=None):
        """neighbour: (t_days, p_psia, T) - a connected reservoir with a known pressure history.
        Gas crosses at  q = T * (p_neighbour - p)  scf/day (into this tank when positive)."""
        self.pvt, self.h = pvt, hist
        self.G = G                      # scf wet gas
        self.pi, self.swi, self.cf, self.cw, self.bw = pi, swi, cf, cw, bw
        self.aq = aquifer or Aquifer()
        self.gpw = hist.gp + pvt.ge * hist.np_     # wet gas produced, scf
        self.bgi = float(pvt.bg(pi))
        self.zi = float(pvt.z(pi))
        self.ce = (cw * swi + cf) / (1.0 - swi)
        self.use, self.w = hist.use, hist.w
        # Pressures that drive the aquifer: switched-off surveys are bridged by
        # interpolating in time between their neighbours.
        self.p_drive = np.interp(hist.t, hist.t[self.use], hist.p[self.use])
        self.nb_T, self.nb_t, self.nb_p = 0.0, None, None
        if neighbour is not None and neighbour[2] and len(neighbour[0]) >= 1:
            o = np.argsort(neighbour[0])
            self.nb_t = np.asarray(neighbour[0], float)[o]
            self.nb_p = np.asarray(neighbour[1], float)[o]
            self.nb_T = float(neighbour[2])
        self.pn = self.neighbour_p(hist.t)
        self.gx_sim = np.zeros(len(hist.t))

    # ---- MBE terms (vectorised on pressure)
    def Eg(self, p):
        return self.pvt.bg(p) - self.bgi

    def Efw(self, p):
        return self.bgi * self.ce * (self.pi - np.asarray(p, float))

    def Et(self, p):
        return self.Eg(p) + self.Efw(p)

    def F(self, p=None):
        p = self.h.p if p is None else p
        return self.gpw * self.pvt.bg(p) + self.h.wp * self.bw

    def We_history(self):
        """Aquifer influx driven by the measured pressures."""
        return self.aq.influx(self.h.t, self.p_drive)

    # ---- neighbouring reservoir
    @property
    def has_neighbour(self):
        return self.nb_T > 0

    def neighbour_p(self, t):
        """Neighbour pressure at times t: interpolated between its surveys, held flat outside."""
        t = np.asarray(t, float)
        if self.nb_t is None:
            return np.full(t.shape, np.nan)
        return np.interp(t, self.nb_t, self.nb_p)

    def gx_step(self, i, p, gx, t=None, pn=None):
        """Cumulative gas received from the neighbour (scf) at step i."""
        if not self.has_neighbour:
            return 0.0
        t = self.h.t if t is None else t
        pn = self.pn if pn is None else pn
        # End-of-step pressure difference (implicit): stays stable when the link is strong
        # and the steps are as long as the gaps between surveys.
        return gx[i - 1] + self.nb_T * (pn[i] - p[i]) * (t[i] - t[i - 1])

    def gx_history(self):
        """Gas received from the neighbour, driven by the measured pressures (scf)."""
        gx = np.zeros(len(self.h.t))
        for i in range(1, len(gx)):
            gx[i] = self.gx_step(i, self.p_drive, gx)
        return gx

    def support_history(self):
        """Aquifer influx plus gas from the neighbour, rb, at the measured pressures."""
        return self.We_history() + self.gx_history() * self.pvt.bg(self.p_drive)

    # ---- forward prediction of pressure from production
    def simulate(self):
        """Solve the MBE for pressure at every history step. Returns (p_sim, We_sim)."""
        h, aq = self.h, self.aq
        n = len(h.t)
        aq.prepare(h.t)
        p = np.full(n, np.nan)
        We = np.zeros(n)
        gx = np.zeros(n)
        p[0] = self.pi
        plo = max(self.pvt.p_range[0], 14.7)
        for i in range(1, n):
            def res(pp):
                p[i] = pp
                we = aq.we_at(i, p, We)
                return (self.G * (self.Eg(pp) + self.Efw(pp)) + we
                        - (self.gpw[i] - self.gx_step(i, p, gx)) * self.pvt.bg(pp) - h.wp[i] * self.bw)
            hi = self.pi
            try:
                r_hi, r_lo = res(hi), res(plo)
                if not (np.isfinite(r_hi) and np.isfinite(r_lo)):
                    raise ValueError
                if r_hi >= 0:
                    sol = hi
                elif r_lo < 0:
                    raise ValueError            # tank cannot deliver this production
                else:
                    sol = brentq(res, plo, hi, xtol=1e-4)
            except ValueError:
                p[i:] = np.nan
                We[i:] = np.nan
                gx[i:] = np.nan
                break
            p[i] = sol
            We[i] = aq.we_at(i, p, We)
            gx[i] = self.gx_step(i, p, gx)
        self.gx_sim = gx
        return p, We

    # ---- drive indices
    def energy(self):
        p = self.p_drive
        we = self.We_history()
        a, b, c = self.G * self.Eg(p), self.G * self.Efw(p), np.maximum(we, 0.0)
        d = np.maximum(self.gx_history() * self.pvt.bg(p), 0.0)
        tot = a + b + c + d
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {"Gas expansion": a / tot, "Rock and connate water compressibility": b / tot,
                   "Water influx": c / tot}
            if self.has_neighbour:
                out["Gas from neighbouring reservoir"] = d / tot
            return out


# ----------------------------------------------------------------- graphical methods
METHODS = [
    "p/z",
    "p/z (overpressured)",
    "Havlena-Odeh (overpressured)",
    "Havlena-Odeh (water drive)",
    "(F-We)/Et (Cole)",
    "Roach (unknown compressibility)",
    "F/Et (Cole - no aquifer)",
]


def _fit(x, y, sel):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = sel & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.ptp(x[m]) == 0:
        return None
    r = stats.linregress(x[m], y[m])
    return {"slope": r.slope, "intercept": r.intercept, "r2": r.rvalue ** 2}


def graphical(tank: Tank, method, sel=None):
    """Return plot data for a diagnostic method.

    Output dict: x, y (history points, initial point excluded where undefined), fit,
    G (scf estimate, or None), kind ('line' | 'flat'), notes, plus axis keys.
    """
    h, pv = tank.h, tank.pvt
    p = h.p
    n = len(p)
    sel = (np.ones(n, bool) if sel is None else np.asarray(sel, bool)) & tank.use
    z = pv.z(p)
    pz, pzi = p / z, tank.pi / tank.zi
    dp = tank.pi - p
    F, Eg, Efw = tank.F(), tank.Eg(p), tank.Efw(p)
    Et = Eg + Efw
    We = tank.support_history() if (tank.aq.active or tank.has_neighbour) else np.zeros(n)
    nz = np.arange(n) > 0
    out = {"method": method, "G": None, "extra": {}, "mask": np.ones(n, bool)}
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "p/z":
            x, y = tank.gpw, pz
            f = _fit(x, y, sel)
            if f and f["slope"] < 0:
                out["G"] = -f["intercept"] / f["slope"]
            out.update(kind="line", xq="gas", yq="pz", xl="Cumulative wet gas produced",
                       yl="p/z (two-phase)", to_zero=True)
        elif method == "p/z (overpressured)":
            x, y = tank.gpw, pz * (1.0 - tank.ce * dp)
            f = _fit(x, y, sel)
            if f and f["slope"] < 0:
                out["G"] = -f["intercept"] / f["slope"]
            out.update(kind="line", xq="gas", yq="pz", xl="Cumulative wet gas produced",
                       yl="(p/z)·[1 − ce·(pi − p)]", to_zero=True)
        elif method == "Havlena-Odeh (overpressured)":
            x, y = tank.bgi * dp / Eg, (F - We) / Eg
            out["mask"] = nz
            f = _fit(x, y, sel & nz)
            if f:
                out["G"] = f["intercept"]
                if f["intercept"] > 0:
                    out["extra"]["ce"] = f["slope"] / f["intercept"]
            out.update(kind="line", xq="hox", yq="gas", xl="Bgi·Δp / Eg",
                       yl="(F − We) / Eg")
        elif method == "Havlena-Odeh (water drive)":
            x, y = We / Et, F / Et
            out["mask"] = nz
            f = _fit(x, y, sel & nz)
            if f:
                out["G"] = f["intercept"]
                out["extra"]["slope"] = f["slope"]
            else:  # no aquifer: x is all zero
                m = sel & nz & np.isfinite(y)
                if m.any():
                    out["G"] = float(np.mean(y[m]))
            out.update(kind="line", xq="gas", yq="gas", xl="We / Et", yl="F / Et")
        elif method in ("(F-We)/Et (Cole)", "F/Et (Cole - no aquifer)"):
            w = We if method.startswith("(F-We)") else 0.0
            x, y = tank.gpw, (F - w) / Et
            out["mask"] = nz
            f = _fit(x, y, sel & nz)
            m = sel & nz & np.isfinite(y)
            if m.any():
                out["G"] = float(np.mean(y[m]))
            out.update(kind="flat", xq="gas", yq="gas", xl="Cumulative wet gas produced",
                       yl="(F − We) / Et" if method.startswith("(F-We)") else "F / Et")
        else:  # Roach
            x = (tank.gpw / dp) * (pzi / pz)
            y = (pzi / pz - 1.0) / dp
            out["mask"] = nz
            f = _fit(x, y, sel & nz)
            if f and f["slope"] > 0:
                out["G"] = 1.0 / f["slope"]
                out["extra"]["ce"] = -f["intercept"]
            out.update(kind="line", xq="roachx", yq="roachy",
                       xl="(Gp / Δp)·(p/z)i / (p/z)", yl="[(p/z)i / (p/z) − 1] / Δp")
    out.update(x=x, y=y, fit=f, sel=sel)
    return out


# ----------------------------------------------------------------- energy-source detection
def detect_drive(tank: Tank, min_dp_frac=0.04):
    """Look for energy support beyond gas expansion using the Cole (no aquifer) plot.

    With no aquifer in the balance, F/Et = G + We/Et. For a volumetric reservoir it is a
    horizontal line at G; any systematic departure from horizontal means another energy
    source is acting. The shape gives the strength (Cole / Pletcher):
    rising = strong water drive, hump = moderate, declining = weak.
    The test fits a quadratic trend and compares it with a constant (F-test).
    """
    p = tank.h.p
    with np.errstate(divide="ignore", invalid="ignore"):
        y = tank.F() / tank.Et(p)
    m = ((np.arange(len(p)) > 0) & ((tank.pi - p) > min_dp_frac * tank.pi) & np.isfinite(y)
         & tank.use)
    res = {"status": "insufficient", "n": int(m.sum())}
    if m.sum() < 5:
        res["message"] = ("Not enough points with meaningful pressure depletion to diagnose the "
                          "drive mechanism (need at least five with more than "
                          f"{min_dp_frac:.0%} pressure drop).")
        return res
    x, yy = tank.gpw[m], y[m]
    xs = (x - x.min()) / max(np.ptp(x), 1e-30)
    n = len(xs)
    coef = np.polyfit(xs, yy, 2)
    fit = np.polyval(coef, xs)
    sse1, sse0 = float(np.sum((yy - fit) ** 2)), float(np.sum((yy - yy.mean()) ** 2))
    fstat = ((sse0 - sse1) / 2.0) / max(sse1 / (n - 3), 1e-300)
    pval = float(stats.f.sf(fstat, 2, n - 3))
    spread = float(np.ptp(fit) / abs(np.mean(yy)))
    imax = int(np.argmax(fit))
    shape = "rising" if imax >= n - 2 else "declining" if imax <= 1 else "hump"
    res.update(spread=spread, pvalue=pval, shape=shape, g_min=float(fit.min()),
               g_max=float(fit.max()), x=x, fit=fit)
    if spread > 0.05 and pval < 0.01:
        res["status"] = "aquifer"
        res["strength"] = {"rising": "strong", "hump": "moderate", "declining": "weak"}[shape]
        txt = {"rising": "keeps rising with production, the signature of a strong water drive",
               "hump": "rises and then turns over, the signature of a moderate water drive",
               "declining": "declines with production, the signature of a weak water drive "
                            "(the same shape appears if rock compressibility is underestimated "
                            "in an overpressured reservoir, so check that too)"}[shape]
        res["message"] = (f"F/Et is not constant: it varies by {spread:.0%} across the history and "
                          f"{txt}. The reservoir has an additional energy source, most likely an "
                          "aquifer.")
    else:
        res["status"] = "volumetric"
        res["message"] = ("F/Et is flat within the scatter of the data: the history is consistent "
                          "with a volumetric (depletion drive) reservoir. No aquifer is needed.")
    return res
