"""Synthetic demonstration case: retrograde condensate with a finite radial aquifer.

The history is generated with the engine itself from known "true" parameters and
then perturbed with gauge noise, so the workflow can be tested end to end.
"""
import numpy as np
import pandas as pd
from .pvt import PVT, correlation_table, wet_gas_gravity, water_props
from .matbal import History
from .model import make_tank

TRUTH = {"G": 250.0, "aq_reD": 5.0, "aq_k": 40.0}

FLUID = dict(sg=0.72, cgr_i=85.0, api=52.0, T=260.0, pd=5200.0, co2=0.02, h2s=0.0, n2=0.01,
             salinity=30000.0, z2="rayes", cvd_zd=0.0)


def default_config():
    bw, cw, mu = water_props(6500.0, FLUID["T"], FLUID["salinity"])
    c = dict(FLUID)
    c.update(pvt_pmax=7000.0,
             pi=6500.0, phi=0.18, swi=0.22, cf=4.0e-6, cw=round(cw, 8), bw=round(bw, 4),
             G=350.0,   # deliberately not the true value (roughly what p/z suggests)
             aq_model="none", aq_geom="radial", aq_inf=False,
             aq_ro=6000.0, aq_reD=4.0, aq_h=80.0, aq_theta=120.0, aq_k=100.0, aq_phi=0.20,
             aq_muw=round(mu, 3), aq_width=8000.0, aq_L=30000.0, aq_kvkh=0.1,
             aq_Wvol=500.0, aq_C=50.0)
    return c


def sample_pvt_table(c=None):
    c = c or default_config()
    return correlation_table(c["sg"], c["cgr_i"], c["api"], c["T"], c["pd"], c["pvt_pmax"],
                             c["co2"], c["h2s"], c["n2"], c["z2"])


BAD_SURVEYS = {19: 160.0, 22: 180.0}    # history row -> error added, psi


def sample_case(seed=7, bad=True):
    """Returns (config, pvt_table, reservoir_history_df, wells_df, pressures_df, start_date).

    With bad=True two surveys carry a gross error (BAD_SURVEYS) so the survey screening
    can be tried: both read high, as if the gauge sat above datum or the correction was wrong."""
    c = default_config()
    tab = sample_pvt_table(c)
    pvt = PVT.from_frame(tab, c["T"], c["api"], wet_gas_gravity(c["sg"], c["cgr_i"], c["api"]))
    start = pd.Timestamp("2014-01-01")
    dates = pd.date_range(start, periods=25, freq="6MS")
    t = (dates - start).days.to_numpy(float)
    yrs = t / 365.25
    rate = np.where(yrs <= 6.0, 52.0, 52.0 * np.exp(-0.16 * (yrs - 6.0)))   # MMscf/d dry gas
    true = dict(c, aq_model="veh", **TRUTH)
    n = len(t)
    gp, np_, wp, p = np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, c["pi"])
    for i in range(1, n):
        dg = 0.5 * (rate[i] + rate[i - 1]) * (t[i] - t[i - 1]) * 1e6
        gp[i] = gp[i - 1] + dg
        np_[i] = np_[i - 1] + dg / 1e6 * float(pvt.cgr(p[i - 1]))
        wp[i] = wp[i - 1] + (0.0 if yrs[i] < 5 else 60.0 * (yrs[i] - 5) ** 1.5 * (t[i] - t[i - 1]))
        h = History(t[:i + 1], p[:i + 1].copy(), gp[:i + 1], np_[:i + 1], wp[:i + 1])
        p[i] = make_tank(true, pvt, h).simulate()[0][i]
    rng = np.random.default_rng(seed)
    p_obs = p + rng.normal(0.0, 12.0, n)
    if bad:
        for i, e in BAD_SURVEYS.items():
            p_obs[i] += e
    hist = pd.DataFrame({"date": dates[1:], "p": np.round(p_obs[1:], 0),
                         "gp": np.round(gp[1:] / 1e6, 1), "np": np.round(np_[1:] / 1e3, 1),
                         "wp": np.round(wp[1:] / 1e3, 1)})
    # split into three wells for the by-well example
    shares = {"A-1": 0.45, "A-2": 0.35, "A-3": 0.20}
    wells = pd.concat([pd.DataFrame({"well": w, "date": hist["date"],
                                     "gp": np.round(hist["gp"] * s, 1),
                                     "np": np.round(hist["np"] * s, 1),
                                     "wp": np.round(hist["wp"] * s, 1)})
                       for w, s in shares.items()], ignore_index=True)
    hist["use"], hist["w"] = True, 1.0
    pressures = hist[["date", "p", "use", "w"]].copy()
    return c, tab, hist, wells, pressures, start.date()
