"""Synthetic oil case: undersaturated oil that falls below its bubble point, with a radial
aquifer. The history is produced by the engine itself from known 'true' values and then
perturbed with gauge noise."""
import numpy as np
import pandas as pd

from .aquifer import Aquifer
from .matbal import History
from .oil import OilPVT, OilRelPerm, OilState, OilTank, OilWell, correlation_table, oil_forecast

OIL_TRUTH = {"o_N": 60.0, "aq_reD": 4.0, "aq_k": 30.0}

OIL_FLUID = dict(o_api=32.0, o_sg=0.75, o_rsi=650.0, o_T=200.0, o_pb=None, o_pmax=5000.0)


def oil_defaults():
    return dict(OIL_FLUID, o_pi=4200.0, o_swi=0.2, o_cf=4e-6, o_cw=3e-6, o_bw=1.02, o_muw=0.35,
                o_phi=0.22, o_N=75.0, o_m=0.0,
                o_sorw=0.25, o_sorg=0.15, o_sgc=0.05, o_krw=0.3, o_kro=1.0, o_krg=0.8,
                o_nw=3.0, o_now=2.0, o_ng=2.0, o_nog=3.0,
                o_years=15.0, o_qtarget=None, o_qmin=200.0, o_wcmax=0.95, o_gormax=None,
                o_bofb=None, o_rsfb=None, o_step="Monthly",
                o_nb_on=False, o_nb_T=None, o_nb_fc="cutoff", o_nb_V=None,
                o_qwi=None, o_vrr=None, o_qgi=None, o_wellrp={})


def oil_sample_case(seed=11):
    """Returns (config, pvt table, pb, history df, wells df, pressures df, tests df, wellcfg df,
    start date)."""
    c = oil_defaults()
    tab, pb = correlation_table(c["o_api"], c["o_sg"], c["o_rsi"], c["o_T"], c["o_pmax"])
    pvt = OilPVT(tab, pb, c["o_T"])
    aq = Aquifer(model="veh", geometry="radial", ct=7e-6, k=OIL_TRUTH["aq_k"], phi=0.2, mu=0.35,
                 h=60.0, ro=4000.0, reD=OIL_TRUTH["aq_reD"], theta=180.0)
    h0 = History(np.array([0.0]), np.array([c["o_pi"]]), np.zeros(1), np.zeros(1), np.zeros(1))
    tank = OilTank(pvt, h0, OIL_TRUTH["o_N"] * 1e6, 0.0, c["o_pi"], c["o_swi"], c["o_cf"], c["o_cw"],
                   c["o_bw"], aq)
    rp = OilRelPerm(swc=c["o_swi"], krw_max=0.25, nw=2.5, krg_max=0.6, ng=2.2, sgc=0.06)
    state = OilState(tank, c["o_muw"])
    wells = [OilWell("P-1", J=6.0, min_bhp=1200.0), OilWell("P-2", J=4.5, min_bhp=1200.0),
             OilWell("P-3", J=3.0, min_bhp=1200.0)]
    df, _ = oil_forecast(tank, np.array([c["o_pi"]]), np.zeros(1), wells, rp, state, 10 * 365.25,
                         dt=30.4375, q_target=12000.0)
    start = pd.Timestamp("2015-01-01")
    pick = df.iloc[6::6].reset_index(drop=True)          # a survey every six months
    rng = np.random.default_rng(seed)
    dates = start + pd.to_timedelta(pick["t"], unit="D")
    hist = pd.DataFrame({"date": dates.dt.normalize(), "p": np.round(pick["p"] + rng.normal(0, 15, len(pick)), 0),
                         "gp": np.round(pick["gp"] / 1e6, 1), "np": np.round(pick["np"] / 1e3, 1),
                         "wp": np.round(pick["wp"] / 1e3, 1), "use": True, "w": 1.0})
    shares = {"P-1": 6.0 / 13.5, "P-2": 4.5 / 13.5, "P-3": 3.0 / 13.5}
    monthly = df.iloc[1:].copy()
    mdates = (start + pd.to_timedelta(monthly["t"], unit="D")).dt.normalize()
    wells_df = pd.concat([pd.DataFrame({"well": w, "date": mdates, "gp": np.round(monthly["gp"] / 1e6 * s, 2),
                                        "np": np.round(monthly["np"] / 1e3 * s, 2),
                                        "wp": np.round(monthly["wp"] / 1e3 * s, 2)})
                          for w, s in shares.items()], ignore_index=True)
    pres = hist[["date", "p", "use", "w"]].copy()
    # one test per well at the start, one after the bubble point is crossed
    tests = []
    from .oil import ipr_shape
    for w in wells:
        for k in (1, len(df) - 13):
            r = df.iloc[k]
            pr_ = float(df.iloc[k - 1]["p"])
            M_ = float(state.ratios(rp, pr_, float(r["sw"]), float(r["sg"]))[2])
            pwf = round(min(1800.0, 0.7 * pr_), 0)
            q = w.J * M_ * float(ipr_shape(pr_, pwf, pb))
            tests.append(dict(well=w.name, date=(start + pd.Timedelta(days=float(r["t"]))).normalize(),
                              ql=round(q, 0), pwf=pwf, pth=np.nan, pr=round(pr_, 0)))
    cfg = pd.DataFrame({"well": [w.name for w in wells], "tvd": 8000.0, "md": np.nan, "tid": 2.992,
                        "wht": 110.0, "min_thp": np.nan, "min_bhp": 1200.0, "qmax": np.nan})
    return c, tab, pb, hist, wells_df, pres, pd.DataFrame(tests), cfg, start.date()
