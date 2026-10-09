"""Run with:  python -m pytest tests   (or simply  python tests/test_engine.py)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbal.aquifer import WD, pD, Aquifer
from mbal.pvt import PVT, z_dak, wet_gas_gravity, cvd_two_phase_z
from mbal.matbal import build_history, graphical, detect_drive, METHODS, aggregate_wells
from mbal.model import make_tank
from mbal.regress import regress, pressure_rms, leave_one_out
from mbal.outliers import trend_screen, residual_screen, runs_test
from mbal.sample import sample_case, TRUTH, BAD_SURVEYS


def test_veh_infinite_radial():
    # van Everdingen & Hurst (1949) table values
    assert np.allclose(WD([1, 10, 100]), [1.569, 7.411, 43.01], rtol=2e-3)
    assert np.allclose(pD([1, 10, 100, 1000])[0], [0.802, 1.651, 2.723, 3.860], rtol=2e-3)


def test_finite_aquifers_reach_capacity():
    assert abs(WD([1e4], "radial", 5.0)[0] - 12.0) < 1e-2      # (reD^2 - 1)/2
    tD = np.array([0.01, 0.1, 0.5, 2.0])
    series = 1 - 8 / np.pi ** 2 * sum(np.exp(-n * n * np.pi ** 2 * tD / 4) / n ** 2
                                      for n in range(1, 400, 2))
    assert np.allclose(WD(tD, "linear"), series, rtol=1e-3)


def test_z_factor():
    assert abs(z_dak(2.0, 1.5) - 0.822) < 0.01     # Standing-Katz chart


def test_carter_tracy_and_fetkovich_track_veh():
    t = np.linspace(0, 3650, 21)
    p = 5000 - 1500 * (t / 3650) ** 0.8
    kw = dict(geometry="radial", ct=7e-6, k=50, phi=0.2, mu=0.3, h=60, ro=5000, reD=5, theta=360)
    veh = Aquifer(model="veh", **kw).influx(t, p)[-1]
    ct = Aquifer(model="carter_tracy", **kw).influx(t, p)[-1]
    fk = Aquifer(model="fetkovich", **kw).influx(t, p)[-1]
    assert abs(ct / veh - 1) < 0.06 and abs(fk / veh - 1) < 0.10


def _case(bad=False):
    c, tab, hist, wells, pres, start = sample_case(bad=bad)
    pvt = PVT.from_frame(tab, c["T"], c["api"], wet_gas_gravity(c["sg"], c["cgr_i"], c["api"]))
    return c, pvt, build_history(hist, start, c["pi"]), (hist, wells, pres)


def test_volumetric_round_trip():
    """With no aquifer, p/z and Cole must return the G used to generate the pressures."""
    c, pvt, h, _ = _case()
    c = dict(c, G=180.0, cf=0.0, cw=0.0)
    h.p = make_tank(c, pvt, h).simulate()[0]
    tank = make_tank(c, pvt, h)
    for m in ("p/z", "F/Et (Cole - no aquifer)", "Roach (unknown compressibility)"):
        assert abs(graphical(tank, m)["G"] / 180e9 - 1) < 2e-3, m
    assert detect_drive(tank)["status"] == "volumetric"


def test_aquifer_detected_and_regressed():
    c, pvt, h, _ = _case()
    assert detect_drive(make_tank(c, pvt, h))["status"] == "aquifer"
    c2 = dict(c, aq_model="veh")
    keys = ["G", "aq_reD", "aq_k"]

    def sim(v):
        return make_tank(dict(c2, **dict(zip(keys, v))), pvt, h).simulate()[0]

    r = regress(sim, h.p, [c2[k] for k in keys], [50, 1.5, 5], [1000, 20, 1000])
    assert r["rms"] < 25.0
    assert abs(r["x"][0] / TRUTH["G"] - 1) < 0.05
    c3 = dict(c2, **dict(zip(keys, r["x"])))
    cole = graphical(make_tank(c3, pvt, h), "(F-We)/Et (Cole)")
    assert abs(cole["G"] / (r["x"][0] * 1e9) - 1) < 0.05


def test_all_methods_run_and_wells_aggregate():
    c, pvt, h, (hist, wells, pres) = _case()
    tank = make_tank(dict(c, aq_model="fetkovich"), pvt, h)
    for m in METHODS:
        g = graphical(tank, m)
        assert len(g["x"]) == len(h.p)
    agg = aggregate_wells(wells, pres)
    assert np.allclose(agg["gp"], hist["gp"], rtol=1e-3, atol=0.3)


def test_survey_screening():
    c, pvt, h, _ = _case(bad=True)
    bad = sorted(BAD_SURVEYS)
    c2 = dict(c, aq_model="veh")
    keys, lo, hi = ["G", "aq_reD", "aq_k"], [50, 1.5, 5], [1000, 20, 1000]
    tank = make_tank(c2, pvt, h)

    def sim(v):
        return make_tank(dict(c2, **dict(zip(keys, v))), pvt, h).simulate()[0]

    # 1. model-free trend screen finds exactly the two bad surveys
    _, z = trend_screen(tank.gpw, h.p)
    assert sorted(np.flatnonzero(np.abs(z) > 3)) == bad
    # 2. ordinary least squares is dragged; the robust loss is not
    x0 = [c2[k] for k in keys]
    ols = regress(sim, h.p, x0, lo, hi)
    rob = regress(sim, h.p, x0, lo, hi, loss="soft_l1")
    assert abs(rob["x"][0] / TRUTH["G"] - 1) < 0.03
    assert abs(rob["x"][0] / TRUTH["G"] - 1) <= abs(ols["x"][0] / TRUTH["G"] - 1) + 1e-3
    # 3. residuals of the robust fit flag the same two surveys
    _, zr = residual_screen(rob["p"], h.p)
    assert sorted(np.flatnonzero(np.abs(zr) > 3)) == bad
    # 4. switching them off gives the clean answer with ordinary least squares
    w = h.w.copy(); w[bad] = 0.0
    off = regress(sim, h.p, x0, lo, hi, w)
    assert off["rms"] < 20 and abs(off["x"][0] / TRUTH["G"] - 1) < 0.03
    # 5. the single most influential survey on the ordinary fit is one of the bad ones
    infl = np.abs(leave_one_out(sim, h.p, ols["x"], lo, hi, h.w))[1:].max(axis=1)
    assert int(np.argmax(infl)) + 1 in bad


def test_runs_test_separates_model_error_from_scatter():
    c, pvt, h, _ = _case()
    good = make_tank(dict(c, aq_model="veh", **TRUTH), pvt, h).simulate()[0]
    assert not runs_test(h.p - good)["systematic"]
    none = make_tank(dict(c, G=300.0), pvt, h).simulate()[0]     # no aquifer: wrong model
    assert runs_test(h.p - none)["systematic"]


def test_switched_off_survey_is_ignored_everywhere():
    c, pvt, h, (hist, wells, pres) = _case(bad=True)
    hist = hist.copy()
    hist.loc[[i - 1 for i in BAD_SURVEYS], "use"] = False
    h2 = build_history(hist, "2014-01-01", c["pi"])
    assert (~h2.use).sum() == 2 and h2.w[~h2.use].sum() == 0
    tank = make_tank(dict(c, aq_model="veh", **TRUTH), pvt, h2)
    assert not graphical(tank, "(F-We)/Et (Cole)")["sel"][sorted(BAD_SURVEYS)].any()
    assert pressure_rms(tank.simulate()[0], h2.p, h2.w) < 20
    pres = pres.copy(); pres.loc[0, "use"] = False
    assert not aggregate_wells(wells, pres)["use"].iloc[0]


def test_cvd_two_phase_z():
    import pandas as pd
    # A two-phase cell built from first principles: constant volume, known moles in each phase.
    pd_, zd = 5000.0, 1.05
    p = np.array([6000.0, 5000.0, 4000.0, 3000.0, 2000.0, 1000.0])
    zg = np.array([1.14, 1.05, 0.93, 0.87, 0.86, 0.91])
    sl = np.array([0.0, 0.0, 12.0, 18.0, 17.0, 14.0])          # % of cell volume
    nl = np.array([0.0, 0.0, 0.10, 0.14, 0.13, 0.11])          # moles in liquid / initial moles
    ng = (p / zg) * (1 - sl / 100) / (pd_ / zd)                # moles in gas / initial moles
    gp = np.where(p >= pd_, 0.0, 1 - ng - nl) * 100
    out, zd_used, notes = cvd_two_phase_z(pd.DataFrame({"p": p, "sl": sl, "gp": gp, "zg": zg}), pd_)
    assert zd_used == zd and not notes
    assert np.allclose(out["z"], [1.14, 1.05] + list(p[2:] / ((pd_ / zd) * (ng + nl)[2:])))
    assert np.allclose(out["liq_mol"], nl * 100, atol=1e-9)
    # p/Z2 is a straight line against cumulative produced, which is what the p/z plot relies on
    assert np.allclose((out["p"] / out["z"])[1:], ((pd_ / zd) * (1 - out["gp"] / 100))[1:])
    # fractions entered instead of percent, and a liquid column that cannot be right, are reported
    _, _, n1 = cvd_two_phase_z(pd.DataFrame({"p": p, "sl": sl, "gp": gp, "zg": zg * 0.5}), pd_, zd)
    assert any("not consistent" in m for m in n1)
    # no dew point row: needs Zd, or interpolates it
    try:
        cvd_two_phase_z(pd.DataFrame({"p": p[2:], "gp": gp[2:]}), pd_)
        assert False
    except ValueError:
        pass
    o2, _, _ = cvd_two_phase_z(pd.DataFrame({"p": p[2:], "gp": gp[2:]}), pd_, zd)
    assert np.allclose(o2["z"], out["z"][2:])


def _sample_forecast_setup(aquifer=True):
    from mbal.pvt import GasZ
    from mbal.relperm import RelPerm, TankState
    from mbal.sample import sample_wells
    from mbal.wells import Well, fit_ipr, tubing_bhp
    c, tab, hist_df, wells, pres, start = sample_case(bad=False)
    pvt = PVT.from_frame(tab, c["T"], c["api"], wet_gas_gravity(c["sg"], c["cgr_i"], c["api"]))
    h = build_history(hist_df, start, c["pi"])
    tank = make_tank(dict(c, aq_model="veh", **TRUTH) if aquifer else dict(c, G=TRUTH["G"]), pvt, h)
    p, we = tank.simulate()
    rp = RelPerm(swc=c["swi"])
    state = TankState(tank, c["aq_muw"])
    gz = GasZ(pvt.sg_wet, c["co2"], c["h2s"], c["n2"])
    tests, cfg = sample_wells(hist_df, c)
    return c, pvt, h, tank, p, we, rp, state, gz, tests, cfg, hist_df


def test_tubing_and_inflow_fit():
    from mbal.pvt import GasZ, water_vapour
    from mbal.wells import Well, fit_ipr, tubing_bhp, well_rate
    gz = GasZ(0.7)
    w = Well("x", tvd=10000.0, tid=2.441, wht=120.0)
    static = tubing_bhp(2000.0, 0.0, gz, 220.0, w)
    # static column: p_bottom = p_top * exp(0.01875 * sg * TVD / (T z))
    zbar = gz(0.5 * (2000 + static), 170.0)
    assert abs(static - 2000.0 * np.exp(0.01875 * 0.7 * 10000 / (629.67 * zbar))) < 2.0
    flowing = tubing_bhp(2000.0, 10.0, gz, 220.0, w)
    assert 300 < flowing - static < 500                 # hand calculation: about 340-400 psi of friction
    assert tubing_bhp(2000.0, 20.0, gz, 220.0, w) > flowing
    # inflow fit recovers C and n from exact data, and one test falls back to n = 1
    x = np.array([1e6, 3e6, 6e6])
    C, n, _ = fit_ipr(2e-5 * x ** 0.8, x)
    assert abs(n - 0.8) < 1e-6 and abs(C / 2e-5 - 1) < 1e-6
    assert fit_ipr([5.0], [1e6])[1] == 1.0
    # rate at a tubing-head limit satisfies inflow and tubing together, and falls with pressure
    w.C, w.n, w.min_thp = C, n, 800.0
    q, pwf, _ = well_rate(w, 4000.0, 1.0, gz, 220.0)
    assert abs(q - C * (4000.0 ** 2 - pwf ** 2) ** n) < 1e-3
    assert abs(pwf - tubing_bhp(800.0, q, gz, 220.0, w)) < 1.0
    assert well_rate(w, 3000.0, 1.0, gz, 220.0)[0] < q
    assert well_rate(w, 900.0, 1.0, gz, 220.0)[0] == 0.0          # cannot lift against the limit
    assert well_rate(w, 4000.0, 0.5, gz, 220.0)[0] < q            # lower mobility, lower rate
    assert 0.5 < water_vapour(3000.0, 200.0) < 1.2                # McKetta-Wehe chart: about 0.8 STB/MMscf


def test_relperm_water_fit():
    from mbal.relperm import RelPerm, fit_water, water_history
    c, pvt, h, tank, p, we, rp, state, *_ = _sample_forecast_setup()
    sw = state.sw(p, we, h.wp)
    assert abs(sw[0] - c["swi"]) < 1e-9 and sw[-1] > sw[0] + 0.02     # the aquifer raises Sw
    krw, krg = rp.gas_water([c["swi"], 1 - rp.sgrw])
    assert krw[0] == 0 and krg[0] == rp.krg_max and krg[1] == 0 and krw[1] == rp.krw_max
    # make a water history from known curves, then recover them
    true = RelPerm(swc=c["swi"], krw_max=0.12, nw=2.2)
    h.wp[:] = 0.0
    for _ in range(6):                                   # Wp feeds back into Sw: iterate to consistency
        h.wp[:] = water_history(state, true, p, we)["wp"]
    got, info = fit_water(state, rp, p, we)
    assert info["ok"]
    fit = water_history(state, got, p, we)["wp"]
    assert abs(fit[-1] / h.wp[-1] - 1) < 0.02
    # with no aquifer the saturation does not move and the fit says so instead of inventing curves
    c2, pvt2, h2, tank2, p2, we2, rp2, state2, *_ = _sample_forecast_setup(aquifer=False)
    assert not fit_water(state2, rp2, p2, we2)[1]["ok"]


def test_forecast():
    import pandas as pd
    from mbal.forecast import forecast
    from mbal.wells import Well, fit_ipr, tubing_bhp
    c, pvt, h, tank, p, we, rp, state, gz, tests, cfg, hist_df = _sample_forecast_setup()
    wells = []
    for r in cfg.itertuples():
        w = Well(r.well, tvd=r.tvd, md=r.md, tid=r.tid, wht=r.wht, min_thp=r.min_thp)
        t = tests[(tests.well == r.well) & tests.pwf.notna()]
        m = np.interp((pd.to_datetime(t.date) - pd.Timestamp(hist_df.date.iloc[0])).dt.days + 181, h.t,
                      state.mobility(rp, p, state.sw(p, we, h.wp)))
        w.C, w.n, _ = fit_ipr(t.qg, m * (t.pr ** 2 - t.pwf ** 2))
        assert abs(w.n - 0.8) < 0.02                      # the sample tests were built with n = 0.8
        wells.append(w)
    df, s = forecast(tank, p, we, wells, rp, state, gz, 20 * 365.25, q_target=45.0, q_min=3.0)
    f = df[df.forecast]
    assert len(f) > 12
    assert np.allclose(df.p[:len(p)], p) and abs(df.gp.iloc[len(p) - 1] - h.gp[-1]) < 1
    assert f.qg.max() <= 45.0 + 1e-6 and f.qg.iloc[-1] < f.qg.iloc[0]      # plateau, then decline
    assert np.all(np.diff(df.gp) >= 0) and s["rf_wet"] < 1.0 and s["rf_wet"] > s["rf_hist"]
    # every forecast point honours the material balance
    i = len(df) - 1
    gpw = df.gp[i] + pvt.ge * df.np[i]
    lhs = tank.G * tank.Et(df.p[i]) + df.we[i]
    assert abs(lhs - gpw * pvt.bg(df.p[i]) - df.wp[i] * tank.bw) / lhs < 1e-4
    # well rates add up, and a tighter tubing-head limit recovers less
    assert np.allclose(f[[f"q_{w.name}" for w in wells]].sum(axis=1), f.qg)
    for w in wells:
        w.min_thp = 1500.0
    df2, s2 = forecast(tank, p, we, wells, rp, state, gz, 20 * 365.25, q_target=45.0, q_min=3.0)
    assert s2["gp_end"] < s["gp_end"]


def test_neighbouring_reservoir():
    from mbal.forecast import forecast
    c, pvt, h, tank0, p0, we0, rp, state, gz, tests, cfg, hist_df = _sample_forecast_setup(aquifer=False)
    base = dict(c, G=TRUTH["G"], nb_on=True)
    # no transmissibility, or a neighbour that is switched off: the closed tank
    assert np.allclose(make_tank(dict(base, nb_T=0.0), pvt, h, neighbour=(h.t, p0)).simulate()[0], p0)
    assert np.allclose(make_tank(dict(base, nb_on=False, nb_T=50.0), pvt, h,
                                 neighbour=(h.t, p0 + 500)).simulate()[0], p0)
    # a neighbour held at initial pressure feeds gas in and holds pressure up; one at low
    # pressure takes gas out
    hi = make_tank(dict(base, nb_T=20.0), pvt, h, neighbour=([0.0], [c["pi"]]))
    p_hi = hi.simulate()[0]
    assert np.all(p_hi[1:] > p0[1:]) and hi.gx_sim[-1] > 0
    lo = make_tank(dict(base, nb_T=0.5), pvt, h, neighbour=([0.0], [300.0]))
    p_lo = lo.simulate()[0]
    assert np.all(p_lo[1:] < p0[1:]) and lo.gx_sim[-1] < 0
    # the gas that crossed is exactly the integral of T * pressure difference
    dp = c["pi"] - p_hi
    assert abs(hi.gx_sim[-1] - 20e3 * np.sum(dp[1:] * np.diff(h.t))) < 1e-3 * hi.gx_sim[-1]
    # and the tank balances with it: G Et = (Gp - Gx) Bg + Wp Bw
    i = len(h.t) - 1
    lhs = hi.G * hi.Et(p_hi[i])
    assert abs(lhs - (hi.gpw[i] - hi.gx_sim[i]) * pvt.bg(p_hi[i]) - h.wp[i] * hi.bw) / lhs < 1e-5
    # an enormous transmissibility pins the tank to the neighbour
    pin = make_tank(dict(base, nb_T=1e5), pvt, h, neighbour=([0.0], [c["pi"] - 300.0])).simulate()[0]
    assert abs(pin[-1] - (c["pi"] - 300.0)) < 15.0
    # regression recovers G and the transmissibility from a history generated with them
    nb = (h.t, np.linspace(c["pi"], 3000.0, len(h.t)))
    true = make_tank(dict(base, nb_T=12.0), pvt, h, neighbour=nb).simulate()[0]
    sim = lambda x: make_tank(dict(base, G=x[0], nb_T=x[1]), pvt, h, neighbour=nb).simulate()[0]   # noqa: E731
    r = regress(sim, true, [400.0, 1.0], [50.0, 0.01], [2000.0, 500.0])
    assert abs(r["x"][0] / TRUTH["G"] - 1) < 0.02 and abs(r["x"][1] / 12.0 - 1) < 0.05
    # the forecast carries the exchange on, with the neighbour held at its last pressure
    from mbal.wells import Well
    w = Well("x", C=2e-5, n=0.8, min_bhp=500.0)
    df, s = forecast(hi, p_hi, hi.simulate()[1], [w], rp, state, gz, 5 * 365.25, nb_mode="constant")
    df0, s0 = forecast(tank0, p0, we0, [w], rp, state, gz, 5 * 365.25)
    assert df.gx.iloc[-1] > df.gx.iloc[len(h.t) - 1] and s["gp_end"] > s0["gp_end"]
    # a neighbour held at constant pressure is an unlimited source; a finite one is not
    lim = make_tank(dict(base, nb_T=5.0), pvt, h, neighbour=(h.t, np.full(len(h.t), c["pi"])))
    p_l, we_l = lim.simulate()
    w2 = Well("y", C=2e-5, n=0.8, min_bhp=300.0)
    long = 60 * 365.25
    dc, sc = forecast(lim, p_l, we_l, [w2], rp, state, gz, long, nb_mode="constant")
    small = 20e9
    dd, sd = forecast(lim, p_l, we_l, [w2], rp, state, gz, long, nb_mode="deplete", nb_G=small)
    dx, sx = forecast(lim, p_l, we_l, [w2], rp, state, gz, long, nb_mode="cutoff")
    gpw = lambda d_: d_.gp.iloc[-1] + pvt.ge * d_.np.iloc[-1]      # noqa: E731
    assert gpw(dc) > lim.G                                          # constant source: more than G
    assert gpw(dd) < lim.G + max(sd["gx_end"], 0) + 1e6             # never more than G + received
    assert sd["gx_end"] - lim.gx_sim[-1] < small                    # cannot take more than it holds
    assert abs(sx["gx_end"] - lim.gx_sim[-1]) < 1.0                 # cut off: nothing more crosses
    assert gpw(dx) < gpw(dd) < gpw(dc) and sd["rf_total"] < 1.0
    f = dd[dd.forecast]
    assert np.all(np.diff(f.pn.to_numpy()) <= 1e-6)                 # the neighbour depletes


def _tpd_text(fun, q, thp, wgr, gor, psig=True):
    """A Prosper-style .tpd file made from fun(thp, q, wgr, gor) -> (bhp, turner flag)."""
    off = 14.696 if psig else 0.0
    j = lambda v: ", ".join(f"{x:.6g}" for x in v)   # noqa: E731
    out = ["# TPD test", "TPDData", "2", "2,0,2,0", "0,0", "58.5,0.7,3.4e+35,3.4e+35,30,9000", "3",
           f"{len(q)}, {len(wgr)}, {len(gor)}, {len(thp)}", "2", "5000, 5105", "4002,18,15,27",
           j(q), j(wgr), j(gor), j(np.asarray(thp) - off)]
    for a in thp:
        for b in gor:
            for c in wgr:
                for d in q:
                    bhp, tur = fun(a, d, c, b)
                    out.append(f"{bhp - off:.4f}, {int(tur)}")
    return "\r\n".join(out)


def test_lift_curves():
    from mbal.pvt import GasZ
    from mbal.vlp import read_tpd, LiftTable
    from mbal.wells import Well, tubing_bhp, well_rate
    gz = GasZ(0.7)
    pipe = Well("p", tvd=9000.0, tid=2.441, wht=120.0)
    q = np.geomspace(0.1, 30, 15)
    thp, wgr, gor = [100.0, 600.0, 1100.0, 1600.0], [0.0, 50.0, 200.0], [5000.0, 20000.0, 100000.0]

    # 1. a table made from the built-in tubing calculation reads back exactly at grid points
    #    and closely in between (linear in THP and WGR, logarithmic in rate and GOR)
    def fun(a, d, c, b):
        return tubing_bhp(a, d, gz, 200.0, pipe) * (1 + c / 1000.0) + 2e5 / b, d < 1.0
    t = read_tpd(_tpd_text(fun, q, thp, wgr, gor), "x.tpd")
    assert t.bhp.shape == (4, 3, 3, 15) and np.allclose(t.thp, thp) and t.info["pressure_units"] == "psig"
    assert abs(t.bhp_at(600.0, q[7], 50.0, 1e6 / 20000.0) - fun(600.0, q[7], 50.0, 20000.0)[0]) < 0.01
    mid = t.bhp_at(850.0, 5.0, 25.0, 1e6 / 20000.0)
    assert abs(mid / fun(850.0, 5.0, 25.0, 20000.0)[0] - 1) < 0.03     # coarse 500 psi THP grid
    assert t.loading_at(600.0, 0.5, 0.0, 20.0) and not t.loading_at(600.0, 5.0, 0.0, 20.0)
    assert t.bhp_at(50.0, 5.0, 0.0, 20.0) == t.bhp_at(100.0, 5.0, 0.0, 20.0)    # held at the edge
    assert abs(LiftTable.from_dict(t.to_dict()).bhp_at(900, 3, 10, 30) - t.bhp_at(900, 3, 10, 30)) < 0.05
    # a well using the table reproduces the built-in result (WGR 0, high GOR term small)
    w_tab = Well("t", C=2e-5, n=0.8, min_thp=600.0, lift=t)
    w_pipe = Well("p", C=2e-5, n=0.8, min_thp=600.0, tvd=9000.0, tid=2.441, wht=120.0)
    q_tab = well_rate(w_tab, 3500.0, 1.0, gz, 200.0, cgr=10.0)[0]
    q_pipe = well_rate(w_pipe, 3500.0, 1.0, gz, 200.0)[0]
    assert abs(q_tab / q_pipe - 1) < 0.03

    # 2. a lift curve that turns up at low rate (liquid loading): the well flows at the
    #    high-rate crossing, and dies when the inflow no longer reaches the curve
    def ushape(a, d, c, b):
        return a + 900.0 + 1500.0 / d + 4.0 * d * d, d < 3.0
    u = read_tpd(_tpd_text(ushape, q, thp, wgr, gor), "u.tpd")
    w = Well("u", C=1e-4, n=0.8, min_thp=600.0, lift=u)
    qq, pwf, load = well_rate(w, 3000.0, 1.0, gz, 200.0, cgr=20.0)
    vl = lambda x: u.bhp_at(600.0, x, 0.0, 20.0)                                   # noqa: E731
    assert qq > 3.0 and abs(pwf - vl(qq)) < 0.5 and not load
    assert abs(qq - w.C * (3000.0 ** 2 - pwf ** 2) ** w.n) < 1e-3
    assert well_rate(w, 1400.0, 1.0, gz, 200.0, cgr=20.0)[0] == 0.0
    # a weak well whose operating point is below the Turner rate is flagged, and shut in on request
    u2 = read_tpd(_tpd_text(lambda a, d, c, b: (ushape(a, d, c, b)[0], d < 6.0), q, thp, wgr, gor))
    w2 = Well("u2", C=2.5e-6, n=1.0, min_thp=100.0, lift=u2)
    q2, _, l2 = well_rate(w2, 1900.0, 1.0, gz, 200.0, cgr=20.0)
    assert 0 < q2 < 6.0 and l2
    assert not well_rate(Well("u3", C=1e-5, n=1.0, min_thp=100.0, lift=u2), 1900.0, 1.0, gz, 200.0,
                         cgr=20.0)[2]
    assert well_rate(w2, 1900.0, 1.0, gz, 200.0, cgr=20.0, stop_loading=True)[0] == 0.0



def test_cce_two_phase_z():
    import pandas as pd
    from mbal.pvt import cce_two_phase_z
    # Oredo-9 A8.0 report (CCE at 180 F): Zd = 0.952 at 4224 psia
    d = pd.DataFrame({"p": [5515, 4835, 4224, 3475, 2427, 1705],
                      "vr": [0.8824, 0.9373, 1.0, 1.1249, 1.4811, 2.0752]})
    out, notes = cce_two_phase_z(d, 4224.0, 0.952)
    assert not notes
    # above the dew point the formula must give back the reported single-phase Z
    assert abs(out.z.iloc[0] - 1.097) < 0.002 and abs(out.z.iloc[1] - 1.021) < 0.002
    assert out.z.iloc[2] == 0.952
    assert np.allclose(out.z, 0.952 * out.p / 4224.0 * out.vr)
    bad = cce_two_phase_z(d.assign(vr=d.vr * 1.02), 4224.0, 0.952)[1]
    assert any("not 1" in m for m in bad)



def test_fractional_flow_match():
    from mbal.relperm import RelPerm, field_points, fit_fw, fw_model, water_history, wgr_from_fw, fit_yield
    c, pvt, h, tank, p, we, rp, state, *_ = _sample_forecast_setup()
    true = RelPerm(swc=c["swi"], krw_max=0.12, nw=2.2, sgrw=0.3)
    for _ in range(6):
        h.wp[:] = water_history(state, true, p, we)["wp"]
    pts = field_points(state, p, we)
    assert len(pts) == len(h.t) - 1 and (pts.fw > 0).sum() > 5
    # water cut and water-gas ratio convert back and forth
    assert np.allclose(wgr_from_fw(state, pts.p, pts.fw), pts.free, rtol=1e-6)
    new, info = fit_fw(state, rp, pts, ["rp_krw", "rp_nw", "rp_sgrw"])
    assert info["ok"] and info["rms"] < 1e-4
    assert abs(new.krw_max / 0.12 - 1) < 0.05 and abs(new.nw - 2.2) < 0.1 and abs(new.sgrw - 0.3) < 0.01
    # no free water: refuses
    h.wp[:] = water_history(state, true, p, we)["wp_vap"]
    assert not fit_fw(state, rp, field_points(state, p, we), ["rp_krw"])[1]["ok"]
    # yield: a straight line through the produced condensate-gas ratio
    fy = fit_yield(pts)
    assert fy is not None and fy[2] < fy[3]


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
