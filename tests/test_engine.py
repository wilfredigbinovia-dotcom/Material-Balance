"""Run with:  python -m pytest tests   (or simply  python tests/test_engine.py)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbal.aquifer import WD, pD, Aquifer
from mbal.pvt import PVT, z_dak, wet_gas_gravity
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


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
