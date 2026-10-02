"""
Streamlit front end for resmb.py.

Run:  streamlit run app.py      (resmb.py must be in the same folder)
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from resmb import (Aquifer, BlackOilPVT, Connection, MultiTank, Tank, build_history, common_origin,
                   gas_mbe, oil_mbe, reservoir_table, standardise_columns)

st.set_page_config(page_title="Reservoir Material Balance", layout="wide")

BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a888a"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e8ea", "axes.edgecolor": "#c5cdd0", "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})

# -----------------------------------------------------------------------------
# Sample data
# -----------------------------------------------------------------------------
OIL_COLS = ["t", "p", "Np", "Gp", "Wp", "Winj", "Ginj", "Bo", "Rs", "Bg", "Bw"]
OIL_GASCAP = pd.DataFrame([
    [0, 3000, 0, 0, 0, 0, 0, 1.315, 510, 0.000953, 1.02],
    [365, 2850, 3.7423, 2395.1, 0, 0, 0, 1.3035, 487, 0.001009, 1.02],
    [730, 2700, 7.1423, 5642.4, 0.02, 0, 0, 1.292, 464, 0.001071, 1.02],
    [1095, 2550, 10.1057, 9701.5, 0.06, 0, 0, 1.2805, 441, 0.001142, 1.02],
    [1460, 2400, 12.8236, 14490.7, 0.11, 0, 0, 1.269, 418, 0.001221, 1.02],
    [1825, 2250, 15.3133, 19907.3, 0.17, 0, 0, 1.2575, 395, 0.001312, 1.02],
    [2190, 2100, 17.3765, 25717.3, 0.24, 0, 0, 1.246, 372, 0.001416, 1.02],
    [2555, 1950, 19.4235, 32243.1, 0.32, 0, 0, 1.2345, 349, 0.001537, 1.02]], columns=OIL_COLS, dtype=float)
OIL_WATERDRIVE = pd.DataFrame([
    [0, 3000, 0, 0, 0, 0, 0, 1.3394, 600, 0.0009644, 1.034],
    [182, 2800, 1.0861, 651.6, 0, 0, 0, 1.3436, 600, 0.001026, 1.034],
    [364, 2650, 2.9163, 1749.8, 0.05, 0, 0, 1.347, 600, 0.001081, 1.035],
    [546, 2500, 5.2994, 3769, 0.15, 0, 0, 1.3389, 580.7, 0.001143, 1.035],
    [728, 2380, 7.3968, 6680.9, 0.3, 0, 0, 1.322, 547.6, 0.001199, 1.035],
    [910, 2280, 9.0032, 9572.4, 0.5, 0, 0, 1.3081, 520.3, 0.001252, 1.036],
    [1092, 2200, 10.0845, 12012.8, 0.8, 0, 0, 1.2971, 498.6, 0.001298, 1.036],
    [1274, 2130, 10.9346, 14250.1, 1.1, 0, 0, 1.2877, 479.8, 0.001342, 1.036],
    [1456, 2070, 11.5714, 16190.9, 1.4, 0, 0, 1.2796, 463.7, 0.001382, 1.036]], columns=OIL_COLS, dtype=float)
GAS_COLS = ["t", "p", "z", "Gp", "Wp", "Bw"]
GAS_SAMPLE = pd.DataFrame([
    [0, 4200, 0.92, 0, 0, 1.02], [180, 3920, 0.905, 3.05, 0, 1.02], [365, 3640, 0.892, 6.371, 0, 1.02],
    [545, 3360, 0.881, 9.819, 0, 1.02], [730, 3080, 0.872, 13.493, 0, 1.02],
    [910, 2800, 0.866, 17.48, 0, 1.02], [1095, 2520, 0.862, 21.616, 0, 1.02],
    [1275, 2240, 0.861, 25.894, 0, 1.02]], columns=GAS_COLS, dtype=float)
AQ_TANK = ["none", "pot", "fetkovich", "veh_radial", "veh_linear"]


def mt_sample():
    tanks = pd.DataFrame([
        dict(name="Main block", fluid="oil", in_place=75.0, m=0.0, pi=3200.0, Swi=0.22, cf=4e-6,
             aquifer="fetkovich", C_or_B=None, Wei=9.0, J=10.0, td_per_day=None, reD=None,
             fit_in_place=True, fit_aquifer=True),
        dict(name="East block", fluid="oil", in_place=40.0, m=0.0, pi=3200.0, Swi=0.22, cf=4e-6,
             aquifer="none", C_or_B=None, Wei=None, J=None, td_per_day=None, reD=None,
             fit_in_place=True, fit_aquifer=False)])
    for c in ("C_or_B", "Wei", "J", "td_per_day", "reD"):
        tanks[c] = tanks[c].astype(float)
    conns = pd.DataFrame([dict(a="Main block", b="East block", T=3.0, fit=True)])
    t = [365, 730, 1095, 1460, 1825, 2190]
    prod = pd.concat([
        pd.DataFrame(dict(tank="Main block", t=t, Np=[2.2, 4.5, 6.6, 8.4, 10, 11.4],
                          Gp=[1320, 2750, 4300, 6100, 8000, 9900], Wp=[0, 0.05, 0.2, 0.5, 0.9, 1.4],
                          p_obs=[2619, 2475, 2321, 2183, 2068, 1948])),
        pd.DataFrame(dict(tank="East block", t=t, Np=[0.5, 1.1, 1.6, 2.0, 2.4, 2.7],
                          Gp=[300, 680, 1050, 1400, 1750, 2050], Wp=[0.0] * 6,
                          p_obs=[2600, 2471, 2370, 2270, 2177, 2091]))], ignore_index=True).astype(
        {"t": float, "Np": float, "Gp": float, "Wp": float, "p_obs": float})
    return tanks, conns, prod


# -----------------------------------------------------------------------------
# Session state. Keys starting with "w_" are widget values kept across pages.
# -----------------------------------------------------------------------------
AQ_DEFAULTS = dict(model="None", geometry="radial", k=150.0, phi=0.2, mu=0.4, ct=7e-6, h=60.0,
                   ro=4000.0, reD=8.0, theta=180.0, L=20000.0, w=6000.0, tune=True)
DEFAULTS = {
    "w_oil_Swi": 0.2, "w_oil_cw": 3e-6, "w_oil_cf": 4e-6, "w_oil_m": 0.3, "w_oil_fitm": False,
    "w_gas_T": 220.0, "w_gas_Swi": 0.25, "w_gas_cw": 3e-6, "w_gas_cf": 0.0, "w_gas_pza": 500.0,
    "w_pvt_api": 35.0, "w_pvt_gg": 0.75, "w_pvt_T": 200.0, "w_pvt_known": "Rsb", "w_pvt_rsb": 600.0,
    "w_pvt_pb": 2500.0, "w_pvt_corr": "standing", "w_pvt_pmin": 200.0, "w_pvt_pmax": 4000.0,
    "w_pvt_step": 200.0, "w_mt_dt": 30.0, "w_mt_cw": 3e-6,
    "w_oil_src": "Type in or use a sample", "w_gas_src": "Type in or use a sample",
    "w_mt_src": "By reservoir (one file)",
    **{f"w_{p}_aq_{k}": v for p in ("oil", "gas") for k, v in AQ_DEFAULTS.items()},
}
ss = st.session_state
for k, v in DEFAULTS.items():
    ss.setdefault(k, v)
for k in list(ss.keys()):                 # re-assign so values survive when a page is not shown
    if k.startswith("w_"):
        ss[k] = ss[k]
if "oil_df" not in ss:
    ss.oil_df, ss.gas_df = OIL_GASCAP.copy(), GAS_SAMPLE.copy()
    ss.mt_tanks, ss.mt_conns, ss.mt_prod = mt_sample()
    ss.ver = 0                            # bumping this resets the table editors
    ss.mt_msg = ""


def set_table(name, df):
    ss[name] = df.reset_index(drop=True)
    ss.ver += 1


# -----------------------------------------------------------------------------
# Shared pieces
# -----------------------------------------------------------------------------
def sci(label, key, **kw):
    return st.number_input(label, key=key, format="%.2e", step=1e-6, min_value=0.0, **kw)


def make_pvt():
    known_rsb = ss.w_pvt_known == "Rsb"
    return BlackOilPVT(api=ss.w_pvt_api, gas_gravity=ss.w_pvt_gg, temp_f=ss.w_pvt_T,
                       rsb=ss.w_pvt_rsb if known_rsb else None,
                       pb=None if known_rsb else ss.w_pvt_pb, correlation=ss.w_pvt_corr)


def pvt_inputs():
    st.subheader("Fluid description")
    st.number_input("Oil gravity, °API", key="w_pvt_api", step=1.0)
    st.number_input("Gas gravity (air = 1)", key="w_pvt_gg", step=0.01)
    st.number_input("Temperature, °F", key="w_pvt_T", step=5.0)
    st.radio("Known value", ["Rsb", "Bubble point"], key="w_pvt_known", horizontal=True)
    if ss.w_pvt_known == "Rsb":
        st.number_input("Rsb, scf/STB", key="w_pvt_rsb", step=10.0)
    else:
        st.number_input("Bubble point, psia", key="w_pvt_pb", step=50.0)
    st.selectbox("pb, Rs, Bo correlation", ["standing", "vasquez-beggs", "glaso"], key="w_pvt_corr",
                 format_func=lambda s: {"standing": "Standing (1947)", "vasquez-beggs": "Vasquez–Beggs (1980)",
                                        "glaso": "Glasø (1980)"}[s])


def aquifer_inputs(prefix):
    """Sidebar aquifer block. Returns an Aquifer or None."""
    key = lambda k: f"w_{prefix}_aq_{k}"
    st.subheader("Aquifer")
    model = st.selectbox("Model", ["None", "Pot", "Schilthuis steady state", "Van Everdingen–Hurst", "Fetkovich"],
                         key=key("model"))
    code = {"None": None, "Pot": "pot", "Schilthuis steady state": "steady",
            "Van Everdingen–Hurst": "veh", "Fetkovich": "fetkovich"}[model]
    if code in (None, "pot", "steady"):
        return Aquifer(model=code) if code else None
    geo = st.radio("Geometry", ["radial", "linear"], key=key("geometry"), horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("k, md", key=key("k"), min_value=0.0, step=10.0)
        st.number_input("μw, cp", key=key("mu"), min_value=0.0, step=0.05)
        st.number_input("h, ft", key=key("h"), min_value=0.0, step=5.0)
    with c2:
        st.number_input("φ, fraction", key=key("phi"), min_value=0.0, step=0.01)
        sci("ct, 1/psi", key("ct"))
    if geo == "radial":
        c1, c2 = st.columns(2)
        c1.number_input("ro, ft", key=key("ro"), min_value=0.0, step=100.0)
        c2.number_input("ra/ro (0 = infinite)", key=key("reD"), min_value=0.0, step=1.0)
        st.number_input("Encroachment angle θ, degrees", key=key("theta"), min_value=0.0, max_value=360.0, step=10.0)
    else:
        c1, c2 = st.columns(2)
        c1.number_input("Length L, ft", key=key("L"), min_value=0.0, step=500.0)
        c2.number_input("Width w, ft", key=key("w"), min_value=0.0, step=500.0)
    st.checkbox("Tune time constant to data" if code == "veh" else "Tune Wei and J to data", key=key("tune"))
    g = lambda k: ss[key(k)]
    return Aquifer(model=code, geometry=geo, k=g("k"), phi=g("phi"), mu_w=g("mu"), ct=g("ct"), h=g("h"),
                   ro=g("ro"), reD=g("reD") if g("reD") > 1 else np.inf, theta=g("theta") or 360.0,
                   L=g("L"), w=g("w"), tune=g("tune"))


def aquifer_metrics(info):
    if not info:
        return []
    m = info["model"]
    if m == "pot":
        return [("Aquifer constant C", f"{info['C']:,.0f} rb/psi", None)]
    if m == "steady":
        return [("Aquifer constant C", f"{info['C']:,.1f} rb/(psi·day)", None)]
    if m == "veh":
        return [("Aquifer constant B", f"{info['B']:,.0f} rb/psi", f"inputs give {info['B_from_inputs']:,.0f}"),
                ("Implied aquifer k", f"{info['implied_k']:,.0f} md",
                 f"time constant ×{info['time_constant_multiplier']:.2f}")]
    return [("Encroachable water Wei", f"{info['Wei'] / 1e6:,.2f} MMbbl",
             f"inputs give {info['Wei_from_inputs'] / 1e6:,.2f}"),
            ("Aquifer productivity J", f"{info['J']:,.1f} rb/(d·psi)", f"inputs give {info['J_from_inputs']:,.1f}")]


def show_metrics(items):
    cols = st.columns(len(items))
    for c, (label, value, note) in zip(cols, items):
        c.metric(label, value)
        if note:
            c.caption(note)


OIL_UNITS = {"STB": 1e-6, "MSTB": 1e-3, "MMSTB": 1.0}                    # to MMSTB
GAS_UNITS = {"scf": 1e-6, "Mscf": 1e-3, "MMscf": 1.0, "Bscf": 1e3}        # to MMscf
GROUPS = {"Each survey date": None, "Month": "M", "Quarter": "Q", "Year": "Y"}
FILE_TYPES = ["csv", "xlsx", "xls"]


def read_any(up):
    up.seek(0)
    return pd.read_excel(up) if up.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(up)


@st.cache_data
def well_templates():
    """Example well files: three wells that add up to the gas-cap sample."""
    dates = pd.date_range("2015-01-31", periods=84, freq="ME")
    start = pd.Timestamp("2015-01-01")
    days = (dates - start).days.to_numpy(dtype=float)
    cum = {c: np.interp(days, OIL_GASCAP.t, OIL_GASCAP[c]) for c in ("Np", "Gp", "Wp")}
    rows = []
    for k, d in enumerate(dates):
        share = {"OB-1": 0.5, "OB-2": 0.3, "OB-3": 0.2} if k >= 12 else {"OB-1": 0.6, "OB-2": 0.4}
        prev = {c: (cum[c][k - 1] if k else 0.0) for c in cum}
        for w, f in share.items():
            rows.append(dict(reservoir="Sand A", well=w, date=d.date().isoformat(),
                             oil=round((cum["Np"][k] - prev["Np"]) * f * 1e6),          # STB
                             gas=round((cum["Gp"][k] - prev["Gp"]) * f * 1e3),          # Mscf
                             water=round((cum["Wp"][k] - prev["Wp"]) * f * 1e6)))       # STB
    pres = []
    for t, p in zip(OIL_GASCAP.t, OIL_GASCAP.p):
        d = (start + pd.Timedelta(days=float(t))).date().isoformat()
        pres += [dict(reservoir="Sand A", well="OB-1", date=d, pressure=p + 6),
                 dict(reservoir="Sand A", well="OB-2", date=d, pressure=p - 6)]
    return pd.DataFrame(rows), pd.DataFrame(pres)


def well_import(prefix, single=True):
    """Upload widgets and options for per-well data. Returns the inputs for build_history, or None."""
    tp, ts = well_templates()
    c1, c2 = st.columns(2)
    up1 = c1.file_uploader("Well production (CSV or Excel)", type=FILE_TYPES, key=f"{prefix}_wprod")
    c1.caption("Columns: well, date (or t in days), oil, gas, water. Optional: reservoir, winj, ginj.")
    c1.download_button("Example production file", tp.to_csv(index=False), "example_well_production.csv",
                       "text/csv", key=f"{prefix}_tp")
    up2 = c2.file_uploader("Pressure surveys (CSV or Excel)", type=FILE_TYPES, key=f"{prefix}_wpres")
    c2.caption("Columns: date (or t in days), pressure. Optional: well, reservoir.")
    c2.download_button("Example pressure file", ts.to_csv(index=False), "example_pressure_surveys.csv",
                       "text/csv", key=f"{prefix}_ts")
    if up1 is None or up2 is None:
        return None
    try:
        prod, pres = standardise_columns(read_any(up1)), standardise_columns(read_any(up2))
    except Exception as e:                                       # unreadable file
        st.error(f"Could not read the file: {e}")
        return None
    o1, o2, o3, o4 = st.columns(4)
    volumes = o1.selectbox("Volumes are", ["Per period (e.g. monthly)", "Cumulative per well"], key=f"{prefix}_vol")
    pdates = o2.selectbox("Each date marks the period", ["End", "Start"], key=f"{prefix}_pd",
                          disabled=volumes.startswith("Cum"))
    oil_u = o3.selectbox("Oil and water unit", list(OIL_UNITS), key=f"{prefix}_ou")
    gas_u = o4.selectbox("Gas unit", list(GAS_UNITS), index=1, key=f"{prefix}_gu")
    o1, o2, o3 = st.columns(3)
    group = o1.selectbox("Average pressure surveys by", list(GROUPS), key=f"{prefix}_grp")
    dayfirst = o2.selectbox("Ambiguous dates are", ["day/month/year", "month/day/year"], key=f"{prefix}_df")
    out = dict(prod=prod, pres=pres, fo=OIL_UNITS[oil_u], fg=GAS_UNITS[gas_u],
               kw=dict(volumes="period" if volumes.startswith("Per") else "cumulative",
                       period_dates=pdates.lower(), group=GROUPS[group], dayfirst=dayfirst.startswith("day")))
    if single:
        pi = o3.number_input("Initial pressure, psia (0 = first survey)", min_value=0.0, value=0.0, step=50.0,
                             key=f"{prefix}_pi")
        sub = prod
        if "reservoir" in prod:
            names = sorted(prod["reservoir"].dropna().astype(str).unique())
            res = st.selectbox("Reservoir", names, key=f"{prefix}_res")
            out["kw"]["reservoir"] = res
            sub = prod[prod["reservoir"].astype(str) == res]
        if "well" in sub:
            names = sorted(sub["well"].dropna().astype(str).unique())
            out["kw"]["wells"] = st.multiselect("Wells to include", names, default=names, key=f"{prefix}_wells")
        out["kw"]["initial_pressure"] = pi or None
    return out


def history_to_table(h, kind, fo, fg):
    """Convert a build_history result to the oil or gas table, with PVT from the correlations page."""
    d = h.copy()
    for c in ("Np", "Wp", "Winj"):
        d[c] *= fo
    for c in ("Gp", "Ginj"):
        d[c] *= fg
    note = ""
    try:
        tab = make_pvt().table(d.p)
        if kind == "oil":
            for c in ("Bo", "Rs", "Bg", "Bw"):
                d[c] = tab[c].values
            note = "Bo, Rs, Bg and Bw come from the PVT correlations page; replace them with lab values if you have them."
        else:
            d["Gp"], d["z"], d["Bw"] = d["Gp"] / 1000.0, tab.z.values, tab.Bw.values
            ss.w_gas_T = float(ss.w_pvt_T)
            note = "z and Bw come from the PVT correlations page, and the temperature was set to match it."
    except ValueError:
        if kind == "gas":
            d["Gp"] = d["Gp"] / 1000.0
        note = "PVT columns are empty: fill them in, or set up the PVT correlations page and use the fill button."
    cols = OIL_COLS if kind == "oil" else GAS_COLS
    for c in cols:
        if c not in d:
            d[c] = np.nan
    return d[cols], note


def data_tools(name, columns, samples, kind):
    """Bring data into a single-tank table: samples, a reservoir file, or per-well files."""
    src = st.radio("Bring in data", ["Type in or use a sample", "By reservoir (one file)", "By well (two files)"],
                   horizontal=True, key=f"w_{kind}_src")
    if src.startswith("Type"):
        cols = st.columns(len(samples) + 3)
        for c, (label, df) in zip(cols, samples.items()):
            if c.button(label, key=f"{name}_{label}"):
                set_table(name, df.copy())
                ss[f"{name}_note"] = ""
                st.rerun()
    elif src.startswith("By reservoir"):
        up = st.file_uploader("Reservoir history (CSV or Excel)", type=FILE_TYPES, key=f"{name}_up")
        st.caption("Columns: t (days) or date, then " + ", ".join(columns[1:]) + ". Cumulative volumes, first row "
                   "initial conditions. Add a 'reservoir' column to keep several reservoirs in one file.")
        if up is not None:
            try:
                raw = standardise_columns(read_any(up), extra=columns)
                c1, c2 = st.columns(2)
                res = None
                if "reservoir" in raw:
                    res = c1.selectbox("Reservoir", sorted(raw["reservoir"].dropna().astype(str).unique()),
                                       key=f"{name}_res")
                dayfirst = c2.selectbox("Ambiguous dates are", ["day/month/year", "month/day/year"],
                                        key=f"{name}_rdf").startswith("day")
                if st.button("Load into table", type="primary", key=f"{name}_load"):
                    if "t" not in raw and "date" not in raw:
                        raise ValueError("The file needs a 't' column (days) or a 'date' column.")
                    tab = reservoir_table(raw, columns, res, dayfirst)
                    set_table(name, tab)
                    ss[f"{name}_note"] = f"Loaded {len(tab)} rows" + (f" for {res}." if res else ".")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not load the file: {e}")
    else:
        w = well_import(name)
        if w is not None and st.button("Build table from wells", type="primary", key=f"{name}_build"):
            try:
                h = build_history(w["prod"], w["pres"], **w["kw"])
                tab, note = history_to_table(h, kind, w["fo"], w["fg"])
                set_table(name, tab)
                n = len(w["kw"].get("wells") or []) or "all"
                ss[f"{name}_note"] = " ".join([f"Built {len(tab)} rows from {n} wells."] + h.attrs["notes"] + [note])
                st.rerun()
            except (ValueError, KeyError) as e:
                st.error(f"Could not build the table: {e}")
    if ss.get(f"{name}_note"):
        st.info(ss[f"{name}_note"])


def mt_import():
    """Load multi-tank production for every reservoir in a reservoir file or in well files."""
    src = st.radio("Source", ["By reservoir (one file)", "By well (two files)"], horizontal=True, key="w_mt_src")
    histories = None
    try:
        if src.startswith("By reservoir"):
            up = st.file_uploader("Reservoir histories (CSV or Excel)", type=FILE_TYPES, key="mt_up")
            st.caption("Columns: reservoir, t (days) or date, p, Np (MMSTB), Gp (MMscf), Wp (MMSTB). "
                       "Each reservoir becomes a tank; its first row sets the initial pressure.")
            if up is not None:
                raw = standardise_columns(read_any(up), extra=["Np", "Gp", "Wp"])
                dayfirst = st.selectbox("Ambiguous dates are", ["day/month/year", "month/day/year"],
                                        key="mt_rdf").startswith("day")
                if st.button("Load into tanks", type="primary", key="mt_load"):
                    if "reservoir" not in raw:
                        raw = raw.assign(reservoir="Reservoir")
                    origin = None
                    if "t" not in raw:
                        origin = pd.to_datetime(raw["date"], format="mixed", dayfirst=dayfirst, errors="coerce").min()
                    histories = {r: reservoir_table(raw, ["t", "p", "Np", "Gp", "Wp"], r, dayfirst, origin)
                                 for r in sorted(raw["reservoir"].dropna().astype(str).unique())}
        else:
            w = well_import("mt", single=False)
            if w is not None and st.button("Build tanks from wells", type="primary", key="mt_build"):
                prod, kw = w["prod"], w["kw"]
                if "reservoir" not in prod:
                    prod = prod.assign(reservoir="Reservoir")
                origin = common_origin(prod, w["pres"], kw["volumes"], kw["period_dates"], kw["dayfirst"])
                histories = {}
                for r in sorted(prod["reservoir"].dropna().astype(str).unique()):
                    h = build_history(prod, w["pres"], reservoir=r, origin=origin, **kw)
                    for c in ("Np", "Wp"):
                        h[c] *= w["fo"]
                    h["Gp"] *= w["fg"]
                    histories[r] = h
    except Exception as e:
        st.error(f"Could not load the data: {e}")
        return
    if not histories:
        return
    old = ss.mt_tanks.set_index("name", drop=False) if len(ss.mt_tanks) else ss.mt_tanks
    tanks, prod_rows = [], []
    for r, h in histories.items():
        h = h.dropna(subset=["t"]).reset_index(drop=True)
        if h.empty:
            continue
        pi = float(h.p.dropna().iloc[0]) if h.p.notna().any() else 3000.0
        row = (old.loc[r].to_dict() if r in old.index else
               dict(name=r, fluid="oil", in_place=50.0, m=0.0, pi=pi, Swi=0.2, cf=4e-6, aquifer="none",
                    C_or_B=np.nan, Wei=np.nan, J=np.nan, td_per_day=np.nan, reD=np.nan,
                    fit_in_place=True, fit_aquifer=False))
        row["pi"] = pi
        tanks.append(row)
        for _, x in h[h.t > 0].iterrows():
            prod_rows.append(dict(tank=r, t=float(x.t), Np=x.Np, Gp=x.Gp, Wp=x.Wp, p_obs=x.p))
    if not tanks:
        st.error("No usable rows found.")
        return
    names = {t["name"] for t in tanks}
    ss.mt_tanks = pd.DataFrame(tanks).reset_index(drop=True)
    ss.mt_conns = ss.mt_conns[ss.mt_conns.a.isin(names) & ss.mt_conns.b.isin(names)].reset_index(drop=True)
    ss.mt_prod = pd.DataFrame(prod_rows, columns=["tank", "t", "Np", "Gp", "Wp", "p_obs"]).astype(
        {c: float for c in ("t", "Np", "Gp", "Wp", "p_obs")})
    ss.mt_msg = (f"Loaded {len(tanks)} tank(s): {', '.join(sorted(names))}. New tanks start with a placeholder "
                 "in-place volume of 50; set it, add connections, then run the history match.")
    ss.ver += 1
    st.rerun()


def fig_ax(n=1, height=3.4):
    fig, ax = plt.subplots(1, n, figsize=(5.6 * n, height))
    return fig, ax


def show(fig):
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def drive_index_chart(tab, parts):
    D = tab.iloc[1:]
    fig, ax = fig_ax(1, 3.0)
    fig.set_size_inches(11, 3.0)
    bottom = np.zeros(len(D))
    labels = [f"{v:,.0f}" for v in D.p]
    for (col, name), c in zip(parts, [BLUE, ORANGE, AQUA, YELLOW] if len(parts) == 4 else [BLUE, AQUA, YELLOW]):
        v = D[col].clip(lower=0).fillna(0).to_numpy()
        ax.bar(labels, v, bottom=bottom, color=c, label=name, width=0.6, edgecolor="white")
        bottom += v
    ax.set(title="Drive indices", xlabel="Survey pressure, psia", ylabel="Fraction of voidage")
    ax.grid(False)
    ax.legend(frameon=False, ncol=len(parts), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    show(fig)


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def page_oil():
    with st.sidebar:
        st.subheader("Rock & fluid")
        st.number_input("Swi, fraction", key="w_oil_Swi", min_value=0.0, max_value=0.95, step=0.01)
        sci("cw, 1/psi", "w_oil_cw")
        sci("cf, 1/psi", "w_oil_cf")
        st.subheader("Gas cap")
        st.number_input("Gas cap ratio m", key="w_oil_m", min_value=0.0, step=0.01, disabled=ss.w_oil_fitm)
        st.checkbox("Estimate m from data", key="w_oil_fitm")
        aq = aquifer_inputs("oil")

    st.caption("F = N·[Eo + m·Eg + Efw] + We.  The first row is initial conditions. "
               "Np, Wp, Winj in MMSTB; Gp, Ginj in MMscf; Bg in rb/scf.")
    with st.expander("Production & PVT history", expanded=True):
        data_tools("oil_df", OIL_COLS, {"Gas-cap sample": OIL_GASCAP, "Water-drive sample": OIL_WATERDRIVE}, "oil")
        data = st.data_editor(ss.oil_df, num_rows="dynamic", width="stretch", key=f"oil_ed_{ss.ver}",
                              column_config={"Bg": st.column_config.NumberColumn(format="%.6f")})
        if st.button("Fill Bo, Rs, Bg, Bw from the PVT correlations page"):
            try:
                filled = make_pvt().fill(data.dropna(subset=["p"]))[OIL_COLS]
                set_table("oil_df", filled)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    data = data.dropna(subset=["p"]).reset_index(drop=True)
    if len(data) < 3:
        st.warning("Enter the initial row and at least two survey rows.")
        return
    try:
        r = oil_mbe(data, Swi=ss.w_oil_Swi, cw=ss.w_oil_cw, cf=ss.w_oil_cf, m=ss.w_oil_m,
                    fit_m=ss.w_oil_fitm, aquifer=aq)
    except (ValueError, np.linalg.LinAlgError) as e:
        st.error(str(e))
        return
    if (data.p[1:] >= data.p[0]).any():
        st.warning("One or more survey pressures are at or above initial pressure; those rows distort the fit.")

    Np_last = float(data.Np.fillna(0).iloc[-1])
    items = [("Oil in place N", f"{r.in_place:,.1f} MMSTB", None),
             ("Gas cap ratio m", f"{r.m:.3f}", "estimated" if ss.w_oil_fitm else "fixed input")]
    if r.m > 0:
        items.append(("Gas cap gas", f"{r.gas_cap:,.1f} Bscf", None))
    items += aquifer_metrics(r.aquifer)
    items += [("Recovery to date", f"{Np_last / r.in_place * 100:.1f} %", f"Np = {Np_last:,.2f} MMSTB"),
              ("Fit quality R²", f"{r.r2:.4f}", f"{len(data) - 1} survey points")]
    show_metrics(items)

    D = r.table.iloc[1:]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = fig_ax()
        x = D.Et.max() * 1.05
        ax.plot([0, x], [0, r.in_place * x], color=ORANGE, lw=2, label="Slope = N")
        ax.plot(D.Et, D.F - D.We, "o", color=BLUE, mec="white", ms=7, label="Survey points")
        ax.set(title="F − We vs Et", xlabel="Et, rb/STB", ylabel="F − We, MMrb")
        ax.legend(frameon=False)
        show(fig)
    with c2:
        fig, ax = fig_ax()
        ax.axhline(r.in_place, color=GREY, ls="--", lw=1.5, label="Fitted N")
        ax.plot(data.Np.iloc[1:], D.F_over_Et, "o", color=BLUE, mec="white", ms=7, label="F/Et")
        ax.set(title="Campbell plot", xlabel="Np, MMSTB", ylabel="F/Et, MMSTB")
        ax.legend(frameon=False)
        show(fig)
        st.caption("Flat points mean depletion drive. A rising trend points to aquifer support.")
    drive_index_chart(r.table, [("DDI", "Depletion"), ("SDI", "Gas cap"), ("CDI", "Rock & water"),
                                ("WDI", "Water influx")])
    st.subheader("Material balance terms")
    st.dataframe(r.table, width="stretch", hide_index=True)
    st.download_button("Download terms as CSV", r.table.to_csv(index=False), "oil_material_balance.csv", "text/csv")


def page_gas():
    with st.sidebar:
        st.subheader("Rock & fluid")
        st.number_input("Temperature, °F", key="w_gas_T", step=5.0)
        st.number_input("Swi, fraction", key="w_gas_Swi", min_value=0.0, max_value=0.95, step=0.01)
        sci("cw, 1/psi", "w_gas_cw")
        sci("cf, 1/psi", "w_gas_cf")
        st.number_input("Abandonment p/z, psia", key="w_gas_pza", min_value=0.0, step=50.0)
        aq = aquifer_inputs("gas")

    st.caption("F = G·[Eg + Efw] + We and p/z = (pi/zi)·(1 − Gp/G).  The first row is initial conditions. "
               "Gp in Bscf, Wp in MMSTB.")
    with st.expander("Production history", expanded=True):
        data_tools("gas_df", GAS_COLS, {"Load sample": GAS_SAMPLE}, "gas")
        data = st.data_editor(ss.gas_df, num_rows="dynamic", width="stretch", key=f"gas_ed_{ss.ver}")
        if st.button("Fill z and Bw from the PVT correlations page"):
            try:
                pv = make_pvt()
                d = data.dropna(subset=["p"]).copy()
                tab = pv.table(d.p)
                d["z"], d["Bw"] = tab.z.values, tab.Bw.values
                ss.w_gas_T = float(ss.w_pvt_T)
                set_table("gas_df", d[GAS_COLS])
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    data = data.dropna(subset=["p"]).reset_index(drop=True)
    if len(data) < 3:
        st.warning("Enter the initial row and at least two survey rows.")
        return
    if not (data.z > 0).all():
        st.error("Every row needs a z-factor.")
        return
    try:
        r = gas_mbe(data, temp_f=ss.w_gas_T, Swi=ss.w_gas_Swi, cw=ss.w_gas_cw, cf=ss.w_gas_cf,
                    aquifer=aq, pz_abandon=ss.w_gas_pza or None)
    except (ValueError, np.linalg.LinAlgError) as e:
        st.error(str(e))
        return
    Gp_last = float(data.Gp.fillna(0).iloc[-1])
    ref = r.in_place if r.aquifer else r.G_pz
    items = [("GIIP from p/z", f"{r.G_pz:,.2f} Bscf", None),
             ("GIIP, Havlena–Odeh", f"{r.in_place:,.2f} Bscf", f"R² {r.r2:.4f}")]
    items += aquifer_metrics(r.aquifer)
    items.append(("Recovery to date", f"{Gp_last / ref * 100:.1f} %", f"Gp = {Gp_last:,.2f} Bscf"))
    if not r.aquifer and math.isfinite(r.Gp_abandon):
        items.append(("Recoverable to abandonment", f"{r.Gp_abandon:,.2f} Bscf", f"at p/z = {ss.w_gas_pza:,.0f}"))
    show_metrics(items)

    D = r.table.iloc[1:]
    c1, c2 = st.columns(2)
    with c1:
        a, _ = r.pz_fit
        fig, ax = fig_ax()
        ax.plot([0, r.G_pz], [a, 0], color=ORANGE, lw=2, label="Fitted line")
        if ss.w_gas_pza:
            ax.axhline(ss.w_gas_pza, color=GREY, ls="--", lw=1.2, label="Abandonment p/z")
        ax.plot(data.Gp, r.table.p_over_z, "o", color=BLUE, mec="white", ms=7, label="Survey points")
        ax.set(title="p/z vs Gp", xlabel="Gp, Bscf", ylabel="p/z, psia", ylim=(0, None))
        ax.legend(frameon=False)
        show(fig)
        st.caption("Valid for volumetric, normally pressured gas.")
    with c2:
        fig, ax = fig_ax()
        ax.axhline(r.in_place, color=GREY, ls="--", lw=1.5, label="Fitted G")
        ax.plot(data.Gp.iloc[1:], D.F_over_Et, "o", color=BLUE, mec="white", ms=7, label="F/Et")
        ax.set(title="Cole plot", xlabel="Gp, Bscf", ylabel="F/Et, Bscf")
        ax.legend(frameon=False)
        show(fig)
        st.caption("Flat points confirm a volumetric reservoir. A rising trend indicates water influx.")
    drive_index_chart(r.table, [("GDI", "Gas expansion"), ("CDI", "Rock & water"), ("WDI", "Water influx")])
    st.subheader("Material balance terms")
    st.dataframe(r.table, width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.6f") for c in ("Bg", "Eg", "Efw", "Et")})
    st.download_button("Download terms as CSV", r.table.to_csv(index=False), "gas_material_balance.csv", "text/csv")


def page_pvt():
    with st.sidebar:
        pvt_inputs()
        st.subheader("Pressure table")
        st.number_input("From, psia", key="w_pvt_pmin", min_value=15.0, step=100.0)
        st.number_input("To, psia", key="w_pvt_pmax", min_value=100.0, step=100.0)
        st.number_input("Step, psi", key="w_pvt_step", min_value=1.0, step=50.0)
    try:
        pvt = make_pvt()
    except ValueError as e:
        st.error(str(e))
        return
    b = pvt.at(pvt.pb)
    show_metrics([("Bubble point", f"{pvt.pb:,.0f} psia", None), ("Rsb", f"{pvt.rsb:,.0f} scf/STB", None),
                  ("Bo at pb", f"{pvt.bob:.4f} rb/STB", None),
                  ("μo at pb", f"{pvt.mu_ob:.3f} cp", f"dead oil {pvt.mu_dead:.2f} cp"),
                  ("z at pb", f"{b['z']:.4f}", f"Bg {b['Bg'] * 1000:.4f} rb/Mscf")])
    pmin, pmax = ss.w_pvt_pmin, max(ss.w_pvt_pmax, ss.w_pvt_pmin + 1)
    dense = pvt.table(np.sort(np.append(np.linspace(pmin, pmax, 150), pvt.pb)))
    specs = [("Rs", "Solution GOR", "scf/STB"), ("Bo", "Oil FVF", "rb/STB"), ("mu_o", "Oil viscosity", "cp"),
             ("z", "Gas z-factor", ""), ("Bg", "Gas FVF", "rb/scf"), ("mu_g", "Gas viscosity", "cp")]
    for row in (specs[:3], specs[3:]):
        for c, (col, title, unit) in zip(st.columns(3), row):
            with c:
                fig, ax = plt.subplots(figsize=(4.2, 2.8))
                ax.plot(dense.p, dense[col], color=BLUE, lw=2)
                if pmin < pvt.pb < pmax:
                    ax.axvline(pvt.pb, color=GREY, ls=":", lw=1.2)
                ax.set(title=title, xlabel="Pressure, psia", ylabel=unit)
                show(fig)
    ps = np.arange(pmin, pmax + 1e-9, ss.w_pvt_step)[:400]
    if pmin < pvt.pb < pmax:
        ps = np.sort(np.append(ps, pvt.pb))
    tab = pvt.table(ps)
    st.subheader("PVT table")
    st.dataframe(tab, width="stretch", hide_index=True, column_config={
        "co": st.column_config.NumberColumn(format="%.3e"), "Bg": st.column_config.NumberColumn(format="%.6f"),
        "mu_g": st.column_config.NumberColumn(format="%.4f")})
    st.download_button("Download table as CSV", tab.to_csv(index=False), "pvt_table.csv", "text/csv")
    st.caption("Above the bubble point Bo and μo use Vasquez–Beggs. Oil viscosity: Beggs–Robinson. "
               "z: Dranchuk–Abou-Kassem with Sutton pseudo-criticals. μg: Lee–Gonzalez–Eakin. Bw: McCain. "
               "Correlations typically carry 5–15% error; prefer lab PVT where it exists.")


def build_multitank(tanks_df, conns_df, prod_df):
    num = lambda v, d=0.0: float(v) if pd.notna(v) else d
    tanks = []
    for _, r in tanks_df.dropna(subset=["name"]).iterrows():
        pr = prod_df[prod_df.tank == r["name"]][["t", "Np", "Gp", "Wp", "p_obs"]].dropna(subset=["t"])
        aq = r["aquifer"] if r["aquifer"] in AQ_TANK else "none"
        tk = Tank(name=str(r["name"]), fluid=r["fluid"] if r["fluid"] in ("oil", "gas") else "oil",
                  in_place=num(r["in_place"]), pi=num(r["pi"]), m=num(r["m"]), Swi=num(r["Swi"]), cf=num(r["cf"]),
                  aquifer=aq, C=num(r["C_or_B"]), Wei=num(r["Wei"]), J=num(r["J"]),
                  td_per_day=num(r["td_per_day"]), reD=num(r["reD"], np.inf),
                  fit_in_place=bool(r["fit_in_place"]), fit_aquifer=bool(r["fit_aquifer"]), production=pr)
        if not (tk.in_place > 0 and tk.pi > 0):
            raise ValueError(f"Tank “{tk.name}” needs a positive in-place volume and initial pressure.")
        if aq == "fetkovich" and not (tk.Wei > 0 and tk.J > 0):
            raise ValueError(f"Tank “{tk.name}” uses a Fetkovich aquifer. Enter Wei and J.")
        if aq.startswith("veh") and not (tk.C >= 0 and tk.td_per_day > 0):
            raise ValueError(f"Tank “{tk.name}” uses a Van Everdingen–Hurst aquifer. Enter B and tD per day.")
        tanks.append(tk)
    if not tanks:
        raise ValueError("Add at least one tank.")
    names = {t.name for t in tanks}
    conns = [Connection(r["a"], r["b"], num(r["T"]), bool(r["fit"])) for _, r in conns_df.iterrows()
             if r["a"] in names and r["b"] in names and r["a"] != r["b"]]
    return MultiTank(tanks, conns, pvt=make_pvt(), cw=ss.w_mt_cw, max_step=max(1.0, ss.w_mt_dt))


def page_multitank():
    with st.sidebar:
        st.subheader("Simulation")
        st.number_input("Max time step, days", key="w_mt_dt", min_value=1.0, step=5.0)
        sci("cw, 1/psi", "w_mt_cw")
        with st.expander("Fluid properties (PVT correlations)"):
            pvt_inputs()
        if st.button("Load sample"):
            ss.mt_tanks, ss.mt_conns, ss.mt_prod = mt_sample()
            ss.mt_msg = ""
            ss.ver += 1
            st.rerun()

    st.caption("Each tank: F = N·Et + We + Σ T·∫(pj − pi)dt, solved for all pressures at every step. "
               "In place is MMSTB for oil tanks and Bscf for gas tanks; Gp is MMscf.")
    with st.expander("Bring in production by reservoir or by well"):
        mt_import()
    with st.expander("Tanks, connections and production", expanded=True):
        st.markdown("**Tanks**")
        tanks_df = st.data_editor(ss.mt_tanks, num_rows="dynamic", width="stretch", key=f"mt_t_{ss.ver}",
                                  column_config={
                                      "fluid": st.column_config.SelectboxColumn(options=["oil", "gas"]),
                                      "aquifer": st.column_config.SelectboxColumn(options=AQ_TANK),
                                      "cf": st.column_config.NumberColumn(format="%.2e"),
                                      "C_or_B": st.column_config.NumberColumn("C or B, rb/psi"),
                                      "Wei": st.column_config.NumberColumn("Wei, MMbbl"),
                                      "J": st.column_config.NumberColumn("J, rb/d/psi"),
                                      "td_per_day": st.column_config.NumberColumn("tD per day", format="%.5f"),
                                      "reD": st.column_config.NumberColumn("ra/ro (blank = ∞)")})
        st.caption("Pot uses C. Fetkovich uses Wei and J. Van Everdingen–Hurst uses B, tD per day and ra/ro.")
        names = [n for n in tanks_df["name"].dropna().tolist()]
        st.markdown("**Connections** — q = T·(p_a − p_b), T in rb/(day·psi)")
        conns_df = st.data_editor(ss.mt_conns, num_rows="dynamic", width="stretch", key=f"mt_c_{ss.ver}",
                                  column_config={"a": st.column_config.SelectboxColumn("from", options=names),
                                                 "b": st.column_config.SelectboxColumn("to", options=names)})
        st.markdown("**Cumulative production and observed pressure** — no row for day 0")
        prod_df = st.data_editor(ss.mt_prod, num_rows="dynamic", width="stretch", key=f"mt_p_{ss.ver}",
                                 column_config={"tank": st.column_config.SelectboxColumn(options=names)})
    try:
        model = build_multitank(tanks_df, conns_df, prod_df)
    except ValueError as e:
        st.error(str(e))
        return

    b1, b2 = st.columns([1, 5])
    if b1.button("Run history match", type="primary"):
        try:
            with st.spinner("Matching observed pressures…"):
                out = model.history_match()
            t2 = tanks_df.dropna(subset=["name"]).reset_index(drop=True).copy()
            for i, tk in enumerate(model.tanks):
                for col, val in (("in_place", tk.in_place), ("C_or_B", tk.C), ("Wei", tk.Wei),
                                 ("J", tk.J), ("td_per_day", tk.td_per_day)):
                    if pd.notna(t2.loc[i, col]):
                        t2.loc[i, col] = float(f"{val:.4g}")
            c2 = conns_df.reset_index(drop=True).copy()
            lookup = {(c.a, c.b): c.T for c in model.connections}
            for i, r in c2.iterrows():
                if (r["a"], r["b"]) in lookup:
                    c2.loc[i, "T"] = float(f"{lookup[(r['a'], r['b'])]:.4g}")
            ss.mt_tanks, ss.mt_conns, ss.mt_prod = t2, c2, prod_df.reset_index(drop=True)
            ss.mt_msg = (f"Match finished after {out['runs']} runs. RMS error "
                         f"{out['rms_before']:.1f} → {out['rms_after']:.1f} psi. The tables show the fitted values.")
            ss.ver += 1
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    b2.caption("Adjusts every parameter ticked fit_in_place, fit_aquifer or fit to match p_obs. "
               "Fitting tank size and aquifer together is not unique; fix what you know.")
    if ss.mt_msg:
        st.success(ss.mt_msg)

    res = model.simulate()
    if not res.converged:
        st.warning(res.message)
    items = [(tk.name, f"{tk.in_place:,.1f} {'Bscf' if tk.fluid == 'gas' else 'MMSTB'}", None) for tk in model.tanks]
    items.append(("Overall match", f"{model.rms_error(res):,.1f} psi RMS" if model.observations and res.converged
                  else "–", f"{len(model.observations)} observed pressures"))
    show_metrics(items)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, tk in enumerate(model.tanks):
        c = SERIES[i % len(SERIES)]
        ax.plot(res.t, res.pressure[tk.name], color=c, lw=2, label=tk.name)
        obs = [(t, p) for j, t, p in model.observations if j == i]
        if obs:
            ax.plot(*zip(*obs), "o", color=c, mec="white", ms=7)
    ax.set(title="Tank pressures (lines simulated, dots observed)", xlabel="Time, days", ylabel="Pressure, psia")
    ax.legend(frameon=False)
    show(fig)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = fig_ax()
        for j, col in enumerate(res.crossflow.columns):
            ax.plot(res.t, res.crossflow[col], lw=2, color=SERIES[(j + len(model.tanks)) % len(SERIES)], label=col)
        ax.set(title="Cumulative crossflow", xlabel="Time, days", ylabel="MMrb")
        if len(res.crossflow.columns):
            ax.legend(frameon=False)
        show(fig)
    with c2:
        fig, ax = fig_ax()
        for i, tk in enumerate(model.tanks):
            if tk.aquifer != "none":
                ax.plot(res.t, res.influx[tk.name], lw=2, color=SERIES[i % len(SERIES)], label=tk.name)
        ax.set(title="Cumulative water influx", xlabel="Time, days", ylabel="We, MMrb")
        if any(tk.aquifer != "none" for tk in model.tanks):
            ax.legend(frameon=False)
        show(fig)
    out = res.pressure.add_suffix(" p, psia").join(res.influx.add_suffix(" We, MMrb")).join(
        res.crossflow.add_suffix(", MMrb")).reset_index()
    st.download_button("Download simulation as CSV", out.to_csv(index=False), "multitank_simulation.csv", "text/csv")


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
st.title("Reservoir Material Balance")
PAGES = {"Oil reservoir": page_oil, "Gas reservoir": page_gas, "Multi-tank": page_multitank,
         "PVT correlations": page_pvt}
with st.sidebar:
    choice = st.radio("Analysis", list(PAGES), key="w_page")
    st.divider()
PAGES[choice]()
