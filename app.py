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

from resmb import Aquifer, BlackOilPVT, Connection, MultiTank, Tank, gas_mbe, oil_mbe

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


def data_tools(name, columns, samples):
    """Sample buttons and CSV upload for a single-tank table."""
    cols = st.columns(len(samples) + 1)
    for c, (label, df) in zip(cols, samples.items()):
        if c.button(label, key=f"{name}_{label}"):
            set_table(name, df.copy())
            st.rerun()
    up = st.file_uploader("Or upload a CSV with columns " + ", ".join(columns), type="csv", key=f"{name}_up")
    if up is not None and ss.get(f"{name}_upid") != up.file_id:
        ss[f"{name}_upid"] = up.file_id
        df = pd.read_csv(up)
        missing = [c for c in columns if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        set_table(name, df[columns].apply(pd.to_numeric, errors="coerce"))
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
        data_tools("oil_df", OIL_COLS, {"Gas-cap sample": OIL_GASCAP, "Water-drive sample": OIL_WATERDRIVE})
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
        data_tools("gas_df", GAS_COLS, {"Load sample": GAS_SAMPLE})
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
