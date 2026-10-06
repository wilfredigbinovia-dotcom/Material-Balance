"""Retrograde gas-condensate material balance tool.

Run with:   streamlit run app.py
"""
import datetime as dt
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mbal import units as U
from mbal.aquifer import GEOMETRIES, MODELS, NEEDS_GEOMETRY, WD
from mbal.matbal import METHODS, aggregate_wells, build_history, detect_drive, graphical
from mbal.importer import parse_workbook, reservoir_tables
from mbal.model import PARAMS, aquifer_params, make_aquifer, make_tank
from mbal.pvt import (PVT, condensate_props, correlation_table, cvd_two_phase_z, gas_equivalent, hall_cf,
                      water_props, wet_gas_gravity)
from mbal.outliers import residual_screen, runs_test, trend_screen
from mbal.regress import leave_one_out, pressure_rms, regress
from mbal.sample import default_config, sample_case

st.set_page_config(page_title="Condensate Material Balance", page_icon="🛢️", layout="wide")
ss = st.session_state

BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85"
RED = "#e34948"
SURVEY = [("use", "Use", "bool"), ("w", "Weight", "weight")]   # pressure survey on/off and weight

TABLES = {  # name -> [(column, label, quantity or None)]
    "pvt": [("p", "Pressure", "pressure"), ("z", "Z-factor (two-phase below dew point)", None),
            ("cgr", "Producing CGR (optional)", "cgr")],
    "cvd": [("p", "Pressure", "pressure"), ("sl", "Retrograde liquid (% of volume at dew point)", None),
            ("gp", "Cum. produced fluid (mol % of initial)", None),
            ("zg", "Gas Z-factor (single phase)", None)],
    "hist": [("date", "Date", "date"), ("p", "Reservoir pressure", "pressure"),
             ("gp", "Cum. gas", "gas_cum"), ("np", "Cum. condensate", "liq_cum"),
             ("wp", "Cum. water", "liq_cum")] + SURVEY,
    "wells": [("well", "Well", "text"), ("date", "Date", "date"), ("gp", "Cum. gas", "gas_cum"),
              ("np", "Cum. condensate", "liq_cum"), ("wp", "Cum. water", "liq_cum")],
    "pres": [("date", "Date", "date"), ("p", "Reservoir pressure", "pressure")] + SURVEY,
}


# ============================================================ state
def empty_table(name):
    d = {}
    for col, _, q in TABLES[name]:
        d[col] = pd.Series(dtype="datetime64[ns]" if q == "date" else "object" if q == "text"
                           else "bool" if q == "bool" else "float64")
    return pd.DataFrame(d)


def sync_tables():
    """Make the latest edited tables the source for the next generation of widgets."""
    ss.tables = {k: v.copy() for k, v in ss.cur.items()}


def refresh():
    sync_tables()
    ss.ver += 1
    ss.pop("reg", None)


def load_sample():
    c, tab, hist, wells, pres, start = sample_case()
    c.update(start_date=start, hist_mode="By reservoir")
    ss.vals = c
    ss.cur = {"pvt": tab, "hist": hist, "wells": wells, "pres": pres, "cvd": empty_table("cvd")}
    ss.ver = ss.get("ver", 0)
    refresh()


def load_blank():
    c = default_config()
    c.update(start_date=dt.date.today().replace(month=1, day=1), hist_mode="By reservoir")
    ss.vals = c
    ss.cur = {k: empty_table(k) for k in TABLES}
    refresh()


def project_json():
    v = dict(ss.vals)
    v["start_date"] = str(v["start_date"])
    tabs = {k: json.loads(d.to_json(orient="split", date_format="iso")) for k, d in ss.cur.items()}
    return json.dumps({"app": "condensate-mbal", "version": 1, "vals": v, "tables": tabs}, indent=1)


def load_project(raw):
    d = json.loads(raw)
    v = default_config()
    v.update(start_date=dt.date.today(), hist_mode="By reservoir")
    v.update(d["vals"])
    v["start_date"] = dt.date.fromisoformat(str(v["start_date"])[:10])
    cur = {}
    for k in TABLES:
        t = d["tables"].get(k)
        df = pd.DataFrame(t["data"], columns=t["columns"]) if t else empty_table(k)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        cur[k] = df
    ss.vals, ss.cur = v, cur
    refresh()


if "vals" not in ss:
    ss.ver = 0
    load_sample()

V = ss.vals

# ============================================================ sidebar
with st.sidebar:
    st.title("Condensate MBAL")
    st.caption("Material balance for retrograde gas-condensate reservoirs")
    unit_choice = st.radio("Units", ["Field", "SI"], horizontal=True, key="unit_radio",
                           on_change=sync_tables)
    SYS = "field" if unit_choice == "Field" else "si"
    st.divider()
    st.button("Load sample case", on_click=load_sample, width="stretch")
    st.button("Start blank", on_click=load_blank, width="stretch")
    st.divider()
    st.download_button("Save project", data=project_json(), file_name="mbal_project.json",
                       mime="application/json", width="stretch")
    up = st.file_uploader("Open project", type="json", key="proj_up")
    if up is not None and ss.get("proj_id") != up.file_id:
        ss.proj_id = up.file_id
        try:
            load_project(up.getvalue().decode("utf-8"))
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that project file: {e}")
    st.divider()
    st.caption("Work through the tabs from left to right. The sample case is synthetic: "
               "250 Bscf wet gas with a finite radial aquifer, and two deliberately bad "
               "pressure surveys to try the screening on.")

KEY = f"{SYS}_{ss.ver}"


# ============================================================ widget helpers
def ulabel(text, q):
    if q in (None, "text", "date", "bool", "weight") or U.label(q, SYS) in ("-",):
        return text
    return f"{text} ({U.label(q, SYS)})"


def num(label, key, q=None, where=st, step=None, minv=None, maxv=None, fmt="%.6g", help=None):
    disp = float(U.to_disp(q, V[key], SYS))
    kw = {}
    if minv is not None:
        kw["min_value"] = float(U.to_disp(q, minv, SYS))
        disp = max(disp, kw["min_value"])
    if maxv is not None:
        kw["max_value"] = float(U.to_disp(q, maxv, SYS))
        disp = min(disp, kw["max_value"])
    val = where.number_input(ulabel(label, q), value=disp, key=f"n_{key}_{KEY}", step=step,
                             format=fmt, help=help, **kw)
    V[key] = float(U.from_disp(q, val, SYS))
    return V[key]


def choice(label, key, options, where=st, horizontal=None, help=None):
    """options: dict value -> label. Stores the value in V[key]."""
    keys = list(options)
    idx = keys.index(V[key]) if V[key] in keys else 0
    wkey = f"c_{key}_{KEY}"
    if horizontal:
        v = where.radio(label, keys, index=idx, key=wkey, horizontal=True, help=help)
    else:
        v = where.selectbox(label, keys, index=idx, format_func=options.get, key=wkey, help=help)
    V[key] = v
    return v


def as_flag(v):
    """Tick-box / CSV value -> bool; blanks count as switched on."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    if isinstance(v, str):
        return v.strip().lower() not in ("0", "false", "no", "n", "off", "")
    return bool(v)


def table(name, height=350):
    """Editable table shown in display units; ss.cur[name] always holds field units."""
    spec = TABLES[name]
    src = ss.tables[name]
    disp, cfg = pd.DataFrame(index=src.index), {}
    for col, lab, q in spec:
        s = src[col] if col in src else pd.Series(np.nan, index=src.index, dtype="object")
        if q == "date":
            disp[col] = pd.to_datetime(s, errors="coerce")
            cfg[col] = st.column_config.DateColumn(lab, format="YYYY-MM-DD")
        elif q == "text":
            disp[col] = s.astype("object")
            cfg[col] = st.column_config.TextColumn(lab)
        elif q == "bool":
            disp[col] = s.map(as_flag).astype(bool)
            cfg[col] = st.column_config.CheckboxColumn(
                lab, default=True, help="Untick to switch this pressure survey off. Its production "
                                        "is kept; its pressure is ignored in every fit.")
        elif q == "weight":
            disp[col] = pd.to_numeric(s, errors="coerce").astype(float).fillna(1.0)
            cfg[col] = st.column_config.NumberColumn(
                lab, default=1.0, min_value=0.0, format="%.3g",
                help="Relative confidence in this survey for regression. 1 is normal; "
                     "2 counts double; 0.5 counts half.")
        else:
            disp[col] = U.to_disp(q, pd.to_numeric(s, errors="coerce").astype(float), SYS)
            cfg[col] = st.column_config.NumberColumn(ulabel(lab, q), format="%.6g")
    ed = st.data_editor(disp, column_config=cfg, num_rows="dynamic", hide_index=True,
                        key=f"t_{name}_{KEY}", height=height, width="stretch")
    out = pd.DataFrame()
    for col, _, q in spec:
        if q in ("date", "text"):
            out[col] = ed[col]
        elif q == "bool":
            out[col] = ed[col].map(as_flag).astype(bool)
        elif q == "weight":
            out[col] = pd.to_numeric(ed[col], errors="coerce").astype(float).fillna(1.0).clip(lower=0.0)
        else:
            out[col] = U.from_disp(q, pd.to_numeric(ed[col], errors="coerce").astype(float), SYS)
    ss.cur[name] = out.reset_index(drop=True)
    return ss.cur[name]


def csv_upload(name, note, expander=True):
    """Import a CSV (columns in the order shown, in the currently selected units)."""
    spec = TABLES[name]
    with (st.expander("Import from CSV") if expander else st.container()):
        st.caption(f"Columns in this order: {', '.join(ulabel(l, q) for _, l, q in spec)}. "
                   f"Values in the currently selected units. {note}")
        f = st.file_uploader("CSV file", type=["csv", "txt"], key=f"up_{name}")
        if f is not None and ss.get(f"upid_{name}") != f.file_id:
            ss[f"upid_{name}"] = f.file_id
            try:
                raw = pd.read_csv(io.BytesIO(f.getvalue()), sep=None, engine="python")
                raw = raw.iloc[:, :len(spec)]
                df = pd.DataFrame()
                for i, (col, _, q) in enumerate(spec):
                    if q == "bool":
                        df[col] = raw.iloc[:, i].map(as_flag) if i < raw.shape[1] else True
                    elif q == "weight":
                        df[col] = (pd.to_numeric(raw.iloc[:, i], errors="coerce").fillna(1.0)
                                   if i < raw.shape[1] else 1.0)
                    elif i >= raw.shape[1]:
                        df[col] = np.nan
                    elif q == "date":
                        df[col] = pd.to_datetime(raw.iloc[:, i], errors="coerce")
                    elif q == "text":
                        df[col] = raw.iloc[:, i].astype(str)
                    else:
                        df[col] = U.from_disp(q, pd.to_numeric(raw.iloc[:, i], errors="coerce"), SYS)
                ss.cur[name] = df
                refresh()
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the file: {e}")


def figure(xl, yl, height=430, legend=True):
    f = go.Figure()
    f.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    xaxis_title=xl, yaxis_title=yl, hovermode="closest")
    return f


def show(fig, key):
    st.plotly_chart(fig, width="stretch", key=f"fig_{key}_{KEY}", config={"displaylogo": False})


def gas_ip(scf):
    return U.to_disp("gas_ip", scf / 1e9, SYS)


def fmt_g(scf):
    if scf is None or not np.isfinite(scf):
        return "n/a"
    return f"{gas_ip(scf):,.1f} {U.label('gas_ip', SYS)}"


PU, GU = U.label("pressure", SYS), U.label("gas_ip", SYS)

t_pvt, t_prod, t_res, t_hm, t_aq, t_reg = st.tabs(
    ["1 · PVT data", "2 · Production history", "3 · Reservoir parameters",
     "4 · History match", "5 · Aquifer", "6 · Regression"])

# ============================================================ 1. PVT
with t_pvt:
    st.subheader("Fluid description")
    c = st.columns(4)
    num("Reservoir temperature", "T", "temperature", c[0], step=5.0)
    num("Dew point pressure", "pd", "pressure", c[1], step=50.0, minv=100.0)
    num("Initial condensate-gas ratio", "cgr_i", "cgr", c[2], step=5.0, minv=0.0)
    num("Condensate gravity", "api", "api", c[3], step=1.0, minv=10.0, maxv=90.0)
    c = st.columns(4)
    num("Separator gas gravity (air = 1)", "sg", None, c[0], step=0.01, minv=0.55, maxv=1.8)
    num("CO₂ (mole fraction)", "co2", None, c[1], step=0.01, minv=0.0, maxv=0.9)
    num("H₂S (mole fraction)", "h2s", None, c[2], step=0.01, minv=0.0, maxv=0.9)
    num("N₂ (mole fraction)", "n2", None, c[3], step=0.01, minv=0.0, maxv=0.9)

    SG_WET = wet_gas_gravity(V["sg"], V["cgr_i"], V["api"])
    GE = gas_equivalent(V["api"])
    sg_o, mw_o = condensate_props(V["api"])
    c = st.columns(4)
    c[0].metric("Well-stream gas gravity", f"{SG_WET:.3f}")
    c[1].metric("Condensate gas equivalent",
                f"{GE if SYS == 'field' else GE * 0.0283168 / 0.158987:,.0f} "
                f"{'scf/STB' if SYS == 'field' else 'sm³/sm³'}")
    c[2].metric("Condensate specific gravity", f"{sg_o:.3f}")
    c[3].metric("Condensate molecular weight", f"{mw_o:.0f}")

    st.subheader("Z-factor table")
    left, right = st.columns([2, 3])
    with left:
        st.caption("Enter the laboratory table (CVD two-phase Z below the dew point), paste from "
                   "a spreadsheet, or fill it from correlations.")
        pvt_df = table("pvt", height=380)
        csv_upload("pvt", "")
        with st.expander("Fill table from correlations"):
            choice("Z-factor below the dew point", "z2",
                   {"rayes": "Two-phase Z, Rayes et al. (rich gas, C7+ ≥ 4 mol%)",
                    "single": "Single-phase Z (lean gas)"})
            num("Maximum table pressure", "pvt_pmax", "pressure", step=100.0, minv=500.0)
            st.caption("Single-phase Z: Dranchuk-Abou-Kassem with Sutton pseudo-criticals and "
                       "Wichert-Aziz correction. The producing CGR curve below the dew point is "
                       "indicative only; replace it with CVD data when available.")

            def _fill_pvt():
                sync_tables()
                ss.cur["pvt"] = correlation_table(V["sg"], V["cgr_i"], V["api"], V["T"], V["pd"],
                                                  V["pvt_pmax"], V["co2"], V["h2s"], V["n2"], V["z2"])
                refresh()
            st.button("Generate table (replaces current table)", on_click=_fill_pvt)
        with st.expander("Calculate two-phase Z from a CVD report"):
            st.caption("For reports that give retrograde liquid deposit, cumulative produced "
                       "fluid and the single-phase (equilibrium gas) Z instead of the two-phase "
                       "Z. Enter the CVD rows, including the dew point row if the report has it. "
                       "Rows above the dew point need only pressure and gas Z.")
            cvd_df = table("cvd", height=260)
            csv_upload("cvd", "", expander=False)
            num("Gas Z-factor at the dew point (0 = take from the table)", "cvd_zd", None,
                step=0.01, minv=0.0, maxv=3.0)
            CVD_OUT = None
            if cvd_df["p"].notna().any():
                try:
                    CVD_OUT, zd_used, cvd_notes = cvd_two_phase_z(cvd_df, V["pd"], V["cvd_zd"] or None)
                    st.caption(f"Z2 = p / [(pd/Zd)(1 − Gp)] with pd = "
                               f"{U.to_disp('pressure', V['pd'], SYS):,.6g} {PU} and Zd = {zd_used:.4f}. "
                               "Liquid remaining is a consistency check from the liquid volume "
                               "and gas Z; it must be positive.")
                    st.dataframe(
                        pd.DataFrame({ulabel("Pressure", "pressure"): U.to_disp("pressure", CVD_OUT["p"], SYS),
                                      "Two-phase Z": CVD_OUT["z"].round(4),
                                      "Gas Z": CVD_OUT["zg"],
                                      "Liquid remaining (mol % of initial)": CVD_OUT["liq_mol"].round(2)}),
                        hide_index=True, width="stretch", height=220)
                    for m in cvd_notes:
                        st.warning(m)
                except Exception as e:  # noqa: BLE001
                    st.warning(str(e))

            def _use_cvd():
                sync_tables()
                new = CVD_OUT.dropna(subset=["z"])[["p", "z"]].sort_values("p").reset_index(drop=True)
                old = ss.cur["pvt"].dropna(subset=["p", "cgr"]).sort_values("p")
                new["cgr"] = np.interp(new["p"], old["p"], old["cgr"]) if len(old) >= 2 else np.nan
                ss.cur["pvt"] = new
                refresh()
            st.button("Use these Z-factors (replaces current table)", on_click=_use_cvd,
                      disabled=CVD_OUT is None or CVD_OUT["z"].notna().sum() < 2)

    PVT_OK, pvt = True, None
    try:
        pvt = PVT.from_frame(pvt_df, V["T"], V["api"], SG_WET)
    except Exception as e:  # noqa: BLE001
        PVT_OK = False
        PVT_ERR = str(e)
    with right:
        if PVT_OK:
            pp = np.linspace(*pvt.p_range, 120)
            px = U.to_disp("pressure", pp, SYS)
            cc = st.columns(2)
            for holder, y, yl, k in (
                    (cc[0], pvt.z(pp), "Z-factor", "z"),
                    (cc[1], U.to_disp("bg", pvt.bg(pp), SYS), f"Bg ({U.label('bg', SYS)})", "bg"),
                    (cc[0], U.to_disp("cgr", pvt.cgr(pp), SYS), f"Producing CGR ({U.label('cgr', SYS)})", "cgr"),
                    (cc[1], pvt.visc(pp), "Gas viscosity (cp)", "mu")):
                f = figure(f"Pressure ({PU})", yl, height=260, legend=False)
                f.add_trace(go.Scatter(x=px, y=y, mode="lines", line=dict(color=BLUE, width=2), name=yl))
                if k in ("z", "cgr"):
                    tp = pvt.p_tab
                    f.add_trace(go.Scatter(x=U.to_disp("pressure", tp, SYS),
                                           y=pvt.z_tab if k == "z" else U.to_disp("cgr", pvt.cgr_tab, SYS),
                                           mode="markers", marker=dict(color=BLUE, size=6), name="Table"))
                f.add_vline(x=U.to_disp("pressure", V["pd"], SYS), line_dash="dot", line_color=GREY,
                            annotation_text="dew point", annotation_font_size=11)
                with holder:
                    show(f, f"pvt_{k}")
        else:
            st.warning(PVT_ERR)

    st.subheader("Formation water")
    c = st.columns(4)
    num("Water formation volume factor", "bw", "fvf_w", c[0], step=0.01, minv=0.9, maxv=1.3)
    num("Water compressibility", "cw", "compress", c[1], step=0.1, minv=0.0)
    num("Water viscosity", "aq_muw", "visc", c[2], step=0.01, minv=0.01)
    num("Water salinity", "salinity", "ppm", c[3], step=5000.0, minv=0.0)

    def _est_water():
        bw, cw, mu = water_props(V["pi"], V["T"], V["salinity"])
        V.update(bw=round(bw, 4), cw=float(f"{cw:.3g}"), aq_muw=round(mu, 3))
        refresh()
    st.button("Estimate water properties from correlations", on_click=_est_water,
              help="McCain Bw, Osif cw, Beggs-Brill viscosity at initial pressure and temperature.")

# ============================================================ 2. Production history
with t_prod:
    c = st.columns([1, 2, 2])
    V["start_date"] = c[0].date_input("Start of production", value=V["start_date"],
                                      key=f"d_start_{KEY}",
                                      help="The reservoir is at initial pressure with zero "
                                           "cumulative production on this date.")
    with c[1]:
        mode = choice("Enter production", "hist_mode",
                      {"By reservoir": "By reservoir", "By well": "By well"}, horizontal=True)
    st.caption("All volumes are cumulative. Gas is separator (dry) gas; the tool converts "
               "condensate to its gas equivalent to obtain wet-gas production. Untick **Use** to "
               "switch a pressure survey off; tab 6 can suggest which ones to review.")
    with st.expander("Import a production and pressure workbook (Excel or CSV)"):
        st.caption("One sheet with Date and cumulative Gas, Condensate and Water, and one with Date "
                   "and Pressure. Reservoir and Well columns are optional. Columns are found by "
                   "their headings, in any order; values must be in the units selected in the "
                   f"sidebar ({U.label('gas_cum', SYS)}, {U.label('liq_cum', SYS)}, {PU}). "
                   "Pressures of several wells on the same date are averaged.")
        wb = st.file_uploader("Workbook", type=["xlsx", "xlsm", "xls", "csv", "txt"], key="up_wb")
        if wb is not None:
            try:
                if ss.get("wb_id") != wb.file_id:
                    ss.wb_data = parse_workbook(wb.getvalue(), wb.name)
                    ss.wb_id = wb.file_id
                wprod, wpres = ss.wb_data
                names = sorted(wprod["reservoir"].unique())
                res_pick = st.selectbox("Reservoir", names, key="wb_res") if len(names) > 1 else names[0]
                WB = reservoir_tables(wprod, wpres, res_pick)
                st.write(f"**{res_pick}**: {WB['wells']['well'].nunique()} well(s), "
                         f"{len(WB['wells'])} production rows, {len(WB['pres'])} survey dates. "
                         f"Start of production will be set to {WB['start_date']}.")
                if len(WB["pre_start"]):
                    st.info("Surveys before production started (use as initial pressure in tab 3): "
                            + "; ".join(f"{U.to_disp('pressure', r.p, SYS):,.6g} {PU} on {r.date:%Y-%m-%d}"
                                        for r in WB["pre_start"].itertuples()))
                wide = WB["spread"][WB["spread"]["n"] > 1].sort_values("range", ascending=False)
                if len(wide) and wide["range"].iloc[0] > 50.0:
                    r = wide.iloc[0]
                    st.warning(f"Wells disagree by {U.to_disp('pressure', r['range'], SYS):,.0f} {PU} on "
                               f"{r['date']:%Y-%m-%d}. The average is used; correct it in the survey "
                               "table if one of the wells is not representative.")
                for m in WB["notes"]:
                    st.warning(m)

                def _use_wb():
                    sync_tables()
                    wl, pr = WB["wells"].copy(), WB["pres"].copy()
                    wl["gp"] = U.from_disp("gas_cum", wl["gp"], SYS)
                    for k in ("np", "wp"):
                        wl[k] = U.from_disp("liq_cum", wl[k], SYS)
                    pr["p"] = U.from_disp("pressure", pr["p"], SYS)
                    ss.cur["wells"], ss.cur["pres"] = wl, pr
                    V["hist_mode"], V["start_date"] = "By well", WB["start_date"]
                    refresh()
                st.button("Load into the tables (replaces current history)", on_click=_use_wb)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not read the workbook: {e}")
    if mode == "By reservoir":
        left, right = st.columns([3, 3])
        with left:
            hist_df = table("hist", height=460)
            csv_upload("hist", "Dates as YYYY-MM-DD. Use and Weight are optional.")
    else:
        left, right = st.columns([3, 3])
        with left:
            st.markdown("**Well cumulative production**")
            wells_df = table("wells", height=300)
            csv_upload("wells", "Dates as YYYY-MM-DD.")
            st.markdown("**Average reservoir pressure surveys**")
            pres_df = table("pres", height=240)
            csv_upload("pres", "Dates as YYYY-MM-DD. Use and Weight are optional.")
        try:
            hist_df = aggregate_wells(wells_df, pres_df)
        except Exception as e:  # noqa: BLE001
            hist_df = empty_table("hist")
            st.warning(f"Could not combine the wells: {e}")

    HIST_OK, hist = True, None
    try:
        hist = build_history(hist_df, V["start_date"], V["pi"])
    except Exception as e:  # noqa: BLE001
        HIST_OK, HIST_ERR = False, str(e)

    with right:
        if HIST_OK:
            dates = pd.Timestamp(V["start_date"]) + pd.to_timedelta(hist.t, unit="D")
            f = figure("", f"Reservoir pressure ({PU})", height=250, legend=False)
            on_ = hist.use[1:]
            py_ = U.to_disp("pressure", hist.p[1:], SYS)
            f.add_trace(go.Scatter(x=dates[1:][on_], y=py_[on_], mode="markers",
                                   marker=dict(color=BLUE, size=8), name="Pressure"))
            f.add_trace(go.Scatter(x=dates[1:][~on_], y=py_[~on_], mode="markers", name="Switched off",
                                   marker=dict(color=GREY, size=8, symbol="circle-open")))
            show(f, "h_p")
            f = figure("", f"Cumulative gas ({U.label('gas_ip', SYS)})", height=250)
            f.add_trace(go.Scatter(x=dates, y=gas_ip(hist.gp), mode="lines",
                                   line=dict(color=BLUE, width=2), name="Separator gas"))
            if PVT_OK:
                f.add_trace(go.Scatter(x=dates, y=gas_ip(hist.gp + GE * hist.np_), mode="lines",
                                       line=dict(color=ORANGE, width=2), name="Wet gas equivalent"))
            show(f, "h_g")
            f = figure("", f"Cumulative liquids ({U.label('liq_cum', SYS)})", height=250)
            f.add_trace(go.Scatter(x=dates, y=U.to_disp("liq_cum", hist.np_ / 1e3, SYS), mode="lines",
                                   line=dict(color=AQUA, width=2), name="Condensate"))
            f.add_trace(go.Scatter(x=dates, y=U.to_disp("liq_cum", hist.wp / 1e3, SYS), mode="lines",
                                   line=dict(color=YELLOW, width=2), name="Water"))
            show(f, "h_l")
            if mode == "By well":
                with st.expander("Combined reservoir history (at pressure survey dates)"):
                    d = hist_df.copy()
                    d["p"] = U.to_disp("pressure", d["p"], SYS)
                    d["gp"] = U.to_disp("gas_cum", d["gp"], SYS)
                    for k in ("np", "wp"):
                        d[k] = U.to_disp("liq_cum", d[k], SYS)
                    d.columns = [ulabel(l, q) for _, l, q in TABLES["hist"]]
                    st.dataframe(d, hide_index=True, width="stretch")
        else:
            st.info(HIST_ERR)

# ============================================================ 3. Reservoir parameters
with t_res:
    st.subheader("Initial reservoir parameters")
    c = st.columns(3)
    num("Initial reservoir pressure", "pi", "pressure", c[0], step=50.0, minv=100.0)
    num("Porosity", "phi", "frac", c[1], step=0.01, minv=0.01, maxv=0.6)
    num("Connate water saturation", "swi", "frac", c[2], step=0.01, minv=0.0, maxv=0.9)
    c = st.columns(3)
    num("Rock compressibility", "cf", "compress", c[0], step=0.5, minv=0.0)
    num("Original gas in place, initial estimate (wet gas)", "G", "gas_ip", c[1], step=10.0, minv=0.001,
        help="Starting value. The history match and regression refine it.")

    def _hall():
        V["cf"] = float(f"{hall_cf(V['phi']):.3g}")
        refresh()
    c[2].write("")
    c[2].button("Estimate rock compressibility (Hall)", on_click=_hall)
    if V["pi"] < V["pd"]:
        st.warning("Initial pressure is below the dew point: the reservoir starts two-phase. "
                   "The two-phase Z-factor must then apply from initial conditions.")
    if PVT_OK:
        lo, hi = pvt.p_range
        if V["pi"] > hi * 1.0001 or V["pi"] < lo:
            st.error(f"Initial pressure is outside the PVT table "
                     f"({U.to_disp('pressure', lo, SYS):,.0f} to {U.to_disp('pressure', hi, SYS):,.0f} {PU}). "
                     "Extend the table in tab 1.")
        zi, bgi = float(pvt.z(V["pi"])), float(pvt.bg(V["pi"]))
        ce = (V["cw"] * V["swi"] + V["cf"]) / (1 - V["swi"])
        hcpv = V["G"] * 1e9 * bgi / 1e6           # MMrb
        frac_dry = 1.0 / (1.0 + GE * V["cgr_i"] / 1e6)
        st.subheader("Derived quantities")
        c = st.columns(4)
        c[0].metric("Initial Z-factor", f"{zi:.4f}")
        c[1].metric(f"Initial Bg ({U.label('bg', SYS)})", f"{U.to_disp('bg', bgi, SYS):.4f}")
        c[2].metric(f"Effective compressibility ({U.label('compress', SYS)})",
                    f"{U.to_disp('compress', ce, SYS):.2f}")
        c[3].metric(f"Initial p/z ({PU})", f"{U.to_disp('pressure', V['pi'] / zi, SYS):,.0f}")
        c = st.columns(4)
        c[0].metric(f"Hydrocarbon pore volume ({U.label('res_vol', SYS)})",
                    f"{U.to_disp('res_vol', hcpv, SYS):,.1f}")
        c[1].metric(f"Pore volume ({U.label('res_vol', SYS)})",
                    f"{U.to_disp('res_vol', hcpv / (1 - V['swi']), SYS):,.1f}")
        c[2].metric(f"Dry gas in place ({GU})", f"{U.to_disp('gas_ip', V['G'] * frac_dry, SYS):,.1f}")
        c[3].metric(f"Condensate in place ({'MMSTB' if SYS == 'field' else '10⁶ sm³'})",
                    f"{U.to_disp('liq_cum', V['G'] * frac_dry * V['cgr_i'] / 1e3, SYS):,.1f}")

# ============================================================ engine
READY = PVT_OK and HIST_OK
if READY:
    lo, hi = pvt.p_range
    if V["pi"] > hi * 1.0001 or hist.p.min() < lo:
        RANGE_WARN = ("Some pressures fall outside the PVT table; Z is held constant beyond the "
                      "table limits. Extend the table in tab 1 for reliable results.")
    else:
        RANGE_WARN = None
    tank0 = make_tank(V, pvt, hist, with_aquifer=False)
    DET = detect_drive(tank0)
    n_pts = len(hist.p)
    DATES = pd.Timestamp(V["start_date"]) + pd.to_timedelta(hist.t, unit="D")
    LAB = np.array([str(d.date()) for d in DATES])
    N_ON = int(hist.use[1:].sum())


def history_points(fig, x, y, flag=None, size=9):
    """History markers: switched-on surveys solid, flagged ones crossed, switched-off hollow."""
    x, y = np.asarray(x), np.asarray(y)
    fl = np.zeros(len(y), bool) if flag is None else flag
    on, off = hist.use & ~fl, ~hist.use & ~fl
    fig.add_trace(go.Scatter(x=x[on], y=y[on], mode="markers", name="History", text=LAB[on],
                             marker=dict(color=BLUE, size=size)))
    if fl.any():
        fig.add_trace(go.Scatter(x=x[fl], y=y[fl], mode="markers", name="Flagged", text=LAB[fl],
                                 marker=dict(color=RED, size=size + 3, symbol="x")))
    if off.any():
        fig.add_trace(go.Scatter(x=x[off], y=y[off], mode="markers", name="Switched off",
                                 text=LAB[off], marker=dict(color=GREY, size=size - 1,
                                                            symbol="circle-open")))


def need_data():
    if not PVT_OK:
        st.info(f"Complete the PVT data first. {PVT_ERR}")
    elif not HIST_OK:
        st.info(f"Complete the production history first. {HIST_ERR}")


def detection_banner(short=False):
    msg = DET["message"]
    if DET["status"] == "aquifer":
        if V["aq_model"] == "none":
            st.warning(f"**Additional energy source detected.** {msg}"
                       + (" Go to tab 5 to define the aquifer." if short else ""))
        else:
            st.success(f"**Additional energy source detected and modelled** with "
                       f"{MODELS[V['aq_model']]}. {msg}")
    elif DET["status"] == "volumetric":
        if V["aq_model"] == "none":
            st.success(f"**No additional energy source detected.** {msg}")
        else:
            st.info(f"**No additional energy source detected**, but an aquifer model is active. {msg}")
    else:
        st.info(msg)


def aquifer_inputs():
    """Aquifer model / geometry / parameter widgets."""
    c = st.columns(2)
    m = choice("Aquifer model (method)", "aq_model", MODELS, c[0])
    if m in NEEDS_GEOMETRY:
        g = choice("Aquifer system (geometry)", "aq_geom", GEOMETRIES, c[1])
    else:
        g = V["aq_geom"]
    if m == "none":
        st.caption("No aquifer: the tank is treated as volumetric.")
        return
    if m == "pot":
        num("Aquifer water volume", "aq_Wvol", "res_vol", step=50.0, minv=0.001,
            help="We = (cw + cf) · W · (pi − p). Instant response; suits small aquifers.")
        return
    if m == "schilthuis":
        num("Aquifer constant", "aq_C", "aq_const", step=5.0, minv=0.001,
            help="dWe/dt = C · (pi − p). Constant-pressure outer boundary.")
        return
    c = st.columns(3)
    if g == "radial":
        num("Reservoir radius", "aq_ro", "length", c[0], step=250.0, minv=10.0)
        inf_ok = m != "fetkovich"
        if inf_ok:
            V["aq_inf"] = c[1].checkbox("Infinite-acting aquifer", value=bool(V["aq_inf"]),
                                        key=f"cb_inf_{KEY}")
        if not (inf_ok and V["aq_inf"]):
            num("Outer/inner radius ratio (re/ro)", "aq_reD", "ratio", c[1], step=0.5, minv=1.05)
        num("Encroachment angle", "aq_theta", "angle", c[2], step=10.0, minv=1.0, maxv=360.0)
        c = st.columns(3)
        num("Aquifer thickness", "aq_h", "length", c[0], step=5.0, minv=0.5)
    elif g == "linear":
        num("Aquifer width", "aq_width", "length", c[0], step=250.0, minv=10.0)
        num("Aquifer length", "aq_L", "length", c[1], step=1000.0, minv=10.0)
        num("Aquifer thickness", "aq_h", "length", c[2], step=5.0, minv=0.5)
        c = st.columns(3)
    else:
        num("Reservoir radius", "aq_ro", "length", c[0], step=250.0, minv=10.0)
        num("Aquifer thickness (below contact)", "aq_h", "length", c[1], step=5.0, minv=0.5)
        num("Vertical anisotropy kv/kh", "aq_kvkh", "ratio", c[2], step=0.05, minv=1e-5, maxv=1.0)
        c = st.columns(3)
        num("Encroachment angle", "aq_theta", "angle", c[0], step=10.0, minv=1.0, maxv=360.0,
            help="360° when water underlies the whole reservoir.")
    num("Aquifer permeability", "aq_k", "perm", c[1], step=10.0, minv=0.001)
    num("Aquifer porosity", "aq_phi", "frac", c[2], step=0.01, minv=0.01, maxv=0.6)
    st.caption(f"Aquifer total compressibility = cw + cf = "
               f"{U.to_disp('compress', V['cw'] + V['cf'], SYS):.2f} {U.label('compress', SYS)}; "
               f"water viscosity {V['aq_muw']:.3g} cp (tab 1)."
               + (" Bottom drive is modelled as vertical linear flow through the reservoir area."
                  if g == "bottom" else ""))


# The aquifer tab is rendered before the history-match tab is computed so that both use
# the same, current aquifer settings.
with t_aq:
    if not READY:
        need_data()
    else:
        detection_banner()
        left, right = st.columns([3, 2])
        with right:
            st.markdown("**Diagnostic: F/Et with no aquifer (Cole plot)**")
            g0 = graphical(tank0, "F/Et (Cole - no aquifer)")
            m0 = g0["mask"] & np.isfinite(g0["y"])
            f = figure(f"Cumulative wet gas produced ({GU})", f"F/Et ({GU})", height=340)
            f.add_trace(go.Scatter(x=gas_ip(g0["x"][m0]), y=gas_ip(g0["y"][m0]), mode="markers",
                                   marker=dict(color=BLUE, size=8), name="History"))
            if "fit" in DET:
                f.add_trace(go.Scatter(x=gas_ip(DET["x"]), y=gas_ip(DET["fit"]), mode="lines",
                                       line=dict(color=ORANGE, width=2), name="Trend"))
            show(f, "aq_cole")
            st.caption("Flat: volumetric. Rising: strong water drive. Hump: moderate. "
                       "Declining: weak water drive or abnormal pressure.")
        with left:
            st.subheader("Aquifer definition")
            if DET["status"] == "aquifer" and V["aq_model"] == "none":
                st.markdown("Select the aquifer model and system below, then enter its parameters.")
            aquifer_inputs()

if READY:
    tank = make_tank(V, pvt, hist)
    P_SIM, WE_SIM = tank.simulate()
    WE_HIST = tank.We_history()
    RMS = pressure_rms(P_SIM, hist.p, hist.w)
    AQ = tank.aq

with t_aq:
    if READY and AQ.active:
        with left:
            st.subheader("Aquifer response over the history")
            c = st.columns(3)
            c[0].metric(f"Cumulative influx ({U.label('res_vol', SYS)})",
                        f"{U.to_disp('res_vol', WE_HIST[-1] / 1e6, SYS):,.2f}")
            if AQ.model in NEEDS_GEOMETRY:
                _, reD_, U_, a_, Wi_, J_ = AQ._geom()
                c[1].metric(f"Aquifer water volume ({U.label('res_vol', SYS)})",
                            "infinite" if not np.isfinite(Wi_) else
                            f"{U.to_disp('res_vol', Wi_ / 1e6, SYS):,.0f}")
                c[2].metric("Dimensionless time at end of history", f"{a_ * hist.t[-1]:.3g}")
            c2 = st.columns(3)
            c2[0].metric(f"Pressure match RMS ({PU})",
                         "n/a" if not np.isfinite(RMS) else f"{U.to_disp('pressure', RMS, SYS):,.1f}")
            c2[1].metric("Influx / withdrawal at end", f"{WE_HIST[-1] / tank.F()[-1]:.0%}")
            st.caption("See tab 4 for the plots with this aquifer and tab 6 to regress its parameters.")

# ============================================================ 4. History match
with t_hm:
    if not READY:
        need_data()
    else:
        detection_banner(short=True)
        if RANGE_WARN:
            st.warning(RANGE_WARN)
        s_gr, s_an, s_en, s_wd = st.tabs(["Graphical plot", "Analytical plot", "Energy plot",
                                           "WD function plot"])
        dates = pd.Timestamp(V["start_date"]) + pd.to_timedelta(hist.t, unit="D")

        # ---------------- graphical
        with s_gr:
            c = st.columns([2, 3])
            method = c[0].selectbox("Method", METHODS, key=f"gm_{KEY}")
            rng = c[1].slider("History points used for the line fit", 1, n_pts - 1, (1, n_pts - 1),
                              key=f"gr_{KEY}")
            sel = np.zeros(n_pts, bool)
            sel[rng[0]:rng[1] + 1] = True
            if method in ("p/z", "p/z (overpressured)"):
                sel[0] = True
            g = graphical(tank, method, sel)

            def conv(q, v):
                v = np.asarray(v, float)
                if q == "gas":
                    return gas_ip(v), GU
                if q == "pz":
                    return U.to_disp("pressure", v, SYS), PU
                if q == "hox":
                    return U.to_disp("pressure", v, SYS), PU
                if q == "roachx":
                    f_ = U.Q["gas_cum"][SYS][1] / U.Q["pressure"][SYS][1]
                    return v / 1e6 * f_, f"{U.label('gas_cum', SYS)}/{'psi' if SYS == 'field' else 'bar'}"
                if q == "roachy":
                    return v * U.Q["compress"][SYS][1], U.label("compress", SYS)
                return v, ""

            xs, xu = conv(g["xq"], g["x"])
            ys, yu = conv(g["yq"], g["y"])
            if method == "Havlena-Odeh (water drive)":
                xs, xu = np.asarray(g["x"], float) / 1e9 * U.Q["gas_ip"][SYS][1], GU
            ok = g["mask"] & np.isfinite(xs) & np.isfinite(ys)
            used, unused = ok & g["sel"], ok & ~g["sel"]
            fig = figure(f"{g['xl']} ({xu})", f"{g['yl']} ({yu})", height=470)
            lab = [str(d.date()) for d in dates]
            fig.add_trace(go.Scatter(x=xs[used], y=ys[used], mode="markers", name="History (in fit)",
                                     text=np.array(lab)[used], marker=dict(color=BLUE, size=9)))
            if unused.any():
                fig.add_trace(go.Scatter(x=xs[unused], y=ys[unused], mode="markers",
                                         name="History (not in fit)", text=np.array(lab)[unused],
                                         marker=dict(color=GREY, size=8, symbol="circle-open")))
            fit = g["fit"]
            if g["kind"] == "flat" and g["G"] is not None and used.any():
                x0, x1 = float(xs[ok].min()), float(xs[ok].max())
                fig.add_trace(go.Scatter(x=[x0, x1], y=[gas_ip(g["G"])] * 2, mode="lines",
                                         name="Mean of fitted points", line=dict(color=ORANGE, width=2)))
            elif fit is not None and used.sum() >= 2:
                xr = np.array([np.nanmin(g["x"][used]), np.nanmax(g["x"][used])], float)
                if g.get("to_zero") and g["G"]:
                    xr = np.array([0.0, g["G"]])
                elif method.startswith("Havlena"):
                    xr[0] = 0.0
                yr = fit["intercept"] + fit["slope"] * xr
                lx, _ = conv(g["xq"], xr)
                ly, _ = conv(g["yq"], yr)
                if method == "Havlena-Odeh (water drive)":
                    lx = xr / 1e9 * U.Q["gas_ip"][SYS][1]
                fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", name="Line fit",
                                         line=dict(color=ORANGE, width=2)))
            show(fig, "gr")

            c = st.columns(4)
            c[0].metric("Gas in place from this plot", fmt_g(g["G"]))
            c[1].metric("Current model G", fmt_g(V["G"] * 1e9))
            if fit is not None:
                c[2].metric("Line fit R²", f"{fit['r2']:.4f}")
            if "ce" in g["extra"]:
                c[3].metric(f"Implied effective compressibility ({U.label('compress', SYS)})",
                            f"{U.to_disp('compress', g['extra']['ce'], SYS):.2f}")
            elif "slope" in g["extra"]:
                c[3].metric("Slope (1.0 when the aquifer is right)", f"{g['extra']['slope']:.3f}")
            elif g["kind"] == "flat" and fit is not None and g["G"]:
                drift = fit["slope"] * np.ptp(g["x"][used]) / g["G"]
                c[3].metric("Trend across fitted points", f"{drift:+.1%}",
                            help="Close to zero when the model is complete.")
            notes = {
                "p/z": "Straight line for a volumetric reservoir; the x-intercept is the gas in place. "
                       "Water influx or rock compaction bend the line and make this estimate too high.",
                "p/z (overpressured)": "Ramagost-Farshad correction for rock and connate-water "
                                       "expansion; ce = (cw·Swi + cf)/(1 − Swi).",
                "Havlena-Odeh (overpressured)": "Intercept = G; slope/intercept = effective "
                                                "compressibility ce. Uses the current aquifer for We.",
                "Havlena-Odeh (water drive)": "Intercept = G; a unit slope confirms the aquifer model. "
                                              "With no aquifer defined all points sit at We/Et = 0.",
                "(F-We)/Et (Cole)": "Horizontal at G when the aquifer model removes all the water influx.",
                "Roach (unknown compressibility)": "Slope = 1/G and intercept = −ce, so G is obtained "
                                                   "without assuming a rock compressibility. Assumes no aquifer.",
                "F/Et (Cole - no aquifer)": "Horizontal at G for a volumetric reservoir; any curvature "
                                            "signals an energy source missing from the balance.",
            }
            st.caption(notes[method])

            with st.expander("Gas in place from every method (same fit range)"):
                rows = []
                for m_ in METHODS:
                    s_ = sel.copy()
                    s_[0] = m_ in ("p/z", "p/z (overpressured)")
                    gi = graphical(tank, m_, s_)
                    rows.append({"Method": m_, f"G ({GU})": None if gi["G"] is None else round(float(gas_ip(gi["G"])), 1),
                                 "R²": None if gi["fit"] is None else round(gi["fit"]["r2"], 4)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        # ---------------- analytical
        with s_an:
            fig = figure(f"Cumulative wet gas produced ({GU})", f"Reservoir pressure ({PU})", height=470)
            if AQ.active:
                p_na = tank0.simulate()[0]
                fig.add_trace(go.Scatter(x=gas_ip(tank.gpw), y=U.to_disp("pressure", p_na, SYS),
                                         mode="lines", name="Model without aquifer",
                                         line=dict(color=GREY, width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=gas_ip(tank.gpw), y=U.to_disp("pressure", P_SIM, SYS), mode="lines",
                                     name="Model" + (f" ({MODELS[AQ.model]})" if AQ.active else " (no aquifer)"),
                                     line=dict(color=ORANGE, width=2)))
            history_points(fig, gas_ip(tank.gpw), U.to_disp("pressure", hist.p, SYS))
            show(fig, "an")
            c = st.columns(4)
            c[0].metric(f"Pressure match RMS ({PU})",
                        "n/a" if not np.isfinite(RMS) else f"{U.to_disp('pressure', RMS, SYS):,.1f}",
                        help="Weighted, over the surveys that are switched on.")
            c[1].metric("Model G", fmt_g(V["G"] * 1e9))
            c[3].metric("Surveys switched on", f"{N_ON} of {n_pts - 1}")
            c[2].metric("Recovery to date (wet gas)", f"{tank.gpw[-1] / (V['G'] * 1e9):.1%}")
            if np.isnan(P_SIM).any():
                st.error("The model cannot deliver the produced volumes with these parameters "
                         "(pressure would fall below the PVT table). Increase the gas in place "
                         "or the aquifer strength.")
            st.caption("The model pressure is solved from the material balance at each date using "
                       "the recorded production. Adjust G in tab 3 and the aquifer in tab 5, or let "
                       "the regression in tab 6 find the match.")
            with st.expander("Results table"):
                st.dataframe(pd.DataFrame({
                    "Date": [d.date() for d in dates],
                    f"Wet gas produced ({GU})": np.round(gas_ip(tank.gpw), 3),
                    f"History pressure ({PU})": np.round(U.to_disp("pressure", hist.p, SYS), 1),
                    f"Model pressure ({PU})": np.round(U.to_disp("pressure", P_SIM, SYS), 1),
                    f"Difference ({PU})": np.round((P_SIM - hist.p) * U.Q["pressure"][SYS][1], 1),
                    "Survey used": hist.use,
                    f"Model water influx ({U.label('res_vol', SYS)})":
                        np.round(U.to_disp("res_vol", WE_SIM / 1e6, SYS), 3),
                }), hide_index=True, width="stretch")

        # ---------------- energy
        with s_en:
            en = tank.energy()
            fig = figure("", "Share of reservoir energy", height=450)
            for (name, y), col in zip(en.items(), (BLUE, YELLOW, AQUA)):
                fig.add_trace(go.Scatter(x=dates[1:], y=y[1:], mode="lines", name=name, stackgroup="e",
                                         line=dict(width=1, color=col),
                                         hovertemplate="%{y:.1%}<extra>" + name + "</extra>"))
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_layout(hovermode="x unified")
            show(fig, "en")
            c = st.columns(3)
            for holder, (name, y) in zip(c, en.items()):
                holder.metric(f"{name}, latest", f"{y[-1]:.1%}")
            st.caption("Drive indices from measured pressures: G·Eg, G·Efw and We as fractions of "
                       "their sum, using the current G and aquifer model.")

        # ---------------- WD
        with s_wd:
            if AQ.model not in NEEDS_GEOMETRY:
                st.info("The WD function applies to aquifers with a defined geometry "
                        "(Hurst-van Everdingen, Carter-Tracy or Fetkovich). Select one in tab 5.")
            else:
                gdim, reD_, U_, a_, Wi_, J_ = AQ._geom()
                tD_hist = a_ * hist.t[1:]
                lo_ = min(tD_hist.min() / 10, 1e-2)
                hi_ = max(tD_hist.max() * 10, 1e2)
                tt = np.logspace(np.log10(lo_), np.log10(hi_), 160)
                fig = figure("Dimensionless time, tD", "Dimensionless influx, WD", height=450)
                if gdim == "radial":
                    for r_ in (1.5, 2, 3, 5, 8, 12, 20):
                        fig.add_trace(go.Scatter(x=tt, y=WD(tt, "radial", r_), mode="lines",
                                                 line=dict(color=GREY, width=1), showlegend=False,
                                                 hovertemplate=f"re/ro = {r_}<br>tD %{{x:.3g}}<br>WD %{{y:.3g}}<extra></extra>"))
                    fig.add_trace(go.Scatter(x=tt, y=WD(tt, "radial", np.inf), mode="lines",
                                             line=dict(color=GREY, width=1, dash="dash"),
                                             name="Infinite acting"))
                    nm = "This aquifer (infinite)" if not np.isfinite(reD_) else f"This aquifer (re/ro = {reD_:.3g})"
                else:
                    nm = "This aquifer (finite linear)"
                fig.add_trace(go.Scatter(x=tt, y=WD(tt, gdim, reD_), mode="lines", name=nm,
                                         line=dict(color=ORANGE, width=2)))
                fig.add_trace(go.Scatter(x=tD_hist, y=WD(tD_hist, gdim, reD_), mode="markers",
                                         name="History dates", text=lab[1:],
                                         marker=dict(color=BLUE, size=8)))
                fig.update_xaxes(type="log")
                fig.update_yaxes(type="log")
                show(fig, "wd")
                st.caption("Constant-terminal-pressure solution of the diffusivity equation, "
                           "We = U·Δp·WD(tD). Grey curves are other re/ro values for reference. "
                           + ("Fetkovich uses the pseudo-steady-state approximation of this response."
                              if AQ.model == "fetkovich" else ""))

# ============================================================ 6. Regression
def set_surveys(upd):
    """upd: {date: (use, weight)}. Writes to the table that holds the pressure surveys."""
    name = "hist" if V["hist_mode"] == "By reservoir" else "pres"
    df = ss.cur[name].copy()
    d = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for i, dd in d.items():
        if dd in upd:
            df.at[i, "use"], df.at[i, "w"] = bool(upd[dd][0]), float(upd[dd][1])
    ss.cur[name] = df
    refresh()


with t_reg:
    if not READY:
        need_data()
    else:
        SIG = hash((hist.p.tobytes(), hist.w.tobytes(), tank.gpw.tobytes()))
        if ss.get("reg") and ss.reg.get("sig") != SIG:
            ss.pop("reg")                       # history changed since the regression was run
        reg = ss.get("reg")
        pf = U.Q["pressure"][SYS][1]
        r_scr, r_fit = st.tabs(["Survey screening", "Parameter regression"])

        # ---------------- survey screening
        with r_scr:
            left, right = st.columns([3, 3])
            with left:
                st.caption(f"Pressures and deviations in {PU}. "
                           "Scores every pressure survey so you can decide which to switch off "
                           "before regressing. Nothing is removed automatically, and a survey that "
                           "is switched off keeps its production record.")
                opts = ["Trend of the data (no model)",
                        "Residuals of the latest regression" if reg else "Residuals of the current model"]
                c = st.columns([3, 2])
                how = c[0].radio("Compare each survey with", opts, key=f"scr_how_{bool(reg)}_{KEY}")
                nsig = c[1].number_input("Flag beyond (standard deviations)", 1.5, 6.0, 3.0, 0.5,
                                         key=f"scr_n_{KEY}")
                by_trend = how == opts[0]
                if by_trend:
                    dev, z = trend_screen(tank.gpw, hist.p, hist.use, nsig)
                    p_ref = None
                else:
                    p_ref = reg["p"] if reg else P_SIM
                    dev, z = residual_screen(p_ref, hist.p, hist.use)
                flag = np.isfinite(z) & (np.abs(z) > nsig)
                flag[0] = False
                new = flag & hist.use
                tbl = pd.DataFrame({
                    "Date": [d.date() for d in DATES[1:]],
                    f"Pressure ({PU})": np.round(U.to_disp("pressure", hist.p[1:], SYS), 1),
                    f"Deviation ({PU})": np.round(dev[1:] * pf, 1),
                    "Score": np.round(z[1:], 1),
                    "Flag": np.where(flag[1:], "suspect", ""),
                    "Use": hist.use[1:], "Weight": hist.w_in[1:]})
                ed_s = st.data_editor(
                    tbl, hide_index=True, width="stretch", height=330,
                    disabled=[c_ for c_ in tbl.columns if c_ not in ("Use", "Weight")],
                    column_config={"Weight": st.column_config.NumberColumn(min_value=0.0, format="%.3g",
                                                                           width="small"),
                                   "Use": st.column_config.CheckboxColumn(width="small"),
                                   "Date": st.column_config.DateColumn(width="small"),
                                   tbl.columns[1]: st.column_config.NumberColumn("Pressure", width="small"),
                                   tbl.columns[2]: st.column_config.NumberColumn("Deviation", width="small"),
                                   "Flag": st.column_config.TextColumn(width="small"),
                                   "Score": st.column_config.NumberColumn(
                                       width="small",
                                       help="Deviation divided by the typical scatter (from the "
                                            "median absolute deviation of the active surveys).")},
                    key=f"scr_ed_{KEY}")
                b = st.columns(3)
                if b[0].button(f"Switch off flagged ({int(new.sum())})", type="primary",
                               disabled=not new.any(), width="stretch"):
                    set_surveys({DATES[i].normalize(): (False, hist.w_in[i]) for i in np.flatnonzero(new)})
                    st.rerun()
                if b[1].button("Apply table edits", width="stretch",
                               help="Applies the Use and Weight columns as edited above."):
                    set_surveys({DATES[i + 1].normalize(): (as_flag(r_["Use"]), r_["Weight"])
                                 for i, r_ in ed_s.reset_index(drop=True).iterrows()})
                    st.rerun()
                if b[2].button("Switch all on", width="stretch", disabled=N_ON == n_pts - 1):
                    set_surveys({DATES[i].normalize(): (True, hist.w_in[i]) for i in range(1, n_pts)})
                    st.rerun()

                if new.sum() > 0.2 * max(N_ON, 1):
                    st.warning(f"{int(new.sum())} of {N_ON} active surveys are flagged. That is too many "
                               "to be bad data: the trend or the model is more likely at fault. "
                               "Do not switch them all off.")
                elif new.any():
                    st.info(f"{int(new.sum())} of {N_ON} active surveys stand out. Check the survey "
                            "reports (shut-in time, gauge, datum correction) before switching them off.")
                else:
                    st.success("No active survey stands out at this threshold.")
                if not by_trend:
                    rt = runs_test((hist.p - p_ref)[hist.use][1:])
                    if rt["systematic"]:
                        st.warning(
                            f"**The mismatch is systematic, not scatter.** The residuals change sign "
                            f"only {max(rt['runs'] - 1, 0)} times in {rt['n']} surveys (longest stretch on "
                            f"one side: {rt['longest']}). That points to the model (aquifer, "
                            "compressibility, PVT, gas in place), and switching surveys off would "
                            "only hide it. Use the trend screen until the model fits.")
                    else:
                        st.caption(f"Runs test: residuals fall in {rt['runs']} runs against about "
                                   f"{rt['expected']:.0f} expected for random scatter, so the model "
                                   "shows no systematic bias.")
                with st.expander("How the screening works"):
                    st.markdown(
                        "- **Trend of the data** compares each survey with a robust straight line "
                        "through its three neighbours on each side of the pressure versus cumulative "
                        "production trend. It needs no model, so it is the right check before any "
                        "match exists.\n"
                        "- **Residuals** compares each survey with the model pressure. Use it after "
                        "regressing, ideally with the robust fitting method so the bad surveys have "
                        "not already pulled the model towards themselves.\n"
                        "- The **score** is the deviation divided by the typical scatter of the active "
                        "surveys. Scatter is estimated from the median absolute deviation, which "
                        "outliers cannot inflate.\n"
                        "- **Weight** scales how strongly a survey counts in the regression "
                        "(for example 2 for a long build-up, 0.5 for a short shut-in).")
            with right:
                fig = figure(f"Cumulative wet gas produced ({GU})", f"Reservoir pressure ({PU})", height=300)
                if p_ref is not None:
                    fig.add_trace(go.Scatter(x=gas_ip(tank.gpw), y=U.to_disp("pressure", p_ref, SYS),
                                             mode="lines", name="Model", line=dict(color=ORANGE, width=2)))
                history_points(fig, gas_ip(tank.gpw), U.to_disp("pressure", hist.p, SYS), new, size=8)
                show(fig, "scr_p")
                fig = figure("", "Score (standard deviations)", height=280, legend=False)
                for m_, col_, sym_, nm_ in ((hist.use & ~flag, BLUE, "circle", "Active"),
                                            (new, RED, "x", "Flagged"),
                                            (~hist.use, GREY, "circle-open", "Switched off")):
                    m_ = m_ & np.isfinite(z)
                    m_[0] = False
                    fig.add_trace(go.Scatter(x=DATES[m_], y=z[m_], mode="markers", name=nm_,
                                             text=LAB[m_], marker=dict(color=col_, size=9, symbol=sym_)))
                for y_ in (nsig, -nsig):
                    fig.add_hline(y=y_, line_dash="dot", line_color=GREY)
                fig.add_hline(y=0, line_color=GREY, line_width=1)
                show(fig, "scr_z")

        # ---------------- parameter regression
        with r_fit:
            st.subheader("Regress on parameters to match pressure history")
            keys = ["G", "cf"] + aquifer_params(V["aq_model"], V["aq_geom"],
                                                bool(V["aq_inf"]) and V["aq_model"] != "fetkovich")
            default_on = {"G"} | set({"pot": ["aq_Wvol"], "schilthuis": ["aq_C"]}.get(
                V["aq_model"], [k for k in keys if k in ("aq_reD", "aq_k", "aq_L")]))
            rows = []
            for k in keys:
                lab_, q, lo_, hi_ = PARAMS[k]
                cur_ = V[k]
                a, b = max(cur_ * 0.2, lo_), min(cur_ * 5.0, hi_)
                if k == "aq_reD":
                    a, b = 1.2, max(20.0, cur_ * 2)
                if k == "aq_theta":
                    a, b = 10.0, 360.0
                rows.append({"Regress": k in default_on, "Parameter": ulabel(lab_, q),
                             "Current value": U.to_disp(q, cur_, SYS),
                             "Minimum": U.to_disp(q, a, SYS), "Maximum": U.to_disp(q, b, SYS)})
            left, right = st.columns([3, 3])
            with left:
                st.caption("Tick the parameters to adjust and set their limits. Regress on as few "
                           "as needed: many aquifer parameters compensate for one another.")
                ed = st.data_editor(
                    pd.DataFrame(rows), hide_index=True, width="stretch",
                    disabled=["Parameter", "Current value"],
                    column_config={c_: st.column_config.NumberColumn(format="%.5g")
                                   for c_ in ("Current value", "Minimum", "Maximum")},
                    key=f"reg_{V['aq_model']}_{V['aq_geom']}_{V['aq_inf']}_{KEY}")
                LOSSES = {"Least squares": "linear", "Robust (soft-L1)": "soft_l1"}
                c = st.columns([2, 3])
                loss_lab = c[0].selectbox(
                    "Fitting method", list(LOSSES), key=f"loss_{KEY}",
                    help="Least squares squares every residual, so one bad survey can pull the whole "
                         "match. The robust method caps the pull of residuals larger than about twice "
                         "the typical scatter, so isolated bad surveys are down-weighted without "
                         "being removed.")
                c[1].caption(f"{N_ON} of {n_pts - 1} pressure surveys are switched on"
                             + ("" if np.allclose(hist.w[hist.use], 1.0) else ", with unequal weights")
                             + ". Change them in Survey screening or in tab 2.")
                go_btn = st.button("Run regression", type="primary")
                if not AQ.active and DET["status"] == "aquifer":
                    st.warning("An additional energy source was detected but no aquifer is defined. "
                               "Regression on G alone will not match the pressure trend; define the "
                               "aquifer in tab 5 first.")

            def make_sim(base, rk):
                def sim(vals):
                    return make_tank(dict(base, **dict(zip(rk, vals))), pvt, hist).simulate()[0]
                return sim

            if go_btn:
                pick = [i for i, r in ed.iterrows() if r["Regress"]]
                if not pick:
                    left.error("Tick at least one parameter.")
                else:
                    rk = [keys[i] for i in pick]
                    qs = [PARAMS[k][1] for k in rk]
                    x0 = [V[k] for k in rk]
                    lo_ = [U.from_disp(q, float(ed.loc[i, "Minimum"]), SYS) for i, q in zip(pick, qs)]
                    hi_ = [U.from_disp(q, float(ed.loc[i, "Maximum"]), SYS) for i, q in zip(pick, qs)]
                    if any(not (0 < a < b) for a, b in zip(lo_, hi_)):
                        left.error("Each minimum must be positive and below its maximum.")
                    elif len(rk) >= N_ON:
                        left.error("More parameters than active pressure surveys.")
                    else:
                        base = dict(V)
                        sim = make_sim(base, rk)
                        with st.spinner("Regressing..."):
                            r = regress(sim, hist.p, x0, lo_, hi_, hist.w, LOSSES[loss_lab])
                        ss.reg = {"keys": rk, "x0": x0, "x": list(map(float, r["x"])),
                                  "rms0": r["rms0"], "rms": r["rms"], "se": r["se_rel"],
                                  "bound": r["at_bound"], "corr": r["corr"], "p": r["p"],
                                  "p0": sim(x0), "lo": lo_, "hi": hi_, "loss": LOSSES[loss_lab],
                                  "loss_lab": loss_lab, "base": base, "sig": SIG}
                        st.rerun()

            if reg:
                with left:
                    c = st.columns(2)
                    c[0].metric(f"RMS before ({PU})", f"{reg['rms0'] * pf:,.1f}")
                    c[1].metric(f"RMS after ({PU})", f"{reg['rms'] * pf:,.1f}",
                                delta=f"{(reg['rms'] - reg['rms0']) * pf:,.1f}", delta_color="inverse")
                    out = []
                    for k, a, b, se, bd in zip(reg["keys"], reg["x0"], reg["x"], reg["se"], reg["bound"]):
                        lab_, q, *_ = PARAMS[k]
                        out.append({"Parameter": ulabel(lab_, q), "Start": U.to_disp(q, a, SYS),
                                    "Matched": U.to_disp(q, b, SYS),
                                    "± 1 std. dev.": "n/a" if not np.isfinite(se) else f"{se:.0%}",
                                    "Note": "at limit" if bd else ""})
                    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch",
                                 column_config={c_: st.column_config.NumberColumn(format="%.5g")
                                                for c_ in ("Start", "Matched")})
                    st.caption(f"Fitting method: {reg['loss_lab']}, on {N_ON} surveys. RMS is over the "
                               "surveys that are switched on.")
                    if any(np.isfinite(s) and s > 0.5 for s in reg["se"]):
                        st.caption("A large standard deviation means the pressure data do not constrain "
                                   "that parameter on its own; fix it or regress on fewer parameters.")

                    def _apply():
                        for k, v in zip(ss.reg["keys"], ss.reg["x"]):
                            V[k] = float(f"{v:.6g}")
                        refresh()
                    st.button("Apply matched values to the model", on_click=_apply, type="primary")

                    rt = runs_test((hist.p - reg["p"])[hist.use][1:])
                    _, z_fit = residual_screen(reg["p"], hist.p, hist.use)
                    n_out = int((hist.use & np.isfinite(z_fit) & (np.abs(z_fit) > 3.0)).sum())
                    if rt["systematic"]:
                        st.warning(f"The residuals are systematic (only {max(rt['runs'] - 1, 0)} sign "
                                   f"changes in {rt['n']} surveys): the model is still missing "
                                   "something. Revisit the aquifer or the regressed parameters "
                                   "rather than switching surveys off.")
                    elif n_out:
                        st.info(f"{n_out} active survey{'s' if n_out > 1 else ''} sit more than 3 standard "
                                "deviations from this match. Review them in Survey screening "
                                "(residuals of the latest regression)."
                                + (" A least-squares fit bends towards bad surveys and can make good "
                                   "neighbours look bad, so rerun with the robust fitting method "
                                   "before trusting these flags." if reg["loss"] == "linear" else ""))

                    if st.button("Influence check (leave one survey out)",
                                 help="Repeats the fit with each survey switched off in turn and "
                                      "reports how far the matched values move. Takes a few seconds."):
                        with st.spinner("Refitting without each survey in turn..."):
                            ss.reg["loo"] = leave_one_out(make_sim(reg["base"], reg["keys"]), hist.p,
                                                          reg["x"], reg["lo"], reg["hi"], hist.w,
                                                          reg["loss"])
                        st.rerun()
                    if "loo" in reg:
                        loo = reg["loo"]
                        worst = np.nan_to_num(np.nanmax(np.abs(np.where(np.isfinite(loo), loo, 0.0)), axis=1))
                        order = [i for i in np.argsort(-worst) if hist.use[i] and i > 0][:6]
                        d = {"Survey removed": [LAB[i] for i in order]}
                        for j_, k in enumerate(reg["keys"]):
                            d[f"Change in {PARAMS[k][0].lower()}"] = [f"{loo[i, j_]:+.1%}" for i in order]
                        st.dataframe(pd.DataFrame(d), hide_index=True, width="stretch")
                        st.caption("The six most influential surveys. A single survey that moves the "
                                   "answer by much more than the others deserves a second look.")
                with right:
                    fig = figure(f"Cumulative wet gas produced ({GU})", f"Reservoir pressure ({PU})", height=440)
                    fig.add_trace(go.Scatter(x=gas_ip(tank.gpw), y=U.to_disp("pressure", reg["p0"], SYS),
                                             mode="lines", name="Before regression",
                                             line=dict(color=GREY, width=2, dash="dash")))
                    fig.add_trace(go.Scatter(x=gas_ip(tank.gpw), y=U.to_disp("pressure", reg["p"], SYS),
                                             mode="lines", name="After regression",
                                             line=dict(color=ORANGE, width=2)))
                    history_points(fig, gas_ip(tank.gpw), U.to_disp("pressure", hist.p, SYS))
                    show(fig, "reg")
            else:
                with right:
                    fig = figure(f"Cumulative wet gas produced ({GU})", f"Reservoir pressure ({PU})", height=440)
                    fig.add_trace(go.Scatter(x=gas_ip(tank.gpw), y=U.to_disp("pressure", P_SIM, SYS),
                                             mode="lines", name="Current model", line=dict(color=ORANGE, width=2)))
                    history_points(fig, gas_ip(tank.gpw), U.to_disp("pressure", hist.p, SYS))
                    show(fig, "reg0")
