"""Retrograde gas-condensate material balance tool.

Run with:   streamlit run app.py
"""
import datetime as dt
import io
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mbal import units as U
from mbal.aquifer import GEOMETRIES, MODELS, NEEDS_GEOMETRY, WD
from mbal.matbal import METHODS, aggregate_wells, build_history, detect_drive, graphical
from mbal.forecast import forecast
from mbal.importer import DATE_FMT, parse_dates, parse_workbook, read_tests, reservoir_tables
from mbal.model import PARAMS, aquifer_params, make_aquifer, make_tank
from mbal.pvt import (GasZ, PVT, condensate_props, correlation_table, cvd_two_phase_z, gas_equivalent, hall_cf,
                      water_props, wet_gas_gravity)
from mbal.outliers import residual_screen, runs_test, trend_screen
from mbal.regress import leave_one_out, pressure_rms, regress
from mbal.relperm import RelPerm, TankState, fit_water, water_history
from mbal.sample import default_config, sample_case, sample_wells
from mbal.vlp import LiftTable, read_tpd
from mbal.wells import Well, fit_ipr, tubing_bhp, well_rate

st.set_page_config(page_title="Condensate Material Balance", page_icon="🛢️", layout="wide")
ss = st.session_state

# Number boxes: no -/+ step buttons (values are typed in).
st.markdown("""<style>
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"],
[data-testid="stNumberInput"] button {display: none !important;}
[data-testid="stNumberInput"] input {-moz-appearance: textfield;}
[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
[data-testid="stNumberInput"] input::-webkit-inner-spin-button {-webkit-appearance: none; margin: 0;}
</style>""", unsafe_allow_html=True)

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
    "nbr": [("date", "Date", "date"), ("p", "Neighbour static pressure", "pressure")],
    "wellcfg": [("well", "Well", "text"), ("tvd", "Depth to perforations, TVD", "length"),
                ("md", "Measured depth (optional)", "length"), ("tid", "Tubing inner diameter", "diam"),
                ("wht", "Flowing wellhead temperature", "temperature"),
                ("min_thp", "Minimum tubing-head pressure", "pressure"),
                ("min_bhp", "Minimum bottomhole pressure", "pressure"),
                ("qmax", "Maximum gas rate (optional)", "gas_rate")],
    "tests": [("well", "Well", "text"), ("date", "Date", "date"), ("qg", "Gas rate", "gas_rate"),
              ("pwf", "Flowing bottomhole pressure", "pressure"),
              ("pth", "Flowing tubing-head pressure", "pressure"),
              ("pr", "Reservoir pressure (optional)", "pressure"),
              ("wgr", "Water-gas ratio (optional)", "cgr")],
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
    ss.pop("rp_fit", None)


def load_sample():
    c, tab, hist, wells, pres, start = sample_case()
    c.update(start_date=start, hist_mode="By reservoir")
    ss.vals = c
    tests, wellcfg = sample_wells(hist, c)
    ss.cur = {k: empty_table(k) for k in TABLES}
    ss.cur.update(pvt=tab, hist=hist, wells=wells, pres=pres, tests=tests, wellcfg=wellcfg)
    ss.ver = ss.get("ver", 0)
    refresh()


FALLBACK = default_config()      # only ever used for inputs the selected model does not need


def blank_config():
    """Every number box empty; only the option selectors keep a setting."""
    c = {k: (None if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
         for k, v in FALLBACK.items()}
    c.update(start_date=None, hist_mode="By reservoir")
    return c


def cfg():
    """Input values for the engine. Empty boxes the current model does not use are filled
    with a placeholder; boxes it does use are checked before the engine is called."""
    return {k: (FALLBACK.get(k) if v is None else v) for k, v in V.items()}


def missing(keys):
    """Labels of the required inputs that are still empty."""
    return [LABELS.get(k, k) for k in keys if V.get(k) is None]


LABELS = {}     # key -> label, filled as the number boxes are drawn


def load_blank():
    ss.vals = blank_config()
    ss.cur = {k: empty_table(k) for k in TABLES}
    refresh()


def project_json():
    v = dict(ss.vals)
    v["start_date"] = None if v["start_date"] is None else str(v["start_date"])
    tabs = {k: json.loads(d.to_json(orient="split", date_format="iso")) for k, d in ss.cur.items()}
    return json.dumps({"app": "condensate-mbal", "version": 1, "vals": v, "tables": tabs}, indent=1)


def load_project(raw):
    d = json.loads(raw)
    v = blank_config()
    v.update(d["vals"])
    v["start_date"] = (dt.date.fromisoformat(str(v["start_date"])[:10])
                       if v["start_date"] not in (None, "None", "") else None)
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
    load_blank()

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
    st.caption("Work through the tabs from left to right. The app starts empty; the sample "
               "case is synthetic: 250 Bscf wet gas with a finite radial aquifer, and two deliberately bad "
               "pressure surveys to try the screening on.")

KEY = f"{SYS}_{ss.ver}"


# ============================================================ widget helpers
def ulabel(text, q):
    if q in (None, "text", "date", "bool", "weight") or U.label(q, SYS) in ("-",):
        return text
    return f"{text} ({U.label(q, SYS)})"


def num(label, key, q=None, where=st, step=None, minv=None, maxv=None, fmt="%.6g", help=None):
    LABELS[key] = label
    disp = None if V[key] is None else float(U.to_disp(q, V[key], SYS))
    kw = {}
    if minv is not None:
        kw["min_value"] = float(U.to_disp(q, minv, SYS))
        disp = None if disp is None else max(disp, kw["min_value"])
    if maxv is not None:
        kw["max_value"] = float(U.to_disp(q, maxv, SYS))
        disp = None if disp is None else min(disp, kw["max_value"])
    val = where.number_input(ulabel(label, q), value=disp, key=f"n_{key}_{KEY}", step=step,
                             format=fmt, help=help, placeholder="", **kw)
    V[key] = None if val is None else float(U.from_disp(q, val, SYS))
    return V[key]


def peek(key, q):
    """Current value of a number box that is drawn further down the page."""
    wkey = f"n_{key}_{KEY}"
    if wkey in ss:
        return None if ss[wkey] is None else float(U.from_disp(q, ss[wkey], SYS))
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
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return True     # also what a cleared tick box holds
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
            # Shown and typed as text, dd/mm/yyyy, and parsed here day first, so a pasted
            # 01/11/2019 is always 1 November whatever the browser's locale.
            dd = parse_dates(s)
            disp[col] = dd.dt.strftime(DATE_FMT).where(dd.notna(), None).astype("object")
            cfg[col] = st.column_config.TextColumn(f"{lab} (dd/mm/yyyy)")
        elif q == "text":
            disp[col] = s.astype("object")
            cfg[col] = st.column_config.TextColumn(lab)
        elif q == "bool":
            # nullable, so that clearing the cell (Delete key, or pasting a blank) is accepted
            disp[col] = s.map(as_flag).astype("boolean")
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
    for col, lab, q in spec:
        if q == "date":
            out[col] = parse_dates(ed[col])
            raw = ed[col].astype("object")
            bad = raw.notna() & raw.astype(str).str.strip().ne("") & out[col].isna()
            if bad.any():
                st.warning(f"{lab}: could not read {', '.join(map(str, raw[bad].head(5)))}"
                           + (" and others" if bad.sum() > 5 else "") + ". Use dd/mm/yyyy.")
        elif q == "text":
            out[col] = ed[col]
        elif q == "bool":
            out[col] = ed[col].map(as_flag).astype(bool)
        elif q == "weight":
            out[col] = pd.to_numeric(ed[col], errors="coerce").astype(float).fillna(1.0).clip(lower=0.0)
        else:
            out[col] = U.from_disp(q, pd.to_numeric(ed[col], errors="coerce").astype(float), SYS)
    ss.cur[name] = out.reset_index(drop=True)
    return ss.cur[name]


def date_order_check(name):
    """Warn when a table's dates look like day and month were swapped on pasting
    (dd/mm read as mm/dd), and offer to swap them back."""
    d = pd.to_datetime(ss.cur[name]["date"], errors="coerce").dropna()
    if len(d) < 3:
        return
    off_first = (d.dt.day != 1).mean()
    if d.dt.day.max() <= 12 and off_first >= 0.5:
        st.warning(f"Every date in this table has a day of 12 or less, and most are not the first of "
                   f"the month (for example {d.iloc[min(1, len(d) - 1)]:%d/%m/%Y}). Day and month may "
                   "have been swapped when the dates were pasted. Check them; a swapped survey date "
                   "puts that pressure against the wrong cumulative production.")

        def _swap():
            sync_tables()
            dd = pd.to_datetime(ss.cur[name]["date"], errors="coerce")
            ok = dd.notna() & (dd.dt.day <= 12)
            new = dd.copy()
            new[ok] = pd.to_datetime(dict(year=dd[ok].dt.year, month=dd[ok].dt.day, day=dd[ok].dt.month))
            ss.cur[name]["date"] = new
            refresh()
        st.button("Swap day and month in this table", on_click=_swap, key=f"swap_{name}_{KEY}")


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
                        df[col] = parse_dates(raw.iloc[:, i])
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
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is not None and len(x) and isinstance(x[0], (pd.Timestamp, dt.date, np.datetime64, str)) \
                and pd.notna(pd.to_datetime(str(x[0]), errors="coerce", format="mixed")):
            fig.update_xaxes(hoverformat="%d/%m/%Y")
            break
    st.plotly_chart(fig, width="stretch", key=f"fig_{key}_{KEY}", config={"displaylogo": False})


def gas_ip(scf):
    return U.to_disp("gas_ip", scf / 1e9, SYS)


def fmt_g(scf):
    if scf is None or not np.isfinite(scf):
        return "n/a"
    return f"{gas_ip(scf):,.1f} {U.label('gas_ip', SYS)}"


PU, GU = U.label("pressure", SYS), U.label("gas_ip", SYS)

t_pvt, t_prod, t_res, t_hm, t_aq, t_reg, t_rp, t_wl, t_fc = st.tabs(
    ["1 · PVT", "2 · Production", "3 · Reservoir",
     "4 · History match", "5 · Aquifer", "6 · Regression",
     "7 · Relative permeability", "8 · Wells", "9 · Forecast"])

# ============================================================ 1. PVT
with t_pvt:
    st.subheader("Fluid description")
    c = st.columns(4, vertical_alignment="bottom")
    num("Reservoir temperature", "T", "temperature", c[0], step=5.0)
    num("Dew point pressure", "pd", "pressure", c[1], step=50.0, minv=100.0)
    num("Initial condensate-gas ratio", "cgr_i", "cgr", c[2], step=5.0, minv=0.0)
    num("Condensate gravity", "api", "api", c[3], step=1.0, minv=10.0, maxv=90.0)
    c = st.columns(4, vertical_alignment="bottom")
    num("Separator gas gravity (air = 1)", "sg", None, c[0], step=0.01, minv=0.55, maxv=1.8)
    none_ok = "Leave empty if there is none."
    num("CO₂ (mole fraction)", "co2", None, c[1], step=0.01, minv=0.0, maxv=0.9, help=none_ok)
    num("H₂S (mole fraction)", "h2s", None, c[2], step=0.01, minv=0.0, maxv=0.9, help=none_ok)
    num("N₂ (mole fraction)", "n2", None, c[3], step=0.01, minv=0.0, maxv=0.9, help=none_ok)

    INERTS = [V[k] or 0.0 for k in ("co2", "h2s", "n2")]
    FLUID_MISS = missing(["T", "pd", "cgr_i", "api", "sg"])
    c = st.columns(4)
    if FLUID_MISS:
        SG_WET = GE = None
        st.info("Enter the fluid description: " + ", ".join(FLUID_MISS).lower() + ".")
    else:
        SG_WET = wet_gas_gravity(V["sg"], V["cgr_i"], V["api"])
        GE = gas_equivalent(V["api"])
        sg_o, mw_o = condensate_props(V["api"])
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
                                                  V["pvt_pmax"], *INERTS, V["z2"])
                refresh()
            st.button("Generate table (replaces current table)", on_click=_fill_pvt,
                      disabled=bool(FLUID_MISS) or V["pvt_pmax"] is None,
                      help="Needs the fluid description above and the maximum table pressure.")
        with st.expander("Calculate two-phase Z from a CVD report"):
            st.caption("For reports that give retrograde liquid deposit, cumulative produced "
                       "fluid and the single-phase (equilibrium gas) Z instead of the two-phase "
                       "Z. Enter the CVD rows, including the dew point row if the report has it. "
                       "Rows above the dew point need only pressure and gas Z.")
            cvd_df = table("cvd", height=260)
            csv_upload("cvd", "", expander=False)
            num("Gas Z-factor at the dew point (empty = take from the table)", "cvd_zd", None,
                step=0.01, minv=0.0, maxv=3.0)
            CVD_OUT = None
            if cvd_df["p"].notna().any() and V["pd"] is None:
                st.info("Enter the dew point pressure under Fluid description first.")
            elif cvd_df["p"].notna().any():
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
        if FLUID_MISS:
            raise ValueError("Enter the fluid description: " + ", ".join(FLUID_MISS).lower() + ".")
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
        elif not FLUID_MISS:
            st.info(PVT_ERR)

# ============================================================ 2. Production history
with t_prod:
    c = st.columns([1, 2, 2])
    V["start_date"] = c[0].date_input("Start of production", value=V["start_date"], format="DD/MM/YYYY",
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
                            + "; ".join(f"{U.to_disp('pressure', r.p, SYS):,.6g} {PU} on {r.date:%d/%m/%Y}"
                                        for r in WB["pre_start"].itertuples()))
                wide = WB["spread"][WB["spread"]["n"] > 1].sort_values("range", ascending=False)
                if len(wide) and wide["range"].iloc[0] > 50.0:
                    r = wide.iloc[0]
                    st.warning(f"Wells disagree by {U.to_disp('pressure', r['range'], SYS):,.0f} {PU} on "
                               f"{r['date']:%d/%m/%Y}. The average is used; correct it in the survey "
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
            date_order_check("hist")
            csv_upload("hist", "Dates as dd/mm/yyyy. Use and Weight are optional.")
    else:
        left, right = st.columns([3, 3])
        with left:
            st.markdown("**Well cumulative production**")
            wells_df = table("wells", height=300)
            csv_upload("wells", "Dates as dd/mm/yyyy.")
            st.markdown("**Average reservoir pressure surveys**")
            pres_df = table("pres", height=240)
            date_order_check("pres")
            csv_upload("pres", "Dates as dd/mm/yyyy. Use and Weight are optional.")
        try:
            hist_df = aggregate_wells(wells_df, pres_df)
        except Exception as e:  # noqa: BLE001
            hist_df = empty_table("hist")
            st.warning(f"Could not combine the wells: {e}")

    HIST_OK, hist = True, None
    try:
        if V["start_date"] is None:
            raise ValueError("Enter the start of production date.")
        pi_now = peek("pi", "pressure")
        hist = build_history(hist_df, V["start_date"], np.nan if pi_now is None else pi_now)
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
            if GE is not None:
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
    c = st.columns(3, vertical_alignment="bottom")
    num("Initial reservoir pressure", "pi", "pressure", c[0], step=50.0, minv=100.0)
    num("Porosity", "phi", "frac", c[1], step=0.01, minv=0.01, maxv=0.6)
    num("Connate water saturation", "swi", "frac", c[2], step=0.01, minv=0.0, maxv=0.9)
    c = st.columns(3, vertical_alignment="bottom")
    num("Rock compressibility", "cf", "compress", c[0], step=0.5, minv=0.0)
    num("Original gas in place, initial estimate (wet gas)", "G", "gas_ip", c[1], step=10.0, minv=0.001,
        help="Starting value. The history match and regression refine it.")

    def _hall():
        V["cf"] = float(f"{hall_cf(V['phi']):.3g}")
        refresh()
    c[2].button("Estimate rock compressibility (Hall)", on_click=_hall, disabled=V["phi"] is None,
                help="Needs the porosity.")

    st.subheader("Formation water")
    c = st.columns(4, vertical_alignment="bottom")
    num("Water formation volume factor", "bw", "fvf_w", c[0], step=0.01, minv=0.9, maxv=1.3)
    num("Water compressibility", "cw", "compress", c[1], step=0.1, minv=0.0)
    num("Water viscosity", "aq_muw", "visc", c[2], step=0.01, minv=0.01)
    num("Water salinity", "salinity", "ppm", c[3], step=5000.0, minv=0.0,
        help="Only used by the estimate below. Empty counts as fresh water.")

    def _est_water():
        bw, cw, mu = water_props(V["pi"], V["T"], V["salinity"] or 0.0)
        V.update(bw=round(bw, 4), cw=float(f"{cw:.3g}"), aq_muw=round(mu, 3))
        refresh()
    st.button("Estimate water properties from correlations", on_click=_est_water,
              disabled=V["T"] is None or V["pi"] is None,
              help="McCain Bw, Osif cw, Beggs-Brill viscosity at initial pressure (above) and "
                   "reservoir temperature (tab 1); both must be entered first.")

    RES_MISS = missing(["pi", "swi", "cf", "G", "bw", "cw"])
    if RES_MISS:
        st.info("Still needed: " + ", ".join(RES_MISS).lower() + ".")
    if V["pi"] is not None and V["pd"] is not None and V["pi"] < V["pd"]:
        st.warning("Initial pressure is below the dew point: the reservoir starts two-phase. "
                   "The two-phase Z-factor must then apply from initial conditions.")
    if PVT_OK and not RES_MISS:
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
READY = PVT_OK and HIST_OK and not RES_MISS
AQ_MISS, NB = [], None
if READY:
    lo, hi = pvt.p_range
    if V["pi"] > hi * 1.0001 or hist.p.min() < lo:
        RANGE_WARN = ("Some pressures fall outside the PVT table; Z is held constant beyond the "
                      "table limits. Extend the table in tab 1 for reliable results.")
    else:
        RANGE_WARN = None
    tank0 = make_tank(cfg(), pvt, hist, with_aquifer=False)
    DET = detect_drive(tank0)
    n_pts = len(hist.p)
    DATES = pd.Timestamp(V["start_date"]) + pd.to_timedelta(hist.t, unit="D")
    LAB = np.array([f"{d:%d/%m/%Y}" for d in DATES])
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
    elif RES_MISS:
        st.info("Complete the reservoir parameters first: " + ", ".join(RES_MISS).lower() + ".")
    elif AQ_MISS:
        st.info("Complete tab 5 first: " + ", ".join(AQ_MISS).lower()
                + ". Or switch that aquifer or neighbouring reservoir off.")


def detection_banner(short=False):
    msg = DET["message"]
    if DET["status"] == "aquifer":
        if V["aq_model"] == "none" and V.get("nb_on"):
            st.success(f"**Additional energy source detected**; a neighbouring reservoir is modelled. {msg}")
        elif V["aq_model"] == "none":
            st.warning(f"**Additional energy source detected.** {msg}"
                       + (" Go to tab 5 to define the aquifer or a neighbouring reservoir." if short else ""))
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
    c = st.columns(2, vertical_alignment="bottom")
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
    c = st.columns(3, vertical_alignment="bottom")
    if g == "radial":
        num("Reservoir radius", "aq_ro", "length", c[0], step=250.0, minv=10.0)
        inf_ok = m != "fetkovich"
        if inf_ok:
            V["aq_inf"] = c[1].checkbox("Infinite-acting aquifer", value=bool(V["aq_inf"]),
                                        key=f"cb_inf_{KEY}")
        if not (inf_ok and V["aq_inf"]):
            num("Outer/inner radius ratio (re/ro)", "aq_reD", "ratio", c[1], step=0.5, minv=1.05)
        num("Encroachment angle", "aq_theta", "angle", c[2], step=10.0, minv=1.0, maxv=360.0)
        c = st.columns(3, vertical_alignment="bottom")
        num("Aquifer thickness", "aq_h", "length", c[0], step=5.0, minv=0.5)
    elif g == "linear":
        num("Aquifer width", "aq_width", "length", c[0], step=250.0, minv=10.0)
        num("Aquifer length", "aq_L", "length", c[1], step=1000.0, minv=10.0)
        num("Aquifer thickness", "aq_h", "length", c[2], step=5.0, minv=0.5)
        c = st.columns(3, vertical_alignment="bottom")
    else:
        num("Reservoir radius", "aq_ro", "length", c[0], step=250.0, minv=10.0)
        num("Aquifer thickness (below contact)", "aq_h", "length", c[1], step=5.0, minv=0.5)
        num("Vertical anisotropy kv/kh", "aq_kvkh", "ratio", c[2], step=0.05, minv=1e-5, maxv=1.0)
        c = st.columns(3, vertical_alignment="bottom")
        num("Encroachment angle", "aq_theta", "angle", c[0], step=10.0, minv=1.0, maxv=360.0,
            help="360° when water underlies the whole reservoir.")
    num("Aquifer permeability", "aq_k", "perm", c[1], step=10.0, minv=0.001)
    num("Aquifer porosity", "aq_phi", "frac", c[2], step=0.01, minv=0.01, maxv=0.6)
    st.caption(f"Aquifer total compressibility = cw + cf = "
               f"{U.to_disp('compress', V['cw'] + V['cf'], SYS):.2f} {U.label('compress', SYS)}; "
               "water viscosity " + ("not entered" if V["aq_muw"] is None else f"{V['aq_muw']:.3g} cp")
               + " (tab 3)."
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
            _inf = bool(V["aq_inf"]) and V["aq_model"] != "fetkovich"
            AQ_MISS = missing(aquifer_params(V["aq_model"], V["aq_geom"], _inf)
                              + (["aq_phi", "aq_muw"] if V["aq_model"] in NEEDS_GEOMETRY else []))
            if AQ_MISS:
                st.info("Still needed for this aquifer: " + ", ".join(AQ_MISS).lower() + ".")

            st.subheader("Neighbouring reservoir")
            V["nb_on"] = st.checkbox(
                "This reservoir exchanges gas with a neighbouring reservoir", value=bool(V.get("nb_on")),
                key=f"cb_nb_{KEY}",
                help="For a reservoir that leaks to or from another sand, across a non-sealing "
                     "fault or through a wellbore. Gas crosses in proportion to the pressure "
                     "difference: rate = transmissibility × (neighbour pressure − this reservoir's "
                     "pressure). It flows in when the neighbour is at higher pressure and out "
                     "when it is lower.")
            if V["nb_on"]:
                c = st.columns([2, 3], vertical_alignment="top")
                with c[0]:
                    num("Transmissibility", "nb_T", "transmiss", minv=0.0,
                        help="Starting value; regress it in tab 6. As a guide, 10 Mscf/d/psi moves "
                             "about 1 Bscf a year across a 300 psi difference.")
                    choice("In the forecast, the neighbour", "nb_fc",
                           {"deplete": "depletes as it gives up gas",
                            "cutoff": "stops exchanging gas",
                            "constant": "stays at its last pressure (unlimited source)"},
                           help="'Depletes' treats the neighbour as a closed gas tank: its p/z falls "
                                "in proportion to the gas it gives up. 'Stays at its last pressure' "
                                "is an unlimited supply and can recover more gas than both "
                                "reservoirs hold, so use it only as an upper bound.")
                    if V["nb_fc"] == "deplete":
                        num("Neighbour gas in place at its last survey", "nb_G", "gas_ip", minv=0.001,
                            help="Gas the neighbour still held at the date of its last pressure survey.")
                    st.caption("Enter the neighbour's static pressures at the same datum as this "
                               "reservoir's. Between surveys the pressure is interpolated; after the "
                               "last survey it is held constant through the rest of the history.")
                with c[1]:
                    nbr_df = table("nbr", height=215)
                    date_order_check("nbr")
                    csv_upload("nbr", "Dates as dd/mm/yyyy.")
                d_ = nbr_df.dropna(subset=["date", "p"])
                if HIST_OK and not d_.empty:
                    last_ = pd.to_datetime(d_["date"]).max()
                    end_ = pd.Timestamp(V["start_date"]) + pd.to_timedelta(hist.t[-1], unit="D")
                    if (end_ - last_).days > 365:
                        st.warning(f"The neighbour's last survey is {last_:%d/%m/%Y}, "
                                   f"{(end_ - last_).days / 365.25:.1f} years before the end of the history. "
                                   f"Its pressure is held at {U.to_disp('pressure', float(d_.sort_values('date')['p'].iloc[-1]), SYS):,.0f} {PU} "
                                   "from then on, which feeds this reservoir as if the neighbour never "
                                   "depleted. Add its later surveys.")
                if V["nb_T"] is None:
                    AQ_MISS = AQ_MISS + ["transmissibility to the neighbouring reservoir"]
                if d_.empty:
                    AQ_MISS = AQ_MISS + ["at least one neighbour pressure"]
                else:
                    NB = ((pd.to_datetime(d_["date"]) - pd.Timestamp(V["start_date"])).dt.days.to_numpy(float),
                          d_["p"].to_numpy(float))
                if V["nb_T"] is None or d_.empty:
                    st.info("Still needed: " + ", ".join(AQ_MISS[-(int(V["nb_T"] is None) + int(d_.empty)):]) + ".")

READY = READY and not AQ_MISS
if READY:
    tank = make_tank(cfg(), pvt, hist, neighbour=NB)
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
    if READY and tank.has_neighbour:
        with left:
            st.subheader("Exchange with the neighbouring reservoir over the history")
            gx_ = tank.gx_sim
            dpn = tank.pn - P_SIM
            c = st.columns(3)
            c[0].metric(f"Net gas received ({GU})", f"{gas_ip(gx_[-1]):+,.2f}",
                        help="Positive: gas has come in from the neighbour. Negative: this "
                             "reservoir has lost gas to it.")
            c[1].metric(f"Pressure difference now ({PU})",
                        f"{U.to_disp('pressure', dpn[-1], SYS):+,.0f}", help="Neighbour minus this reservoir.")
            c[2].metric(f"Transfer rate now ({U.label('gas_rate', SYS)})",
                        f"{U.to_disp('gas_rate', tank.nb_T * dpn[-1] / 1e6, SYS):+,.2f}")
            f = figure("", f"Pressure ({PU})", height=300)
            f.add_trace(go.Scatter(x=DATES, y=U.to_disp("pressure", P_SIM, SYS), mode="lines",
                                   name="This reservoir (model)", line=dict(color=ORANGE, width=2)))
            f.add_trace(go.Scatter(x=DATES, y=U.to_disp("pressure", tank.pn, SYS), mode="lines",
                                   name="Neighbour (interpolated)", line=dict(color=GREY, width=2, dash="dash")))
            nd_ = pd.Timestamp(V["start_date"]) + pd.to_timedelta(NB[0], unit="D")
            f.add_trace(go.Scatter(x=nd_, y=U.to_disp("pressure", NB[1], SYS), mode="markers",
                                   name="Neighbour surveys", marker=dict(color=GREY, size=8)))
            history_points(f, DATES, U.to_disp("pressure", hist.p, SYS))
            show(f, "nb_p")
            f = figure("", f"Cumulative gas received ({GU})", height=240, legend=False)
            f.add_trace(go.Scatter(x=DATES, y=gas_ip(gx_), mode="lines", line=dict(color=BLUE, width=2)))
            f.add_hline(y=0, line_color=GREY, line_width=1)
            show(f, "nb_g")

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
            c = st.columns([2, 3], vertical_alignment="bottom")
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
            lab = [f"{d:%d/%m/%Y}" for d in dates]
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
            if AQ.active or tank.has_neighbour:
                p_na = tank0.simulate()[0]
                fig.add_trace(go.Scatter(x=gas_ip(tank.gpw), y=U.to_disp("pressure", p_na, SYS),
                                         mode="lines", name="Closed tank (no aquifer or neighbour)",
                                         line=dict(color=GREY, width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=gas_ip(tank.gpw), y=U.to_disp("pressure", P_SIM, SYS), mode="lines",
                                     name="Model" + (f" ({MODELS[AQ.model]})" if AQ.active else " (no aquifer)")
                                     + (" with neighbour" if tank.has_neighbour else ""),
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
                    "Date": [f"{d:%d/%m/%Y}" for d in dates],
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
            for (name, y), col in zip(en.items(), (BLUE, YELLOW, AQUA, ORANGE)):
                fig.add_trace(go.Scatter(x=dates[1:], y=y[1:], mode="lines", name=name, stackgroup="e",
                                         line=dict(width=1, color=col),
                                         hovertemplate="%{y:.1%}<extra>" + name + "</extra>"))
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_layout(hovermode="x unified")
            show(fig, "en")
            c = st.columns(len(en))
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
            wt = pd.to_numeric(upd[dd][1], errors="coerce")
            df.at[i, "use"], df.at[i, "w"] = as_flag(upd[dd][0]), 1.0 if pd.isna(wt) else float(wt)
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
                c = st.columns([3, 2], vertical_alignment="bottom")
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
                    "Date": [f"{d:%d/%m/%Y}" for d in DATES[1:]],
                    f"Pressure ({PU})": np.round(U.to_disp("pressure", hist.p[1:], SYS), 1),
                    f"Deviation ({PU})": np.round(dev[1:] * pf, 1),
                    "Score": np.round(z[1:], 1),
                    "Flag": np.where(flag[1:], "suspect", ""),
                    "Use": pd.array(hist.use[1:], dtype="boolean"), "Weight": hist.w_in[1:]})
                ed_s = st.data_editor(
                    tbl, hide_index=True, width="stretch", height=330,
                    disabled=[c_ for c_ in tbl.columns if c_ not in ("Use", "Weight")],
                    column_config={"Weight": st.column_config.NumberColumn(min_value=0.0, format="%.3g",
                                                                           width="small"),
                                   "Use": st.column_config.CheckboxColumn(width="small"),
                                   "Date": st.column_config.TextColumn(width="small"),
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
            if tank.has_neighbour:
                keys.append("nb_T")
            default_on = {"G"} | set({"pot": ["aq_Wvol"], "schilthuis": ["aq_C"]}.get(
                V["aq_model"], [k for k in keys if k in ("aq_reD", "aq_k", "aq_L")])) | {"nb_T"}
            rows = []
            for k in keys:
                lab_, q, lo_, hi_ = PARAMS[k]
                cur_ = V[k]
                a, b = max(cur_ * 0.2, lo_), min(cur_ * 5.0, hi_)
                if k == "aq_reD":
                    a, b = 1.2, max(20.0, cur_ * 2)
                if k == "aq_theta":
                    a, b = 10.0, 360.0
                if k == "nb_T":
                    a, b = max(cur_ * 0.02, lo_), min(cur_ * 50.0, hi_)
                rows.append({"Regress": k in default_on, "Parameter": ulabel(lab_, q),
                             "Current value": U.to_disp(q, cur_, SYS),
                             "Minimum": U.to_disp(q, a, SYS), "Maximum": U.to_disp(q, b, SYS)})
            left, right = st.columns([3, 3])
            with left:
                st.caption("Tick the parameters to adjust and set their limits. Regress on as few "
                           "as needed: many aquifer parameters compensate for one another.")
                ed = st.data_editor(
                    pd.DataFrame(rows).astype({"Regress": "boolean"}), hide_index=True, width="stretch",
                    disabled=["Parameter", "Current value"],
                    column_config={c_: st.column_config.NumberColumn(format="%.5g")
                                   for c_ in ("Current value", "Minimum", "Maximum")},
                    key=f"reg_{V['aq_model']}_{V['aq_geom']}_{V['aq_inf']}_{V.get('nb_on')}_{KEY}")
                LOSSES = {"Least squares": "linear", "Robust (soft-L1)": "soft_l1"}
                c = st.columns([2, 3], vertical_alignment="bottom")
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
                if not AQ.active and not tank.has_neighbour and DET["status"] == "aquifer":
                    st.warning("An additional energy source was detected but no aquifer is defined. "
                               "Regression on G alone will not match the pressure trend; define the "
                               "aquifer in tab 5 first.")

            def make_sim(base, rk):
                def sim(vals):
                    return make_tank(dict(base, **dict(zip(rk, vals))), pvt, hist, neighbour=NB).simulate()[0]
                return sim

            if go_btn:
                pick = [i for i, r in ed.iterrows() if pd.notna(r["Regress"]) and bool(r["Regress"])]
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
                        base = cfg()
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


# ============================================================ 7. Relative permeability
RP_TYPICAL = dict(rp_sgrw=0.25, rp_krw=0.3, rp_krg=1.0, rp_nw=3.0, rp_ng=2.0,
                  rp_soc=0.2, rp_no=3.0, rp_ngo=2.0)
RP, STATE, RP_MISS = None, None, []
with t_rp:
    st.caption("Tank-scale curves used by the forecast. They decide how much water the wells make "
               "as the aquifer advances, how much gas is trapped behind it, and how far "
               "retrograde condensate reduces well deliverability. They do not change the "
               "history match of tabs 4 to 6.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Gas and water**")
        c = st.columns(3, vertical_alignment="bottom")
        num("Residual gas saturation", "rp_sgrw", "frac", c[0], minv=0.0, maxv=0.6,
            help="Gas left behind where water has swept. Sets the saturation at which gas stops flowing.")
        num("Water end point, krw", "rp_krw", None, c[1], minv=1e-4, maxv=1.0,
            help="Water relative permeability at residual gas.")
        num("Gas end point, krg", "rp_krg", None, c[2], minv=0.01, maxv=1.0,
            help="Gas relative permeability at connate water.")
        c = st.columns(3, vertical_alignment="bottom")
        num("Water exponent", "rp_nw", None, c[0], minv=1.0, maxv=8.0)
        num("Gas exponent", "rp_ng", None, c[1], minv=1.0, maxv=8.0)
        st.markdown("**Gas and condensate**")
        c = st.columns(3, vertical_alignment="bottom")
        num("Critical condensate saturation", "rp_soc", "frac", c[0], minv=0.0, maxv=0.6,
            help="Condensate in the reservoir flows only above this saturation.")
        num("Condensate exponent", "rp_no", None, c[1], minv=1.0, maxv=8.0)
        num("Gas exponent", "rp_ngo", None, c[2], minv=1.0, maxv=8.0)
        V["rp_vap"] = st.checkbox(
            "Include water vapour condensing from the gas", value=bool(V.get("rp_vap", True)),
            key=f"cb_vap_{KEY}",
            help="Reservoir gas carries water vapour that condenses at surface (Bukacek "
                 "correlation). It is produced with no aquifer at all, so it is separated from "
                 "free water before the curves are fitted.")

        def _rp_typical():
            V.update(RP_TYPICAL)
            refresh()
        st.button("Fill typical values", on_click=_rp_typical,
                  help="Corey values often used when there is no core data. Replace or fit them.")
        st.caption("Connate water saturation is taken from tab 3.")

    LABELS.update(rp_ng="gas exponent (gas and water)", rp_ngo="gas exponent (gas and condensate)")
    RP_MISS = missing(list(RP_TYPICAL))
    if not RP_MISS and V["swi"] is not None:
        if V["swi"] + V["rp_sgrw"] >= 0.95:
            RP_MISS = ["connate water plus residual gas must leave some movable gas"]
        else:
            RP = RelPerm(swc=V["swi"], sgrw=V["rp_sgrw"], krw_max=V["rp_krw"], krg_max=V["rp_krg"],
                         nw=V["rp_nw"], ng=V["rp_ng"], soc=V["rp_soc"], no=V["rp_no"], ngo=V["rp_ngo"])
    if READY and V["aq_muw"] is not None:
        STATE = TankState(tank, V["aq_muw"], ss.cur["cvd"], V["pd"])
    with left:
        if RP_MISS:
            st.info("Still needed: " + ", ".join(RP_MISS).lower() + ".")
        if not READY:
            need_data()
        elif V["aq_muw"] is None:
            st.info("Enter the water viscosity in tab 3 (Formation water).")
        elif not np.all(np.isfinite(P_SIM)):
            st.warning("The model does not reproduce the whole history; fix the match first.")
        elif RP is not None:
            def _fit_rp():
                new, info = fit_water(STATE, RP, P_SIM, WE_SIM, V["rp_vap"])
                ss.rp_fit = info
                if info["ok"]:
                    V.update(rp_krw=float(f"{new.krw_max:.4g}"), rp_nw=float(f"{new.nw:.4g}"))
                    refresh()
                    ss.rp_fit = info
            st.button("Fit water end point and exponent to the water history", on_click=_fit_rp,
                      type="primary")
            info = ss.get("rp_fit")
            if info and info["ok"]:
                st.success(f"Fitted. Cumulative water is matched to within "
                           f"{U.to_disp('liq_cum', info['rms'] / 1e3, SYS):,.2f} {U.label('liq_cum', SYS)}."
                           + (" A value reached its limit, so the history only weakly fixes the "
                              "curve." if info["at_limit"] else ""))
            elif info:
                st.warning(info["reason"])
            if not STATE.has_condensate:
                st.caption("No retrograde liquid data (CVD panel in tab 1), so the effect of "
                           "condensate on gas flow is left out.")
            elif float(STATE.so(hist.p.min())) > 0:
                so_now = float(STATE.so(P_SIM[-1]))
                st.caption(f"Condensate saturation in the tank is now {so_now:.1%} of pore volume: "
                           + ("above" if so_now > RP.soc else "below")
                           + " the critical saturation, so reservoir condensate is "
                           + ("mobile." if so_now > RP.soc else "not mobile and the produced yield "
                              "follows the CGR column of the PVT table."))
    with right:
        if RP is not None:
            cc = st.columns(2)
            sw_ = np.linspace(RP.swc, 1.0 - RP.sgrw, 60)
            krw_, krg_ = RP.gas_water(sw_)
            f = figure("Water saturation", "Relative permeability", height=280)
            f.add_trace(go.Scatter(x=sw_, y=krg_, mode="lines", name="Gas", line=dict(color=ORANGE, width=2)))
            f.add_trace(go.Scatter(x=sw_, y=krw_, mode="lines", name="Water", line=dict(color=BLUE, width=2)))
            with cc[0]:
                show(f, "rp_gw")
            so_ = np.linspace(0.0, 1.0 - RP.swc, 60)
            f = figure("Condensate saturation", "Relative permeability", height=280)
            f.add_trace(go.Scatter(x=so_, y=RP.krg_max * RP.cond_mult(so_), mode="lines", name="Gas",
                                   line=dict(color=ORANGE, width=2)))
            f.add_trace(go.Scatter(x=so_, y=RP.kro(so_), mode="lines", name="Condensate",
                                   line=dict(color=AQUA, width=2)))
            with cc[1]:
                show(f, "rp_go")
            if STATE is not None and np.all(np.isfinite(P_SIM)):
                WH = water_history(STATE, RP, P_SIM, WE_SIM, V["rp_vap"])
                lu = U.label("liq_cum", SYS)
                f = figure("", f"Cumulative water ({lu})", height=300)
                f.add_trace(go.Scatter(x=DATES, y=U.to_disp("liq_cum", hist.wp / 1e3, SYS), mode="markers",
                                       name="Measured", marker=dict(color=BLUE, size=8)))
                f.add_trace(go.Scatter(x=DATES, y=U.to_disp("liq_cum", WH["wp"] / 1e3, SYS), mode="lines",
                                       name="Model total", line=dict(color=ORANGE, width=2)))
                f.add_trace(go.Scatter(x=DATES, y=U.to_disp("liq_cum", WH["wp_vap"] / 1e3, SYS), mode="lines",
                                       name="Water vapour only", line=dict(color=GREY, width=2, dash="dash")))
                show(f, "rp_wp")
                c = st.columns(3)
                c[0].metric("Tank water saturation now", f"{WH['sw'][-1]:.1%}",
                            f"{WH['sw'][-1] - WH['sw'][0]:+.1%} since start", delta_color="off")
                c[1].metric("Gas flows until water saturation", f"{1 - RP.sgrw:.0%}")
                c[2].metric(f"Water vapour in gas now ({U.label('cgr', SYS)})",
                            f"{U.to_disp('cgr', WH['vap'][-1], SYS):.2f}")

# ============================================================ 8. Wells
WELLS, GZ, WELL_ERR = [], None, None


def _f(v):
    v = pd.to_numeric(v, errors="coerce")
    return float(v) if pd.notna(v) else np.nan


with t_wl:
    st.caption("Each well needs an inflow relation, fitted here to its tests, and a pressure limit "
               "to flow against. A test may give the flowing bottomhole pressure directly, or the "
               "tubing-head pressure, which is converted with the completion data below.")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Completion and limits**")
        cfg_df = table("wellcfg", height=180)
        st.caption("Depth, tubing size and wellhead temperature are needed only to use tubing-head "
                   "pressures. Give a minimum tubing-head pressure, a minimum bottomhole pressure, "
                   "or both.")
        st.caption("Where a test has no reservoir pressure, it is interpolated in time between the "
                   "pressure surveys of tab 2.")
        st.markdown("**Well tests**")
        tests_df = table("tests", height=300)
        with st.expander("Import well tests (Excel or CSV)"):
            st.caption("Columns are found by their headings: well, date, gas rate, flowing bottomhole "
                       "pressure, tubing-head pressure, reservoir (static) pressure. Values in the "
                       f"units selected in the sidebar ({U.label('gas_rate', SYS)}, {PU}).")
            tf = st.file_uploader("Well test file", type=["xlsx", "xlsm", "xls", "csv", "txt"], key="up_tests")
            if tf is not None and ss.get("tests_id") != tf.file_id:
                try:
                    d = read_tests(tf.getvalue(), tf.name)
                    d["qg"] = U.from_disp("gas_rate", d["qg"], SYS)
                    for k in ("pwf", "pth", "pr"):
                        d[k] = U.from_disp("pressure", d[k], SYS)
                    ss.tests_id = tf.file_id
                    sync_tables()
                    names = [str(x) for x in d["well"].unique()]
                    have = set(ss.cur["wellcfg"]["well"].dropna().astype(str))
                    add = pd.DataFrame({"well": [n_ for n_ in names if n_ not in have]})
                    ss.cur["tests"] = d
                    if len(add):
                        ss.cur["wellcfg"] = pd.concat([ss.cur["wellcfg"], add], ignore_index=True)
                    refresh()
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Could not read the file: {e}")

        st.markdown("**Lift curves**")
        st.caption("Optional. A Prosper lift-curve table (.tpd) replaces the built-in tubing "
                   "calculation for its well: it carries the well model's multiphase correlation, so "
                   "condensate and water in the tubing and liquid loading are included. Export the "
                   "table with tubing-head pressure, water-gas ratio and total GOR as sensitivity "
                   "variables. The .vlp file Prosper writes alongside is a binary copy of the same "
                   "table; load the .tpd.")
        V.setdefault("lift", {})
        lift_names = [str(x).strip() for x in cfg_df["well"].dropna() if str(x).strip()]
        choice("Pressures in the lift-curve files are", "lift_gauge",
               {"psig": "psig (Prosper oilfield units)", "psia": "psia"}, horizontal=True)
        ups = st.file_uploader("Lift-curve files", type=["tpd"], accept_multiple_files=True,
                               key=f"up_lift_{ss.ver}")
        pend = []
        for f_ in ups or []:
            try:
                tab_ = read_tpd(f_.getvalue().decode("latin-1"), f_.name, V["lift_gauge"] == "psig")
            except Exception as e:  # noqa: BLE001
                st.warning(f"{f_.name}: {e}")
                continue
            toks = [t for t in re.split(r"[^0-9a-z]+", f_.name.lower().rsplit(".", 1)[0]) if t]
            norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())   # noqa: E731
            hits = [n_ for n_ in lift_names if norm(n_)
                    and any(t == norm(n_) or (len(norm(n_)) >= 2 and t.endswith(norm(n_))) for t in toks)]
            guess = max(hits, key=len) if hits else None
            opts = lift_names or ["(enter the wells in the table above first)"]
            pick = st.selectbox(f"Well for {f_.name}", opts,
                                index=opts.index(guess) if guess in opts else 0, key=f"lw_{f_.name}_{ss.ver}")
            pend.append((pick, tab_))

        def _attach(items):
            for nm_, tb_ in items:
                V["lift"][nm_] = tb_.to_dict()
            refresh()
        if pend and lift_names:
            st.button(f"Attach {len(pend)} lift-curve table(s)", type="primary", on_click=_attach, args=(pend,))
        if V["lift"]:
            rows_l = []
            for nm_, d_ in V["lift"].items():
                rows_l.append({"Well": nm_, "File": d_.get("name", ""),
                               "Correlation": d_.get("info", {}).get("correlation", ""),
                               f"Rates ({U.label('gas_rate', SYS)})":
                                   f"{U.to_disp('gas_rate', min(d_['q']), SYS):.3g} to {U.to_disp('gas_rate', max(d_['q']), SYS):.3g}",
                               f"Tubing-head pressures ({PU})":
                                   f"{U.to_disp('pressure', min(d_['thp']), SYS):,.0f} to {U.to_disp('pressure', max(d_['thp']), SYS):,.0f}",
                               "Water-gas ratios (STB/MMscf)": f"{min(d_['wgr']):.4g} to {max(d_['wgr']):.4g}",
                               "Condensate-gas ratios (STB/MMscf)": f"{1e6 / max(d_['gor']):.3g} to {1e6 / min(d_['gor']):.3g}"})
            st.dataframe(pd.DataFrame(rows_l), hide_index=True, width="stretch")
            c = st.columns([2, 1], vertical_alignment="bottom")
            rm = c[0].selectbox("Remove lift curves from", list(V["lift"]), key=f"lrm_{KEY}")

            def _rm(nm_):
                V["lift"].pop(nm_, None)
                refresh()
            c[1].button("Remove", on_click=_rm, args=(rm,))

    if not READY:
        with right:
            need_data()
    elif RP is None or STATE is None:
        with right:
            st.info("Complete tab 7 (relative permeability) first.")
    elif not np.all(np.isfinite(P_SIM)):
        with right:
            st.warning("The model does not reproduce the whole history; fix the match first.")
    else:
        GZ = GasZ(SG_WET, *INERTS)
        SW_H = STATE.sw(P_SIM, WE_SIM, hist.wp)
        M_H = STATE.mobility(RP, P_SIM, SW_H)
        cgr_now = float(pvt.cgr(P_SIM[-1])) if np.nanmax(pvt.cgr_tab) > 0 else V["cgr_i"]
        CGR_CONST = None if np.nanmax(pvt.cgr_tab) > 0 else V["cgr_i"]
        # field water-gas ratio over each history interval (STB/MMscf), for tests without one
        with np.errstate(divide="ignore", invalid="ignore"):
            WGR_H = np.diff(hist.wp) / np.diff(hist.gp) * 1e6
        WGR_H = np.nan_to_num(np.concatenate([WGR_H[:1], WGR_H]), nan=0.0)
        sw_now = float(SW_H[-1])
        WGR_NOW = (float(STATE.wgr_free(RP, P_SIM[-1], sw_now)) * (1.0 + GE * cgr_now / 1e6)
                   + (float(STATE.wgr_vapour(P_SIM[-1])) if V["rp_vap"] else 0.0))
        tdf = tests_df.dropna(subset=["well", "qg"]).copy()
        tdf["well"] = tdf["well"].astype(str).str.strip()
        names = [str(x).strip() for x in cfg_df["well"].dropna() if str(x).strip()]
        names += [x for x in tdf["well"].unique() if x not in names]
        rows, PTS = [], {}
        for nm in names:
            r = cfg_df[cfg_df["well"].astype(str).str.strip() == nm]
            r = r.iloc[0] if len(r) else {}
            w = Well(nm, tvd=_f(r.get("tvd")), md=_f(r.get("md")), tid=_f(r.get("tid")),
                     wht=_f(r.get("wht")), min_thp=_f(r.get("min_thp")), min_bhp=_f(r.get("min_bhp")),
                     qmax=_f(r.get("qmax")))
            if nm in V.get("lift", {}):
                w.lift = LiftTable.from_dict(V["lift"][nm])
            qs, xs, skipped = [], [], 0
            for t_ in tdf[tdf["well"] == nm].itertuples():
                day = ((pd.Timestamp(t_.date) - pd.Timestamp(V["start_date"])).days
                       if pd.notna(t_.date) else hist.t[-1])
                pr_ = (_f(t_.pr) if np.isfinite(_f(t_.pr))
                       else float(np.interp(day, hist.t[hist.use], hist.p[hist.use])))
                m_ = float(np.interp(day, hist.t, M_H))
                cgr_ = float(pvt.cgr(pr_)) if CGR_CONST is None else CGR_CONST
                wet_ = 1.0 + GE * cgr_ / 1e6
                wgr_ = (_f(getattr(t_, "wgr", np.nan)) if np.isfinite(_f(getattr(t_, "wgr", np.nan)))
                        else float(np.interp(day, hist.t, WGR_H)))
                if np.isfinite(_f(t_.pwf)):
                    pwf_ = _f(t_.pwf)
                elif np.isfinite(_f(t_.pth)) and w.has_tubing:
                    pwf_ = tubing_bhp(_f(t_.pth), _f(t_.qg), GZ, V["T"], w, wet_, cgr_, wgr_)
                else:
                    skipped += 1
                    continue
                qs.append(_f(t_.qg))
                xs.append(m_ * (pr_ ** 2 - pwf_ ** 2))
            w.C, w.n, w.note = fit_ipr(qs, xs)
            w.n_tests = int(np.sum(np.array(xs) > 0)) if xs else 0
            if skipped:
                w.note = (w.note + "; " if w.note else "") + (
                    f"{skipped} tubing-head test(s) not used: depth, tubing size and wellhead "
                    "temperature are needed")
            if np.isfinite(w.C) and w.control is None:
                w.note = (w.note + "; " if w.note else "") + "no pressure limit set, so it is left out of the forecast"
            PTS[nm] = (np.array(qs), np.array(xs))
            WELLS.append(w)
            wet_now = 1.0 + GE * cgr_now / 1e6
            q_now, pwf_now, load_now = well_rate(w, P_SIM[-1], M_H[-1], GZ, V["T"], wet_now, cgr_now, WGR_NOW)
            aof = w.C * (M_H[-1] * P_SIM[-1] ** 2) ** w.n if np.isfinite(w.C) else np.nan
            rows.append({"Well": nm, "Tests used": w.n_tests,
                         "C": w.C, "n": w.n,
                         f"Open-flow potential now ({U.label('gas_rate', SYS)})": U.to_disp("gas_rate", aof, SYS),
                         f"Rate at limit now ({U.label('gas_rate', SYS)})": U.to_disp("gas_rate", q_now, SYS),
                         f"Flowing BHP at limit ({PU})": U.to_disp("pressure", pwf_now, SYS),
                         "Limit": {"thp": "tubing head", "bhp": "bottomhole", None: "none"}[w.control],
                         "Tubing": ("lift curves" if w.lift is not None else
                                    "built-in" if w.has_tubing else "-"),
                         "Loading now": ("yes" if load_now else "no") if w.lift is not None else "-",
                         "Note": w.note})
        with right:
            if not WELLS:
                st.info("Enter the wells and their tests on the left.")
            else:
                f = figure("Mobility × (pr² − pwf²)  (psi²)", f"Gas rate ({U.label('gas_rate', SYS)})", height=380)
                for i, w in enumerate(WELLS):
                    col = [BLUE, ORANGE, AQUA, YELLOW, RED, GREY][i % 6]
                    qs, xs = PTS[w.name]
                    ok = xs > 0
                    if ok.any():
                        f.add_trace(go.Scatter(x=xs[ok], y=U.to_disp("gas_rate", qs[ok], SYS), mode="markers",
                                               name=f"{w.name} tests", marker=dict(color=col, size=9)))
                    if np.isfinite(w.C) and ok.any():
                        xx = np.geomspace(xs[ok].min() * 0.5, max(xs[ok].max(), M_H[-1] * P_SIM[-1] ** 2), 40)
                        f.add_trace(go.Scatter(x=xx, y=U.to_disp("gas_rate", w.C * xx ** w.n, SYS), mode="lines",
                                               name=f"{w.name} fit", line=dict(color=col, width=2)))
                f.update_xaxes(type="log")
                f.update_yaxes(type="log")
                show(f, "ipr")
                st.caption("Deliverability plot. Each line is q = C · [M (pr² − pwf²)]ⁿ fitted to that "
                           "well's tests; M corrects for the change in gas mobility since initial "
                           "conditions.")
                op_w = [w for w in WELLS if np.isfinite(w.C) and w.control == "thp"]
                if op_w:
                    wsel = st.selectbox("Operating point now for", [w.name for w in op_w], key=f"opw_{KEY}")
                    w = next(x for x in op_w if x.name == wsel)
                    pr_n, m_n = P_SIM[-1], M_H[-1]
                    wet_n = 1.0 + GE * cgr_now / 1e6
                    qmax_ = w.C * (m_n * pr_n ** 2) ** w.n
                    qq = np.geomspace(max(qmax_ * 1e-3, 0.05), qmax_, 80)
                    ipr_p = np.sqrt(np.maximum(pr_n ** 2 - (qq / w.C) ** (1 / w.n) / m_n, 0.0))
                    vlp_p = np.array([tubing_bhp(w.min_thp, x, GZ, V["T"], w, wet_n, cgr_now, WGR_NOW) for x in qq])
                    f = figure(f"Gas rate ({U.label('gas_rate', SYS)})", f"Flowing bottomhole pressure ({PU})", height=340)
                    f.add_trace(go.Scatter(x=U.to_disp("gas_rate", qq, SYS), y=U.to_disp("pressure", ipr_p, SYS),
                                           mode="lines", name="Inflow", line=dict(color=BLUE, width=2)))
                    f.add_trace(go.Scatter(x=U.to_disp("gas_rate", qq, SYS), y=U.to_disp("pressure", vlp_p, SYS),
                                           mode="lines", name="Lift" + (" (table)" if w.lift is not None else " (built-in)"),
                                           line=dict(color=ORANGE, width=2)))
                    if w.lift is not None:
                        ld = np.array([w.lift.loading_at(w.min_thp, x, WGR_NOW, cgr_now) for x in qq])
                        if ld.any():
                            f.add_trace(go.Scatter(x=U.to_disp("gas_rate", qq[ld], SYS),
                                                   y=U.to_disp("pressure", vlp_p[ld], SYS), mode="markers",
                                                   name="Below Turner rate", marker=dict(color=RED, size=5)))
                    qo, po, lo_ = well_rate(w, pr_n, m_n, GZ, V["T"], wet_n, cgr_now, WGR_NOW)
                    if qo > 0:
                        f.add_trace(go.Scatter(x=[U.to_disp("gas_rate", qo, SYS)], y=[U.to_disp("pressure", po, SYS)],
                                               mode="markers", name="Operating point",
                                               marker=dict(color=AQUA, size=13, symbol="diamond")))
                    show(f, "opp")
                    st.caption(f"At today's model reservoir pressure ({U.to_disp('pressure', pr_n, SYS):,.0f} {PU}), "
                               f"tubing-head pressure {U.to_disp('pressure', w.min_thp, SYS):,.0f} {PU}, "
                               f"condensate-gas ratio {cgr_now:.1f} and water-gas ratio {WGR_NOW:.2f} STB/MMscf. "
                               + ("No crossing: the well cannot lift its liquids at this pressure." if qo <= 0 else ""))
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"C": st.column_config.NumberColumn(format="%.3e"),
                                        "n": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Wells without lift curves use the built-in tubing calculation: single-phase gas "
                       "with the well-stream gravity and no liquid loading, so their late-life rates are "
                       "optimistic. Tests without a water-gas ratio use the field value from the "
                       "production history at the test date.")

# ============================================================ 9. Forecast
with t_fc:
    c = st.columns(4, vertical_alignment="bottom")
    num("Forecast length (years)", "fc_years", None, c[0], minv=0.5, maxv=60.0)
    choice("Time step", "fc_step", {"Monthly": "Monthly", "Quarterly": "Quarterly", "Yearly": "Yearly"}, c[1])
    num("Field gas rate target (optional)", "fc_qtarget", "gas_rate", c[2], minv=0.0,
        help="Wells are choked back in proportion when together they can deliver more. "
             "Empty means every well produces at its own limit.")
    num("Stop below field gas rate (optional)", "fc_qmin", "gas_rate", c[3], minv=0.0)
    V["fc_stop_load"] = st.checkbox(
        "Shut in a well when it loads up with liquid (wells with lift curves)",
        value=bool(V.get("fc_stop_load", True)), key=f"cb_load_{KEY}",
        help="Uses the Turner critical-velocity flag in the lift-curve table. Unticked, a loading "
             "well keeps producing at the rate the curves give.")
    active = [w for w in WELLS if np.isfinite(w.C) and w.control is not None]
    if not READY:
        need_data()
    elif RP is None or STATE is None:
        st.info("Complete tab 7 (relative permeability) first.")
    elif not active:
        st.info("Complete tab 8 first: at least one well needs usable tests and a pressure limit.")
    elif V["fc_years"] is None:
        st.info("Enter the forecast length.")
    elif tank.has_neighbour and V.get("nb_fc", "deplete") == "deplete" and V.get("nb_G") is None:
        st.info("Enter the neighbouring reservoir's gas in place in tab 5, or choose how it behaves "
                "in the forecast.")
    else:
        try:
            dt_ = {"Monthly": 30.4375, "Quarterly": 91.3125, "Yearly": 365.25}[V["fc_step"]]
            FC, FS = forecast(tank, P_SIM, WE_SIM, active, RP, STATE, GZ, V["fc_years"] * 365.25, dt_,
                              V["fc_qtarget"], V["fc_qmin"], V["rp_vap"], CGR_CONST,
                              bool(V.get("fc_stop_load", True)), V.get("nb_fc", "deplete"),
                              (V.get("nb_G") or 0.0) * 1e9)
        except Exception as e:  # noqa: BLE001
            FC = None
            st.warning(str(e))
        if FC is not None:
            n0 = FS["n_hist"]
            fdates = pd.Timestamp(V["start_date"]) + pd.to_timedelta(FC["t"], unit="D")
            fut = FC["forecast"].to_numpy()
            end_date = f"{fdates[len(FC) - 1]:%d/%m/%Y}"
            gw_end = FC["gp"].iloc[-1] + GE * FC["np"].iloc[-1]
            c = st.columns(5)
            c[0].metric(f"Ultimate gas recovery ({GU})", f"{gas_ip(FC['gp'].iloc[-1]):,.1f}",
                        f"{gas_ip(FC['gp'].iloc[-1] - hist.gp[-1]):,.1f} still to produce", delta_color="off",
                        help="Separator gas, history plus forecast. Gas in place and the recovery "
                             "factor are on a wet-gas basis (condensate converted to gas).")
            if tank.has_neighbour:
                c[1].metric("Recovery factor (wet gas)", f"{FS['rf_total']:.1%}",
                            f"{gas_ip(FS['gx_end']):+,.1f} {GU} net from neighbour", delta_color="off",
                            help="Wet gas produced divided by this reservoir's gas in place plus the "
                                 "net gas received from the neighbour.")
            else:
                c[1].metric("Recovery factor (wet gas)", f"{FS['rf_wet']:.1%}",
                            f"{FS['rf_hist']:.1%} to date", delta_color="off")
            c[2].metric(f"Ultimate condensate ({U.label('liq_cum', SYS)})",
                        f"{U.to_disp('liq_cum', FC['np'].iloc[-1] / 1e3, SYS):,.0f}")
            c[3].metric(f"Ultimate water ({U.label('liq_cum', SYS)})",
                        f"{U.to_disp('liq_cum', FC['wp'].iloc[-1] / 1e3, SYS):,.0f}")
            c[4].metric(f"Reservoir pressure at end ({PU})", f"{U.to_disp('pressure', FS['p_end'], SYS):,.0f}")
            (st.warning if FS["stopped"] else st.info)(
                f"Forecast ends {end_date}: {FS['reason']}.")
            ld_msgs = []
            for w in active:
                lcol = FC[f"load_{w.name}"].to_numpy(bool) & fut
                if lcol.any():
                    ld_msgs.append(f"{w.name} from {fdates[np.argmax(lcol)]:%d/%m/%Y}")
            if ld_msgs:
                st.caption("Below the Turner critical rate (liquid loading): " + "; ".join(ld_msgs)
                           + (". Those wells are shut in at that point." if V.get("fc_stop_load", True) else "."))
            if n0 == len(FC):
                st.warning("No forecast steps could be taken: the wells cannot flow against their "
                           "limits at the current reservoir pressure. Check tab 8.")
            if CGR_CONST is not None:
                st.caption("The PVT table has no producing CGR column, so the initial CGR is held "
                           "constant. Enter the CGR column in tab 1 to let the yield fall below the dew point.")
            # history rates from the cumulative record, for continuity on the plots
            hq = np.diff(hist.gp) / np.diff(hist.t) / 1e6
            gu, lu_r = U.label("gas_rate", SYS), U.label("liq_rate", SYS)
            left, right = st.columns(2)
            f = figure("", f"Gas rate ({gu})", height=320)
            f.add_trace(go.Scatter(x=DATES[1:], y=U.to_disp("gas_rate", hq, SYS), mode="lines", name="History",
                                   line=dict(color=GREY, width=2, shape="vh")))
            f.add_trace(go.Scatter(x=fdates[fut], y=U.to_disp("gas_rate", FC["qg"][fut], SYS), mode="lines",
                                   name="Field forecast", line=dict(color=BLUE, width=3)))
            for i, w in enumerate(active):
                f.add_trace(go.Scatter(x=fdates[fut], y=U.to_disp("gas_rate", FC[f"q_{w.name}"][fut], SYS),
                                       mode="lines", name=w.name,
                                       line=dict(color=[ORANGE, AQUA, YELLOW, RED, GREY][i % 5], width=1.5)))
            with left:
                show(f, "fc_q")
            f = figure("", f"Reservoir pressure ({PU})", height=320)
            f.add_trace(go.Scatter(x=DATES[hist.use], y=U.to_disp("pressure", hist.p[hist.use], SYS),
                                   mode="markers", name="Measured", marker=dict(color=BLUE, size=8)))
            f.add_trace(go.Scatter(x=fdates, y=U.to_disp("pressure", FC["p"], SYS), mode="lines",
                                   name="Model and forecast", line=dict(color=ORANGE, width=2)))
            if tank.has_neighbour:
                f.add_trace(go.Scatter(x=fdates, y=U.to_disp("pressure", FC["pn"], SYS), mode="lines",
                                       name="Neighbour", line=dict(color=GREY, width=2, dash="dash")))
            with right:
                show(f, "fc_p")
            f = figure("", f"Liquid rate ({lu_r})", height=300)
            f.add_trace(go.Scatter(x=fdates[fut], y=U.to_disp("liq_rate", FC["qc"][fut], SYS), mode="lines",
                                   name="Condensate", line=dict(color=AQUA, width=2)))
            f.add_trace(go.Scatter(x=fdates[fut], y=U.to_disp("liq_rate", FC["qw"][fut], SYS), mode="lines",
                                   name="Water", line=dict(color=BLUE, width=2)))
            with left:
                show(f, "fc_l")
            f = figure("", "Saturation (fraction of pore volume)", height=300)
            f.add_trace(go.Scatter(x=fdates[fut], y=FC["sw"][fut], mode="lines", name="Water",
                                   line=dict(color=BLUE, width=2)))
            f.add_trace(go.Scatter(x=fdates[fut], y=FC["so"][fut], mode="lines", name="Condensate",
                                   line=dict(color=AQUA, width=2)))
            with right:
                show(f, "fc_s")
            out = pd.DataFrame({
                "Date": [f"{d:%d/%m/%Y}" for d in fdates[fut]],
                f"Reservoir pressure ({PU})": np.round(U.to_disp("pressure", FC["p"][fut], SYS), 1),
                f"Gas rate ({gu})": np.round(U.to_disp("gas_rate", FC["qg"][fut], SYS), 3),
                f"Condensate rate ({lu_r})": np.round(U.to_disp("liq_rate", FC["qc"][fut], SYS), 1),
                f"Water rate ({lu_r})": np.round(U.to_disp("liq_rate", FC["qw"][fut], SYS), 1),
                f"Cum. gas ({U.label('gas_cum', SYS)})": np.round(U.to_disp("gas_cum", FC["gp"][fut] / 1e6, SYS), 1),
                f"Cum. condensate ({U.label('liq_cum', SYS)})": np.round(U.to_disp("liq_cum", FC["np"][fut] / 1e3, SYS), 1),
                f"Cum. water ({U.label('liq_cum', SYS)})": np.round(U.to_disp("liq_cum", FC["wp"][fut] / 1e3, SYS), 1),
                **{f"{w.name} gas rate ({gu})": np.round(U.to_disp("gas_rate", FC[f"q_{w.name}"][fut], SYS), 3)
                   for w in active},
                **{f"{w.name} flowing BHP ({PU})": np.round(U.to_disp("pressure", FC[f"pwf_{w.name}"][fut], SYS), 0)
                   for w in active}})
            with st.expander("Forecast table"):
                st.dataframe(out, hide_index=True, width="stretch", height=320)
                st.download_button("Download forecast (CSV)", out.to_csv(index=False),
                                   file_name="forecast.csv", mime="text/csv")
