"""Oil reservoir pages of the material balance app.

render(g) is called by app.py with its globals, so the widget helpers, unit handling, session
state and project file machinery are shared with the gas-condensate pages.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mbal.aquifer import MODELS, NEEDS_GEOMETRY, Aquifer
from mbal.importer import parse_workbook, reservoir_tables
from dataclasses import replace as dc_replace
from mbal.matbal import aggregate_wells, build_history, detect_drive
from mbal.multiphase import bb_bhp, bb_lift_table
from mbal.outliers import residual_screen, trend_screen
from mbal.model import aquifer_params
from mbal.oil import (OIL_FIT, OIL_METHODS, OilPVT, OilRelPerm, OilState, OilTank, OilWell,
                      correlation_table, fit_oil_ratios, fit_pi, ipr_shape, lab_table, oil_forecast,
                      oil_graphical, oil_points, oil_well_points, oil_well_rate)
from mbal.pvt import hall_cf, water_props
from mbal.regress import pressure_rms, regress
from mbal.vlp import OilLiftTable, read_tpd_oil

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#8a8a85"]
BLUE, ORANGE, AQUA, YELLOW, RED, GREY = PAL

O_PARAMS = {   # regressable: key -> (label, quantity, lower, upper)
    "o_N": ("Oil initially in place", "oil_ip", 1e-3, 1e5),
    "o_m": ("Gas cap ratio m", "ratio", 1e-4, 20.0),
    "o_cf": ("Rock compressibility", "compress", 1e-7, 1e-3),
    "o_nb_T": ("Transmissibility to neighbouring reservoir", "transmiss_res", 1e-4, 1e5),
}


def render(g):
    st, ss, V, U = g["st"], g["ss"], g["V"], g["U"]
    SYS, KEY = g["SYS"], g["KEY"]
    num, choice, table, csv_upload = g["num"], g["choice"], g["table"], g["csv_upload"]
    figure, show, ulabel, missing = g["figure"], g["show"], g["ulabel"], g["missing"]
    refresh, sync_tables, date_order_check = g["refresh"], g["sync_tables"], g["date_order_check"]
    LABELS, PARAMS = g["LABELS"], g["PARAMS"]
    PU = U.label("pressure", SYS)
    OU = U.label("oil_ip", SYS)

    def oil_ip(stb):
        return U.to_disp("oil_ip", np.asarray(stb, float) / 1e6, SYS)

    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(
        ["1 · PVT", "2 · Production", "3 · Reservoir", "4 · History match", "5 · Aquifer",
         "6 · Regression", "7 · Relative permeability", "8 · Wells", "9 · Forecast"])

    # ===================================================== 1. PVT
    with t1:
        st.subheader("Fluid description")
        c = st.columns(4, vertical_alignment="bottom")
        num("Reservoir temperature", "o_T", "temperature", c[0])
        num("Bubble point pressure", "o_pb", "pressure", c[1], minv=14.7)
        num("Stock-tank oil gravity", "o_api", "api", c[2], minv=5.0, maxv=80.0)
        num("Solution gas gravity (air = 1)", "o_sg", None, c[3], minv=0.55, maxv=1.8)
        c = st.columns(4, vertical_alignment="bottom")
        num("Solution GOR at the bubble point", "o_rsi", "gor", c[0], minv=0.0)
        st.subheader("Black-oil table")
        left, right = st.columns([2, 3])
        with left:
            st.caption("Bo, Rs, Bg and viscosities against pressure, from the lab report, from "
                       "correlations, or pasted from another program's black-oil table.")
            pvt_df = table("opvt", height=330)
            csv_upload("opvt", "")
            with st.expander("Fill from correlations"):
                num("Maximum table pressure", "o_pmax", "pressure", minv=500.0)
                st.caption("Standing Rs, Bo and bubble point; Vasquez-Beggs above the bubble point; "
                           "Beggs-Robinson oil viscosity; DAK gas Z and Lee-Gonzalez-Eakin gas viscosity. "
                           "If the bubble point is entered, Rs is scaled to reach the solution GOR there; "
                           "if it is empty, Standing's bubble point is used and written into the box.")
                need = missing(["o_T", "o_api", "o_sg", "o_rsi", "o_pmax"])

                def _corr():
                    sync_tables()
                    tab, pb = correlation_table(V["o_api"], V["o_sg"], V["o_rsi"], V["o_T"], V["o_pmax"], V["o_pb"])
                    ss.cur["opvt"] = tab
                    V["o_pb"] = round(pb, 1)
                    refresh()
                st.button("Generate table (replaces current table)", on_click=_corr, disabled=bool(need),
                          help="Needs temperature, oil and gas gravity, solution GOR and the maximum pressure.")
            with st.expander("Fill from a lab report (differential liberation + separator test)"):
                st.caption("The differential liberation is corrected to separator conditions with the "
                           "separator test's Bo and Rs at the bubble point: Bo = Bod·Bofb/Bodb, "
                           "Rs = Rsfb − (Rsdb − Rsd)·Bofb/Bodb. Above the bubble point Bo = Bofb·Vr from "
                           "the constant composition expansion.")
                c = st.columns(2, vertical_alignment="bottom")
                num("Bofb (separator test)", "o_bofb", "fvf_o", c[0], minv=0.5)
                num("Rsfb (separator test)", "o_rsfb", "gor", c[1], minv=0.0)
                st.markdown("**Differential liberation (at and below the bubble point)**")
                dl_df = table("odl", height=220)
                csv_upload("odl", "", expander=False)
                st.markdown("**Constant composition expansion (above the bubble point)**")
                cce_df = table("occe", height=180)
                need = missing(["o_pb", "o_bofb", "o_rsfb", "o_T", "o_sg", "o_api"])

                def _lab():
                    sync_tables()
                    try:
                        tab, notes = lab_table(ss.cur["odl"], V["o_pb"], V["o_bofb"], V["o_rsfb"], ss.cur["occe"],
                                               V["o_T"], V["o_sg"], V["o_api"])
                    except ValueError as e:
                        ss.o_lab_msg = [str(e)]
                        return
                    ss.cur["opvt"] = tab
                    refresh()
                    ss.o_lab_msg = notes
                st.button("Build the black-oil table (replaces current table)", on_click=_lab,
                          disabled=bool(need) or dl_df["p"].notna().sum() < 2,
                          help="Needs the bubble point, Bofb, Rsfb, temperature, gas and oil gravity "
                               "(for any missing gas properties and viscosities).")
                for m_ in ss.get("o_lab_msg", []):
                    st.warning(m_)
        PVT_OK, pvt, PVT_ERR = False, None, ""
        try:
            if V["o_pb"] is None or V["o_T"] is None:
                raise ValueError("Enter the bubble point pressure and reservoir temperature.")
            pvt = OilPVT(pvt_df, V["o_pb"], V["o_T"])
            PVT_OK = True
        except Exception as e:  # noqa: BLE001
            PVT_ERR = str(e)
        with right:
            if PVT_OK:
                pp = np.linspace(*pvt.p_range, 150)
                px = U.to_disp("pressure", pp, SYS)
                cc = st.columns(2)
                for holder, y, yl, k in ((cc[0], U.to_disp("fvf_o", pvt.bo(pp), SYS), f"Bo ({U.label('fvf_o', SYS)})", "bo"),
                                         (cc[1], U.to_disp("gor", pvt.rs(pp), SYS), f"Rs ({U.label('gor', SYS)})", "rs"),
                                         (cc[0], U.to_disp("bg_o", pvt.bg(pp) * 1e3, SYS), f"Bg ({U.label('bg_o', SYS)})", "bg"),
                                         (cc[1], pvt.muo(pp), "Oil viscosity (cp)", "muo")):
                    f = figure(f"Pressure ({PU})", yl, height=250, legend=False)
                    f.add_trace(go.Scatter(x=px, y=y, mode="lines", line=dict(color=BLUE, width=2)))
                    f.add_vline(x=U.to_disp("pressure", pvt.pb, SYS), line_dash="dot", line_color=GREY,
                                annotation_text="bubble point", annotation_font_size=11)
                    with holder:
                        show(f, f"o_pvt_{k}")
            else:
                st.info(PVT_ERR)

    # ===================================================== 2. Production
    with t2:
        c = st.columns([1, 2, 2])
        V["start_date"] = c[0].date_input("Start of production", value=V["start_date"], format="DD/MM/YYYY",
                                          key=f"o_start_{KEY}")
        with c[1]:
            mode = choice("Enter production", "hist_mode", {"By reservoir": "By reservoir", "By well": "By well"},
                          horizontal=True)
        st.caption("All volumes are cumulative: oil (stock tank), gas (all produced gas, separator and "
                   "stock tank) and water. Untick Use to switch a pressure survey off.")
        with st.expander("Import a production and pressure workbook (Excel or CSV)"):
            st.caption("One sheet with Date and cumulative Gas, Oil and Water, one with Date and Pressure; "
                       "Reservoir and Well columns optional. Pressures of several wells on one date are averaged.")
            wb = st.file_uploader("Workbook", type=["xlsx", "xlsm", "xls", "csv", "txt"], key="o_up_wb")
            if wb is not None:
                try:
                    wprod, wpres = parse_workbook(wb.getvalue(), wb.name)
                    names = sorted(wprod["reservoir"].unique())
                    res = st.selectbox("Reservoir", names, key="o_wb_res") if len(names) > 1 else names[0]
                    WB = reservoir_tables(wprod, wpres, res)
                    st.write(f"**{res}**: {WB['wells']['well'].nunique()} well(s), {len(WB['pres'])} survey dates; "
                             f"start of production {WB['start_date']:%d/%m/%Y}.")

                    def _use():
                        sync_tables()
                        wl, pr = WB["wells"].copy(), WB["pres"].copy()
                        wl["gp"] = U.from_disp("gas_cum", wl["gp"], SYS)
                        for k in ("np", "wp"):
                            wl[k] = U.from_disp("liq_cum", wl[k], SYS)
                        pr["p"] = U.from_disp("pressure", pr["p"], SYS)
                        ss.cur["wells"], ss.cur["pres"] = wl, pr
                        V["hist_mode"], V["start_date"] = "By well", WB["start_date"]
                        refresh()
                    st.button("Load into the tables", on_click=_use)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Could not read the workbook: {e}")
        left, right = st.columns([3, 3])
        with left:
            if mode == "By reservoir":
                hist_df = table("hist", height=420)
                date_order_check("hist")
                csv_upload("hist", "Dates as dd/mm/yyyy.")
            else:
                st.markdown("**Well cumulative production**")
                wells_df = table("wells", height=280)
                csv_upload("wells", "Dates as dd/mm/yyyy.")
                st.markdown("**Average reservoir pressure surveys**")
                pres_df = table("pres", height=220)
                date_order_check("pres")
                csv_upload("pres", "Dates as dd/mm/yyyy.")
                hist_df = aggregate_wells(wells_df, pres_df)
        HIST_OK, hist, HIST_ERR = False, None, ""
        try:
            if V["start_date"] is None:
                raise ValueError("Enter the start of production date.")
            pi_now = V["o_pi"] if V["o_pi"] is not None else np.nan
            hist = build_history(hist_df, V["start_date"], pi_now)
            HIST_OK = True
        except Exception as e:  # noqa: BLE001
            HIST_ERR = str(e)
        with right:
            if HIST_OK:
                dates = pd.Timestamp(V["start_date"]) + pd.to_timedelta(hist.t, unit="D")
                f = figure("", f"Reservoir pressure ({PU})", height=240, legend=False)
                f.add_trace(go.Scatter(x=dates[1:], y=U.to_disp("pressure", hist.p[1:], SYS), mode="markers",
                                       marker=dict(color=BLUE, size=8)))
                show(f, "o_hp")
                f = figure("", f"Cumulative ({U.label('liq_cum', SYS)})", height=240)
                f.add_trace(go.Scatter(x=dates, y=U.to_disp("liq_cum", hist.np_ / 1e3, SYS), mode="lines",
                                       name="Oil", line=dict(color=AQUA, width=2)))
                f.add_trace(go.Scatter(x=dates, y=U.to_disp("liq_cum", hist.wp / 1e3, SYS), mode="lines",
                                       name="Water", line=dict(color=BLUE, width=2)))
                show(f, "o_hl")
                with np.errstate(divide="ignore", invalid="ignore"):
                    rp_ = np.diff(hist.gp) / np.diff(hist.np_)
                f = figure("", f"Producing GOR over each interval ({U.label('gor', SYS)})", height=240, legend=False)
                f.add_trace(go.Scatter(x=dates[1:], y=U.to_disp("gor", rp_, SYS), mode="lines+markers",
                                       line=dict(color=ORANGE, width=2, shape="vh")))
                show(f, "o_hg")
            else:
                st.info(HIST_ERR)

    # ===================================================== 3. Reservoir
    with t3:
        st.subheader("Initial reservoir parameters")
        c = st.columns(3, vertical_alignment="bottom")
        num("Initial reservoir pressure", "o_pi", "pressure", c[0], minv=14.7)
        num("Porosity", "o_phi", "frac", c[1], minv=0.01, maxv=0.6)
        num("Connate water saturation", "o_swi", "frac", c[2], minv=0.0, maxv=0.9)
        c = st.columns(3, vertical_alignment="bottom")
        num("Rock compressibility", "o_cf", "compress", c[0], minv=0.0)
        num("Oil initially in place, initial estimate", "o_N", "oil_ip", c[1], minv=0.001,
            help="Starting value; the history match and regression refine it.")
        num("Gas cap ratio m (gas cap / oil zone volume)", "o_m", "ratio", c[2], minv=0.0,
            help="0 for no gas cap. A gas cap needs the reservoir to be at its bubble point initially.")

        def _hall():
            V["o_cf"] = float(f"{hall_cf(V['o_phi']):.3g}")
            refresh()
        st.button("Estimate rock compressibility (Hall)", on_click=_hall, disabled=V["o_phi"] is None)
        st.subheader("Formation water")
        c = st.columns(4, vertical_alignment="bottom")
        num("Water formation volume factor", "o_bw", "fvf_w", c[0], minv=0.9, maxv=1.3)
        num("Water compressibility", "o_cw", "compress", c[1], minv=0.0)
        num("Water viscosity", "o_muw", "visc", c[2], minv=0.01)

        def _water():
            bw, cw, mu = water_props(V["o_pi"], V["o_T"], 0.0)
            V.update(o_bw=round(bw, 4), o_cw=float(f"{cw:.3g}"), o_muw=round(mu, 3))
            refresh()
        c[3].button("Estimate (McCain, Osif, Beggs-Brill)", on_click=_water,
                    disabled=V["o_pi"] is None or V["o_T"] is None)
        RES_MISS = missing(["o_pi", "o_swi", "o_cf", "o_N", "o_m", "o_bw", "o_cw"])
        if RES_MISS:
            st.info("Still needed: " + ", ".join(RES_MISS).lower() + ".")
        if PVT_OK and not RES_MISS:
            if V["o_m"] > 0 and V["o_pi"] > pvt.pb * 1.01:
                st.warning("A gas cap is entered but the initial pressure is above the bubble point. "
                           "A gas cap means the oil was saturated at discovery: initial pressure equal "
                           "to the bubble point.")
            st.caption(("Undersaturated at discovery: " if V["o_pi"] > pvt.pb * 1.001 else "Saturated at discovery: ")
                       + f"initial pressure {U.to_disp('pressure', V['o_pi'], SYS):,.0f} {PU}, "
                       f"bubble point {U.to_disp('pressure', pvt.pb, SYS):,.0f} {PU}.")
            boi, rsi, bgi = float(pvt.bo(V["o_pi"])), float(pvt.rs(V["o_pi"])), float(pvt.bg(V["o_pi"]))
            c = st.columns(4)
            c[0].metric(f"Boi ({U.label('fvf_o', SYS)})", f"{boi:.4f}")
            c[1].metric(f"Rsi ({U.label('gor', SYS)})", f"{U.to_disp('gor', rsi, SYS):,.0f}")
            c[2].metric(f"Bgi ({U.label('bg_o', SYS)})", f"{U.to_disp('bg_o', bgi * 1e3, SYS):.4f}")
            hcpv = V["o_N"] * 1e6 * boi * (1 + V["o_m"]) / 1e6
            c[3].metric(f"Hydrocarbon pore volume ({U.label('res_vol', SYS)})", f"{U.to_disp('res_vol', hcpv, SYS):,.1f}")
            if V["o_m"] > 0:
                gcap = V["o_m"] * V["o_N"] * 1e6 * boi / bgi
                st.caption(f"Gas cap free gas in place: {U.to_disp('gas_ip', gcap / 1e9, SYS):,.1f} "
                           f"{U.label('gas_ip', SYS)}; solution gas: {U.to_disp('gas_ip', V['o_N'] * rsi / 1e3, SYS):,.1f}.")

    READY = PVT_OK and HIST_OK and not RES_MISS
    FB = g["FALLBACK"]
    cf_ = lambda k: FB.get(k) if V.get(k) is None else V[k]       # noqa: E731

    def make_aq(vals):
        if vals["aq_model"] == "none":
            return Aquifer()
        inf = bool(vals.get("aq_inf")) and vals["aq_geom"] == "radial" and vals["aq_model"] != "fetkovich"
        get = lambda k: FB.get(k) if vals.get(k) is None else vals[k]    # noqa: E731
        return Aquifer(model=vals["aq_model"], geometry=vals["aq_geom"], ct=get("o_cw") + get("o_cf"),
                       k=get("aq_k"), phi=get("aq_phi"), mu=get("o_muw"), h=get("aq_h"), ro=get("aq_ro"),
                       reD=np.inf if inf else get("aq_reD"), theta=get("aq_theta"), width=get("aq_width"),
                       L=get("aq_L"), kvkh=get("aq_kvkh"), Wvol=get("aq_Wvol") * 1e6, C=get("aq_C"))

    def make_tank(vals, with_aq=True):
        nb = None
        if with_aq and vals.get("o_nb_on") and vals.get("o_nb_T") and NB is not None:
            nb = (NB[0], NB[1], vals["o_nb_T"])
        return OilTank(pvt, hist, vals["o_N"] * 1e6, vals["o_m"], vals["o_pi"], vals["o_swi"], vals["o_cf"],
                       vals["o_cw"], vals["o_bw"], make_aq(vals) if with_aq else Aquifer(), neighbour=nb)

    def need():
        if not PVT_OK:
            st.info(f"Complete the PVT data first. {PVT_ERR}")
        elif not HIST_OK:
            st.info(f"Complete the production history first. {HIST_ERR}")
        elif RES_MISS:
            st.info("Complete the reservoir parameters first: " + ", ".join(RES_MISS).lower() + ".")
        elif AQ_MISS:
            st.info("Complete the aquifer in tab 5 first: " + ", ".join(AQ_MISS).lower() + ".")

    AQ_MISS, NB = [], None
    if READY:
        tank0 = make_tank(dict(V), with_aq=False)
        # The Campbell test is noisy above the bubble point, where Et is tiny: use the points
        # below it when there are enough.
        below = hist.p < pvt.pb
        if (below & tank0.use).sum() >= 5:
            tank0.use = tank0.use & (below | (np.arange(len(hist.p)) == 0))
            DET_NOTE = "Only surveys below the bubble point are used for this test."
        else:
            DET_NOTE = ""
        DET = detect_drive(tank0)
        tank0.use = hist.use
        DATES = pd.Timestamp(V["start_date"]) + pd.to_timedelta(hist.t, unit="D")

    # ===================================================== 5. Aquifer (before 4: same settings)
    with t5:
        if not READY:
            need()
        else:
            if DET["status"] == "aquifer":
                st.warning(f"**Additional energy source detected.** {DET['message']} In an oil reservoir "
                           "the same signature also comes from a gas cap larger than entered. " + DET_NOTE)
            else:
                st.info(DET["message"] + " " + DET_NOTE)
            left, right = st.columns([3, 2])
            with right:
                st.markdown("**Diagnostic: Campbell plot, no aquifer**")
                gq = oil_graphical(tank0, "Campbell: F/Et against Np")
                m0 = gq["mask"] & np.isfinite(gq["y"])
                f = figure(f"Cumulative oil ({U.label('liq_cum', SYS)})", f"F/Et ({OU})", height=320)
                f.add_trace(go.Scatter(x=U.to_disp("liq_cum", gq["x"][m0] / 1e3, SYS), y=oil_ip(gq["y"][m0]),
                                       mode="markers", marker=dict(color=BLUE, size=8), name="History"))
                show(f, "o_camp")
                st.caption("Flat: volumetric with the entered gas cap. Rising: water drive or a larger gas cap.")
            with left:
                st.subheader("Aquifer definition")
                g["aquifer_inputs"]("o_cw", "o_cf", "o_muw")
                inf = bool(V["aq_inf"]) and V["aq_model"] != "fetkovich"
                AQ_MISS = missing(aquifer_params(V["aq_model"], V["aq_geom"], inf)
                                  + (["aq_phi", "o_muw"] if V["aq_model"] in NEEDS_GEOMETRY else []))
                if AQ_MISS:
                    st.info("Still needed for this aquifer: " + ", ".join(AQ_MISS).lower() + ".")
                st.subheader("Neighbouring reservoir")
                V["o_nb_on"] = st.checkbox(
                    "This reservoir exchanges fluid with a neighbouring reservoir", value=bool(V.get("o_nb_on")),
                    key=f"o_cb_nb_{KEY}",
                    help="Reservoirs sharing an aquifer, or connected across a leaking fault. Fluid crosses "
                         "at transmissibility × pressure difference and is counted as water.")
                if V["o_nb_on"]:
                    c = st.columns([2, 3])
                    with c[0]:
                        num("Transmissibility", "o_nb_T", "transmiss_res", minv=0.0,
                            help="Starting value; regress it in tab 6.")
                        choice("In the forecast, the neighbour", "o_nb_fc",
                               {"deplete": "depletes as it gives up fluid", "cutoff": "stops exchanging",
                                "constant": "stays at its last pressure (unlimited)"})
                        if V["o_nb_fc"] == "deplete":
                            num("Neighbour pore volume", "o_nb_V", "res_vol", minv=0.001,
                                help="Its pressure falls by the volume given up divided by "
                                     "(cw + cf) × this volume.")
                    with c[1]:
                        nbr_df = table("nbr", height=200)
                        date_order_check("nbr")
                    d_ = nbr_df.dropna(subset=["date", "p"])
                    if V["o_nb_T"] is None:
                        AQ_MISS = AQ_MISS + ["transmissibility to the neighbouring reservoir"]
                    if d_.empty:
                        AQ_MISS = AQ_MISS + ["at least one neighbour pressure"]
                    else:
                        NB = ((pd.to_datetime(d_["date"]) - pd.Timestamp(V["start_date"])).dt.days.to_numpy(float),
                              d_["p"].to_numpy(float))
    READY = READY and not AQ_MISS
    if READY:
        tank = make_tank(dict(V))
        P_SIM, WE_SIM = tank.simulate()
        RMS = pressure_rms(P_SIM, hist.p, hist.w)

    # ===================================================== 4. History match
    with t4:
        if not READY:
            need()
        else:
            s_gr, s_an, s_en = st.tabs(["Graphical plot", "Analytical plot", "Energy plot"])
            with s_gr:
                meth = st.selectbox("Method", OIL_METHODS, key=f"o_meth_{KEY}")
                gr = oil_graphical(tank, meth)
                m_ = gr["mask"] & np.isfinite(gr["x"]) & np.isfinite(gr["y"])
                f = figure(gr["xl"], gr["yl"], height=430)
                f.add_trace(go.Scatter(x=gr["x"][m_], y=gr["y"][m_], mode="markers", name="History",
                                       marker=dict(color=BLUE, size=9)))
                if gr["fit"] is not None:
                    xx = np.linspace(0 if gr["kind"] == "line" else np.nanmin(gr["x"][m_]), np.nanmax(gr["x"][m_]), 50)
                    f.add_trace(go.Scatter(x=xx, y=gr["fit"]["slope"] * xx + gr["fit"]["intercept"], mode="lines",
                                           name="Fit", line=dict(color=ORANGE, width=2)))
                elif gr["kind"] == "flat" and gr["N"] is not None:
                    f.add_hline(y=gr["N"], line_color=ORANGE)
                show(f, "o_gr")
                c = st.columns(3)
                c[0].metric(f"Oil in place from this plot ({OU})",
                            "n/a" if gr["N"] is None else f"{oil_ip(gr['N']):,.2f}")
                c[1].metric(f"Current model N ({OU})", f"{V['o_N']:,.2f}")
                if "m" in gr["extra"]:
                    c[2].metric("Gas cap ratio from this plot", f"{gr['extra']['m']:.3f}")
                st.caption("F against Et is a straight line through the origin with slope N when the "
                           "reservoir is volumetric. The gas cap plot gives N as intercept and m·N as slope "
                           "(compressibility neglected).")
            with s_an:
                f = figure(f"Cumulative oil ({U.label('liq_cum', SYS)})", f"Reservoir pressure ({PU})", height=430)
                f.add_trace(go.Scatter(x=U.to_disp("liq_cum", hist.np_ / 1e3, SYS), y=U.to_disp("pressure", P_SIM, SYS),
                                       mode="lines", name="Model", line=dict(color=ORANGE, width=2)))
                on = hist.use
                f.add_trace(go.Scatter(x=U.to_disp("liq_cum", hist.np_[on] / 1e3, SYS), y=U.to_disp("pressure", hist.p[on], SYS),
                                       mode="markers", name="History", marker=dict(color=BLUE, size=9)))
                if (~on).any():
                    f.add_trace(go.Scatter(x=U.to_disp("liq_cum", hist.np_[~on] / 1e3, SYS),
                                           y=U.to_disp("pressure", hist.p[~on], SYS), mode="markers",
                                           name="Switched off", marker=dict(color=GREY, size=8, symbol="circle-open")))
                f.add_hline(y=U.to_disp("pressure", pvt.pb, SYS), line_dash="dot", line_color=GREY,
                            annotation_text="bubble point")
                show(f, "o_an")
                c = st.columns(3)
                c[0].metric(f"Pressure match RMS ({PU})", "n/a" if not np.isfinite(RMS) else f"{U.to_disp('pressure', RMS, SYS):,.1f}")
                c[1].metric("Recovery to date", f"{hist.np_[-1] / (V['o_N'] * 1e6):.1%}")
                c[2].metric("Cumulative water influx (MMrb)", f"{WE_SIM[-1] / 1e6:,.2f}" if np.isfinite(WE_SIM[-1]) else "n/a")
            with s_en:
                en = tank.energy()
                f = figure("", "Share of reservoir energy", height=420)
                for (nm, y), col in zip(en.items(), PAL):
                    f.add_trace(go.Scatter(x=DATES[1:], y=y[1:], mode="lines", name=nm, stackgroup="e",
                                           line=dict(width=1, color=col)))
                f.update_yaxes(tickformat=".0%", range=[0, 1])
                show(f, "o_en")

    # ===================================================== 6. Regression
    with t6:
        if not READY:
            need()
        else:
            inf = bool(V["aq_inf"]) and V["aq_model"] != "fetkovich"
            keys = ["o_N", "o_m", "o_cf"] + aquifer_params(V["aq_model"], V["aq_geom"], inf)
            if tank.has_neighbour:
                keys.append("o_nb_T")
            lab = {**{k: v for k, v in O_PARAMS.items()}, **PARAMS}
            rows = []
            for k in keys:
                l_, q_, lo_, hi_ = lab[k]
                cur = V[k] if V[k] is not None else lo_
                a, b = max(cur * 0.2, lo_), min(max(cur, lo_) * 5.0, hi_)
                if k == "o_m":
                    a, b = 0.0, max(2.0, cur * 3)
                if k == "aq_reD":
                    a, b = 1.2, max(20.0, cur * 2)
                if k == "aq_theta":
                    a, b = 10.0, 360.0
                if k == "o_nb_T":
                    a, b = max(cur * 0.02, lo_), min(cur * 50.0, hi_)
                rows.append({"Regress": k in ("o_N",), "Parameter": ulabel(l_, q_),
                             "Current value": U.to_disp(q_, cur, SYS), "Minimum": U.to_disp(q_, a, SYS),
                             "Maximum": U.to_disp(q_, b, SYS)})
            left, right = st.columns([3, 3])
            with left:
                ed = st.data_editor(pd.DataFrame(rows).astype({"Regress": "boolean"}), hide_index=True, width="stretch",
                                    disabled=["Parameter", "Current value"], key=f"o_reg_{V['aq_model']}_{KEY}")
                n_on = int(hist.use[1:].sum())
                st.caption(f"{n_on} active pressure surveys.")
                if st.button("Run regression", type="primary"):
                    pick = [i for i, r in ed.iterrows() if pd.notna(r["Regress"]) and bool(r["Regress"])]
                    if not pick:
                        st.error("Tick at least one parameter.")
                    elif len(pick) >= n_on:
                        st.error("More parameters than active pressure surveys.")
                    else:
                        rk = [keys[i] for i in pick]
                        qs = [lab[k][1] for k in rk]
                        lo = [U.from_disp(q, float(ed.loc[i, "Minimum"]), SYS) for i, q in zip(pick, qs)]
                        hi = [U.from_disp(q, float(ed.loc[i, "Maximum"]), SYS) for i, q in zip(pick, qs)]
                        lo = [max(x, 1e-6) for x in lo]
                        base = {k: cf_(k) for k in V}
                        x0 = [base[k] if base[k] else l_ * 1.01 for k, l_ in zip(rk, lo)]

                        def sim(vals):
                            return make_tank(dict(base, **dict(zip(rk, vals)))).simulate()[0]
                        with st.spinner("Regressing..."):
                            r = regress(sim, hist.p, x0, lo, hi, hist.w)
                        ss.o_reg = {"keys": rk, "x0": x0, "x": list(map(float, r["x"])), "rms0": r["rms0"],
                                    "rms": r["rms"], "p": r["p"]}
                reg = ss.get("o_reg")
                if reg:
                    out = pd.DataFrame({"Parameter": [lab[k][0] for k in reg["keys"]],
                                        "Before": [U.to_disp(lab[k][1], a, SYS) for k, a in zip(reg["keys"], reg["x0"])],
                                        "After": [U.to_disp(lab[k][1], b, SYS) for k, b in zip(reg["keys"], reg["x"])]})
                    st.dataframe(out, hide_index=True, width="stretch")
                    st.metric(f"Pressure RMS ({PU})", f"{U.to_disp('pressure', reg['rms'], SYS):,.1f}",
                              f"{U.to_disp('pressure', reg['rms'] - reg['rms0'], SYS):+,.1f}", delta_color="inverse")

                    def _accept():
                        for k, v in zip(reg["keys"], reg["x"]):
                            V[k] = float(f"{v:.6g}")
                        ss.pop("o_reg", None)
                        refresh()
                    st.button("Accept these values", on_click=_accept)
            with right:
                f = figure(f"Cumulative oil ({U.label('liq_cum', SYS)})", f"Reservoir pressure ({PU})", height=430)
                f.add_trace(go.Scatter(x=U.to_disp("liq_cum", hist.np_ / 1e3, SYS), y=U.to_disp("pressure", P_SIM, SYS),
                                       mode="lines", name="Current model", line=dict(color=GREY, width=2, dash="dash")))
                if reg:
                    f.add_trace(go.Scatter(x=U.to_disp("liq_cum", hist.np_ / 1e3, SYS), y=U.to_disp("pressure", reg["p"], SYS),
                                           mode="lines", name="After regression", line=dict(color=ORANGE, width=2)))
                f.add_trace(go.Scatter(x=U.to_disp("liq_cum", hist.np_[hist.use] / 1e3, SYS),
                                       y=U.to_disp("pressure", hist.p[hist.use], SYS), mode="markers", name="History",
                                       marker=dict(color=BLUE, size=9)))
                show(f, "o_reg")
            st.divider()
            st.subheader("Survey screening")
            c = st.columns([3, 2], vertical_alignment="bottom")
            how = c[0].radio("Compare each survey with",
                             ["Its neighbours on the pressure trend (no model)", "The current model"],
                             horizontal=True, key=f"o_scr_{KEY}")
            nsig = c[1].number_input("Flag beyond (standard deviations)", 1.5, 6.0, 3.0, 0.5, key=f"o_scr_n_{KEY}")
            if how.startswith("Its"):
                dev, z = trend_screen(hist.np_, hist.p, hist.use, nsig)
            else:
                dev, z = residual_screen(P_SIM, hist.p, hist.use)
            flag = np.abs(np.nan_to_num(z)) > nsig
            flag[0] = False
            n_pts = len(hist.p)
            tbl = pd.DataFrame({"Date": [f"{d:%d/%m/%Y}" for d in DATES[1:]],
                                f"Pressure ({PU})": np.round(U.to_disp("pressure", hist.p[1:], SYS), 1),
                                f"Deviation ({PU})": np.round(np.nan_to_num(dev[1:]) * U.to_disp("pressure", 1.0, SYS), 1),
                                "Score": np.round(np.nan_to_num(z[1:]), 1),
                                "Flag": np.where(flag[1:], "suspect", ""),
                                "Use": pd.array(hist.use[1:], dtype="boolean"), "Weight": hist.w_in[1:]})
            ed_s = st.data_editor(tbl, hide_index=True, width="stretch", height=280,
                                  disabled=[c_ for c_ in tbl.columns if c_ not in ("Use", "Weight")],
                                  key=f"o_scr_ed_{KEY}")

            def _set(upd):
                name = "hist" if V["hist_mode"] == "By reservoir" else "pres"
                sync_tables()
                df = ss.cur[name].copy()
                d = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
                for i_, dd in d.items():
                    if dd in upd:
                        df.at[i_, "use"], df.at[i_, "w"] = upd[dd]
                ss.cur[name] = df
                refresh()
            b = st.columns(3)
            new_flags = flag & hist.use
            b[0].button(f"Switch off flagged ({int(new_flags.sum())})", disabled=not new_flags.any(),
                        on_click=_set, args=({DATES[i_].normalize(): (False, float(hist.w_in[i_]))
                                              for i_ in np.flatnonzero(new_flags)},))
            b[1].button("Apply table edits", on_click=lambda: _set(
                {DATES[i_ + 1].normalize(): (bool(r_["Use"]) if pd.notna(r_["Use"]) else True,
                                             float(r_["Weight"]) if pd.notna(r_["Weight"]) else 1.0)
                 for i_, r_ in ed_s.reset_index(drop=True).iterrows()}))
            b[2].button("Switch all on", on_click=_set,
                        args=({DATES[i_].normalize(): (True, float(hist.w_in[i_])) for i_ in range(1, n_pts)},))
            st.caption("The trend screen compares each pressure with a robust line through its neighbours "
                       "against cumulative oil; the model screen compares it with the current match. "
                       "Switched-off surveys keep their production but their pressure is ignored.")

    # ===================================================== 7. Relative permeability
    RP, STATE, WELL_RP = None, None, {}
    RP_KEYS = ["o_sorw", "o_sorg", "o_sgc", "o_krw", "o_kro", "o_krg", "o_nw", "o_now", "o_ng", "o_nog"]
    with t7:
        st.caption("Tank-scale curves for the forecast: they set the producing GOR and water cut as the "
                   "tank's gas and water saturations change, and the fall in well productivity.")
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("**Oil and water**")
            c = st.columns(3, vertical_alignment="bottom")
            num("Residual oil to water", "o_sorw", "frac", c[0], minv=0.0, maxv=0.7)
            num("Water end point, krw", "o_krw", None, c[1], minv=1e-4, maxv=1.0)
            num("Oil end point, kro", "o_kro", None, c[2], minv=0.01, maxv=1.0)
            c = st.columns(3, vertical_alignment="bottom")
            num("Water exponent", "o_nw", None, c[0], minv=1.0, maxv=8.0)
            num("Oil exponent (with water)", "o_now", None, c[1], minv=1.0, maxv=8.0)
            st.markdown("**Gas and oil**")
            c = st.columns(3, vertical_alignment="bottom")
            num("Critical gas saturation", "o_sgc", "frac", c[0], minv=0.0, maxv=0.4)
            num("Residual oil to gas", "o_sorg", "frac", c[1], minv=0.0, maxv=0.7)
            num("Gas end point, krg", "o_krg", None, c[2], minv=1e-3, maxv=1.0)
            c = st.columns(3, vertical_alignment="bottom")
            num("Gas exponent", "o_ng", None, c[0], minv=1.0, maxv=8.0)
            num("Oil exponent (with gas)", "o_nog", None, c[1], minv=1.0, maxv=8.0)

            def _typ():
                V.update(o_sorw=0.25, o_sorg=0.15, o_sgc=0.05, o_krw=0.3, o_kro=1.0, o_krg=0.8,
                         o_nw=3.0, o_now=2.0, o_ng=2.0, o_nog=3.0)
                refresh()
            st.button("Fill typical values", on_click=_typ)
            st.caption("Connate water saturation is taken from tab 3. Oil relative permeability with both "
                       "gas and water present is the product of the two curves, normalised to the end point.")
            RPM = missing(RP_KEYS)
            if RPM:
                st.info("Still needed: " + ", ".join(RPM).lower() + ".")
            elif V["o_swi"] is not None:
                RP = OilRelPerm(swc=V["o_swi"], sorw=V["o_sorw"], sorg=V["o_sorg"], sgc=V["o_sgc"],
                                krw_max=V["o_krw"], kro_max=V["o_kro"], krg_max=V["o_krg"], nw=V["o_nw"],
                                now=V["o_now"], ng=V["o_ng"], nog=V["o_nog"])
        if READY and V["o_muw"] is not None:
            STATE = OilState(tank, V["o_muw"])
        with right:
            if RP is not None:
                cc = st.columns(2)
                sw_ = np.linspace(RP.swc, 1 - RP.sorw, 60)
                krw_, krow_ = RP.ow(sw_)
                f = figure("Water saturation", "Relative permeability", height=270)
                f.add_trace(go.Scatter(x=sw_, y=krow_, mode="lines", name="Oil", line=dict(color=AQUA, width=2)))
                f.add_trace(go.Scatter(x=sw_, y=krw_, mode="lines", name="Water", line=dict(color=BLUE, width=2)))
                with cc[0]:
                    show(f, "o_kw")
                sg_ = np.linspace(0, 1 - RP.swc - RP.sorg, 60)
                krg_, krog_ = RP.go(sg_)
                f = figure("Gas saturation", "Relative permeability", height=270)
                f.add_trace(go.Scatter(x=sg_, y=krog_, mode="lines", name="Oil", line=dict(color=AQUA, width=2)))
                f.add_trace(go.Scatter(x=sg_, y=krg_, mode="lines", name="Gas", line=dict(color=ORANGE, width=2)))
                with cc[1]:
                    show(f, "o_kg")
        WELL_RP = {}
        if RP is not None:
            for nm_, d_ in (V.get("o_wellrp") or {}).items():
                WELL_RP[nm_] = dc_replace(RP, **{OIL_FIT[k][0]: v for k, v in d_.items()})
        if READY and RP is not None and STATE is not None and np.all(np.isfinite(P_SIM)):
            st.divider()
            st.subheader("Fractional flow match")
            st.caption("Measured producing GOR and water cut against the tank's gas and water saturations "
                       "from the history match, with the model curves. Fit the gas curves to the GOR and "
                       "the water curves to the water cut.")
            by_well = V["hist_mode"] == "By well" and ss.cur["wells"].dropna(subset=["well", "date"]).shape[0] > 0
            src = {"field": "Reservoir, between pressure surveys"}
            if by_well:
                src["wells"] = "Each well, each production record"
            V.setdefault("o_ff_src", "wells" if by_well else "field")
            if V["o_ff_src"] not in src:
                V["o_ff_src"] = "field"
            lf, rf = st.columns([2, 3], gap="large")
            with lf:
                choice("Points", "o_ff_src", src)
                PTS = (oil_well_points(STATE, P_SIM, WE_SIM, ss.cur["wells"], V["start_date"])
                       if V["o_ff_src"] == "wells" else oil_points(STATE, RP, P_SIM, WE_SIM))
                labs = {"o_krw": "krw", "o_nw": "water exponent", "o_sorw": "residual oil to water",
                        "o_now": "oil exponent (water)", "o_krg": "krg", "o_ng": "gas exponent",
                        "o_sgc": "critical gas saturation", "o_nog": "oil exponent (gas)"}
                kw = st.multiselect("Water curve parameters", ["o_krw", "o_nw", "o_sorw", "o_now"],
                                    default=["o_krw", "o_nw"], format_func=labs.get, key=f"o_kw_{KEY}")
                kg = st.multiselect("Gas curve parameters", ["o_krg", "o_ng", "o_sgc", "o_nog"],
                                    default=["o_krg", "o_ng"], format_func=labs.get, key=f"o_kg_{KEY}")
                st.caption(f"{len(PTS)} points.")

                def _fit(keys_, phase):
                    try:
                        new, info = fit_oil_ratios(STATE, RP, PTS, keys_, phase)
                    except ValueError as e:
                        ss.o_ff = dict(ok=False, reason=str(e))
                        return
                    if info["ok"]:
                        V.update({k: float(f"{getattr(new, OIL_FIT[k][0]):.4g}") for k in keys_})
                        refresh()
                    info["phase"] = phase
                    ss.o_ff = info
                c = st.columns(2)
                c[0].button("Fit water curves", on_click=_fit, args=(kw, "water"), disabled=not kw or not len(PTS))
                c[1].button("Fit gas curves", on_click=_fit, args=(kg, "gas"), disabled=not kg or not len(PTS))
                V.setdefault("o_wellrp", {})
                if V["o_ff_src"] == "wells":
                    st.caption("Wells see gas and water at different times. Fitting each well's own curves "
                               "lets the forecast give each well its own GOR and water cut; wells not "
                               "fitted use the curves above.")

                    def _fit_each(keys_, phase):
                        msgs = []
                        for nm_, g_ in PTS.groupby("well"):
                            base_ = dc_replace(RP, **{OIL_FIT[k][0]: v for k, v in V["o_wellrp"].get(nm_, {}).items()})
                            try:
                                new, info = fit_oil_ratios(STATE, base_, g_, keys_, phase)
                            except ValueError as e:
                                msgs.append(f"{nm_}: {e}")
                                continue
                            if info["ok"]:
                                V["o_wellrp"].setdefault(nm_, {}).update(
                                    {k: float(f"{getattr(new, OIL_FIT[k][0]):.4g}") for k in keys_})
                                msgs.append(f"{nm_}: fitted, mismatch "
                                            + (f"{info['rms0']:.2%} → {info['rms']:.2%}" if phase == "water"
                                               else f"{info['rms0']:,.0f} → {info['rms']:,.0f} scf/STB"))
                            else:
                                msgs.append(f"{nm_}: {info['reason']}")
                        refresh()
                        ss.o_ff_each = msgs
                    c = st.columns(2)
                    c[0].button("Fit water curves, each well", on_click=_fit_each, args=(kw, "water"), disabled=not kw)
                    c[1].button("Fit gas curves, each well", on_click=_fit_each, args=(kg, "gas"), disabled=not kg)
                    for m_ in ss.get("o_ff_each", []):
                        st.caption(m_)
                if V["o_wellrp"]:
                    st.dataframe(pd.DataFrame([{"Well": k, **{labs[a]: b for a, b in d.items()}}
                                               for k, d in V["o_wellrp"].items()]), hide_index=True, width="stretch")

                    def _clear():
                        V["o_wellrp"] = {}
                        refresh()
                    st.button("Clear the well curves", on_click=_clear)
                info = ss.get("o_ff")
                if info and info.get("ok"):
                    unit = "water cut" if info["phase"] == "water" else "scf/STB"
                    fmt = (lambda v: f"{v:.2%}") if info["phase"] == "water" else (lambda v: f"{v:,.0f} {unit}")
                    st.success(f"Fitted the {info['phase']} curves to {info['n']} points: mismatch "
                               f"{fmt(info['rms0'])} before, {fmt(info['rms'])} after."
                               + (" At a limit: " + ", ".join(labs[k] for k in info["at_limit"]) if info["at_limit"] else ""))
                elif info:
                    st.warning(info["reason"])
            with rf:
                if len(PTS):
                    gor_m, wor_m = np.empty(len(PTS)), np.empty(len(PTS))
                    for nm_, g_ in PTS.groupby("well"):
                        r_ = WELL_RP.get(nm_, RP)
                        a_, b_, _ = STATE.ratios(r_, g_["p"].to_numpy(), g_["sw"].to_numpy(), g_["sg"].to_numpy())
                        gor_m[PTS.index.get_indexer(g_.index)] = a_
                        wor_m[PTS.index.get_indexer(g_.index)] = b_
                    dts = pd.Timestamp(V["start_date"]) + pd.to_timedelta(PTS["t"], unit="D")
                    f = figure("", f"Producing GOR ({U.label('gor', SYS)})", height=300)
                    for i_, (nm_, g_) in enumerate(PTS.groupby("well")):
                        f.add_trace(go.Scatter(x=dts.loc[g_.index], y=U.to_disp("gor", g_["gor"], SYS), mode="markers",
                                               name=f"{nm_} measured", marker=dict(color=PAL[i_ % 6], size=6)))
                    o_ = np.argsort(PTS["t"].to_numpy())
                    for i_, (nm_, g_) in enumerate(PTS.groupby("well")):
                        ix = PTS.index.get_indexer(g_.index)
                        oo = np.argsort(g_["t"].to_numpy())
                        f.add_trace(go.Scatter(x=dts.loc[g_.index].to_numpy()[oo], y=U.to_disp("gor", gor_m[ix][oo], SYS),
                                               mode="lines", name=f"{nm_} model", line=dict(color=PAL[i_ % 6], width=1.5)))
                    f.add_trace(go.Scatter(x=dts.to_numpy()[o_], y=U.to_disp("gor", PTS["rs"].to_numpy()[o_], SYS),
                                           mode="lines", name="Solution GOR", line=dict(color=GREY, width=1, dash="dot")))
                    show(f, "o_ffg")
                    f = figure("Tank water saturation", "Water cut", height=280)
                    for i_, (nm_, g_) in enumerate(PTS.groupby("well")):
                        f.add_trace(go.Scatter(x=g_["sw"], y=g_["fw"], mode="markers", name=str(nm_),
                                               marker=dict(color=PAL[i_ % 6], size=6)))
                    swc = np.linspace(RP.swc, 1 - RP.sorw, 100)
                    _, wor_c, _ = STATE.ratios(RP, np.full_like(swc, P_SIM[-1]), swc, np.full_like(swc, float(PTS["sg"].iloc[-1])))
                    f.add_trace(go.Scatter(x=swc, y=wor_c / (1 + wor_c), mode="lines", name="Model, pressure now",
                                           line=dict(color=GREY, width=2)))
                    f.update_xaxes(range=[RP.swc - 0.01, min(1 - RP.sorw, float(PTS["sw"].max()) + 0.1)])
                    show(f, "o_ffw")

    # ===================================================== 8. Wells
    WELLS = []
    with t8:
        st.caption("Each well needs a productivity index, fitted to its tests, and a pressure limit. "
                   "Inflow is a straight line above the bubble point and Vogel below it. A tubing-head "
                   "limit uses the well's Prosper lift curves if attached, otherwise the built-in Beggs-Brill "
                   "calculation from depth, tubing size and wellhead temperature.")
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Limits**")
            cfg_df = table("owellcfg", height=160)
            st.markdown("**Well tests**")
            tests_df = table("otests", height=240)
            date_order_check("otests")
            st.markdown("**Lift curves (Prosper .tpd, oil)**")
            V.setdefault("o_lift", {})
            V.setdefault("lift_gauge", "psig")
            choice("Pressures in the lift-curve files are", "lift_gauge", {"psig": "psig", "psia": "psia"}, horizontal=True)
            names_cfg = [str(x).strip() for x in cfg_df["well"].dropna() if str(x).strip()]
            ups = st.file_uploader("Lift-curve files", type=["tpd"], accept_multiple_files=True, key=f"o_up_lift_{ss.ver}")
            pend = []
            for f_ in ups or []:
                try:
                    tb = read_tpd_oil(f_.getvalue().decode("latin-1"), f_.name, V["lift_gauge"] == "psig")
                except Exception as e:  # noqa: BLE001
                    st.warning(f"{f_.name}: {e}")
                    continue
                opts = names_cfg or ["(enter the wells above first)"]
                guess = next((n for n in opts if "".join(ch for ch in n.lower() if ch.isalnum())
                              in "".join(ch for ch in f_.name.lower() if ch.isalnum())), opts[0])
                pick = st.selectbox(f"Well for {f_.name}", opts, index=opts.index(guess), key=f"o_lw_{f_.name}_{ss.ver}")
                pend.append((pick, tb))

            def _attach(items):
                for n_, t_ in items:
                    V["o_lift"][n_] = t_.to_dict()
                refresh()
            if pend and names_cfg:
                st.button(f"Attach {len(pend)} table(s)", type="primary", on_click=_attach, args=(pend,))
            if V["o_lift"]:
                st.caption("Attached: " + ", ".join(f"{k} ({d.get('name', '')}, {d.get('rate_kind', '')} rate)"
                                                    for k, d in V["o_lift"].items()))
        if not READY:
            with right:
                need()
        elif RP is None or STATE is None:
            with right:
                st.info("Complete tab 7 (relative permeability) first.")
        else:
            sw_h, so_h, sg_h = STATE.sats(P_SIM, WE_SIM, hist.np_, hist.wp)
            gor_h, wor_h, M_h = STATE.ratios(RP, P_SIM, sw_h, sg_h)

            def M_hw(nm_):
                return STATE.ratios(WELL_RP.get(nm_, RP), P_SIM, sw_h, sg_h)[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                gor_meas = np.nan_to_num(np.concatenate([[pvt.rs(V["o_pi"])], np.diff(hist.gp) / np.diff(hist.np_)]), nan=float(pvt.rs(V["o_pi"])))
                wc_meas = np.nan_to_num(np.concatenate([[0.0], np.diff(hist.wp) / (np.diff(hist.wp) + np.diff(hist.np_))]))
            tdf = tests_df.dropna(subset=["well", "ql"]).copy()
            tdf["well"] = tdf["well"].astype(str).str.strip()
            names = names_cfg + [x for x in tdf["well"].unique() if x not in names_cfg]
            rows = []
            for nm in names:
                r = cfg_df[cfg_df["well"].astype(str).str.strip() == nm]
                r = r.iloc[0] if len(r) else {}
                gv = lambda k: float(pd.to_numeric(r.get(k), errors="coerce")) if len(r) and pd.notna(pd.to_numeric(r.get(k), errors="coerce")) else np.nan   # noqa: E731
                w = OilWell(nm, min_thp=gv("min_thp"), min_bhp=gv("min_bhp"), qmax=gv("qmax"))
                tub = dict(tvd=gv("tvd"), md=gv("md"), tid=gv("tid"), wht=gv("wht"))
                has_tub = all(np.isfinite(tub[k]) and tub[k] > 0 for k in ("tvd", "tid", "wht"))
                lift_src = "-"
                if nm in V["o_lift"]:
                    w.lift = OilLiftTable.from_dict(V["o_lift"][nm])
                    lift_src = "lift curves"
                elif has_tub and np.isfinite(w.min_thp) and V["o_api"] and V["o_sg"]:
                    # built-in Beggs-Brill, tabulated once per completion and cached
                    ck = ("bb", nm, round(w.min_thp, 1), *[round(tub[k], 3) if np.isfinite(tub[k]) else None for k in tub],
                          V["o_api"], V["o_sg"], V["o_muw"], V["o_bw"], hash(pvt_df.to_csv()))
                    cache = ss.setdefault("o_bb_cache", {})
                    if ck not in cache:
                        with st.spinner(f"Tabulating Beggs-Brill lift curves for {nm}..."):
                            cache[ck] = bb_lift_table([w.min_thp], tub["tvd"], tub["md"], tub["tid"], tub["wht"], V["o_T"],
                                                      pvt, V["o_api"], V["o_sg"], V["o_muw"] or 0.5, V["o_bw"] or 1.02)
                    w.lift = cache[ck]
                    lift_src = "built-in (Beggs-Brill)"
                q_, pr_, pwf_, mob_, skipped = [], [], [], [], 0
                for t_ in tdf[tdf["well"] == nm].itertuples():
                    day = (pd.Timestamp(t_.date) - pd.Timestamp(V["start_date"])).days if pd.notna(t_.date) else hist.t[-1]
                    prt = float(t_.pr) if pd.notna(t_.pr) else float(np.interp(day, hist.t[hist.use], hist.p[hist.use]))
                    wct, gort = float(np.interp(day, hist.t, wc_meas)), float(np.interp(day, hist.t, gor_meas))
                    if pd.notna(t_.pwf):
                        pw = float(t_.pwf)
                    elif pd.notna(t_.pth) and nm in V["o_lift"]:
                        pw = w.lift.bhp_at(float(t_.pth), float(t_.ql), 100 * wct, gort)
                    elif pd.notna(t_.pth) and has_tub and V["o_api"] and V["o_sg"]:
                        pw = bb_bhp(float(t_.pth), float(t_.ql), wct, gort, tub["tvd"], tub["md"], tub["tid"], tub["wht"],
                                    V["o_T"], pvt, V["o_api"], V["o_sg"], V["o_muw"] or 0.5, V["o_bw"] or 1.02)
                    else:
                        skipped += 1
                        continue
                    q_.append(float(t_.ql))
                    pr_.append(prt)
                    pwf_.append(pw)
                    mob_.append(float(np.interp(day, hist.t, M_hw(nm))))
                w.J, w.note = fit_pi(q_, pr_, pwf_, pvt.pb, mob_) if q_ else (np.nan, "no tests")
                w.n_tests = len(q_)
                if skipped:
                    w.note = (w.note + "; " if w.note else "") + (f"{skipped} tubing-head test(s) need lift curves or "
                                                                  "depth, tubing size and wellhead temperature")
                WELLS.append(w)
                gw_, ww_, Mw_ = STATE.ratios(WELL_RP.get(nm, RP), P_SIM[-1], sw_h[-1], sg_h[-1])
                wc_now = float(ww_ / (1 + ww_))
                qn, pwn, _ = oil_well_rate(w, P_SIM[-1], float(Mw_), pvt.pb, 100 * wc_now, float(gw_))
                rows.append({"Well": nm, "Tests used": w.n_tests,
                             f"PI ({U.label('pi_liq', SYS)})": U.to_disp("pi_liq", w.J * float(Mw_), SYS) if np.isfinite(w.J) else np.nan,
                             f"Liquid rate at limit now ({U.label('liq_rate', SYS)})": U.to_disp("liq_rate", qn, SYS),
                             f"Flowing BHP ({PU})": U.to_disp("pressure", pwn, SYS),
                             "Limit": {"thp": "tubing head", "bhp": "bottomhole", None: "none"}[w.control],
                             "Tubing": lift_src, "Curves": "own" if nm in WELL_RP else "reservoir",
                             f"GOR now ({U.label('gor', SYS)})": U.to_disp("gor", float(gw_), SYS),
                             "Water cut now": round(wc_now, 4), "Note": w.note})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("PI is shown at today's oil mobility; it falls as gas and water saturations "
                           "rise. Tests without a reservoir pressure use the surveys interpolated in time.")
            with right:
                op = [w for w in WELLS if np.isfinite(w.J) and w.control is not None]
                if op:
                    ws = st.selectbox("Inflow and lift now for", [w.name for w in op], key=f"o_opw_{KEY}")
                    w = next(x for x in op if x.name == ws)
                    prn = P_SIM[-1]
                    g_n, w_n, Mn = (float(x) for x in STATE.ratios(WELL_RP.get(w.name, RP), prn, sw_h[-1], sg_h[-1]))
                    wc_now = w_n / (1 + w_n)
                    gor_h = np.full_like(gor_h, g_n)
                    pw = np.linspace(0, prn, 80)
                    qi = np.array([w.J * Mn * float(ipr_shape(prn, x, pvt.pb)) for x in pw])
                    f = figure(f"Liquid rate ({U.label('liq_rate', SYS)})", f"Flowing bottomhole pressure ({PU})", height=340)
                    f.add_trace(go.Scatter(x=U.to_disp("liq_rate", qi, SYS), y=U.to_disp("pressure", pw, SYS),
                                           mode="lines", name="Inflow", line=dict(color=BLUE, width=2)))
                    if w.lift is not None and np.isfinite(w.min_thp) and w.control == "thp":
                        qq = np.geomspace(max(qi.max() * 1e-3, 1.0), qi.max(), 60)
                        f.add_trace(go.Scatter(x=U.to_disp("liq_rate", qq, SYS),
                                               y=U.to_disp("pressure", np.array([w.lift.bhp_at(w.min_thp, x, 100 * wc_now, float(gor_h[-1])) for x in qq]), SYS),
                                               mode="lines", name="Lift curve", line=dict(color=ORANGE, width=2)))
                    qo_, po_, _ = oil_well_rate(w, prn, Mn, pvt.pb, 100 * wc_now, float(gor_h[-1]))
                    if qo_ > 0:
                        f.add_trace(go.Scatter(x=[U.to_disp("liq_rate", qo_, SYS)], y=[U.to_disp("pressure", po_, SYS)],
                                               mode="markers", name="Operating point",
                                               marker=dict(color=AQUA, size=13, symbol="diamond")))
                    show(f, "o_op")

    # ===================================================== 9. Forecast
    with t9:
        c = st.columns(5, vertical_alignment="bottom")
        num("Forecast length (years)", "o_years", None, c[0], minv=0.5, maxv=60.0)
        V.setdefault("o_step", "Monthly")
        choice("Time step", "o_step", {"Monthly": "Monthly", "Quarterly": "Quarterly", "Yearly": "Yearly"}, c[1])
        num("Field oil target (optional)", "o_qtarget", "liq_rate", c[2], minv=0.0)
        num("Stop below field oil rate (optional)", "o_qmin", "liq_rate", c[3], minv=0.0)
        num("Stop above water cut (optional)", "o_wcmax", "frac", c[4], minv=0.0, maxv=1.0)
        c = st.columns(5, vertical_alignment="bottom")
        num("Stop above producing GOR (optional)", "o_gormax", "gor", c[0], minv=0.0)
        num("Water injection rate (optional)", "o_qwi", "liq_rate", c[1], minv=0.0)
        num("Or voidage replacement ratio (optional)", "o_vrr", "ratio", c[2], minv=0.0, maxv=3.0,
            help="Water injection equal to this fraction of the reservoir volume produced; overrides the rate.")
        num("Gas injection rate (optional)", "o_qgi", "gas_rate", c[3], minv=0.0)
        act = [w for w in WELLS if np.isfinite(w.J) and w.control is not None]
        if not READY:
            need()
        elif RP is None or STATE is None:
            st.info("Complete tab 7 first.")
        elif not act:
            st.info("Complete tab 8 first: at least one well needs a test and a pressure limit.")
        elif V["o_years"] is None:
            st.info("Enter the forecast length.")
        else:
            try:
                dt_ = {"Monthly": 30.4375, "Quarterly": 91.3125, "Yearly": 365.25}[V["o_step"]]
                if tank.has_neighbour and V.get("o_nb_fc") == "deplete" and not V.get("o_nb_V"):
                    raise ValueError("Enter the neighbouring reservoir's pore volume in tab 5, or choose how it "
                                     "behaves in the forecast.")
                FC, FS = oil_forecast(tank, P_SIM, WE_SIM, act, RP, STATE, V["o_years"] * 365.25, dt_,
                                      V["o_qtarget"], V["o_qmin"], V["o_wcmax"], V["o_gormax"],
                                      well_rp=WELL_RP, qwi=V.get("o_qwi") or 0.0, vrr=V.get("o_vrr"),
                                      qgi=(V.get("o_qgi") or 0.0) * 1e6, nb_mode=V.get("o_nb_fc", "cutoff"),
                                      nb_vol=(V.get("o_nb_V") or 0.0) * 1e6)
            except Exception as e:  # noqa: BLE001
                FC = None
                st.warning(str(e))
            if FC is not None:
                fut = FC["forecast"].to_numpy()
                fd = pd.Timestamp(V["start_date"]) + pd.to_timedelta(FC["t"], unit="D")
                lu, lr = U.label("liq_cum", SYS), U.label("liq_rate", SYS)
                c = st.columns(5)
                c[0].metric(f"Ultimate oil ({lu})", f"{U.to_disp('liq_cum', FS['np_end'] / 1e3, SYS):,.0f}",
                            f"{U.to_disp('liq_cum', (FS['np_end'] - hist.np_[-1]) / 1e3, SYS):,.0f} still to produce", delta_color="off")
                c[1].metric("Oil recovery factor", f"{FS['rf']:.1%}", f"{FS['rf_hist']:.1%} to date", delta_color="off")
                c[2].metric(f"Ultimate gas ({U.label('gas_ip', SYS)})", f"{U.to_disp('gas_ip', FS['gp_end'] / 1e9, SYS):,.1f}")
                c[3].metric(f"Ultimate water ({lu})", f"{U.to_disp('liq_cum', FS['wp_end'] / 1e3, SYS):,.0f}")
                c[4].metric(f"Reservoir pressure at end ({PU})", f"{U.to_disp('pressure', FS['p_end'], SYS):,.0f}")
                (st.warning if FS["stopped"] else st.info)(f"Forecast ends {fd[len(FC) - 1]:%d/%m/%Y}: {FS['reason']}.")
                extra = []
                if FS["wi_end"] > hist.wi[-1]:
                    extra.append(f"water injected {U.to_disp('liq_cum', (FS['wi_end'] - hist.wi[-1]) / 1e3, SYS):,.0f} {lu}")
                if FS["gi_end"] > hist.gi[-1]:
                    extra.append(f"gas injected {U.to_disp('gas_ip', (FS['gi_end'] - hist.gi[-1]) / 1e9, SYS):,.1f} {U.label('gas_ip', SYS)}")
                if tank.has_neighbour:
                    extra.append(f"net volume from the neighbour {FS['qx_end'] / 1e6:+,.2f} MMrb")
                if extra:
                    st.caption("Over the forecast: " + "; ".join(extra) + ".")
                with np.errstate(divide="ignore", invalid="ignore"):
                    hq = np.diff(hist.np_) / np.diff(hist.t)
                l_, r_ = st.columns(2)
                f = figure("", f"Oil rate ({lr})", height=320)
                f.add_trace(go.Scatter(x=DATES[1:], y=U.to_disp("liq_rate", hq, SYS), mode="lines", name="History",
                                       line=dict(color=GREY, width=2, shape="vh")))
                f.add_trace(go.Scatter(x=fd[fut], y=U.to_disp("liq_rate", FC["qo"][fut], SYS), mode="lines",
                                       name="Forecast", line=dict(color=AQUA, width=3)))
                for i_, w in enumerate(act):
                    f.add_trace(go.Scatter(x=fd[fut], y=U.to_disp("liq_rate", FC[f"qo_{w.name}"][fut], SYS),
                                           mode="lines", name=w.name, line=dict(color=PAL[(i_ + 1) % 6], width=1.5)))
                with l_:
                    show(f, "o_fq")
                f = figure("", f"Reservoir pressure ({PU})", height=320)
                f.add_trace(go.Scatter(x=DATES[hist.use], y=U.to_disp("pressure", hist.p[hist.use], SYS), mode="markers",
                                       name="Measured", marker=dict(color=BLUE, size=8)))
                f.add_trace(go.Scatter(x=fd, y=U.to_disp("pressure", FC["p"], SYS), mode="lines", name="Model and forecast",
                                       line=dict(color=ORANGE, width=2)))
                f.add_hline(y=U.to_disp("pressure", pvt.pb, SYS), line_dash="dot", line_color=GREY, annotation_text="bubble point")
                with r_:
                    show(f, "o_fp")
                f = figure("", f"Producing GOR ({U.label('gor', SYS)})", height=300, legend=False)
                f.add_trace(go.Scatter(x=fd[fut], y=U.to_disp("gor", FC["gor"][fut], SYS), mode="lines",
                                       line=dict(color=ORANGE, width=2)))
                with l_:
                    show(f, "o_fg")
                f = figure("", "Water cut", height=300, legend=False)
                f.add_trace(go.Scatter(x=fd[fut], y=FC["wc"][fut], mode="lines", line=dict(color=BLUE, width=2)))
                f.update_yaxes(tickformat=".1%")
                with r_:
                    show(f, "o_fw")
                out = pd.DataFrame({
                    "Date": [f"{d:%d/%m/%Y}" for d in fd[fut]],
                    f"Reservoir pressure ({PU})": np.round(U.to_disp("pressure", FC["p"][fut], SYS), 1),
                    f"Oil rate ({lr})": np.round(U.to_disp("liq_rate", FC["qo"][fut], SYS), 1),
                    f"Gas rate ({U.label('gas_rate', SYS)})": np.round(U.to_disp("gas_rate", FC["qg"][fut] / 1e6, SYS), 3),
                    f"Water rate ({lr})": np.round(U.to_disp("liq_rate", FC["qw"][fut], SYS), 1),
                    f"Producing GOR ({U.label('gor', SYS)})": np.round(U.to_disp("gor", FC["gor"][fut], SYS), 0),
                    "Water cut": FC["wc"][fut].round(4),
                    f"Cum. oil ({lu})": np.round(U.to_disp("liq_cum", FC["np"][fut] / 1e3, SYS), 1),
                    f"Cum. gas ({U.label('gas_cum', SYS)})": np.round(U.to_disp("gas_cum", FC["gp"][fut] / 1e6, SYS), 1),
                    f"Cum. water ({lu})": np.round(U.to_disp("liq_cum", FC["wp"][fut] / 1e3, SYS), 1),
                    f"Water injection ({lr})": np.round(U.to_disp("liq_rate", FC["qwi"][fut], SYS), 1),
                    f"Gas injection ({U.label('gas_rate', SYS)})": np.round(U.to_disp("gas_rate", FC["qgi"][fut] / 1e6, SYS), 3),
                    **{f"{w.name} oil rate ({lr})": np.round(U.to_disp("liq_rate", FC[f"qo_{w.name}"][fut], SYS), 1) for w in act}})
                with st.expander("Forecast table"):
                    st.dataframe(out, hide_index=True, width="stretch", height=300)
                    st.download_button("Download forecast (CSV)", out.to_csv(index=False), file_name="oil_forecast.csv",
                                       mime="text/csv")
