# Condensate MBAL

Material balance tool for retrograde gas-condensate reservoirs (oil and dry-gas modules to follow).

## Run

    pip install -r requirements.txt
    streamlit run app.py

The app opens in your browser with every input empty. Each tab lists what it still needs.
"Load sample case" in the sidebar loads a synthetic case
(250 Bscf wet gas, finite radial aquifer with re/ro = 5 and k = 40 md).

## Workflow

1. **PVT data** - fluid description and Z-factor table (lab CVD table, CSV import, or correlations).
   If the CVD report has no two-phase Z, it is calculated from the cumulative produced fluid (see Method).
2. **Production history** - cumulative gas, condensate, water and reservoir pressure, by reservoir or by well.
   An Excel or CSV workbook (production sheet and pressure sheet, columns found by their headings) can be
   imported directly; pressures of several wells on the same date are averaged.
3. **Reservoir parameters** - initial pressure, porosity, connate water, rock compressibility, first estimate of gas in place.
4. **History match** - graphical plots (p/z, p/z overpressured, Havlena-Odeh overpressured, Havlena-Odeh water drive,
   Cole (F-We)/Et, Roach, Cole F/Et), analytical plot (pressure vs cumulative production), energy plot, WD function plot.
5. **Aquifer and neighbouring reservoir** - automatic detection of an extra energy source, then model (small pot, Schilthuis, Hurst-van Everdingen,
   Carter-Tracy, Fetkovich) and system (radial, linear, bottom drive).
6. **Regression** - survey screening, then a least-squares or robust match of model pressure to
   history on any ticked parameters.

### Pressure survey screening

Each survey has a **Use** tick box and a **Weight** (tab 2). A survey that is switched off keeps its
production record but its pressure is ignored in the regression, the RMS, the graphical fits and the
aquifer detection, and is drawn hollow on the plots. Tab 6 scores the surveys so you can decide:

* **Trend screen (no model)** - each pressure is compared with a Theil-Sen line through its three
  neighbours on each side of the p versus Gp trend; the worst is set aside and the screen repeated.
* **Residual screen** - each pressure is compared with the model; scatter comes from the median
  absolute deviation, which outliers cannot inflate.
* **Runs test** - warns when the residuals are one-sided for long stretches, which is model error
  and must not be "fixed" by switching surveys off.
* **Robust fitting (soft-L1)** - down-weights isolated bad surveys without removing them.
* **Influence check** - refits with each survey left out and reports how far the answer moves.

The sample case contains two deliberately bad surveys (160 and 180 psi high).

7. **Relative permeability** - tank-scale Corey curves (gas-water, gas-condensate), with a fit of the water
   curve to the produced-water history after water vapour from the gas has been set aside.
8. **Wells** - completion data, pressure limits and well tests. Prosper lift-curve tables (.tpd) can be attached per well;
   they replace the built-in tubing calculation and add Turner liquid-loading checks. Tests may give flowing bottomhole pressure or
   tubing-head pressure (converted with the tubing calculation). A back-pressure inflow equation is fitted per well.
9. **Forecast** - the matched tank stepped forward under the well limits and an optional field rate target:
   gas, condensate and water rates, reservoir pressure, saturations and recovery.

Units switch between Field and SI in the sidebar. Projects save to and load from a JSON file.

## Method

    F = G (Eg + Efw) + We
    F   = Gp,wet * Bg + Wp * Bw
    Eg  = Bg - Bgi                 (two-phase Z below the dew point)
    Efw = Bgi (cw*Swi + cf)/(1 - Swi) * (pi - p)
    Gp,wet = Gp,separator + GE * Np    with GE = 133,316 * SGo / Mo  scf/STB

G is the wet (well-stream) gas initially in place.

* Single-phase Z: Dranchuk-Abou-Kassem, Sutton pseudo-criticals, Wichert-Aziz correction.
* Two-phase Z: Rayes-Piper-McCain-Poston (1992), scaled to be continuous at the dew point.
* Two-phase Z from a CCE (constant mass) study: Z2 = Zd (p / pd) Vr, Vr = V / Vd. The same formula gives
  the single-phase Z above the dew point.
* Two-phase Z from a CVD report: Z2 = p / [(pd/Zd)(1 - Gp)], Gp = cumulative well-stream produced as a
  fraction of the fluid at the dew point, Zd = gas Z at the dew point. The retrograde liquid volume SL and
  the equilibrium gas Z are used as a check: the liquid moles left, (1 - Gp) - (p/Zg)(1 - SL)/(pd/Zd),
  must be positive.
* Aquifer dimensionless functions (WD, pD) are computed by Stehfest inversion of the Laplace-space
  solutions, so any re/ro can be used without table look-up.
* Bottom drive is modelled as vertical linear flow through the reservoir area (kv = k * kv/kh).
* Neighbouring reservoir (tab 5): for a tank that leaks to or from another sand, gas crosses at
  q = T (p_neighbour - p), with the neighbour's measured static pressures interpolated in time. The
  balance becomes G (Eg + Efw) + We = (Gp,wet - Gx) Bg + Wp Bw, Gx being the cumulative gas received.
  T can be regressed in tab 6. In the forecast the neighbour either depletes as a closed gas tank
  (its gas in place at the last survey is entered; p/z falls with the gas it gives up), stops
  exchanging, or stays at its last pressure. The last is an unlimited source and only an upper bound.
* Aquifer detection: quadratic trend test (F-test) on the Cole no-aquifer plot F/Et vs Gp.

### Forecast method

    Sw  = [PVi Swi (1 + cw dp) + We - Wp Bw] / [PVi (1 - cf dp)]        tank water saturation
    So  = retrograde liquid (CVD) x (1 - Swi)                           tank condensate saturation
    krw = krw_max Swn^nw,  krg = krg_max (1 - Swn)^ng,  Swn = (Sw - Swc)/(1 - Swc - Sgrw)
    M   = krg(Sw, So) / (mu z), relative to initial conditions          gas mobility
    q   = C [M (pr^2 - pwf^2)]^n                                        well inflow
    pwf^2 = e^s pth^2 + 6.67e-4 q^2 f T^2 z^2 (e^s - 1)(MD/TVD)/d^5     tubing (average T and Z)
    WGR = (krw / mu_w Bw) / (krg / mu_g Bg)  +  water vapour (Bukacek)

* Rates over a step use the pressure and saturations at the start of the step; the end pressure
  follows from the material balance, including the aquifer.
* The curves are tank averages (pseudo curves), not core curves. Trapped gas is represented by gas
  ceasing to flow as the tank water saturation approaches 1 - Sgrw.
* Lift curves (.tpd): bottomhole pressure is interpolated linearly in tubing-head pressure and
  water-gas ratio, and in the logarithm of rate and GOR (GOR = 10^6 / CGR). The operating point is
  the highest-rate crossing of inflow and lift curve; no crossing means the well cannot lift its
  liquids. Prosper writes pressures in psig; they are converted to psia on import.
* Not modelled: liquid loading in the tubing for wells without lift curves, near-well condensate banking beyond the tank-average
  saturation, coning, and differences in water arrival between wells.

## Files

    app.py            Streamlit interface
    mbal/pvt.py       PVT correlations and table handling
    mbal/aquifer.py   aquifer models
    mbal/matbal.py    material balance, graphical methods, drive detection
    mbal/regress.py   regression (weights, robust loss, leave-one-out)
    mbal/outliers.py  survey screening statistics
    mbal/importer.py  production / pressure workbook import
    mbal/relperm.py   relative permeability, tank saturations, water fit
    mbal/wells.py     tubing pressure drop and inflow fit
    mbal/vlp.py       Prosper .tpd lift-curve reader and interpolation
    mbal/forecast.py  forecast stepping
    mbal/model.py     configuration -> engine objects
    mbal/sample.py    synthetic sample case
    tests/            engine checks:  python tests/test_engine.py
