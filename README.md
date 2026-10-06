# Condensate MBAL

Material balance tool for retrograde gas-condensate reservoirs (oil and dry-gas modules to follow).

## Run

    pip install -r requirements.txt
    streamlit run app.py

The app opens in your browser with a synthetic sample case loaded
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
5. **Aquifer** - automatic detection of an extra energy source, then model (small pot, Schilthuis, Hurst-van Everdingen,
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
* Two-phase Z from a CVD report: Z2 = p / [(pd/Zd)(1 - Gp)], Gp = cumulative well-stream produced as a
  fraction of the fluid at the dew point, Zd = gas Z at the dew point. The retrograde liquid volume SL and
  the equilibrium gas Z are used as a check: the liquid moles left, (1 - Gp) - (p/Zg)(1 - SL)/(pd/Zd),
  must be positive.
* Aquifer dimensionless functions (WD, pD) are computed by Stehfest inversion of the Laplace-space
  solutions, so any re/ro can be used without table look-up.
* Bottom drive is modelled as vertical linear flow through the reservoir area (kv = k * kv/kh).
* Aquifer detection: quadratic trend test (F-test) on the Cole no-aquifer plot F/Et vs Gp.

## Files

    app.py            Streamlit interface
    mbal/pvt.py       PVT correlations and table handling
    mbal/aquifer.py   aquifer models
    mbal/matbal.py    material balance, graphical methods, drive detection
    mbal/regress.py   regression (weights, robust loss, leave-one-out)
    mbal/outliers.py  survey screening statistics
    mbal/importer.py  production / pressure workbook import
    mbal/model.py     configuration -> engine objects
    mbal/sample.py    synthetic sample case
    tests/            engine checks:  python tests/test_engine.py
