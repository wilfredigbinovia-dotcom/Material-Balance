# resmb

Reservoir material balance in Python, in field units.

- **Single-tank analysis** for oil and gas reservoirs (Havlena–Odeh straight line, p/z, drive indices)
- **Aquifer models**: pot, Schilthuis steady state, Van Everdingen–Hurst (radial or linear), Fetkovich
- **Multi-tank simulator** with transmissibility between tanks and pressure history matching
- **PVT correlations** for black oil, gas and water

The calculations are in one file, `resmb.py`. `app.py` is a Streamlit app on top of it, and `examples.py` runs five worked cases as a script.

## Install

Python 3.9 or later.

```bash
pip install -r requirements.txt
python examples.py
```

`examples.py` prints the results of each case and saves four charts as PNG files next to the script. It takes about ten seconds.

Tested with Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11 and streamlit 1.64. The minimum versions in `requirements.txt` are not tested.

## Streamlit app

```bash
streamlit run app.py
```

Run `app.py`, not `resmb.py`: `resmb.py` is the library and shows nothing on its own. Keep both files in the same folder.

The sidebar switches between four pages: Oil reservoir, Gas reservoir, Multi-tank and PVT correlations. Tables can be edited in place or loaded from a CSV, and results can be downloaded as CSV.

To deploy on Streamlit Community Cloud, put `app.py`, `resmb.py` and `requirements.txt` in the root of a GitHub repository and set the main file to `app.py`.

## Units

| Quantity | Unit |
|---|---|
| Pressure | psia |
| Time | days |
| Np, Wp, Winj | MMSTB |
| Gp, Ginj (oil analysis, all multi-tank tanks) | MMscf |
| Gp (`gas_mbe`) | Bscf |
| Bo, Bw | rb/STB |
| Rs | scf/STB |
| Bg | rb/scf |
| Compressibility | 1/psi |
| F, We, crossflow | MMrb |
| Oil in place | MMSTB |
| Gas in place | Bscf |
| Aquifer constant C or B | rb/psi |
| Aquifer productivity J | rb/(day·psi) |
| Wei in `Aquifer` results | bbl |
| Wei in `Tank` | MMbbl |
| Transmissibility T | rb/(day·psi) |

## Oil reservoir

Put the production and PVT history in a DataFrame. **Row 0 is initial conditions** (zero production, initial pressure).

Required columns: `p, Np, Gp, Bo, Rs, Bg`. Optional: `t, Wp, Winj, Ginj, Bw`. Volumes are cumulative.

```python
import pandas as pd
from resmb import oil_mbe

data = pd.read_csv("history.csv")
result = oil_mbe(data, Swi=0.2, cw=3e-6, cf=4e-6, m=0.3)

print(result.in_place)   # N, MMSTB
print(result.gas_cap)    # Bscf
print(result.r2)
print(result.table)      # F, Eo, Eg, Efw, Et, We, F/Et and drive indices per row
```

Set `fit_m=True` to estimate the gas cap ratio from the data instead of fixing it. The estimate is sensitive to scatter; fix `m` from logs or seismic where you can.

## Aquifers

Pass an `Aquifer` to `oil_mbe` or `gas_mbe`. The `t` column is needed for every model except `pot`.

```python
from resmb import Aquifer

aq = Aquifer(model="veh", geometry="radial",
             k=150, phi=0.2, mu_w=0.4, ct=7e-6, h=60,
             ro=4000, reD=8, theta=180, tune=True)
result = oil_mbe(data, Swi=0.2, aquifer=aq)
print(result.aquifer)    # B, td_per_day, implied_k, ...
```

| `model` | Fitted | Needs |
|---|---|---|
| `"pot"` | C | nothing |
| `"steady"` | C | `t` |
| `"veh"` | B, and the time constant if `tune=True` | rock properties and geometry |
| `"fetkovich"` | Wei and J if `tune=True`, otherwise taken from the inputs | rock properties, finite `reD` |

Geometry is `"radial"` (`ro`, `reD`, `theta`; leave `reD` at its default for an infinite aquifer) or `"linear"` (`L`, `w`, closed outer end).

`Aquifer.constants(pi)` returns the time constant, influx constant, water volume and productivity implied by the rock properties. `WD(tD, geometry, reD)` gives the Van Everdingen–Hurst dimensionless influx on its own.

## Gas reservoir

Required columns: `p, z, Gp`. Optional: `t, Wp, Bw`.

```python
from resmb import gas_mbe

result = gas_mbe(data, temp_f=220, Swi=0.25, cw=3e-6, cf=0.0, pz_abandon=500)
print(result.G_pz)        # gas in place from the p/z line, Bscf
print(result.in_place)    # gas in place from Havlena–Odeh, Bscf
print(result.Gp_abandon)  # recoverable at the abandonment p/z, Bscf
```

## PVT correlations

```python
from resmb import BlackOilPVT

pvt = BlackOilPVT(api=35, gas_gravity=0.75, temp_f=200, rsb=600, correlation="standing")
pvt.pb                        # bubble point, psia
pvt.at(2500)                  # dict: Rs, Bo, co, mu_o, z, Bg, mu_g, Bw
pvt.table(range(500, 4001, 500))
data = pvt.fill(data)         # writes Bo, Rs, Bg, Bw and z from the 'p' column
```

Give either `rsb` or `pb`. `correlation` is `"standing"`, `"vasquez-beggs"` or `"glaso"`.

| Property | Correlation |
|---|---|
| Bubble point, Rs, Bo | Standing, Vasquez–Beggs or Glasø |
| Bo and oil viscosity above the bubble point | Vasquez–Beggs |
| Dead and saturated oil viscosity | Beggs–Robinson |
| Gas z-factor | Dranchuk–Abou-Kassem with Sutton pseudo-criticals |
| Gas viscosity | Lee–Gonzalez–Eakin |
| Bw | McCain, fresh water |

Correlations typically carry 5–15% error. Use lab PVT where it exists.

## Multi-tank

Each tank has its own volume, aquifer and production table (`t, Np, Gp, Wp`, plus `p_obs` where a pressure was measured). Production tables do not include a row for day 0. Fluid properties come from a `BlackOilPVT`.

```python
from resmb import Tank, Connection, MultiTank

tanks = [
    Tank("Main block", in_place=75, pi=3200, Swi=0.22, cf=4e-6,
         aquifer="fetkovich", Wei=9, J=10,
         fit_in_place=True, fit_aquifer=True, production=main_df),
    Tank("East block", in_place=40, pi=3200, Swi=0.22, cf=4e-6,
         fit_in_place=True, production=east_df),
]
model = MultiTank(tanks, [Connection("Main block", "East block", T=3, fit=True)],
                  pvt=pvt, cw=3e-6, max_step=30)

run = model.simulate()        # run.pressure, run.influx, run.crossflow (DataFrames)
match = model.history_match() # updates tanks and connections in place
print(match["rms_before"], match["rms_after"], match["fitted"])
```

| `Tank.aquifer` | Parameters |
|---|---|
| `"none"` | |
| `"pot"` | `C` |
| `"fetkovich"` | `Wei` (MMbbl), `J` |
| `"veh_radial"` | `C` (= B), `td_per_day`, `reD` |
| `"veh_linear"` | `C` (= B), `td_per_day` |

`fluid` is `"oil"` (default) or `"gas"`; `in_place` is MMSTB or Bscf accordingly.

`history_match` adjusts every parameter flagged `fit_in_place`, `fit_aquifer` or `fit` to minimise the squared difference from `p_obs`.

## Limitations

- **Matches are not unique.** Fitting tank size and aquifer strength together can give a good pressure match with the wrong values, because a smaller tank with a stronger aquifer behaves much like a larger tank with a weaker one. Fix what you know independently and fit the rest.
- **A good fit does not prove an aquifer exists.** Fitting an aquifer to a volumetric reservoir returns meaningless aquifer values. Check the Campbell plot (oil) or Cole plot (gas) first.
- **Crossflow between tanks is reservoir volume only.** The simulator does not track how much of the transferred volume is oil and how much is gas.
- **One PVT description for all tanks** in a `MultiTank` model.
- **Dry gas only** in `gas_mbe`; no condensate or abnormal-pressure correction.

## Files

| File | Contents |
|---|---|
| `resmb.py` | The library |
| `app.py` | Streamlit app |
| `examples.py` | Five worked cases with charts |
| `requirements.txt` | Dependencies |
