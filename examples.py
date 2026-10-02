"""
Worked examples for resmb.py. Run:  python examples.py
Prints results and saves four charts as PNG files next to this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resmb import (Aquifer, BlackOilPVT, Connection, MultiTank, Tank, WD, gas_mbe, oil_mbe)

OUT = Path(__file__).parent
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a888a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e8ea", "axes.edgecolor": "#c5cdd0", "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.5g}")

OIL_COLS = ["t", "p", "Np", "Gp", "Wp", "Winj", "Ginj", "Bo", "Rs", "Bg", "Bw"]


def section(title):
    print("\n" + title + "\n" + "-" * len(title))


# -----------------------------------------------------------------------------
# 1. PVT correlations
# -----------------------------------------------------------------------------
section("1. PVT correlations (35 API, gas gravity 0.75, 200 F, Rsb 600 scf/STB)")
pvt = BlackOilPVT(api=35, gas_gravity=0.75, temp_f=200, rsb=600, correlation="standing")
print(f"Bubble point {pvt.pb:,.0f} psia, Bo at pb {pvt.bob:.4f} rb/STB, oil viscosity at pb {pvt.mu_ob:.3f} cp")
print(pvt.table(np.arange(500, 4001, 500)).to_string(index=False))

# -----------------------------------------------------------------------------
# 2. Oil reservoir with a gas cap, no aquifer
# -----------------------------------------------------------------------------
section("2. Oil reservoir with gas cap (true N = 150 MMSTB, m = 0.3)")
oil = pd.DataFrame([
    [0, 3000, 0, 0, 0, 0, 0, 1.315, 510, 0.000953, 1.02],
    [365, 2850, 3.7423, 2395.1, 0, 0, 0, 1.3035, 487, 0.001009, 1.02],
    [730, 2700, 7.1423, 5642.4, 0.02, 0, 0, 1.292, 464, 0.001071, 1.02],
    [1095, 2550, 10.1057, 9701.5, 0.06, 0, 0, 1.2805, 441, 0.001142, 1.02],
    [1460, 2400, 12.8236, 14490.7, 0.11, 0, 0, 1.269, 418, 0.001221, 1.02],
    [1825, 2250, 15.3133, 19907.3, 0.17, 0, 0, 1.2575, 395, 0.001312, 1.02],
    [2190, 2100, 17.3765, 25717.3, 0.24, 0, 0, 1.246, 372, 0.001416, 1.02],
    [2555, 1950, 19.4235, 32243.1, 0.32, 0, 0, 1.2345, 349, 0.001537, 1.02]], columns=OIL_COLS)
r_oil = oil_mbe(oil, Swi=0.2, cw=3e-6, cf=4e-6, m=0.3)
print(f"N = {r_oil.in_place:.1f} MMSTB, gas cap {r_oil.gas_cap:.1f} Bscf, R2 = {r_oil.r2:.4f}")
print(r_oil.table[["p", "F", "Et", "DDI", "SDI", "CDI", "WDI"]].to_string(index=False))

# -----------------------------------------------------------------------------
# 3. Water-drive oil reservoir: Van Everdingen-Hurst and Fetkovich
# -----------------------------------------------------------------------------
section("3. Water-drive oil reservoir (true N = 100 MMSTB, B = 752 rb/psi, k = 150 md)")
wd = pd.DataFrame([
    [0, 3000, 0, 0, 0, 0, 0, 1.3394, 600, 0.0009644, 1.034],
    [182, 2800, 1.0861, 651.6, 0, 0, 0, 1.3436, 600, 0.001026, 1.034],
    [364, 2650, 2.9163, 1749.8, 0.05, 0, 0, 1.347, 600, 0.001081, 1.035],
    [546, 2500, 5.2994, 3769, 0.15, 0, 0, 1.3389, 580.7, 0.001143, 1.035],
    [728, 2380, 7.3968, 6680.9, 0.3, 0, 0, 1.322, 547.6, 0.001199, 1.035],
    [910, 2280, 9.0032, 9572.4, 0.5, 0, 0, 1.3081, 520.3, 0.001252, 1.036],
    [1092, 2200, 10.0845, 12012.8, 0.8, 0, 0, 1.2971, 498.6, 0.001298, 1.036],
    [1274, 2130, 10.9346, 14250.1, 1.1, 0, 0, 1.2877, 479.8, 0.001342, 1.036],
    [1456, 2070, 11.5714, 16190.9, 1.4, 0, 0, 1.2796, 463.7, 0.001382, 1.036]], columns=OIL_COLS)
rock = dict(geometry="radial", k=150, phi=0.2, mu_w=0.4, ct=7e-6, h=60, ro=4000, reD=8, theta=180)
r_none = oil_mbe(wd, Swi=0.2)
r_veh = oil_mbe(wd, Swi=0.2, aquifer=Aquifer(model="veh", tune=True, **rock))
r_fet = oil_mbe(wd, Swi=0.2, aquifer=Aquifer(model="fetkovich", tune=True, **rock))
print(f"No aquifer        : N = {r_none.in_place:6.1f} MMSTB  (R2 {r_none.r2:.4f})  <- overstated")
print(f"Van Everdingen-H. : N = {r_veh.in_place:6.1f} MMSTB, B = {r_veh.aquifer['B']:.0f} rb/psi, "
      f"implied k = {r_veh.aquifer['implied_k']:.0f} md")
print(f"Fetkovich         : N = {r_fet.in_place:6.1f} MMSTB, Wei = {r_fet.aquifer['Wei'] / 1e6:.1f} MMbbl, "
      f"J = {r_fet.aquifer['J']:.1f} rb/d/psi")
print("W_D check, infinite radial, tD = 1, 10, 100:", np.round(WD(np.array([1.0, 10.0, 100.0])), 3))

# -----------------------------------------------------------------------------
# 4. Gas reservoir
# -----------------------------------------------------------------------------
section("4. Volumetric gas reservoir (true G = 60 Bscf)")
gas = pd.DataFrame([
    [0, 4200, 0.92, 0, 0, 1.02], [180, 3920, 0.905, 3.05, 0, 1.02], [365, 3640, 0.892, 6.371, 0, 1.02],
    [545, 3360, 0.881, 9.819, 0, 1.02], [730, 3080, 0.872, 13.493, 0, 1.02],
    [910, 2800, 0.866, 17.48, 0, 1.02], [1095, 2520, 0.862, 21.616, 0, 1.02],
    [1275, 2240, 0.861, 25.894, 0, 1.02]], columns=["t", "p", "z", "Gp", "Wp", "Bw"])
r_gas = gas_mbe(gas, temp_f=220, Swi=0.25, cw=3e-6, cf=0.0, pz_abandon=500)
print(f"G from p/z = {r_gas.G_pz:.2f} Bscf, G from Havlena-Odeh = {r_gas.in_place:.2f} Bscf, "
      f"recoverable to p/z 500 = {r_gas.Gp_abandon:.2f} Bscf")

# -----------------------------------------------------------------------------
# 5. Multi-tank history match
# -----------------------------------------------------------------------------
section("5. Two connected oil tanks, history match (true N = 90 / 40 MMSTB, T = 8)")
t = [365, 730, 1095, 1460, 1825, 2190]
main = pd.DataFrame(dict(t=t, Np=[2.2, 4.5, 6.6, 8.4, 10, 11.4], Gp=[1320, 2750, 4300, 6100, 8000, 9900],
                         Wp=[0, 0.05, 0.2, 0.5, 0.9, 1.4], p_obs=[2619, 2475, 2321, 2183, 2068, 1948]))
east = pd.DataFrame(dict(t=t, Np=[0.5, 1.1, 1.6, 2.0, 2.4, 2.7], Gp=[300, 680, 1050, 1400, 1750, 2050],
                         Wp=[0] * 6, p_obs=[2600, 2471, 2370, 2270, 2177, 2091]))
tanks = [
    Tank("Main block", in_place=75, pi=3200, Swi=0.22, cf=4e-6, aquifer="fetkovich", Wei=9, J=10,
         fit_in_place=True, fit_aquifer=True, production=main),
    Tank("East block", in_place=40, pi=3200, Swi=0.22, cf=4e-6, fit_in_place=True, production=east)]
# For a Van Everdingen-Hurst aquifer instead:
#   Tank(..., aquifer="veh_radial", C=500, td_per_day=0.05, reD=6, ...)
model = MultiTank(tanks, [Connection("Main block", "East block", T=3, fit=True)], pvt=pvt, cw=3e-6)
start = model.simulate()
match = model.history_match()
final = model.simulate()
print(f"RMS pressure error {match['rms_before']:.1f} -> {match['rms_after']:.1f} psi in {match['runs']} runs")
for k, v in match["fitted"].items():
    print(f"  {k:28s} {v:8.2f}")

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
D = r_oil.table.iloc[1:]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot([0, D.Et.max() * 1.05], [0, r_oil.in_place * D.Et.max() * 1.05], color=ORANGE, lw=2, label="Fitted line")
ax[0].plot(D.Et, D.F, "o", color=BLUE, label="Survey points")
ax[0].set(title=f"F vs Et (slope = N = {r_oil.in_place:.1f} MMSTB)", xlabel="Et, rb/STB", ylabel="F, MMrb")
ax[0].legend(frameon=False)
ax[1].axhline(r_oil.in_place, color=GREY, ls="--", lw=1.5, label="Fitted N")
ax[1].plot(oil.Np[1:], D.F_over_Et, "o", color=BLUE, label="F/Et")
ax[1].set(title="Campbell plot", xlabel="Np, MMSTB", ylabel="F/Et, MMSTB")
ax[1].legend(frameon=False)
bottom = np.zeros(len(D))
for col, name, c in [("DDI", "Depletion", BLUE), ("SDI", "Gas cap", ORANGE), ("CDI", "Rock & water", AQUA),
                     ("WDI", "Water influx", YELLOW)]:
    v = D[col].clip(lower=0).to_numpy()
    ax[2].bar(D.p.astype(int).astype(str), v, bottom=bottom, color=c, label=name, width=0.6, edgecolor="white")
    bottom += v
ax[2].set(title="Drive indices", xlabel="Survey pressure, psia", ylabel="Fraction of voidage")
ax[2].legend(frameon=False, ncol=2, fontsize=8)
ax[2].grid(False)
fig.tight_layout(); fig.savefig(OUT / "oil_material_balance.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
a, b = r_gas.pz_fit
ax[0].plot([0, r_gas.G_pz], [a, 0], color=ORANGE, lw=2, label="Fitted line")
ax[0].plot(gas.Gp, gas.p / gas.z, "o", color=BLUE, label="Survey points")
ax[0].set(title=f"p/z vs Gp (G = {r_gas.G_pz:.1f} Bscf)", xlabel="Gp, Bscf", ylabel="p/z, psia", ylim=(0, None))
ax[0].legend(frameon=False)
Dg = r_gas.table.iloc[1:]
ax[1].axhline(r_gas.in_place, color=GREY, ls="--", lw=1.5, label="Fitted G")
ax[1].plot(gas.Gp[1:], Dg.F_over_Et, "o", color=BLUE, label="F/Et")
ax[1].set(title="Cole plot", xlabel="Gp, Bscf", ylabel="F/Et, Bscf")
ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "gas_material_balance.png", dpi=150); plt.close(fig)

tab = pvt.table(np.linspace(200, 4000, 150))
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for axx, (col, title, unit) in zip(axes.flat, [("Rs", "Solution GOR", "scf/STB"), ("Bo", "Oil FVF", "rb/STB"),
                                               ("mu_o", "Oil viscosity", "cp"), ("z", "Gas z-factor", ""),
                                               ("Bg", "Gas FVF", "rb/scf"), ("mu_g", "Gas viscosity", "cp")]):
    axx.plot(tab.p, tab[col], color=BLUE, lw=2)
    axx.axvline(pvt.pb, color=GREY, ls=":", lw=1.2)
    axx.set(title=title, xlabel="Pressure, psia", ylabel=unit)
fig.tight_layout(); fig.savefig(OUT / "pvt_correlations.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for axx, res, title in [(ax[0], start, f"Before match (RMS {match['rms_before']:.0f} psi)"),
                        (ax[1], final, f"After match (RMS {match['rms_after']:.1f} psi)")]:
    for tk, c in zip(tanks, [BLUE, ORANGE]):
        axx.plot(res.t, res.pressure[tk.name], color=c, lw=2, label=tk.name)
        axx.plot(tk.production.t, tk.production.p_obs, "o", color=c, mec="white", ms=7)
    axx.set(title=title, xlabel="Time, days")
    axx.legend(frameon=False)
ax[0].set_ylabel("Pressure, psia")
fig.tight_layout(); fig.savefig(OUT / "multitank_history_match.png", dpi=150); plt.close(fig)
print("\nCharts saved:", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))
