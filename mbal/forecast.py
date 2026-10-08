"""Production forecast: the matched tank stepped forward under well and field limits."""
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .wells import well_rate


def forecast(tank, p_sim, we_sim, wells, rp, state, gz, days, dt=30.4375, q_target=None,
             q_min=None, vapour=True, cgr_const=None, stop_loading=False):
    """Step the tank from the end of history.

    tank          matched Tank (with its aquifer)          p_sim, we_sim  model history
    wells         list of Well with fitted inflow          rp             RelPerm
    state         TankState                                gz             GasZ of the well stream
    days          length of the forecast                   dt             step, days
    q_target      field separator-gas target, MMscf/d      q_min          stop below this rate                   stop_loading   shut in wells that load up
    cgr_const     producing CGR to use when the PVT table has no CGR column, STB/MMscf

    Rates over a step are evaluated at the pressure and saturations at the start of the step;
    the pressure at the end of the step then follows from the material balance.
    Returns (DataFrame, summary dict).
    """
    h, pv = tank.h, tank.pvt
    if not np.all(np.isfinite(p_sim)):
        raise ValueError("The model does not reproduce the whole history (the tank runs out of "
                         "gas). Fix the history match before forecasting.")
    n0 = len(h.t)
    nf = int(np.ceil(days / dt))
    t = np.concatenate([h.t, h.t[-1] + dt * np.arange(1, nf + 1)])
    N = len(t)
    aq = replace(tank.aq).prepare(t)
    p, We, gx = np.full(N, np.nan), np.zeros(N), np.zeros(N)
    p[:n0], We[:n0] = p_sim, we_sim
    gx[:n0] = tank.gx_sim
    pn = tank.neighbour_p(t)          # held at its last survey through the forecast
    gp, npc, wp = (np.concatenate([a, np.zeros(nf)]) for a in (h.gp, h.np_, h.wp))
    gpw = gp + pv.ge * npc
    cols = {k: np.full(N, np.nan) for k in ("qg", "qc", "qw", "sw", "so", "wgr", "cgr", "M")}
    qwell = {w.name: np.full(N, np.nan) for w in wells}
    pwfw = {w.name: np.full(N, np.nan) for w in wells}
    loadw = {w.name: np.zeros(N, bool) for w in wells}
    plo = max(pv.p_range[0], 14.7)
    reason, last = "end of forecast period", N - 1
    for i in range(n0, N):
        pr = p[i - 1]
        sw = float(state.sw(pr, We[i - 1], wp[i - 1]))
        M = float(state.mobility(rp, pr, sw))
        cgr = float(pv.cgr(pr)) if cgr_const is None else cgr_const
        wet = 1.0 + pv.ge * cgr / 1e6
        wgr = float(state.wgr_free(rp, pr, sw)) * wet + (float(state.wgr_vapour(pr)) if vapour else 0.0)
        res = [well_rate(w, pr, M, gz, pv.T, wet, cgr, wgr, stop_loading) for w in wells]
        q = np.array([r[0] for r in res])
        tot = q.sum()
        if q_target and tot > q_target:
            q *= q_target / tot
            tot = q_target
        if tot <= max(q_min or 0.0, 1e-6):
            if tot > 1e-6:
                reason = "field rate fell below the minimum rate"
            elif any(r[2] for r in res):
                reason = "the remaining wells have loaded up with liquid"
            else:
                reason = "wells can no longer flow against their pressure limits"
            last = i - 1
            break
        step = t[i] - t[i - 1]
        qg, qc = tot * 1e6, tot * cgr                 # scf/d, STB/d
        qw = tot * wgr                                # STB/d
        gp[i], npc[i], wp[i] = gp[i - 1] + qg * step, npc[i - 1] + qc * step, wp[i - 1] + qw * step
        gpw[i] = gp[i] + pv.ge * npc[i]

        def bal(pp):
            p[i] = pp
            return (tank.G * tank.Et(pp) + aq.we_at(i, p, We)
                    - (gpw[i] - tank.gx_step(i, p, gx, t, pn)) * pv.bg(pp) - wp[i] * tank.bw)

        r_hi, r_lo = bal(tank.pi), bal(plo)
        if not (np.isfinite(r_hi) and np.isfinite(r_lo)) or r_lo < 0:
            reason, last = "reservoir pressure fell below the PVT table", i - 1
            p[i] = np.nan
            break
        p[i] = tank.pi if r_hi >= 0 else brentq(bal, plo, tank.pi, xtol=1e-3)
        We[i] = aq.we_at(i, p, We)
        gx[i] = tank.gx_step(i, p, gx, t, pn)
        for k, v in (("qg", tot), ("qc", qc), ("qw", qw), ("sw", sw), ("so", float(state.so(pr))),
                     ("wgr", wgr), ("cgr", cgr), ("M", M)):
            cols[k][i] = v
        for w, qq, r in zip(wells, q, res):
            qwell[w.name][i] = qq
            loadw[w.name][i] = r[2]
            pwfw[w.name][i] = r[1] if qq >= r[0] * 0.999 else np.sqrt(
                max(pr * pr - (qq / w.C) ** (1.0 / w.n) / M, 0.0)) if qq > 0 else np.nan
    sl = slice(0, last + 1)
    df = pd.DataFrame({"t": t[sl], "p": p[sl], "gp": gp[sl], "np": npc[sl], "wp": wp[sl],
                       "we": We[sl], "gx": gx[sl], **{k: v[sl] for k, v in cols.items()},
                       **{f"q_{k}": v[sl] for k, v in qwell.items()},
                       **{f"pwf_{k}": v[sl] for k, v in pwfw.items()},
                       **{f"load_{k}": v[sl] for k, v in loadw.items()}})
    df["forecast"] = np.arange(len(df)) >= n0
    summary = dict(reason=reason, n_hist=n0, t_end=float(t[last]), p_end=float(p[last]),
                   gp_end=float(gp[last]), np_end=float(npc[last]), wp_end=float(wp[last]),
                   gx_end=float(gx[last]), rf_wet=float(gpw[last] / tank.G), rf_hist=float(gpw[n0 - 1] / tank.G),
                   stopped=last < N - 1)
    return df, summary
