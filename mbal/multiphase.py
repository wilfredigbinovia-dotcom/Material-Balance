"""Vertical multiphase flow in oil wells: Beggs and Brill (1973) with the Payne holdup correction.

The tubing is marched from the wellhead down in segments. In each segment the black-oil table
gives the in-situ oil, free gas and water rates and densities; Beggs-Brill gives the liquid
holdup and the two-phase friction factor. Acceleration is neglected and temperature varies
linearly from the flowing wellhead temperature to reservoir temperature.
"""
import numpy as np

from .vlp import OilLiftTable

G = 32.174
ROUGH = 0.0006            # in
SIGMA_O, SIGMA_W = 30.0, 60.0   # dyn/cm


def _moody(re, eps_d):
    re = max(re, 1.0)
    if re < 2000:
        return 64.0 / re
    return 0.25 / np.log10(eps_d / 3.7 + 5.74 / re ** 0.9) ** 2


def _holdup(lam, nfr, nlv, theta=90.0):
    """Beggs-Brill liquid holdup (uphill) with the Payne correction."""
    lam = min(max(lam, 1e-6), 1.0)
    if lam >= 0.999:
        return 1.0
    L1, L2 = 316 * lam ** 0.302, 0.0009252 * lam ** -2.4684
    L3, L4 = 0.1 * lam ** -1.4516, 0.5 * lam ** -6.738
    coef = {"seg": (0.98, 0.4846, 0.0868, (0.011, -3.768, 3.539, -1.614)),
            "int": (0.845, 0.5351, 0.0173, (2.96, 0.305, -0.4473, 0.0978)),
            "dis": (1.065, 0.5824, 0.0609, None)}

    def hl(reg):
        a, b, c, cc = coef[reg]
        h0 = max(a * lam ** b / max(nfr, 1e-9) ** c, lam)
        C = 0.0
        if cc is not None:
            e, f, g_, h = cc
            arg = e * lam ** f * max(nlv, 1e-9) ** g_ * max(nfr, 1e-9) ** h
            C = max((1 - lam) * np.log(arg), 0.0) if arg > 0 else 0.0
        s = np.sin(np.radians(1.8 * theta))
        return min(max(h0 * (1 + C * (s - s ** 3 / 3.0)) * 0.924, lam), 1.0)    # 0.924: Payne uphill factor

    if (lam < 0.01 and nfr < L1) or (lam >= 0.01 and nfr < L2):
        return hl("seg")
    if lam >= 0.01 and L2 <= nfr <= L3:
        A = (L3 - nfr) / (L3 - L2)
        return A * hl("seg") + (1 - A) * hl("int")
    if (0.01 <= lam < 0.4 and L3 < nfr <= L1) or (lam >= 0.4 and L3 < nfr <= L4):
        return hl("int")
    return hl("dis")


def bb_gradient(p, T, qo, qw, gor, d_in, pvt, api, sg, muw=0.5, bw=1.02, theta=90.0):
    """Pressure gradient (psi/ft) at pressure p, temperature T for surface rates qo, qw (STB/d)
    and producing GOR (scf/STB)."""
    A = np.pi / 4 * (d_in / 12) ** 2
    bo, rs, bg = float(pvt.bo(p)), float(pvt.rs(p)), float(pvt.bg(p))
    muo, mug = float(pvt.muo(p)), float(pvt.mug(p))
    free = max(gor - rs, 0.0)
    ql = (qo * bo + qw * bw) * 5.615 / 86400.0            # ft3/s
    qg = qo * free * bg * 5.615 / 86400.0
    vsl, vsg = ql / A, qg / A
    vm = vsl + vsg
    sgo = 141.5 / (api + 131.5)
    rho_o = (62.4 * sgo + 0.0136 * rs * sg) / bo
    rho_w = 62.4 * 1.02 / bw
    rho_g = 0.0764 * sg / (bg * 5.615)
    fo = qo * bo / max(qo * bo + qw * bw, 1e-9)
    rho_l = fo * rho_o + (1 - fo) * rho_w
    mu_l = fo * muo + (1 - fo) * muw
    sig = fo * SIGMA_O + (1 - fo) * SIGMA_W
    if vm <= 1e-9:
        return rho_l / 144.0
    lam = vsl / vm
    D = d_in / 12
    nfr = vm * vm / (G * D)
    nlv = 1.938 * vsl * (rho_l / sig) ** 0.25
    hl = _holdup(lam, nfr, nlv, theta)
    rho_s = rho_l * hl + rho_g * (1 - hl)
    rho_n = rho_l * lam + rho_g * (1 - lam)
    mu_n = (mu_l * lam + mug * (1 - lam)) * 6.72e-4
    fn = _moody(rho_n * vm * D / mu_n, ROUGH / d_in)
    y = lam / max(hl * hl, 1e-12)
    if 1.0 < y < 1.2:
        S = np.log(2.2 * y - 1.2)
    elif y > 0 and abs(y - 1) > 1e-9:
        ly = np.log(y)
        S = ly / (-0.0523 + 3.182 * ly - 0.8725 * ly ** 2 + 0.01853 * ly ** 4)
    else:
        S = 0.0
    ftp = fn * np.exp(S)
    return (rho_s * np.sin(np.radians(theta)) + ftp * rho_n * vm * vm / (2 * G * D)) / 144.0


def bb_bhp(thp, q_liq, wc, gor, tvd, md, d_in, wht, t_res, pvt, api, sg, muw=0.5, bw=1.02, nseg=40):
    """Flowing bottomhole pressure (psia) from tubing-head pressure. q_liq in STB/d, wc as a
    fraction, gor in scf/STB."""
    md = md if np.isfinite(md) and md >= tvd else tvd
    theta = np.degrees(np.arcsin(min(tvd / md, 1.0)))
    qo, qw = q_liq * (1 - wc), q_liq * wc
    dz = md / nseg
    p = thp
    for k in range(nseg):
        T = wht + (t_res - wht) * (k + 0.5) / nseg
        g1 = bb_gradient(p, T, qo, qw, gor, d_in, pvt, api, sg, muw, bw, theta)
        g2 = bb_gradient(p + g1 * dz / 2, T, qo, qw, gor, d_in, pvt, api, sg, muw, bw, theta)
        p += g2 * dz
    return float(p)


def bb_lift_table(thps, tvd, md, d_in, wht, t_res, pvt, api, sg, muw=0.5, bw=1.02, qmax=20000.0,
                  name="Beggs and Brill"):
    """Tabulate Beggs-Brill as a lift-curve table so the forecast can interpolate it quickly."""
    q = np.geomspace(10.0, qmax, 18)
    wc = np.array([0.0, 30.0, 60.0, 80.0, 95.0])
    rs_lo = max(float(pvt.rs(pvt.p_range[0])), 20.0)
    gor = np.geomspace(rs_lo, 30000.0, 7)
    thps = np.atleast_1d(np.asarray(thps, float))
    if len(thps) == 1:
        thps = np.array([thps[0], thps[0] * 1.5 + 50.0])
    bhp = np.empty((len(thps), len(gor), len(wc), len(q)))
    for a, th in enumerate(thps):
        for b, gr in enumerate(gor):
            for c, w in enumerate(wc):
                for d, qq in enumerate(q):
                    bhp[a, b, c, d] = bb_bhp(th, qq, w / 100.0, gr, tvd, md, d_in, wht, t_res, pvt, api, sg,
                                             muw, bw, nseg=25)
    return OilLiftTable(q=q, thp=thps, wc=wc, gor=gor, bhp=bhp, rate_kind="liquid", name=name,
                        info={"correlation": "Beggs and Brill (built in)"})
