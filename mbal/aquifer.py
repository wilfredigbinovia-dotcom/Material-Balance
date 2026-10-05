"""Aquifer influx models.

Dimensionless functions come from numerical Laplace inversion (Stehfest) of the
diffusivity-equation solutions, so no table look-ups are needed and any reD works.

    We = U * dp * WD(tD)          constant terminal pressure (van Everdingen & Hurst)
    radial :  tD = 0.006328 k t / (phi mu ct ro^2),  U = 1.119 phi ct h ro^2 (theta/360)
    linear :  tD = 0.006328 k t / (phi mu ct L^2),   U = phi ct w h L / 5.615
    bottom :  vertical linear flow through area pi ro^2 (theta/360) over the aquifer
              thickness, with kv = k * (kv/kh)
"""
from dataclasses import dataclass, field
from math import factorial
import numpy as np
from scipy.special import ive, kve

MODELS = {
    "none": "None (volumetric)",
    "pot": "Small pot",
    "schilthuis": "Schilthuis steady state",
    "veh": "Hurst-van Everdingen (unsteady state)",
    "carter_tracy": "Carter-Tracy",
    "fetkovich": "Fetkovich (pseudo-steady state)",
}
GEOMETRIES = {"radial": "Radial", "linear": "Linear", "bottom": "Bottom drive"}
NEEDS_GEOMETRY = ("veh", "carter_tracy", "fetkovich")


def _stehfest_coeffs(N=12):
    V = np.zeros(N)
    h = N // 2
    for i in range(1, N + 1):
        s = 0.0
        for k in range((i + 1) // 2, min(i, h) + 1):
            s += (k ** h * factorial(2 * k)
                  / (factorial(h - k) * factorial(k) * factorial(k - 1)
                     * factorial(i - k) * factorial(2 * k - i)))
        V[i - 1] = (-1) ** (i + h) * s
    return V


_V = _stehfest_coeffs(12)
_I = np.arange(1, 13)


def _invert(fbar, t):
    """Stehfest inversion of fbar(s) at times t (array, t > 0)."""
    t = np.atleast_1d(np.asarray(t, float))
    a = np.log(2.0) / t
    s = np.outer(_I, a)
    return (a * (_V[:, None] * fbar(s)).sum(axis=0))


def _kernel(geometry, reD):
    """k(s) = s * qD_bar(s) for a closed outer boundary (reD = inf -> infinite radial)."""
    if geometry == "radial":
        def k(s):
            u = np.sqrt(s)
            k0, k1 = kve(0, u), kve(1, u)
            if not np.isfinite(reD):
                return u * k1 / k0
            x = reD * u
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                r = kve(1, x) / ive(1, x) * np.exp(-2.0 * (x - u))
            r = np.where(np.isfinite(r), r, 0.0)
            return u * (k1 - r * ive(1, u)) / (k0 + r * ive(0, u))
        return k

    def k(s):  # finite linear, sealed far end
        u = np.sqrt(s)
        return u * np.tanh(u)
    return k


def WD(tD, geometry="radial", reD=np.inf):
    """Dimensionless cumulative influx, constant terminal pressure."""
    tD = np.atleast_1d(np.asarray(tD, float))
    out = np.zeros_like(tD)
    m = tD > 0
    if m.any():
        k = _kernel(geometry, reD)
        out[m] = _invert(lambda s: k(s) / s ** 2, tD[m])
    return np.maximum(out, 0.0)


def pD(tD, geometry="radial", reD=np.inf):
    """Dimensionless pressure and its derivative, constant terminal rate."""
    tD = np.atleast_1d(np.asarray(tD, float))
    k = _kernel(geometry, reD)
    return _invert(lambda s: 1.0 / (s * k(s)), tD), _invert(lambda s: 1.0 / k(s), tD)


@dataclass
class Aquifer:
    model: str = "none"
    geometry: str = "radial"
    ct: float = 7e-6          # cw + cf, 1/psi
    k: float = 100.0          # md (horizontal)
    phi: float = 0.2
    mu: float = 0.3           # cp
    h: float = 50.0           # ft, aquifer thickness (radial/linear); also bottom-drive thickness
    ro: float = 5000.0        # ft, reservoir radius
    reD: float = 5.0          # re/ro ; np.inf for infinite acting
    theta: float = 360.0      # encroachment angle
    width: float = 5000.0     # ft (linear)
    L: float = 20000.0        # ft (linear)
    kvkh: float = 0.1         # bottom drive
    Wvol: float = 0.0         # rb, pot aquifer water volume
    C: float = 0.0            # rb/day/psi, Schilthuis constant
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- derived constants
    @property
    def active(self):
        return self.model != "none"

    def _geom(self):
        """(dimensionless-geometry, reD, U [rb/psi], tD per day, Wi [rb], J [rb/d/psi])."""
        f = self.theta / 360.0
        if self.geometry == "radial":
            U = 1.119 * self.phi * self.ct * self.h * self.ro ** 2 * f
            a = 0.006328 * self.k / (self.phi * self.mu * self.ct * self.ro ** 2)
            if np.isfinite(self.reD):
                Wi = np.pi * (self.reD ** 2 - 1) * self.ro ** 2 * self.h * self.phi * f / 5.615
                J = 0.00708 * self.k * self.h * f / (self.mu * max(np.log(self.reD) - 0.75, 0.05))
            else:
                Wi, J = np.inf, 0.0
            return "radial", self.reD, U, a, Wi, J
        if self.geometry == "linear":
            area, length, kk = self.width * self.h, self.L, self.k
        else:  # bottom drive: vertical linear flow
            area, length, kk = np.pi * self.ro ** 2 * f, self.h, self.k * self.kvkh
        Wi = area * length * self.phi / 5.615
        U = Wi * self.ct
        a = 0.006328 * kk / (self.phi * self.mu * self.ct * length ** 2)
        J = 3.0 * 0.001127 * kk * area / (self.mu * length)
        return "linear", np.inf, U, a, Wi, J

    def tD(self, t):
        return self._geom()[3] * np.asarray(t, float)

    def prepare(self, t):
        """Pre-compute everything that depends on the time grid only."""
        t = np.asarray(t, float)
        c = {"t": t}
        if self.model in NEEDS_GEOMETRY:
            g, reD, U, a, Wi, J = self._geom()
            tD = a * t
            c.update(U=U, tD=tD, Wi=Wi, J=J)
            if self.model == "veh":
                n = len(t)
                M = np.zeros((n, n))
                iu = np.tril_indices(n, -1)
                M[iu] = WD(tD[iu[0]] - tD[iu[1]], g, reD)
                c["WD"] = M
            elif self.model == "carter_tracy":
                p, dp = np.zeros(len(t)), np.zeros(len(t))
                p[1:], dp[1:] = pD(tD[1:], g, reD)
                c["pD"], c["dpD"] = p, dp
        self._cache = c
        return self

    def we_at(self, n, p, We):
        """Cumulative influx (rb) at step n given pressures p[0..n] and We[0..n-1]."""
        if n == 0 or self.model == "none":
            return 0.0
        c = self._cache
        t = c["t"]
        m = self.model
        if m == "pot":
            return self.ct * self.Wvol * (p[0] - p[n])
        if m == "schilthuis":
            return We[n - 1] + self.C * (p[0] - 0.5 * (p[n - 1] + p[n])) * (t[n] - t[n - 1])
        if m == "veh":
            dp = np.empty(n)
            dp[0] = 0.5 * (p[0] - p[1])
            if n > 1:
                dp[1:] = 0.5 * (p[0:n - 1] - p[2:n + 1])
            return c["U"] * float(dp @ c["WD"][n, :n])
        if m == "carter_tracy":
            tD, pd_, dpd = c["tD"], c["pD"][n], c["dpD"][n]
            num = c["U"] * (p[0] - p[n]) - We[n - 1] * dpd
            den = pd_ - tD[n - 1] * dpd
            return We[n - 1] + num / den * (tD[n] - tD[n - 1])
        if m == "fetkovich":
            cap = self.ct * c["Wi"]                 # rb/psi
            if not np.isfinite(cap) or cap <= 0:
                return float("nan")
            pa = p[0] - We[n - 1] / cap
            pbar = 0.5 * (p[n - 1] + p[n])
            return We[n - 1] + cap * (pa - pbar) * (1.0 - np.exp(-c["J"] * (t[n] - t[n - 1]) / cap))
        raise ValueError(m)

    def influx(self, t, p):
        """Cumulative influx series for a known pressure history."""
        self.prepare(t)
        We = np.zeros(len(t))
        for n in range(1, len(t)):
            We[n] = self.we_at(n, p, We)
        return We
