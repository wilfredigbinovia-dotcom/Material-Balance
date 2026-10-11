"""Vertical lift performance tables exported by Prosper (.tpd text files).

A table gives the flowing bottomhole pressure for every combination of gas rate,
tubing-head pressure, water-gas ratio and total gas-oil ratio, as calculated by the
well model with its own multiphase correlation. Interpolation is linear in tubing-head
pressure and water-gas ratio and in the logarithm of rate and gas-oil ratio (the grids
are geometric). Values outside the table are held at its edges.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PSIG = 14.696
# Prosper variable codes
VAR_NAMES = {27: "thp", 18: "wgr", 15: "gor", 4002: "q"}
COL_BHP, COL_TURNER = 5000, 5105


@dataclass
class LiftTable:
    q: np.ndarray            # MMscf/d gas
    thp: np.ndarray          # psia
    wgr: np.ndarray          # STB/MMscf
    gor: np.ndarray          # scf/STB (total gas per stock-tank condensate)
    bhp: np.ndarray          # psia, shape (thp, gor, wgr, q)
    turner: np.ndarray       # 1 where the gas velocity is below the Turner critical velocity
    name: str = ""
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        axes = (self.thp, np.log(self.gor), self.wgr, np.log(self.q))
        self._axes = [a if len(a) > 1 else np.array([a[0], a[0] + 1.0]) for a in axes]
        bhp, tur = self.bhp, self.turner.astype(float)
        for k, a in enumerate(axes):            # single-valued variables: duplicate the slice
            if len(a) == 1:
                bhp = np.concatenate([bhp, bhp], axis=k)
                tur = np.concatenate([tur, tur], axis=k)
        self._f = RegularGridInterpolator(self._axes, bhp)
        self._t = RegularGridInterpolator(self._axes, tur)

    def _pt(self, thp, q, wgr, cgr):
        gor = 1e6 / cgr if cgr and cgr > 0 else self.gor.max()
        x = [thp, np.log(max(gor, 1e-9)), wgr, np.log(max(q, 1e-9))]
        return [float(np.clip(v, a[0], a[-1])) for v, a in zip(x, self._axes)]

    def bhp_at(self, thp, q, wgr=0.0, cgr=0.0):
        """Flowing bottomhole pressure (psia). q in MMscf/d, cgr in STB/MMscf."""
        return float(self._f([self._pt(thp, q, wgr, cgr)])[0])

    def loading_at(self, thp, q, wgr=0.0, cgr=0.0):
        return float(self._t([self._pt(thp, q, wgr, cgr)])[0]) > 0.5

    def to_dict(self):
        return dict(q=self.q.tolist(), thp=self.thp.tolist(), wgr=self.wgr.tolist(),
                    gor=self.gor.tolist(), bhp=np.round(self.bhp, 2).ravel().tolist(),
                    turner=self.turner.astype(int).ravel().tolist(), name=self.name, info=self.info)

    @classmethod
    def from_dict(cls, d):
        shape = (len(d["thp"]), len(d["gor"]), len(d["wgr"]), len(d["q"]))
        return cls(np.array(d["q"]), np.array(d["thp"]), np.array(d["wgr"]), np.array(d["gor"]),
                   np.array(d["bhp"], float).reshape(shape), np.array(d["turner"]).reshape(shape),
                   d.get("name", ""), d.get("info", {}))


def read_tpd(text, name="", gauge=True):
    """Parse a Prosper .tpd file. gauge=True: pressures in the file are psig (Prosper oilfield
    units) and are converted to psia."""
    lines = [l.strip() for l in text.replace("\r", "").split("\n")]
    head = {}
    for l in lines:
        if l.startswith("#") and ":" in l:
            k, v = l[1:].split(":", 1)
            head.setdefault(k.strip(), v.strip())
    data = [l for l in lines if l and not l.startswith("#")]
    if not data or data[0] != "TPDData":
        raise ValueError("Not a Prosper TPD file (it should start with a TPDData record).")
    nums = lambda s: [float(x) for x in s.split(",") if x.strip()]   # noqa: E731
    try:
        nvar = int(nums(data[5])[0])
        counts = [int(x) for x in nums(data[6])]
        ncol = int(nums(data[7])[0])
        cols = [int(x) for x in nums(data[8])]
        types = [int(x) for x in nums(data[9])]
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not read the TPD header: {e}")
    fixed = nums(data[4])          # oil gravity, gas gravity, WGR, GOR, top TVD, bottom TVD
    if len(counts) != nvar + 1 or len(types) != nvar + 1 or len(cols) != ncol:
        raise ValueError("The TPD header is inconsistent (counts and variable types do not match).")
    if types[0] != 4002:
        raise ValueError(f"The table's rate variable is type {types[0]}; only gas rate tables "
                         "(type 4002) can be used for a gas-condensate well.")
    unknown = [t for t in types[1:] if t not in VAR_NAMES]
    if unknown:
        raise ValueError(f"Sensitivity variable type(s) {unknown} are not supported. Export the "
                         "lift curves with tubing-head pressure, water-gas ratio and total GOR.")
    pos = 10
    values = []
    for n in counts:
        v = nums(data[pos])
        if len(v) != n:
            raise ValueError("A list of sensitivity values has the wrong length.")
        values.append(np.array(v))
        pos += 1
    rows = np.array([nums(l) for l in data[pos:]])
    npt = int(np.prod(counts))
    if rows.shape != (npt, ncol):
        raise ValueError(f"Expected {npt} result rows of {ncol} values, found {rows.shape}.")
    if COL_BHP not in cols:
        raise ValueError("The table has no flowing bottomhole pressure column (5000).")
    # Results: rate varies fastest, then variable 3, 2, 1 (as listed after the rate).
    shape = tuple(counts[::-1])                       # (var1, var2, ..., rate)
    bhp = rows[:, cols.index(COL_BHP)].reshape(shape)
    tur = (rows[:, cols.index(COL_TURNER)].reshape(shape) if COL_TURNER in cols
           else np.zeros(shape))
    names = [VAR_NAMES[t] for t in types[::-1]]       # axis names in array order, rate last
    axis_vals = dict(zip(names, values[::-1]))
    # Missing variables are fixed at the header value
    defaults = {"thp": np.array([0.0]), "wgr": np.array([fixed[2] if fixed[2] < 1e30 else 0.0]),
                "gor": np.array([fixed[3] if fixed[3] < 1e30 else 1e6])}
    order = ["thp", "gor", "wgr", "q"]
    for k in order:
        if k not in axis_vals:
            axis_vals[k] = defaults[k]
            bhp, tur = bhp[np.newaxis], tur[np.newaxis]
            names = [k] + names
    perm = [names.index(k) for k in order]
    bhp, tur = np.transpose(bhp, perm), np.transpose(tur, perm)
    off = PSIG if gauge else 0.0
    if "thp" not in [VAR_NAMES[t] for t in types[1:]]:
        raise ValueError("The table does not vary tubing-head pressure, so it cannot be used "
                         "with a tubing-head pressure limit.")
    info = {"correlation": head.get("Vertical Lift Correlation", ""),
            "generated": head.get("Generated on", ""), "fluid": head.get("Fluid", ""),
            "bottom_tvd": fixed[5] if len(fixed) > 5 else None,
            "pressure_units": "psig" if gauge else "psia"}
    return LiftTable(q=axis_vals["q"], thp=axis_vals["thp"] + off, wgr=axis_vals["wgr"],
                     gor=axis_vals["gor"], bhp=bhp + off, turner=tur > 0.5, name=name, info=info)


# ----------------------------------------------------------------- oil wells
def _role(name):
    n = name.lower()
    if "top node" in n or "wellhead" in n or "tubing head" in n or "thp" in n or "whp" in n:
        return "thp"
    if "water cut" in n:
        return "wc"
    if "water gas" in n:
        return "wgr"
    if "gor" in n or "gas oil" in n:
        return "gor"
    if "rate" in n:
        return "q"
    return None


@dataclass
class OilLiftTable:
    """Prosper lift curves for an oil well: bottomhole pressure against liquid (or oil) rate,
    tubing-head pressure, water cut (%) and GOR (scf/STB)."""
    q: np.ndarray
    thp: np.ndarray
    wc: np.ndarray
    gor: np.ndarray
    bhp: np.ndarray            # (thp, gor, wc, q)
    rate_kind: str = "liquid"  # 'liquid' or 'oil'
    name: str = ""
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        axes = [self.thp, np.log(np.maximum(self.gor, 1e-6)), self.wc, np.log(self.q)]
        b = self.bhp
        for k, a in enumerate(list(axes)):
            if len(a) == 1:
                axes[k] = np.array([a[0], a[0] + 1.0])
                b = np.concatenate([b, b], axis=k)
        self._axes = axes
        self._f = RegularGridInterpolator(axes, b)

    def bhp_at(self, thp, q_liquid, wc_pct, gor):
        q = q_liquid if self.rate_kind == "liquid" else q_liquid * (1 - wc_pct / 100.0)
        x = [thp, np.log(max(gor, 1e-6)), wc_pct, np.log(max(q, 1e-9))]
        return float(self._f([[float(np.clip(v, a[0], a[-1])) for v, a in zip(x, self._axes)]])[0])

    def to_dict(self):
        return dict(q=self.q.tolist(), thp=self.thp.tolist(), wc=self.wc.tolist(), gor=self.gor.tolist(),
                    bhp=np.round(self.bhp, 2).ravel().tolist(), rate_kind=self.rate_kind,
                    name=self.name, info=self.info)

    @classmethod
    def from_dict(cls, d):
        shape = (len(d["thp"]), len(d["gor"]), len(d["wc"]), len(d["q"]))
        return cls(np.array(d["q"]), np.array(d["thp"]), np.array(d["wc"]), np.array(d["gor"]),
                   np.array(d["bhp"], float).reshape(shape), d.get("rate_kind", "liquid"),
                   d.get("name", ""), d.get("info", {}))


def read_tpd_oil(text, name="", gauge=True):
    """Parse a Prosper .tpd file made for an oil well. Variables are recognised from the
    file's own comment lines (rate, top node pressure, water cut, GOR)."""
    lines = [l.strip() for l in text.replace("\r", "").split("\n")]
    names = {}
    head = {}
    for l in lines:
        if not l.startswith("#"):
            continue
        s = l[1:].strip()
        if ":" in s:
            k, v = s.split(":", 1)
            head.setdefault(k.strip(), v.strip())
        if "=" in s and ("Variable" in s):
            right = s.split("=", 1)[1].strip()
            if "-" in right:
                code, nm = right.split("-", 1)
                try:
                    names[int(code.strip())] = nm.strip()
                except ValueError:
                    pass
    data = [l for l in lines if l and not l.startswith("#")]
    if not data or data[0] != "TPDData":
        raise ValueError("Not a Prosper TPD file (it should start with a TPDData record).")
    nums = lambda s: [float(x) for x in s.split(",") if x.strip()]   # noqa: E731
    nvar = int(nums(data[5])[0])
    counts = [int(x) for x in nums(data[6])]
    ncol = int(nums(data[7])[0])
    cols = [int(x) for x in nums(data[8])]
    types = [int(x) for x in nums(data[9])]
    fixed = nums(data[4])
    roles = []
    for t in types:
        r = _role(names.get(t, "")) if t in names else {27: "thp", 18: "wgr", 15: "gor", 4002: "q"}.get(t)
        roles.append(r)
    if roles[0] != "q":
        raise ValueError("Could not recognise the table's rate variable.")
    rname = names.get(types[0], "").lower()
    if "gas" in rname:
        raise ValueError("This is a gas-rate table; use it for a gas-condensate well.")
    rate_kind = "oil" if "oil" in rname else "liquid"
    bad = [names.get(t, str(t)) for t, r in zip(types[1:], roles[1:]) if r not in ("thp", "wc", "gor")]
    if bad:
        raise ValueError("Unsupported sensitivity variable(s): " + ", ".join(bad)
                         + ". Export with tubing-head pressure, water cut and GOR.")
    pos, values = 10, []
    for n in counts:
        values.append(np.array(nums(data[pos])))
        pos += 1
    rows = np.array([nums(l) for l in data[pos:]])
    if rows.shape != (int(np.prod(counts)), ncol):
        raise ValueError("The number of result rows does not match the table size.")
    bhp = rows[:, cols.index(COL_BHP)].reshape(tuple(counts[::-1]))
    axis_names = roles[::-1]
    vals = dict(zip(axis_names, values[::-1]))
    defaults = {"thp": np.array([0.0]), "wc": np.array([fixed[2] if len(fixed) > 2 and fixed[2] < 1e30 else 0.0]),
                "gor": np.array([fixed[3] if len(fixed) > 3 and fixed[3] < 1e30 else 500.0])}
    for k in ("thp", "gor", "wc", "q"):
        if k not in vals:
            vals[k] = defaults[k]
            bhp = bhp[np.newaxis]
            axis_names = [k] + axis_names
    if "thp" not in roles[1:]:
        raise ValueError("The table does not vary tubing-head pressure.")
    bhp = np.transpose(bhp, [axis_names.index(k) for k in ("thp", "gor", "wc", "q")])
    off = PSIG if gauge else 0.0
    info = {"correlation": head.get("Vertical Lift Correlation", ""), "rate": names.get(types[0], ""),
            "pressure_units": "psig" if gauge else "psia"}
    return OilLiftTable(q=vals["q"], thp=vals["thp"] + off, wc=vals["wc"], gor=vals["gor"], bhp=bhp + off,
                        rate_kind=rate_kind, name=name, info=info)
