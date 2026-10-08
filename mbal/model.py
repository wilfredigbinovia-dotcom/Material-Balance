"""Glue between a flat configuration dict (field units) and the engine objects."""
import numpy as np
from .aquifer import Aquifer
from .matbal import Tank

# Regressable parameters: key -> (label, display quantity, lower limit, upper limit)
PARAMS = {
    "G":        ("Gas initially in place (wet)", "gas_ip", 1e-3, 1e6),
    "cf":       ("Rock compressibility", "compress", 1e-7, 1e-3),
    "aq_ro":    ("Reservoir radius", "length", 10.0, 1e6),
    "aq_reD":   ("Outer/inner radius ratio", "ratio", 1.05, 200.0),
    "aq_h":     ("Aquifer thickness", "length", 1.0, 1e4),
    "aq_theta": ("Encroachment angle", "angle", 1.0, 360.0),
    "aq_k":     ("Aquifer permeability", "perm", 1e-3, 1e5),
    "aq_width": ("Aquifer width", "length", 10.0, 1e6),
    "aq_L":     ("Aquifer length", "length", 10.0, 1e7),
    "aq_kvkh":  ("Vertical anisotropy kv/kh", "ratio", 1e-5, 1.0),
    "aq_Wvol":  ("Aquifer water volume", "res_vol", 1e-3, 1e7),
    "aq_C":     ("Aquifer constant", "aq_const", 1e-3, 1e7),
    "nb_T":     ("Transmissibility to neighbouring reservoir", "transmiss", 1e-4, 1e5),
}


def aquifer_params(model, geometry, infinite=False):
    """Keys of the parameters used by a given aquifer model."""
    if model == "none":
        return []
    if model == "pot":
        return ["aq_Wvol"]
    if model == "schilthuis":
        return ["aq_C"]
    if geometry == "radial":
        keys = ["aq_ro", "aq_reD", "aq_h", "aq_theta", "aq_k"]
        if infinite:
            keys.remove("aq_reD")
        return keys
    if geometry == "linear":
        return ["aq_width", "aq_L", "aq_h", "aq_k"]
    return ["aq_ro", "aq_h", "aq_theta", "aq_k", "aq_kvkh"]


def make_aquifer(c):
    inf = bool(c.get("aq_inf")) and c["aq_geom"] == "radial" and c["aq_model"] != "fetkovich"
    return Aquifer(
        model=c["aq_model"], geometry=c["aq_geom"], ct=c["cw"] + c["cf"], k=c["aq_k"],
        phi=c["aq_phi"], mu=c["aq_muw"], h=c["aq_h"], ro=c["aq_ro"],
        reD=np.inf if inf else c["aq_reD"], theta=c["aq_theta"], width=c["aq_width"],
        L=c["aq_L"], kvkh=c["aq_kvkh"], Wvol=c["aq_Wvol"] * 1e6, C=c["aq_C"])


def make_tank(c, pvt, hist, with_aquifer=True, neighbour=None):
    """neighbour: (t_days, p_psia) of a connected reservoir; used when c['nb_on'] is set, with
    transmissibility c['nb_T'] in Mscf/day/psi. Left out together with the aquifer."""
    aq = make_aquifer(c) if with_aquifer else Aquifer()
    nb = None
    if with_aquifer and neighbour is not None and c.get("nb_on") and c.get("nb_T"):
        nb = (neighbour[0], neighbour[1], c["nb_T"] * 1e3)
    return Tank(pvt, hist, G=c["G"] * 1e9, pi=c["pi"], swi=c["swi"], cf=c["cf"], cw=c["cw"],
                bw=c["bw"], aquifer=aq, neighbour=nb)
