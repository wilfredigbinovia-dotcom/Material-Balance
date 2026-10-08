"""Field <-> SI display conversion. The engine always works in field units."""

# quantity: {system: (label, multiplier, offset)}   display = internal*mult + offset
Q = {
    "pressure":    {"field": ("psia", 1.0, 0.0),            "si": ("bara", 0.0689476, 0.0)},
    "temperature": {"field": ("°F", 1.0, 0.0),              "si": ("°C", 5.0 / 9.0, -160.0 / 9.0)},
    "gas_cum":     {"field": ("MMscf", 1.0, 0.0),           "si": ("10⁶ sm³", 0.0283168, 0.0)},
    "gas_ip":      {"field": ("Bscf", 1.0, 0.0),            "si": ("10⁹ sm³", 0.0283168, 0.0)},
    "liq_cum":     {"field": ("MSTB", 1.0, 0.0),            "si": ("10³ sm³", 0.158987, 0.0)},
    "cgr":         {"field": ("STB/MMscf", 1.0, 0.0),       "si": ("sm³/10⁶ sm³", 5.61458, 0.0)},
    "length":      {"field": ("ft", 1.0, 0.0),              "si": ("m", 0.3048, 0.0)},
    "compress":    {"field": ("10⁻⁶/psi", 1e6, 0.0),        "si": ("10⁻⁶/bar", 1e6 * 14.5038, 0.0)},
    "bg":          {"field": ("rb/Mscf", 1e3, 0.0),         "si": ("rm³/10³ sm³", 1e3 * 5.61458, 0.0)},
    "res_vol":     {"field": ("MMrb", 1.0, 0.0),            "si": ("10⁶ rm³", 0.158987, 0.0)},
    "aq_const":    {"field": ("rb/day/psi", 1.0, 0.0),      "si": ("rm³/day/bar", 0.158987 * 14.5038, 0.0)},
    "perm":        {"field": ("md", 1.0, 0.0),              "si": ("md", 1.0, 0.0)},
    "visc":        {"field": ("cp", 1.0, 0.0),              "si": ("mPa·s", 1.0, 0.0)},
    "frac":        {"field": ("fraction", 1.0, 0.0),        "si": ("fraction", 1.0, 0.0)},
    "angle":       {"field": ("degrees", 1.0, 0.0),         "si": ("degrees", 1.0, 0.0)},
    "ratio":       {"field": ("-", 1.0, 0.0),               "si": ("-", 1.0, 0.0)},
    "fvf_w":       {"field": ("rb/STB", 1.0, 0.0),          "si": ("rm³/sm³", 1.0, 0.0)},
    "api":         {"field": ("°API", 1.0, 0.0),            "si": ("°API", 1.0, 0.0)},
    "ppm":         {"field": ("ppm", 1.0, 0.0),             "si": ("ppm", 1.0, 0.0)},
    "gas_rate":    {"field": ("MMscf/d", 1.0, 0.0),         "si": ("10³ sm³/d", 28.3168, 0.0)},
    "liq_rate":    {"field": ("STB/d", 1.0, 0.0),           "si": ("sm³/d", 0.158987, 0.0)},
    "diam":        {"field": ("in", 1.0, 0.0),              "si": ("mm", 25.4, 0.0)},
    "transmiss":   {"field": ("Mscf/d/psi", 1.0, 0.0),      "si": ("10³ sm³/d/bar", 0.0283168 * 14.5038, 0.0)},
    "pz":          {"field": ("psia", 1.0, 0.0),            "si": ("bara", 0.0689476, 0.0)},
}


def label(q, system):
    return Q[q][system][0]


def to_disp(q, x, system):
    if q is None:
        return x
    _, m, o = Q[q][system]
    return x * m + o


def from_disp(q, x, system):
    if q is None:
        return x
    _, m, o = Q[q][system]
    return (x - o) / m
