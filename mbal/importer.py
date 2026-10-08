"""Read a production / pressure workbook (Excel or CSV) by its column headings."""
import io
import re
import numpy as np
import pandas as pd

DATE_FMT = "%d/%m/%Y"     # how dates are shown and typed throughout the app


def parse_dates(s):
    """Dates typed, pasted or imported -> datetime64. Text is read day first (dd/mm/yyyy, also
    dd-mm-yyyy, dd.mm.yyyy, 1 Nov 2019); ISO text (yyyy-mm-dd) and real dates from Excel are
    taken as they are. Anything unreadable becomes NaT."""
    s = pd.Series(s)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.tz_localize(None) if getattr(s.dt, "tz", None) else s
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for i, v in s.items():
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
            continue
        if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
            out[i] = pd.Timestamp(v).tz_localize(None) if pd.Timestamp(v).tzinfo else pd.Timestamp(v)
            continue
        txt = str(v).strip()
        iso = re.match(r"^\d{4}-\d{1,2}-\d{1,2}", txt)
        try:
            out[i] = pd.to_datetime(txt[:10] if iso else txt, dayfirst=not iso)
        except (ValueError, TypeError, OverflowError):
            pass
    return out

# column -> words looked for in the heading (first match wins, checked in this order)
KEYS = [("reservoir", ("reservoir", "tank", "sand")), ("well", ("well", "string", "completion")),
        ("date", ("date", "month", "time")), ("p", ("press", "psi", "bar")),
        ("gp", ("gas",)), ("np", ("cond", "oil", "liquid")), ("wp", ("water", "wat"))]


def _columns(df):
    """Map recognised headings to standard names; unrecognised columns are dropped."""
    out, used = {}, set()
    for name, words in KEYS:
        for col in df.columns:
            h = str(col).strip().lower()
            if col not in used and any(w in h for w in words):
                out[name] = col
                used.add(col)
                break
    d = df[list(out.values())].copy()
    d.columns = list(out)
    return d


def read_sheets(raw, filename):
    """bytes -> list of DataFrames with standard column names."""
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
    else:
        sheets = {"csv": pd.read_csv(io.BytesIO(raw), sep=None, engine="python")}
    return [_columns(s.dropna(how="all")) for s in sheets.values() if len(s.columns)]


def parse_workbook(raw, filename):
    """Returns (production, pressures) long tables.

    production: reservoir, well, date, gp, np, wp     (cumulative, as in the file)
    pressures:  reservoir, well, date, p
    A sheet with a gas column is production; a sheet with a pressure column is pressure
    (one sheet may be both). Missing reservoir / well columns are filled with a single name.
    """
    prod, pres = [], []
    for d in read_sheets(raw, filename):
        if "date" not in d:
            continue
        d["date"] = parse_dates(d["date"])
        d = d.dropna(subset=["date"])
        for k, default in (("reservoir", "Reservoir"), ("well", "Reservoir")):
            d[k] = d[k].astype(str).str.strip() if k in d else default
        for k in ("gp", "np", "wp", "p"):
            if k in d:
                d[k] = pd.to_numeric(d[k], errors="coerce")
        if "gp" in d:
            x = d.dropna(subset=["gp"]).copy()
            for k in ("np", "wp"):
                x[k] = x[k].fillna(0.0) if k in x else 0.0
            prod.append(x[["reservoir", "well", "date", "gp", "np", "wp"]])
        if "p" in d:
            pres.append(d.dropna(subset=["p"])[["reservoir", "well", "date", "p"]])
    if not prod:
        raise ValueError("No production sheet found: it needs a Date column and a Gas column.")
    if not pres:
        raise ValueError("No pressure sheet found: it needs a Date column and a Pressure column.")
    return pd.concat(prod, ignore_index=True), pd.concat(pres, ignore_index=True)


def reservoir_tables(prod, pres, reservoir):
    """Tables for one reservoir, ready for the By-well input.

    Returns dict(wells, pres, start_date, pre_start, spread, notes):
      wells      well, date, gp, np, wp
      pres       date, p (mean of the wells surveyed on that date), use, w
      start_date one reporting period before the first production record
      pre_start  surveys taken before production started (candidates for initial pressure)
      spread     per date: number of wells surveyed and max - min pressure
    """
    w = prod[prod["reservoir"] == reservoir].sort_values(["well", "date"])
    s = pres[pres["reservoir"] == reservoir]
    if w.empty:
        raise ValueError(f"No production rows for reservoir {reservoir}.")
    notes = []
    first = w["date"].min()
    gaps = w.groupby("well")["date"].diff().dropna()
    step = gaps.median() if len(gaps) else pd.Timedelta(days=30)
    # Monthly cumulatives stamped on the first of the month: back off one calendar month.
    if w["date"].dt.day.eq(1).all() and pd.Timedelta(days=27) <= step <= pd.Timedelta(days=32):
        start = first - pd.DateOffset(months=1)
    else:
        start = first - step
    if float(w.loc[w["date"] == first, "gp"].sum()) <= 0:
        start = first
    falling = [k for k, g in w.groupby("well") if (g["gp"].diff() < -1e-9).any()]
    if falling:
        notes.append("Gas does not rise continuously for " + ", ".join(falling) + ": the file "
                     "must hold cumulative volumes, not monthly volumes or rates.")
    g = s.groupby("date")["p"]
    avg = g.mean().reset_index().sort_values("date")
    spread = pd.DataFrame({"date": avg["date"].to_numpy(), "n": g.count().to_numpy(),
                           "range": (g.max() - g.min()).to_numpy()})
    pre = avg[avg["date"] <= start]
    late = avg[avg["date"] > w["date"].max()]
    if len(late):
        notes.append(f"{len(late)} survey(s) are dated after the last production record "
                     f"({w['date'].max():%d/%m/%Y}); cumulative production is held constant "
                     "to those dates.")
    surveys = avg[avg["date"] > start].copy()
    surveys["use"], surveys["w"] = True, 1.0
    return dict(wells=w[["well", "date", "gp", "np", "wp"]].reset_index(drop=True),
                pres=surveys.reset_index(drop=True), start_date=pd.Timestamp(start).date(),
                pre_start=pre.reset_index(drop=True), spread=spread, notes=notes)


TEST_KEYS = [("well", ("well", "string", "completion")), ("date", ("date", "time")),
             ("pwf", ("bhp", "pwf", "bottom", "downhole", "gauge")),
             ("pth", ("thp", "whp", "tubing", "wellhead", "well head", "pth", "head")),
             ("pr", ("reservoir", "static", "shut", "average")),
             ("wgr", ("wgr", "water")),
             ("qg", ("gas", "rate", "qg"))]


def read_tests(raw, filename):
    """Well tests from Excel or CSV by column heading -> well, date, qg, pwf, pth, pr."""
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        sheets = list(pd.read_excel(io.BytesIO(raw), sheet_name=None).values())
    else:
        sheets = [pd.read_csv(io.BytesIO(raw), sep=None, engine="python")]
    out = []
    for df in sheets:
        cols, used = {}, set()
        for name, words in TEST_KEYS:
            for col in df.columns:
                if col not in used and any(w in str(col).strip().lower() for w in words):
                    cols[name] = col
                    used.add(col)
                    break
        if "qg" not in cols or not ({"pwf", "pth"} & set(cols)):
            continue
        d = pd.DataFrame({k: df[v] for k, v in cols.items()})
        d["well"] = d["well"].astype(str).str.strip() if "well" in d else "Well"
        d["date"] = parse_dates(d["date"]) if "date" in d else pd.NaT
        for k in ("qg", "pwf", "pth", "pr", "wgr"):
            d[k] = pd.to_numeric(d[k], errors="coerce") if k in d else np.nan
        out.append(d.dropna(subset=["qg"])[["well", "date", "qg", "pwf", "pth", "pr", "wgr"]])
    if not out:
        raise ValueError("No well test sheet found: it needs a gas rate column and a flowing "
                         "bottomhole or tubing-head pressure column.")
    return pd.concat(out, ignore_index=True)
