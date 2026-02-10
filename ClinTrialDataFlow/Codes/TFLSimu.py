# adam_to_tfl.py
from __future__ import annotations
import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional scipy for exact binomial CI
try:
    from scipy.stats import beta
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------
# IO
# -----------------------
def read_csv(folder: str, name: str) -> pd.DataFrame:
    p = os.path.join(folder, f"{name}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def write_csv(df: pd.DataFrame, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, name), index=False)

# -----------------------
# Stats helpers
# -----------------------
def fmt_mean_sd(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    return f"{x.mean():.1f} ({x.std(ddof=1):.1f})"

def fmt_median_iqr(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    return f"{x.median():.1f} [{q1:.1f}, {q3:.1f}]"

def binom_ci_cp(k: int, n: int, alpha: float = 0.05):
    # Clopper-Pearson exact CI
    if n == 0:
        return (np.nan, np.nan)
    if not HAS_SCIPY:
        return binom_ci_wilson(k, n, alpha)
    lo = 0.0 if k == 0 else beta.ppf(alpha/2, k, n-k+1)
    hi = 1.0 if k == n else beta.ppf(1-alpha/2, k+1, n-k)
    return (lo, hi)

def binom_ci_wilson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # approx 97.5% quantile
    phat = k / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))) / denom
    return (max(0.0, center-half), min(1.0, center+half))

# -----------------------
# Kaplan-Meier (no lifelines)
# -----------------------
def km_curve(time: np.ndarray, event: np.ndarray):
    """
    time: days (float) >=0
    event: 1=event, 0=censor
    Returns a DataFrame with columns: t, n_risk, n_event, surv
    """
    df = pd.DataFrame({"time": time, "event": event}).dropna()
    df = df[df["time"] >= 0].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["t","n_risk","n_event","surv"])
    df = df.sort_values("time")

    times = np.sort(df["time"].unique())
    surv = 1.0
    rows = []
    n = len(df)
    for t in times:
        at_t = df[df["time"] == t]
        d = int((at_t["event"] == 1).sum())
        c = int((at_t["event"] == 0).sum())
        n_risk = int((df["time"] >= t).sum())  # those not yet failed/censored before t
        # KM updates only at event times
        if n_risk > 0 and d > 0:
            surv *= (1 - d / n_risk)
        rows.append({"t": float(t), "n_risk": n_risk, "n_event": d, "n_cens": c, "surv": surv})
    return pd.DataFrame(rows)

def km_median(km: pd.DataFrame):
    if len(km) == 0:
        return np.nan
    # median is first time survival <= 0.5
    hit = km[km["surv"] <= 0.5]
    if len(hit) == 0:
        return np.nan
    return float(hit["t"].iloc[0])

def km_surv_at(km: pd.DataFrame, t: float):
    if len(km) == 0:
        return np.nan
    km2 = km[km["t"] <= t]
    if len(km2) == 0:
        return 1.0
    return float(km2["surv"].iloc[-1])

def plot_km(km_by_arm: dict, title: str, out_png: str):
    plt.figure()
    for arm, km in km_by_arm.items():
        if len(km) == 0:
            continue
        # step plot
        t = np.r_[0.0, km["t"].values]
        s = np.r_[1.0, km["surv"].values]
        plt.step(t, s, where="post", label=str(arm))
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -----------------------
# TFL builders
# -----------------------
def tfl_baseline(adsl: pd.DataFrame) -> pd.DataFrame:
    # basic baseline summary by TRT01A
    df = adsl.copy()
    df["TRT01A"] = df["TRT01A"].fillna("")

    out_rows = []

    for arm, g in df.groupby("TRT01A", dropna=False):
        n = len(g)
        out_rows.append({"ARM": arm, "PARAM": "N", "VALUE": str(n)})
        out_rows.append({"ARM": arm, "PARAM": "Age, mean (SD)", "VALUE": fmt_mean_sd(g.get("AGE", pd.Series(dtype=float)))})
        out_rows.append({"ARM": arm, "PARAM": "Age, median [Q1,Q3]", "VALUE": fmt_median_iqr(g.get("AGE", pd.Series(dtype=float)))})

        # SEX
        if "SEX" in g.columns:
            for level in ["M", "F"]:
                cnt = int((g["SEX"] == level).sum())
                out_rows.append({"ARM": arm, "PARAM": f"Sex={level}", "VALUE": f"{cnt} ({cnt/n*100:.1f}%)" if n else ""})

        # RACE
        if "RACE" in g.columns:
            for level, cnt in g["RACE"].fillna("UNK").value_counts().items():
                out_rows.append({"ARM": arm, "PARAM": f"Race={level}", "VALUE": f"{int(cnt)} ({cnt/n*100:.1f}%)" if n else ""})

    return pd.DataFrame(out_rows)

def tfl_orr(adslrs: pd.DataFrame) -> pd.DataFrame:
    """
    ORR: responders = CR/PR
    Primary uses BORC; also reports BORUNC as sensitivity.
    """
    df = adslrs.copy()
    df["TRT01A"] = df["TRT01A"].fillna("")
    for col in ["BORC","BORUNC","BOR"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.upper()

    rows = []
    for arm, g in df.groupby("TRT01A"):
        n = len(g)
        for endpoint in ["BORC", "BORUNC"]:
            if endpoint not in g.columns:
                continue
            resp = g[endpoint].isin(["CR","PR"])
            k = int(resp.sum())
            p = (k/n) if n else np.nan
            lo, hi = binom_ci_cp(k, n)
            rows.append({
                "ARM": arm,
                "ENDPOINT": endpoint,
                "N": n,
                "Responders (CR+PR)": k,
                "ORR %": "" if n==0 else f"{100*p:.1f}",
                "95% CI %": "" if n==0 else f"{100*lo:.1f}, {100*hi:.1f}",
            })
    return pd.DataFrame(rows)

def tfl_tte(adtte: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Summarize PFS/OS by arm:
      - Events
      - Median KM (days)
      - S(t) at 6 and 12 months
    """
    df = adtte.copy()
    df["TRT01A"] = df["TRT01A"].fillna("")
    df["PARAMCD"] = df["PARAMCD"].fillna("").astype(str).str.upper()
    df["AVAL"] = pd.to_numeric(df["AVAL"], errors="coerce")
    df["CNSR"] = pd.to_numeric(df["CNSR"], errors="coerce")
    df["EVENT"] = np.where(df["CNSR"] == 0, 1, 0)

    summary_rows = []
    km_plots = {}  # (paramcd) -> dict(arm -> km_df)

    for paramcd in ["PFS", "OS"]:
        sub = df[df["PARAMCD"] == paramcd].copy()
        km_by_arm = {}
        for arm, g in sub.groupby("TRT01A"):
            time = g["AVAL"].to_numpy(dtype=float)
            event = g["EVENT"].to_numpy(dtype=int)
            km = km_curve(time, event)
            km_by_arm[arm] = km

            n = len(g)
            nevt = int(event.sum())
            med = km_median(km)
            s6 = km_surv_at(km, 183.0)
            s12 = km_surv_at(km, 365.0)

            summary_rows.append({
                "PARAMCD": paramcd,
                "ARM": arm,
                "N": n,
                "Events": nevt,
                "Median (days)": "" if np.isnan(med) else f"{med:.1f}",
                "KM at 6 mo (%)": "" if np.isnan(s6) else f"{100*s6:.1f}",
                "KM at 12 mo (%)": "" if np.isnan(s12) else f"{100*s12:.1f}",
            })

        km_plots[paramcd] = km_by_arm

    return pd.DataFrame(summary_rows), km_plots


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inadam", required=True, help="ADaM folder containing adsl/adslrs/adtte csv")
    ap.add_argument("--out", required=True, help="Output folder for TFL")
    args = ap.parse_args()

    adsl = read_csv(args.inadam, "adsl")
    adslrs = read_csv(args.inadam, "adslrs")  # expects BORC/BORUNC
    adtte = read_csv(args.inadam, "adtte")

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    # Table 1: Baseline
    tab_base = tfl_baseline(adsl)
    write_csv(tab_base, outdir, "tfl_baseline.csv")

    # Table 2: ORR
    tab_orr = tfl_orr(adslrs)
    write_csv(tab_orr, outdir, "tfl_orr.csv")

    # Table 3: TTE summary + plots
    tab_tte, km_plots = tfl_tte(adtte)
    write_csv(tab_tte, outdir, "tfl_tte_summary.csv")

    # KM plots
    if "PFS" in km_plots:
        plot_km(km_plots["PFS"], "Kaplan–Meier: PFS", os.path.join(outdir, "km_pfs.png"))
    if "OS" in km_plots:
        plot_km(km_plots["OS"], "Kaplan–Meier: OS", os.path.join(outdir, "km_os.png"))

    print(f"Done -> {outdir}")
    print("Wrote: tfl_baseline.csv, tfl_orr.csv, tfl_tte_summary.csv, km_pfs.png, km_os.png")
    if not HAS_SCIPY:
        print("NOTE: scipy not available -> ORR CI uses Wilson approximation (not Clopper-Pearson).")

if __name__ == "__main__":
    main()
