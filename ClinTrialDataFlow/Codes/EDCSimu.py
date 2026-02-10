from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from typing import Callable, Any

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except Exception:
    HAS_PYREADSTAT = False

import json
from dataclasses import asdict

# =========================
# Config
# =========================
@dataclass
class Cfg:
    seed: int = 20260210
    n: int = 150
    studyid: str = "TEST-001-001"
    site_n: int = 8
    arms: Tuple[str, ...] = ("PBO", "TRT")
    alloc: Tuple[int, ...] = (1, 1)

    # raw-like visits (tumor usually q6-8w)
    visits: Tuple[Tuple[str, int], ...] = (
        ("SCR", -14),
        ("BASE", 0),
        ("C1D1", 0),
        ("W6", 42),
        ("W12", 84),
        ("W18", 126),
        ("W24", 168),
    )

    dropout_rate: float = 0.12
    form_missing_rate: float = 0.03
    item_missing_rate: float = 0.02


def rng(cfg: Cfg) -> np.random.Generator:
    return np.random.default_rng(cfg.seed)


def parse_visits(visits_str: str):
    """
    visits_str example:
      "SCR:-14,BASE:0,C1D1:0,W6:42,W12:84,W18:126,W24:168"
    """
    out = []
    for part in visits_str.split(","):
        part = part.strip()
        if not part:
            continue
        name, dy = part.split(":")
        out.append((name.strip(), int(dy.strip())))
    return tuple(out)

def parse_cfg() -> Cfg:
    ap = argparse.ArgumentParser()

    # basic
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None, help="optional JSON config file")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--studyid", type=str, default=None)
    ap.add_argument("--site-n", type=int, default=None)

    # design-ish
    ap.add_argument("--arms", type=str, default=None, help='comma separated, e.g. "PBO,TRT"')
    ap.add_argument("--alloc", type=str, default=None, help='comma separated ints, e.g. "1,1"')

    # visits
    ap.add_argument("--visits", type=str, default=None,
                    help='e.g. "SCR:-14,BASE:0,C1D1:0,W6:42,W12:84"')

    # missingness
    ap.add_argument("--dropout-rate", type=float, default=None)
    ap.add_argument("--form-missing-rate", type=float, default=None)
    ap.add_argument("--item-missing-rate", type=float, default=None)

    args = ap.parse_args()

    # start from defaults
    cfg = Cfg()

    # 1) load JSON overrides if provided
    if args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            j = json.load(f)
        # only apply keys that exist in Cfg
        for k, v in j.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # 2) apply CLI overrides (highest priority)
    if args.seed is not None: cfg.seed = args.seed
    if args.n is not None: cfg.n = args.n
    if args.studyid is not None: cfg.studyid = args.studyid
    if args.site_n is not None: cfg.site_n = args.site_n

    if args.arms is not None:
        cfg.arms = tuple([x.strip() for x in args.arms.split(",") if x.strip()])
    if args.alloc is not None:
        cfg.alloc = tuple([int(x.strip()) for x in args.alloc.split(",") if x.strip()])

    if args.visits is not None:
        cfg.visits = parse_visits(args.visits)

    if args.dropout_rate is not None: cfg.dropout_rate = args.dropout_rate
    if args.form_missing_rate is not None: cfg.form_missing_rate = args.form_missing_rate
    if args.item_missing_rate is not None: cfg.item_missing_rate = args.item_missing_rate

    # basic sanity
    if len(cfg.arms) != len(cfg.alloc):
        raise ValueError(f"arms({len(cfg.arms)}) and alloc({len(cfg.alloc)}) must have same length.")
    if cfg.n <= 0:
        raise ValueError("n must be positive.")

    return cfg, args.out

def _wchoice(g: np.random.Generator, items: List[str], w: List[float], n: int) -> np.ndarray:
    w = np.array(w, dtype=float)
    w = w / w.sum()
    return g.choice(items, size=n, replace=True, p=w)


def fmt_date(x) -> pd.Series:
    # accept Series / array-like of datetime
    return pd.to_datetime(x, errors="coerce").dt.strftime("%Y-%m-%d")


def inject_missing(df: pd.DataFrame, cols: List[str], g: np.random.Generator, rate: float) -> pd.DataFrame:
    if rate <= 0 or len(df) == 0:
        return df
    out = df.copy()
    m = g.random((len(out), len(cols))) < rate
    for j, c in enumerate(cols):
        out.loc[m[:, j], c] = np.nan
    return out


# =========================
# Writers
# =========================
def write_table(df: pd.DataFrame, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    if HAS_PYREADSTAT and hasattr(pyreadstat, "write_xport"):
        xpt_path = os.path.join(out_dir, f"{name}.xpt")
        pyreadstat.write_xport(df, xpt_path, table_name=name[:8].upper())


def write_formats_placeholder(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cat = os.path.join(out_dir, "formats.sas7bcat")
    if not os.path.exists(cat):
        with open(cat, "wb") as f:
            f.write(b"")


# =========================
# Core roster + visit frame
# =========================
def sim_subject(cfg: Cfg, g: np.random.Generator) -> pd.DataFrame:
    sites = [f"CN{i:03d}" for i in range(1, cfg.site_n + 1)]
    site = g.choice(sites, size=cfg.n, replace=True)
    running = np.arange(1, cfg.n + 1)
    subjid = np.array([f"{site[i]}{running[i]:03d}" for i in range(cfg.n)])

    rfstdt = pd.to_datetime("2025-01-01") + pd.to_timedelta(g.integers(0, 160, size=cfg.n), unit="D")
    arm = _wchoice(g, list(cfg.arms), list(cfg.alloc), cfg.n)

    sex = g.choice(["M", "F"], size=cfg.n, p=[0.55, 0.45])
    age = np.clip(g.normal(60, 9, size=cfg.n).round(), 18, 85).astype(int)
    ecog = g.choice([0, 1, 2], size=cfg.n, p=[0.35, 0.55, 0.10])

    dropout = g.random(cfg.n) < cfg.dropout_rate
    drop_day = np.where(dropout, g.integers(30, 220, size=cfg.n), np.nan)

    subject = pd.DataFrame({
        "STUDYID": cfg.studyid,
        "SUBJID": subjid,
        "SITEID": site,
        "USUBJID": [f"{cfg.studyid}-{s}" for s in subjid],
        "RFSTDTC": rfstdt.strftime("%Y-%m-%d"),
        "ARM": arm,
        "SEX": sex,
        "AGE": age,
        "ECOG0": ecog,          # baseline ECOG
        "_DROPOUTFL": np.where(dropout, "Y", "N"),
        "_DROPDAY": drop_day,
    })
    return subject


def sim_sv(cfg: Cfg, subject: pd.DataFrame, g: np.random.Generator) -> pd.DataFrame:
    rfstdt = pd.to_datetime(subject["RFSTDTC"])
    rows = []
    for vname, vdy in cfg.visits:
        jitter = g.integers(-3, 4, size=len(subject))  # +/-3 days
        vdt = rfstdt + pd.to_timedelta(vdy + jitter, unit="D")
        rows.append(pd.DataFrame({
            "USUBJID": subject["USUBJID"].values,
            "VISIT": vname,
            "VISITDY": vdy,
            "VISITDTC": vdt.dt.strftime("%Y-%m-%d"),
        }))
    sv = pd.concat(rows, ignore_index=True)

    # apply dropout cut
    dropday = subject.set_index("USUBJID")["_DROPDAY"]
    rfmap = subject.set_index("USUBJID")["RFSTDTC"]
    sv["RFSTDTC"] = sv["USUBJID"].map(rfmap)
    sv["ADY"] = (pd.to_datetime(sv["VISITDTC"]) - pd.to_datetime(sv["RFSTDTC"])).dt.days
    sv["DROPDAY"] = sv["USUBJID"].map(dropday)
    sv = sv[(sv["DROPDAY"].isna()) | (sv["ADY"] <= sv["DROPDAY"])].drop(columns=["RFSTDTC", "ADY", "DROPDAY"])

    # form missing
    if cfg.form_missing_rate > 0 and len(sv) > 0:
        miss = g.random(len(sv)) < cfg.form_missing_rate
        sv = sv.loc[~miss].reset_index(drop=True)

    sv["STUDYID"] = cfg.studyid
    return sv


# =========================
# Key oncology: TU + RS
# =========================
def sim_tu(cfg: Cfg, subject: pd.DataFrame, sv: pd.DataFrame, g: np.random.Generator) -> pd.DataFrame:
    """
    Tumor identification table:
    - 1-5 lesions per subject
    - baseline + follow-ups (lesion persists)
    """
    lesions_pool = ["LIVER", "LUNG", "LYMPH NODE", "BONE", "OTHER"]
    rows = []
    for usubjid in subject["USUBJID"].values:
        n_les = int(g.integers(1, 6))
        sites = g.choice(lesions_pool, size=n_les, replace=True, p=[0.20, 0.25, 0.25, 0.15, 0.15])
        for i in range(n_les):
            rows.append({
                "STUDYID": cfg.studyid,
                "USUBJID": usubjid,
                "TUSEQ": i + 1,
                "TULOC": sites[i],
                "TUTYPE": g.choice(["TARGET", "NON-TARGET"], p=[0.65, 0.35]),
                "TULNKID": f"LESION{i+1:02d}",   # link id for RS
            })
    tu = pd.DataFrame(rows)
    tu = inject_missing(tu, ["TULOC"], g, cfg.item_missing_rate)
    return tu


def sim_rs(cfg: Cfg, subject: pd.DataFrame, sv: pd.DataFrame, tu: pd.DataFrame, g: np.random.Generator) -> pd.DataFrame:
    """
    Tumor response assessment:
    - per subject per response visit (BASE/W6/W12/...)
    - produce OVRLRESP with a simple progression tendency over time
    """
    resp_vis = sv[sv["VISIT"].isin(["BASE", "W6", "W12", "W18", "W24"])].copy()
    arm_map = subject.set_index("USUBJID")["ARM"]

    # subject-level latent "benefit": TRT slightly better
    benefit = {u: (0.15 if arm_map[u] == "TRT" else 0.0) + g.normal(0, 0.08) for u in subject["USUBJID"].values}

    rows = []
    for u, grp in resp_vis.groupby("USUBJID"):
        grp = grp.sort_values("VISITDY")
        prog_hazard = 0.08  # baseline hazard
        progressed = False
        for j, r in grp.reset_index(drop=True).iterrows():
            vdy = int(r["VISITDY"])
            # time increases -> more likely PD
            t = max(vdy, 0) / 84.0  # scaled
            p_pd = min(0.75, prog_hazard + 0.10 * t - benefit[u])
            p_crpr = max(0.05, 0.18 + benefit[u] - 0.04 * t)
            p_sd = 0.35
            # normalize
            rem = max(1e-6, 1 - (p_pd + p_crpr + p_sd))
            p_unk = rem

            if progressed:
                ov = "PD"
            else:
                ov = g.choice(["CR", "PR", "SD", "PD", "NE"], p=_norm([p_crpr / 2, p_crpr / 2, p_sd, p_pd, p_unk]))
                if ov == "PD":
                    progressed = True

            rows.append({
                "STUDYID": cfg.studyid,
                "USUBJID": u,
                "RSSEQ": len([x for x in rows if x["USUBJID"] == u]) + 1,
                "RSVISIT": r["VISIT"],
                "RSVISITDY": r["VISITDY"],
                "RSDTC": r["VISITDTC"],
                "OVRLRESP": ov,  # overall response
                "NEWLIND": "Y" if (ov == "PD" and g.random() < 0.25) else "N",
            })

    rs = pd.DataFrame(rows)
    rs = inject_missing(rs, ["OVRLRESP"], g, cfg.item_missing_rate)
    return rs


def _norm(x):
    x = np.array(x, dtype=float)
    x = np.clip(x, 1e-9, None)
    return (x / x.sum()).tolist()


# =========================
# Other “rawdata-like” tables (lightweight)
# =========================
def sim_vs(cfg: Cfg, subject: pd.DataFrame, sv: pd.DataFrame, g: np.random.Generator) -> pd.DataFrame:
    df = sv.copy()
    sbp = np.clip(g.normal(120, 12, size=len(df)), 80, 220).round(0)
    wt = np.clip(g.normal(67, 12, size=len(df)), 35, 140).round(1)

    vs = pd.concat([
        df.assign(VSTEST="SYSBP", VSORRES=sbp, VSORRESU="mmHg"),
        df.assign(VSTEST="WEIGHT", VSORRES=wt, VSORRESU="kg"),
    ], ignore_index=True)
    vs["STUDYID"] = cfg.studyid
    vs["VSSEQ"] = vs.groupby("USUBJID").cumcount() + 1
    vs = vs[["STUDYID","USUBJID","VSSEQ","VISIT","VISITDY","VISITDTC","VSTEST","VSORRES","VSORRESU"]]
    return inject_missing(vs, ["VSORRES"], g, cfg.item_missing_rate)


def sim_lb(cfg: Cfg, subject: pd.DataFrame, sv: pd.DataFrame, g: np.random.Generator) -> pd.DataFrame:
    df = sv.copy()
    arm = subject.set_index("USUBJID")["ARM"]
    df["ARM"] = df["USUBJID"].map(arm)

    post = df["VISITDY"] > 0
    alt = np.clip(g.normal(25, 10, size=len(df)) + np.where((df["ARM"]=="TRT") & post, 3.0, 0.0), 5, None).round(1)
    ast = np.clip(g.normal(23, 9, size=len(df)) + np.where((df["ARM"]=="TRT") & post, 2.0, 0.0), 5, None).round(1)

    lb = pd.concat([
        df.assign(LBTEST="ALT", LBORRES=alt, LBORRESU="U/L"),
        df.assign(LBTEST="AST", LBORRES=ast, LBORRESU="U/L"),
    ], ignore_index=True)
    lb["STUDYID"] = cfg.studyid
    lb["LBSEQ"] = lb.groupby("USUBJID").cumcount() + 1
    lb = lb[["STUDYID","USUBJID","LBSEQ","VISIT","VISITDY","VISITDTC","LBTEST","LBORRES","LBORRESU"]]
    return inject_missing(lb, ["LBORRES"], g, cfg.item_missing_rate)


def sim_ae(cfg: Cfg, subject: pd.DataFrame, g: np.random.Generator) -> pd.DataFrame:
    base_prob = 0.45
    arm = subject["ARM"].values
    prob = np.where(arm == "TRT", np.minimum(0.90, base_prob * 1.25), base_prob)
    has = g.random(len(subject)) < prob

    terms = ["NAUSEA", "FATIGUE", "ALT INCREASED", "AST INCREASED", "HYPOGLYCAEMIA"]
    rows = []
    for i, u in enumerate(subject["USUBJID"].values):
        if not has[i]:
            continue
        rf = pd.to_datetime(subject.loc[i, "RFSTDTC"])
        n = int(g.integers(1, 4))
        for j in range(n):
            t = g.choice(terms)
            sd = int(g.integers(1, 160))
            ed = sd + int(g.integers(1, 20))
            rows.append({
                "STUDYID": cfg.studyid,
                "USUBJID": u,
                "AESEQ": j + 1,
                "AETERM": t,
                "AESTDTC": (rf + pd.to_timedelta(sd, unit="D")).strftime("%Y-%m-%d"),
                "AEENDTC": (rf + pd.to_timedelta(ed, unit="D")).strftime("%Y-%m-%d"),
                "AESEV": g.choice(["MILD","MODERATE","SEVERE"], p=[0.55,0.35,0.10]),
                "AEREL": g.choice(["RELATED","NOT RELATED",""], p=[0.45,0.45,0.10]),
            })
    ae = pd.DataFrame(rows)
    if len(ae) == 0:
        ae = pd.DataFrame(columns=["STUDYID","USUBJID","AESEQ","AETERM","AESTDTC","AEENDTC","AESEV","AEREL"])
    return inject_missing(ae, ["AEREL"], g, cfg.item_missing_rate)


def sim_trt_tables(cfg: Cfg, subject: pd.DataFrame, g: np.random.Generator) -> Dict[str, pd.DataFrame]:
    # minimal: trt (assignment), trnt (treatment exposure summary), trnew (dose modification), trnew optional
    trt = subject[["STUDYID","USUBJID","ARM","RFSTDTC"]].copy()
    trt = trt.rename(columns={"ARM":"TRT01A"})

    # trnt: pretend total dose and exposure days
    exp_days = np.where(subject["_DROPOUTFL"].values=="Y",
                        subject["_DROPDAY"].fillna(168).astype(float),
                        168.0)
    trnt = pd.DataFrame({
        "STUDYID": cfg.studyid,
        "USUBJID": subject["USUBJID"].values,
        "TRT01A": trt["TRT01A"].values,
        "EXPDAYS": exp_days.round(0),
        "TOTDOSEMG": (exp_days * 200).round(0),
    })

    # trnew: dose changes (small rate)
    m = g.random(len(subject)) < 0.08
    trnew = pd.DataFrame({
        "STUDYID": cfg.studyid,
        "USUBJID": subject.loc[m, "USUBJID"].values,
        "ACTION": "DOSE REDUCTION",
        "REASON": g.choice(["AE","LAB ABN","OTHER"], size=m.sum(), replace=True),
    })
    return {"trt": trt, "trnt": trnt, "trnew": trnew}


def sim_uncollect_unoccurre(cfg: Cfg, subject: pd.DataFrame, g: np.random.Generator) -> Dict[str, pd.DataFrame]:
    # missing form tracker tables (raw-like)
    # uncollect: visit-level missing; unoccurre: event not occurred
    m1 = g.random(len(subject)) < 0.10
    uncollect = pd.DataFrame({
        "STUDYID": cfg.studyid,
        "USUBJID": subject.loc[m1, "USUBJID"].values,
        "FORM": g.choice(["RS","LB","VS","AE"], size=m1.sum(), replace=True),
        "REASON": g.choice(["MISSED VISIT","TECH ISSUE","WITHDRAWAL"], size=m1.sum(), replace=True),
    })

    m2 = g.random(len(subject)) < 0.12
    unoccurre = pd.DataFrame({
        "STUDYID": cfg.studyid,
        "USUBJID": subject.loc[m2, "USUBJID"].values,
        "EVENT": g.choice(["SAE","DEATH","PD"], size=m2.sum(), replace=True),
        "FLAG": "N",
    })
    return {"uncollect": uncollect, "unoccurre": unoccurre}


def sim_pr_tables(cfg: Cfg, subject: pd.DataFrame, g: np.random.Generator) -> Dict[str, pd.DataFrame]:
    # simplified procedure / radiotherapy / surgery / follow-up tables
    # names you listed: prcnd, prcrt, prcsurg, prfurt, prfusurg, prrt, prtsurg
    # keep few vars but realistic
    rf = pd.to_datetime(subject.set_index("USUBJID")["RFSTDTC"])
    us = subject["USUBJID"].values

    def mk(name, prob, label):
        m = g.random(len(us)) < prob
        sel = us[m]
        if len(sel) == 0:
            return pd.DataFrame(columns=["STUDYID","USUBJID","PRTYPE","PRSTDTC","PRENDTC"])
        sd = (rf.loc[sel] + pd.to_timedelta(g.integers(-30, 120, size=len(sel)), unit="D"))
        ed = sd + pd.to_timedelta(g.integers(1, 30, size=len(sel)), unit="D")
        return pd.DataFrame({
            "STUDYID": cfg.studyid,
            "USUBJID": sel,
            "PRTYPE": label,
            "PRSTDTC": sd.dt.strftime("%Y-%m-%d"),
            "PRENDTC": ed.dt.strftime("%Y-%m-%d"),
        })

    return {
        "prcnd": mk("prcnd", 0.55, "CONCOMITANT PROCEDURE"),
        "prcrt": mk("prcrt", 0.20, "CHEMOTHERAPY"),
        "prcsurg": mk("prcsurg", 0.18, "SURGERY"),
        "prfurt": mk("prfurt", 0.35, "FOLLOW-UP"),
        "prfusurg": mk("prfusurg", 0.10, "FOLLOW-UP SURGERY"),
        "prrt": mk("prrt", 0.22, "RADIOTHERAPY"),
        "prtsurg": mk("prtsurg", 0.08, "RADIOTHERAPY+SURGERY"),
    }


def sim_repf_rppfu_rsecog(cfg: Cfg, subject: pd.DataFrame, sv: pd.DataFrame, g: np.random.Generator) -> Dict[str, pd.DataFrame]:
    # repf: response/eval performance (toy)
    # rppfu: progression/phone followup (toy)
    # rsecog: ECOG repeated assessments
    resp_vis = sv[sv["VISIT"].isin(["BASE","W6","W12","W18","W24"])].copy()

    repf = resp_vis.copy()
    repf["STUDYID"] = cfg.studyid
    repf["REPF_SCORE"] = np.clip(g.normal(80, 12, size=len(repf)), 30, 100).round(0)
    repf = repf[["STUDYID","USUBJID","VISIT","VISITDY","VISITDTC","REPF_SCORE"]]

    # phone follow-up per subject (0-2 records)
    rows = []
    for u in subject["USUBJID"].values:
        k = int(g.integers(0, 3))
        if k == 0:
            continue
        rf = pd.to_datetime(subject.loc[subject["USUBJID"]==u, "RFSTDTC"]).iloc[0]
        for j in range(k):
            dt = (rf + pd.to_timedelta(int(g.integers(60, 240)), unit="D")).strftime("%Y-%m-%d")
            rows.append({"STUDYID": cfg.studyid, "USUBJID": u, "PPFUSEQ": j+1, "PPFUDTC": dt, "CONTACT": "PHONE"})
    rppfu = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["STUDYID","USUBJID","PPFUSEQ","PPFUDTC","CONTACT"])

    # ECOG repeated
    base_ecog = subject.set_index("USUBJID")["ECOG0"]
    ec = resp_vis.copy()
    ec["STUDYID"] = cfg.studyid
    ec["ECOG"] = ec["USUBJID"].map(base_ecog).astype(float) + np.where(ec["VISITDY"]>0, g.choice([0,0,0,1], size=len(ec)), 0)
    ec["ECOG"] = np.clip(ec["ECOG"], 0, 4).astype(int)
    rsecog = ec[["STUDYID","USUBJID","VISIT","VISITDY","VISITDTC","ECOG"]]

    return {"repf": repf, "rppfu": rppfu, "rsecog": rsecog}


def sim_ss_sualco_sucigr(cfg: Cfg, subject: pd.DataFrame, g: np.random.Generator) -> Dict[str, pd.DataFrame]:
    # ss: subject status summary
    ss = subject[["STUDYID","USUBJID","RFSTDTC"]].copy()
    ss["STATUS"] = np.where(subject["_DROPOUTFL"].values=="Y", "DISCONTINUED", "ONGOING/COMPLETED")

    # alcohol/cigarette simple baseline flags
    sualco = pd.DataFrame({
        "STUDYID": cfg.studyid,
        "USUBJID": subject["USUBJID"].values,
        "ALCOHOL": g.choice(["Y","N",""], size=len(subject), p=[0.35,0.60,0.05]),
    })
    sucigr = pd.DataFrame({
        "STUDYID": cfg.studyid,
        "USUBJID": subject["USUBJID"].values,
        "SMOKE": g.choice(["Y","N","FORMER",""], size=len(subject), p=[0.25,0.55,0.15,0.05]),
    })
    return {"ss": ss, "sualco": sualco, "sucigr": sucigr}



# =========================
# Build pack
# =========================
# 每个 generator 接收 (cfg, ctx, rng) -> DataFrame 或 Dict[str, DataFrame]
Gen = Callable[[Cfg, Dict[str, pd.DataFrame], np.random.Generator], Any]

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # base context
    "subject": {"deps": [], "gen": lambda cfg, ctx, g: sim_subject(cfg, g)},
    "sv":      {"deps": ["subject"], "gen": lambda cfg, ctx, g: sim_sv(cfg, ctx["subject"], g)},

    # key oncology
    "tu": {"deps": ["subject","sv"], "gen": lambda cfg, ctx, g: sim_tu(cfg, ctx["subject"], ctx["sv"], g)},
    "rs": {"deps": ["subject","sv","tu"], "gen": lambda cfg, ctx, g: sim_rs(cfg, ctx["subject"], ctx["sv"], ctx["tu"], g)},

    # safety / labs / vitals
    "ae": {"deps": ["subject"], "gen": lambda cfg, ctx, g: sim_ae(cfg, ctx["subject"], g)},
    "lb": {"deps": ["subject","sv"], "gen": lambda cfg, ctx, g: sim_lb(cfg, ctx["subject"], ctx["sv"], g)},
    "vs": {"deps": ["subject","sv"], "gen": lambda cfg, ctx, g: sim_vs(cfg, ctx["subject"], ctx["sv"], g)},

    # derived splits (simple examples)
    "vs1":  {"deps": ["vs"], "gen": lambda cfg, ctx, g: ctx["vs"].copy()},
    "vswt": {"deps": ["vs"], "gen": lambda cfg, ctx, g: ctx["vs"][ctx["vs"]["VSTEST"]=="WEIGHT"].copy()},

    # packs that return multiple datasets
    "trt_pack": {"deps": ["subject"], "gen": lambda cfg, ctx, g: sim_trt_tables(cfg, ctx["subject"], g)},
    "un_pack":  {"deps": ["subject"], "gen": lambda cfg, ctx, g: sim_uncollect_unoccurre(cfg, ctx["subject"], g)},
    "pr_pack":  {"deps": ["subject"], "gen": lambda cfg, ctx, g: sim_pr_tables(cfg, ctx["subject"], g)},
    "rep_pack": {"deps": ["subject","sv"], "gen": lambda cfg, ctx, g: sim_repf_rppfu_rsecog(cfg, ctx["subject"], ctx["sv"], g)},
    "ss_pack":  {"deps": ["subject"], "gen": lambda cfg, ctx, g: sim_ss_sualco_sucigr(cfg, ctx["subject"], g)},

    # misc
    "formats": {"deps": [], "gen": lambda cfg, ctx, g: pd.DataFrame({"FMTNAME":["$ARM"], "START":["TRT"], "LABEL":["Treatment"]})},
    "invalid": {"deps": [], "gen": lambda cfg, ctx, g: pd.DataFrame(columns=["DATASET","USUBJID","ISSUE","VAR","VALUE"])},
    "pc": {"deps": [], "gen": lambda cfg, ctx, g: pd.DataFrame(columns=["STUDYID","USUBJID","PCSEQ","PCDTC","CONC","CONCU"])},
    "pcf": {"deps": ["pc"], "gen": lambda cfg, ctx, g: ctx["pc"].copy()},
    "pe": {"deps": [], "gen": lambda cfg, ctx, g: pd.DataFrame(columns=["STUDYID","USUBJID","PESEQ","PEDTC","SCORE","SCOREU"])},
}

def topo_order(reg: Dict[str, Dict[str, Any]]) -> List[str]:
    # simple topological sort
    seen, temp, out = set(), set(), []

    def dfs(k):
        if k in seen:
            return
        if k in temp:
            raise ValueError(f"Cyclic dependency at {k}")
        temp.add(k)
        for d in reg[k].get("deps", []):
            if d in reg:
                dfs(d)
        temp.remove(k)
        seen.add(k)
        out.append(k)

    for k in reg:
        dfs(k)
    return out

def build_pack(cfg: Cfg) -> Dict[str, pd.DataFrame]:
    g = rng(cfg)
    ctx: Dict[str, pd.DataFrame] = {}
    pack: Dict[str, pd.DataFrame] = {}

    for key in topo_order(DATASET_REGISTRY):
        spec = DATASET_REGISTRY[key]
        gen = spec["gen"]
        obj = gen(cfg, ctx, g)

        # if returns dict -> merge into pack+ctx
        if isinstance(obj, dict):
            for k2, df2 in obj.items():
                pack[k2] = df2
                ctx[k2] = df2
        else:
            pack[key] = obj
            ctx[key] = obj

    # drop internal vars from subject (if exist)
    if "subject" in pack:
        for col in ["_DROPOUTFL","_DROPDAY"]:
            if col in pack["subject"].columns:
                pack["subject"] = pack["subject"].drop(columns=[col])

    return pack

# =========================
# Data Dictionary
# =========================

DATA_DICTIONARY_BOOK = {

    "title": "Clinical Trial Raw Data Dictionary",
    "version": "1.0",
    "language": "English",
    "standard_reference": [
        "ICH E6(R2)",
        "CDISC SDTM (conceptual alignment)",
        "RECIST v1.1 (oncology response)"
    ],

    "study_context": """
    This data dictionary describes simulated raw clinical trial datasets
    generated by an EDC-like data simulator. The structure and variables
    are designed to resemble sponsor-defined raw data commonly used in
    oncology clinical trials prior to SDTM mapping.
    """,

    "datasets": {

        # =====================================================
        # SUBJECT
        # =====================================================
        "subject": {
            "label": "Subject-Level Roster",
            "description": """
            One record per subject. This dataset represents the core subject
            roster combining demographics, treatment assignment, and baseline
            disease status. It serves as the anchor dataset for all other domains.
            """,
            "keys": ["USUBJID"],
            "variables": {
                "STUDYID": {
                    "label": "Study Identifier",
                    "type": "Character",
                    "source": "Protocol",
                    "example": "KEYNOTE-001-001"
                },
                "USUBJID": {
                    "label": "Unique Subject Identifier",
                    "type": "Character",
                    "source": "Derived",
                    "description": "Globally unique identifier for each subject"
                },
                "SITEID": {
                    "label": "Study Site Identifier",
                    "type": "Character",
                    "source": "EDC"
                },
                "RFSTDTC": {
                    "label": "Date of First Study Treatment",
                    "type": "ISO 8601 Date",
                    "source": "Derived"
                },
                "ARM": {
                    "label": "Treatment Arm",
                    "type": "Character",
                    "controlled_terms": ["PBO", "TRT"],
                    "source": "Randomization"
                },
                "SEX": {
                    "label": "Biological Sex",
                    "type": "Character",
                    "controlled_terms": ["M", "F"]
                },
                "AGE": {
                    "label": "Age at Baseline (Years)",
                    "type": "Integer"
                },
                "ECOG0": {
                    "label": "Baseline ECOG Performance Status",
                    "type": "Integer",
                    "controlled_terms": [0, 1, 2, 3, 4],
                    "notes": "Baseline functional status in oncology trials"
                }
            }
        },

        # =====================================================
        # SV – VISITS
        # =====================================================
        "sv": {
            "label": "Subject Visit Schedule",
            "description": """
            Planned and actual visit-level information for each subject.
            Used as the temporal backbone for assessments such as labs,
            tumor response, and vital signs.
            """,
            "keys": ["USUBJID", "VISIT"],
            "variables": {
                "USUBJID": {"label": "Unique Subject Identifier", "type": "Character"},
                "VISIT": {
                    "label": "Visit Name",
                    "type": "Character",
                    "example": "W12"
                },
                "VISITDY": {
                    "label": "Planned Study Day Relative to First Dose",
                    "type": "Integer"
                },
                "VISITDTC": {
                    "label": "Visit Date",
                    "type": "ISO 8601 Date"
                }
            }
        },

        # =====================================================
        # TU – TUMOR IDENTIFICATION
        # =====================================================
        "tu": {
            "label": "Tumor Identification",
            "description": """
            Identifies individual tumor lesions for each subject.
            Lesions defined here are referenced by response assessments (RS).
            """,
            "keys": ["USUBJID", "TUSEQ"],
            "variables": {
                "USUBJID": {"label": "Unique Subject Identifier", "type": "Character"},
                "TUSEQ": {
                    "label": "Tumor Sequence Number",
                    "type": "Integer"
                },
                "TULOC": {
                    "label": "Tumor Location",
                    "type": "Character",
                    "example": "LUNG"
                },
                "TUTYPE": {
                    "label": "Tumor Type",
                    "type": "Character",
                    "controlled_terms": ["TARGET", "NON-TARGET"]
                },
                "TULNKID": {
                    "label": "Tumor Link Identifier",
                    "type": "Character",
                    "notes": "Used to link tumor lesions with response assessments"
                }
            }
        },

        # =====================================================
        # RS – RESPONSE ASSESSMENT
        # =====================================================
        "rs": {
            "label": "Tumor Response Assessment",
            "description": """
            Overall tumor response assessments performed at scheduled visits.
            Responses are based on RECIST-like logic and reflect disease progression
            or treatment benefit over time.
            """,
            "keys": ["USUBJID", "RSSEQ"],
            "variables": {
                "USUBJID": {"label": "Unique Subject Identifier", "type": "Character"},
                "RSSEQ": {
                    "label": "Response Assessment Sequence Number",
                    "type": "Integer"
                },
                "RSVISIT": {
                    "label": "Assessment Visit",
                    "type": "Character"
                },
                "RSVISITDY": {
                    "label": "Study Day of Response Assessment",
                    "type": "Integer"
                },
                "RSDTC": {
                    "label": "Response Assessment Date",
                    "type": "ISO 8601 Date"
                },
                "OVRLRESP": {
                    "label": "Overall Tumor Response",
                    "type": "Character",
                    "controlled_terms": ["CR", "PR", "SD", "PD", "NE"],
                    "notes": "Overall response per RECIST v1.1 concepts"
                },
                "NEWLIND": {
                    "label": "New Lesion Indicator",
                    "type": "Character",
                    "controlled_terms": ["Y", "N"]
                }
            }
        },

        # =====================================================
        # AE – ADVERSE EVENTS
        # =====================================================
        "ae": {
            "label": "Adverse Events",
            "description": """
            Records treatment-emergent adverse events occurring after
            first study treatment.
            """,
            "keys": ["USUBJID", "AESEQ"],
            "variables": {
                "AETERM": {
                    "label": "Adverse Event Term",
                    "type": "Character"
                },
                "AESTDTC": {
                    "label": "Adverse Event Start Date",
                    "type": "ISO 8601 Date"
                },
                "AEENDTC": {
                    "label": "Adverse Event End Date",
                    "type": "ISO 8601 Date"
                },
                "AESEV": {
                    "label": "Adverse Event Severity",
                    "type": "Character",
                    "controlled_terms": ["MILD", "MODERATE", "SEVERE"]
                },
                "AEREL": {
                    "label": "Causality to Study Drug",
                    "type": "Character",
                    "controlled_terms": ["RELATED", "NOT RELATED"]
                }
            }
        }
    }
}

def export_dictionary(dict_obj, out_dir: str):
    """
    Export variable-level data dictionary as CSV.
    Supports:
      (A) BOOK structure: {"title":..., "datasets": {...}}
      (B) Plain structure: {"ds": {"variables": {...}}, ...}
    """
    os.makedirs(out_dir, exist_ok=True)

    # detect structure
    if isinstance(dict_obj, dict) and "datasets" in dict_obj and isinstance(dict_obj["datasets"], dict):
        datasets = dict_obj["datasets"]
        book_title = dict_obj.get("title", "")
        book_version = dict_obj.get("version", "")
    else:
        datasets = dict_obj
        book_title = ""
        book_version = ""

    rows = []
    for ds_name, meta in datasets.items():
        if not isinstance(meta, dict):
            continue

        ds_label = meta.get("label", "")
        ds_desc = (meta.get("description", "") or "").strip()
        keys = meta.get("keys", [])

        vars_meta = meta.get("variables", {})
        if not isinstance(vars_meta, dict):
            continue

        for var, vmeta in vars_meta.items():
            if not isinstance(vmeta, dict):
                continue

            rows.append({
                "Book Title": book_title,
                "Book Version": book_version,
                "Dataset": ds_name,
                "Dataset Label": ds_label,
                "Dataset Keys": ",".join(keys) if isinstance(keys, (list, tuple)) else str(keys),
                "Dataset Description": ds_desc,
                "Variable": var,
                "Variable Label": vmeta.get("label", ""),
                "Type": vmeta.get("type", ""),
                "Controlled Terms": ",".join(map(str, vmeta.get("controlled_terms", []))),
                "Source": vmeta.get("source", ""),
                "Notes": vmeta.get("notes", ""),
                "Description": vmeta.get("description", ""),
                "Example": vmeta.get("example", ""),
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "data_dictionary.csv")
    df.to_csv(out_path, index=False)
    print(f"Data dictionary exported: {out_path}")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", required=True)
#     ap.add_argument("--n", type=int, default=150)
#     ap.add_argument("--seed", type=int, default=20260210)
#     args = ap.parse_args()
#
#     cfg = Cfg(n=args.n, seed=args.seed)
#     pack = build_pack(cfg)
#
#     for name, df in pack.items():
#         write_table(df, args.out, name)
#
#     write_formats_placeholder(args.out)
#     print(f"Done. Wrote {len(pack)} datasets (CSV + XPT if available) to: {args.out}")

def main():
    cfg, out_dir = parse_cfg()
    pack = build_pack(cfg)

    for name, df in pack.items():
        write_table(df, out_dir, name)

    write_formats_placeholder(out_dir)
    print("Config used:\n", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    print(f"Done. Wrote {len(pack)} datasets to: {out_dir}")

    export_dictionary(DATA_DICTIONARY_BOOK, out_dir)

if __name__ == "__main__":
    main()
