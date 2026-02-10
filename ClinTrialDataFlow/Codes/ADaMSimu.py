# sdtm_to_adam_plus.py
from __future__ import annotations
import argparse
import os
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except Exception:
    HAS_PYREADSTAT = False


# ----------------------------
# Helpers
# ----------------------------
def read_csv(in_dir: str, name: str) -> pd.DataFrame:
    p = os.path.join(in_dir, f"{name}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def parse_date(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce")

def iso_date(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.strftime("%Y-%m-%d")

def write_out(df: pd.DataFrame, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    if HAS_PYREADSTAT and hasattr(pyreadstat, "write_xport"):
        xpt_path = os.path.join(out_dir, f"{name}.xpt")
        pyreadstat.write_xport(df, xpt_path, table_name=name[:8].upper())

def compute_ady(dt: pd.Series, trtsdt: pd.Series) -> pd.Series:
    dt = parse_date(dt)
    trtsdt = parse_date(trtsdt)
    return (dt - trtsdt).dt.days + 1

def ensure_cols(df: pd.DataFrame, cols: list[str], fill=""):
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

# response ranking (best -> worst)
RESP_ORDER = {"CR": 1, "PR": 2, "SD": 3, "NE": 4, "PD": 5}

def resp_rank(x: str) -> int:
    return RESP_ORDER.get(str(x).upper(), 99)


# ----------------------------
# Load SDTM
# ----------------------------
def load_sdtm(in_dir: str) -> Dict[str, pd.DataFrame]:
    need = ["dm","sv","ae","lb","vs","rs","ex","ds"]
    return {n: read_csv(in_dir, n) for n in need}


# ----------------------------
# ADSL (minimal)
# ----------------------------
def make_adsl(dm: pd.DataFrame, ex: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    adsl = dm.copy()

    ex2 = ex.copy()
    ex2["EXSTDTC"] = parse_date(ex2.get("EXSTDTC", ""))
    ex2["EXENDTC"] = parse_date(ex2.get("EXENDTC", ""))

    trt = (ex2.groupby("USUBJID")
              .agg(TRTSDT=("EXSTDTC","min"),
                   TRTEDT=("EXENDTC","max"))
              .reset_index())
    adsl = adsl.merge(trt, on="USUBJID", how="left")

    # planned/actual treatment (toy)
    if "ARM" in adsl.columns:
        adsl["TRT01P"] = adsl["ARM"]
        adsl["TRT01A"] = adsl["ARM"]
    else:
        adsl["TRT01P"] = ""
        adsl["TRT01A"] = ""

    adsl["TRT01AN"] = np.where(adsl["TRT01A"].astype(str).str.upper().eq("TRT"), 1, 0)
    adsl["SAFFL"] = np.where(adsl["TRTSDT"].notna(), "Y", "N")
    adsl["ITTFL"] = "Y"

    # DS last status
    ds2 = ds.copy()
    ds2["DSSTDTC"] = parse_date(ds2.get("DSSTDTC",""))
    if "DSDECOD" in ds2.columns:
        ds_last = (ds2.sort_values(["USUBJID","DSSTDTC"])
                     .groupby("USUBJID", as_index=False)
                     .tail(1)[["USUBJID","DSDECOD","DSSTDTC"]])
        adsl = adsl.merge(ds_last, on="USUBJID", how="left", suffixes=("","_DS"))
    else:
        adsl["DSDECOD"] = ""
        adsl["DSSTDTC"] = pd.NaT

    keep = [
        "STUDYID","USUBJID","SUBJID","SITEID",
        "AGE","SEX","RACE","COUNTRY",
        "ARMCD","ARM",
        "TRT01P","TRT01A","TRT01AN",
        "TRTSDT","TRTEDT",
        "SAFFL","ITTFL",
        "DSDECOD"
    ]
    adsl = ensure_cols(adsl, keep, "")
    adsl = adsl[keep].copy()
    # keep dates as ISO strings in outputs (common in csv)
    adsl["TRTSDT"] = iso_date(adsl["TRTSDT"])
    adsl["TRTEDT"] = iso_date(adsl["TRTEDT"])
    return adsl.sort_values(["SITEID","USUBJID"]).reset_index(drop=True)


# ----------------------------
# ADRS (BDS)
# ----------------------------
def make_adrs(rs: pd.DataFrame, sv: pd.DataFrame, adsl: pd.DataFrame) -> pd.DataFrame:
    """
    ADRS built from SDTM RS:
      - Use only RSTESTCD=OVRLRESP.
      - One record per subject per assessment visit.
      - Provide AVISIT/AVISITN, ADT/ADY, AVALC, ABLFL, ASEQ.
    """
    x = rs.copy()
    x = x[x["RSTESTCD"].astype(str).str.upper().eq("OVRLRESP")].copy()

    # merge ADSL anchors
    adsl_key = adsl[["USUBJID","TRT01A","TRT01AN","TRTSDT"]].copy()
    x = x.merge(adsl_key, on="USUBJID", how="left")

    # ADT/ADY
    x["ADT"] = parse_date(x.get("RSDTC",""))
    x["TRTSDT_DT"] = parse_date(x.get("TRTSDT",""))
    x["ADY"] = compute_ady(x["ADT"], x["TRTSDT_DT"])

    # --- SV date column compatibility (VISITDTC vs SVSTDTC) ---
    date_col = "VISITDTC" if "VISITDTC" in sv.columns else ("SVSTDTC" if "SVSTDTC" in sv.columns else None)
    if date_col is not None:
        sv2 = sv[["USUBJID","VISIT", date_col]].copy().rename(columns={date_col:"VISITDTC"})
    else:
        sv2 = sv[["USUBJID","VISIT"]].copy()
        sv2["VISITDTC"] = pd.NA

    # Global visit order for AVISITN (prefer SVDY if exists)
    if "SVDY" in sv.columns:
        vord = sv[["VISIT","SVDY"]].drop_duplicates().sort_values(["SVDY","VISIT"]).reset_index(drop=True)
    else:
        vord = sv[["VISIT"]].drop_duplicates().sort_values(["VISIT"]).reset_index(drop=True)
    vord["AVISITN"] = np.arange(1, len(vord) + 1)

    # In our RS SDTM, visit name is in column VISIT
    x = x.rename(columns={"VISIT":"AVISIT"})
    x = x.merge(vord.rename(columns={"VISIT":"AVISIT"}), on="AVISIT", how="left")

    # PARAM
    x["PARAMCD"] = "OVRLRESP"
    x["PARAM"] = "Overall Response (RECIST-like)"
    x["AVALC"] = x.get("RSORRES","").astype(str)
    x["AVAL"] = np.nan

    # baseline flag: on/before Day 1
    x["ABLFL"] = ""
    x.loc[(x["ADY"].notna()) & (x["ADY"] <= 1), "ABLFL"] = "Y"

    # ASEQ
    x = x.sort_values(["USUBJID","ADT","AVISITN"]).reset_index(drop=True)
    x["ASEQ"] = x.groupby("USUBJID").cumcount() + 1

    keep = [
        "STUDYID","USUBJID","TRT01A","TRT01AN",
        "PARAMCD","PARAM",
        "AVISIT","AVISITN",
        "ADT","ADY",
        "AVAL","AVALC",
        "ABLFL",
        "ASEQ"
    ]
    x = ensure_cols(x, keep, "")
    out = x[keep].copy()
    out["ADT"] = iso_date(out["ADT"])
    return out


# ----------------------------
# ADSLRS (subject-level response summary)
# ----------------------------
def make_adslrs(adrs: pd.DataFrame, adsl: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact subject-level response summary:
      - BOR (best overall response) from ADRS AVALC
      - BORDT (date of BOR)
      - PDDT (date of first PD)
      - PD_FL
    """
    x = adrs.copy()
    # only OVRLRESP records
    x = x[x["PARAMCD"].astype(str).str.upper().eq("OVRLRESP")].copy()

    # parse ADT
    x["ADT_DT"] = parse_date(x["ADT"])
    x["RANK"] = x["AVALC"].map(resp_rank)

    def per_subj(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("ADT_DT")
        # best response = min rank
        g_valid = g[g["ADT_DT"].notna() & g["AVALC"].notna()].copy()
        if len(g_valid) == 0:
            bor = ""
            bordt = pd.NaT
        else:
            best_rank = g_valid["RANK"].min()
            best_rows = g_valid[g_valid["RANK"] == best_rank]
            # if ties: earliest date
            best_rows = best_rows.sort_values("ADT_DT")
            bor = str(best_rows["AVALC"].iloc[0]).upper()
            bordt = best_rows["ADT_DT"].iloc[0]

        pd_rows = g_valid[g_valid["AVALC"].astype(str).str.upper().eq("PD")]
        if len(pd_rows) > 0:
            pddt = pd_rows["ADT_DT"].iloc[0]
            pdfl = "Y"
        else:
            pddt = pd.NaT
            pdfl = "N"

        return pd.Series({"BOR": bor, "BORDT": bordt, "PDDT": pddt, "PDFL": pdfl})

    summ = x.groupby("USUBJID", as_index=False).apply(per_subj)
    if isinstance(summ.index, pd.MultiIndex):
        summ = summ.reset_index(drop=True)

    out = adsl.merge(summ, on="USUBJID", how="left")
    out["BORDT"] = iso_date(out["BORDT"])
    out["PDDT"] = iso_date(out["PDDT"])

    keep = [
        "STUDYID","USUBJID","SUBJID","SITEID",
        "TRT01A","TRT01AN","TRTSDT","TRTEDT",
        "BOR","BORDT","PDFL","PDDT"
    ]
    out = ensure_cols(out, keep, "")
    return out[keep].sort_values(["SITEID","USUBJID"]).reset_index(drop=True)

def make_adslrs_confirmed(adrs: pd.DataFrame, adsl: pd.DataFrame) -> pd.DataFrame:
    """
    Create subject-level response summary with:
      - BOR    : Best Overall Response (unconfirmed)
      - BORUNC : Same as BOR
      - BORC   : Confirmed BOR (CR/PR requires confirmation >=28 days)
    """

    df = adrs.copy()
    df = df[df["PARAMCD"].astype(str).str.upper().eq("OVRLRESP")].copy()

    df["ADT_DT"] = pd.to_datetime(df["ADT"], errors="coerce")
    df["RESP"] = df["AVALC"].astype(str).str.upper()

    RESP_RANK = {"CR": 1, "PR": 2, "SD": 3, "NE": 4, "PD": 5}
    df["RANK"] = df["RESP"].map(RESP_RANK)

    out_rows = []

    for usubjid, g in df.groupby("USUBJID"):
        g = g.sort_values("ADT_DT")

        # ---------- BOR / BORUNC ----------
        valid = g[g["RANK"].notna()]
        if len(valid) == 0:
            bor = ""
        else:
            best_rank = valid["RANK"].min()
            bor = valid.loc[valid["RANK"] == best_rank, "RESP"].iloc[0]

        borunc = bor

        # ---------- BORC ----------
        borc = ""

        # candidate PR/CR rows
        cand = g[g["RESP"].isin(["CR", "PR"])].copy()

        confirmed = False
        for i, row in cand.iterrows():
            t0 = row["ADT_DT"]
            later = g[
                (g["ADT_DT"] >= t0 + pd.Timedelta(days=28)) &
                (g["RESP"].isin(["CR", "PR"]))
            ]
            if len(later) > 0:
                borc = row["RESP"]
                confirmed = True
                break

        if not confirmed:
            # fallback
            if bor in ["CR", "PR"]:
                if "SD" in valid["RESP"].values:
                    borc = "SD"
                else:
                    borc = bor
            else:
                borc = bor

        out_rows.append({
            "USUBJID": usubjid,
            "BOR": bor,
            "BORUNC": borunc,
            "BORC": borc
        })

    summ = pd.DataFrame(out_rows)

    # merge back to ADSL
    out = adsl.merge(summ, on="USUBJID", how="left")

    keep = [
        "STUDYID","USUBJID","SUBJID","SITEID",
        "TRT01A","TRT01AN","TRTSDT","TRTEDT",
        "BOR","BORUNC","BORC"
    ]
    out = ensure_cols(out, keep, "")
    return out[keep].sort_values(["SITEID","USUBJID"]).reset_index(drop=True)


# ----------------------------
# Enhanced ADTTE (PFS + OS)
# ----------------------------
def make_adtte(rs: pd.DataFrame, ds: pd.DataFrame, ex: pd.DataFrame, adsl: pd.DataFrame) -> pd.DataFrame:
    """
    Produce multiple time-to-event parameters:
      - PFS: event = first PD from RS OVRLRESP; censor = last RS assessment date otherwise.
      - OS : event = death if DSDECOD contains DEATH/DIED; censor = last contact date otherwise.

    Output columns:
      STUDYID, USUBJID, TRT01A, TRT01AN,
      PARAMCD, PARAM,
      STARTDT, ADT, AVAL, CNSR,
      EVNTDESC, CNSRREAS,
      LSTALVDT
    """
    # anchors
    base = adsl[["STUDYID","USUBJID","TRT01A","TRT01AN","TRTSDT","TRTEDT"]].copy()
    base["TRTSDT_DT"] = parse_date(base["TRTSDT"])
    base["TRTEDT_DT"] = parse_date(base["TRTEDT"])

    # RS overall response dates
    rs2 = rs.copy()
    rs2 = rs2[rs2["RSTESTCD"].astype(str).str.upper().eq("OVRLRESP")].copy()
    rs2["ADT_DT"] = parse_date(rs2.get("RSDTC",""))
    rs2["VISIT"] = rs2.get("VISIT","")

    # last rs date
    last_rs = (rs2.groupby("USUBJID")["ADT_DT"].max().reset_index().rename(columns={"ADT_DT":"LASTRSDT"}))

    # first PD date
    pd_rs = rs2[rs2["RSORRES"].astype(str).str.upper().eq("PD")].copy()
    first_pd = (pd_rs.sort_values(["USUBJID","ADT_DT"])
                    .groupby("USUBJID", as_index=False)
                    .first()[["USUBJID","ADT_DT"]]
                    .rename(columns={"ADT_DT":"PDDT"}))

    # DS death date (if any)
    ds2 = ds.copy()
    ds2["DSSTDTC_DT"] = parse_date(ds2.get("DSSTDTC",""))
    ds2["DSDECOD_UP"] = ds2.get("DSDECOD","").astype(str).str.upper()
    death = ds2[ds2["DSDECOD_UP"].str.contains("DEATH|DIED", regex=True, na=False)].copy()
    death = (death.sort_values(["USUBJID","DSSTDTC_DT"])
                  .groupby("USUBJID", as_index=False)
                  .first()[["USUBJID","DSSTDTC_DT"]]
                  .rename(columns={"DSSTDTC_DT":"DTHDT"}))

    # last contact date (simple): max(TRTEDT, last RS)
    base = base.merge(last_rs, on="USUBJID", how="left")
    base["LSTALVDT_DT"] = base[["TRTEDT_DT","LASTRSDT"]].max(axis=1)

    # ----- PFS -----
    tmp = base.merge(first_pd, on="USUBJID", how="left")
    tmp["ADT_DT"] = tmp["PDDT"]
    tmp["CNSR"] = np.where(tmp["ADT_DT"].notna(), 0, 1)
    tmp.loc[tmp["CNSR"] == 1, "ADT_DT"] = tmp.loc[tmp["CNSR"] == 1, "LASTRSDT"]

    tmp["AVAL"] = (tmp["ADT_DT"] - tmp["TRTSDT_DT"]).dt.days + 1
    tmp.loc[tmp["AVAL"].isna(), "AVAL"] = np.nan

    pfs = pd.DataFrame({
        "STUDYID": tmp["STUDYID"],
        "USUBJID": tmp["USUBJID"],
        "TRT01A": tmp["TRT01A"],
        "TRT01AN": tmp["TRT01AN"],
        "PARAMCD": "PFS",
        "PARAM": "Progression-Free Survival (PD from RS)",
        "STARTDT": iso_date(tmp["TRTSDT_DT"]),
        "ADT": iso_date(tmp["ADT_DT"]),
        "AVAL": tmp["AVAL"],
        "CNSR": tmp["CNSR"],
        "EVNTDESC": np.where(tmp["CNSR"]==0, "Disease progression (PD)", ""),
        "CNSRREAS": np.where(tmp["CNSR"]==1, "No PD observed; censored at last RS assessment", ""),
        "LSTALVDT": iso_date(tmp["LSTALVDT_DT"])
    })

    # ----- OS -----
    tmp2 = base.merge(death, on="USUBJID", how="left")
    tmp2["ADT_DT"] = tmp2["DTHDT"]
    tmp2["CNSR"] = np.where(tmp2["ADT_DT"].notna(), 0, 1)
    tmp2.loc[tmp2["CNSR"] == 1, "ADT_DT"] = tmp2.loc[tmp2["CNSR"] == 1, "LSTALVDT_DT"]

    tmp2["AVAL"] = (tmp2["ADT_DT"] - tmp2["TRTSDT_DT"]).dt.days + 1
    tmp2.loc[tmp2["AVAL"].isna(), "AVAL"] = np.nan

    osdf = pd.DataFrame({
        "STUDYID": tmp2["STUDYID"],
        "USUBJID": tmp2["USUBJID"],
        "TRT01A": tmp2["TRT01A"],
        "TRT01AN": tmp2["TRT01AN"],
        "PARAMCD": "OS",
        "PARAM": "Overall Survival",
        "STARTDT": iso_date(tmp2["TRTSDT_DT"]),
        "ADT": iso_date(tmp2["ADT_DT"]),
        "AVAL": tmp2["AVAL"],
        "CNSR": tmp2["CNSR"],
        "EVNTDESC": np.where(tmp2["CNSR"]==0, "Death", ""),
        "CNSRREAS": np.where(tmp2["CNSR"]==1, "No death recorded; censored at last contact", ""),
        "LSTALVDT": iso_date(tmp2["LSTALVDT_DT"])
    })

    out = pd.concat([pfs, osdf], ignore_index=True)
    out = out.sort_values(["USUBJID","PARAMCD"]).reset_index(drop=True)
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insdtm", required=True, help="Input SDTM folder (csv)")
    ap.add_argument("--out", required=True, help="Output ADaM folder")
    args = ap.parse_args()

    sdtm = load_sdtm(args.insdtm)
    dm, sv, ae, lb, vs, rs, ex, ds = (sdtm["dm"], sdtm["sv"], sdtm["ae"], sdtm["lb"],
                                     sdtm["vs"], sdtm["rs"], sdtm["ex"], sdtm["ds"])

    # ADSL
    adsl = make_adsl(dm, ex, ds)
    write_out(adsl, args.out, "adsl")

    # ADRS + ADSLRS
    adrs = make_adrs(rs, sv, adsl)
    write_out(adrs, args.out, "adrs")

    adslrs = make_adslrs_confirmed(adrs, adsl)
    write_out(adslrs, args.out, "adslrs")

    # ADTTE (PFS + OS)
    adtte = make_adtte(rs, ds, ex, adsl)
    write_out(adtte, args.out, "adtte")

    print(f"Done -> {args.out}")
    print("Wrote: adsl, adrs, adslrs, adtte (PFS+OS)")
    if not (HAS_PYREADSTAT and hasattr(pyreadstat, "write_xport")):
        print("NOTE: XPT not written (pyreadstat.write_xport unavailable); CSV written only.")


if __name__ == "__main__":
    main()
