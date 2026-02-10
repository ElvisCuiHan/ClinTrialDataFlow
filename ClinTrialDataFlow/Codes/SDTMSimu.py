# rawzip_to_sdtm.py
from __future__ import annotations
import argparse
import os
import zipfile
from typing import Dict, Optional
import pandas as pd
import numpy as np

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except Exception:
    HAS_PYREADSTAT = False


# ----------------------------
# IO helpers
# ----------------------------
def read_csv_from_zip(z: zipfile.ZipFile, path: str) -> pd.DataFrame:
    return pd.read_csv(z.open(path))


# def list_datasets_in_zip(z: zipfile.ZipFile, prefix: str) -> Dict[str, str]:
#     """
#     Return map: dataset_name -> csv_path inside zip
#     Prefer CSV if both CSV and XPT exist.
#     """
#     csvs = {}
#     for n in z.namelist():
#         if not n.startswith(prefix):
#             continue
#         if n.endswith(".csv") and "/__MACOSX/" not in n and "__MACOSX" not in n:
#             base = os.path.basename(n).split(".")[0]
#             csvs[base] = n
#     return csvs

def list_datasets_in_dir(dir_path: str) -> Dict[str, str]:
    """
    Return map: dataset_name -> csv_path in directory
    """
    ds = {}
    for fn in os.listdir(dir_path):
        if fn.endswith(".csv"):
            name = fn.replace(".csv", "")
            ds[name] = os.path.join(dir_path, fn)
    return ds

def write_outputs(df: pd.DataFrame, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    # XPT optional
    if HAS_PYREADSTAT and hasattr(pyreadstat, "write_xport"):
        xpt_path = os.path.join(out_dir, f"{name}.xpt")
        pyreadstat.write_xport(df, xpt_path, table_name=name[:8].upper())


def parse_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


# ----------------------------
# SDTM builders
# ----------------------------
def make_dm(subject: pd.DataFrame, trnt: Optional[pd.DataFrame]) -> pd.DataFrame:
    dm = subject.copy()

    # RFENDTC from EXPDAYS if available
    if trnt is not None and "EXPDAYS" in trnt.columns:
        tmp = trnt[["USUBJID", "EXPDAYS"]].copy()
        dm = dm.merge(tmp, on="USUBJID", how="left")
        rf = parse_date(dm["RFSTDTC"])
        dm["RFENDTC"] = (rf + pd.to_timedelta(dm["EXPDAYS"].fillna(0).astype(float), unit="D")).dt.strftime("%Y-%m-%d")
        dm = dm.drop(columns=["EXPDAYS"])
    else:
        dm["RFENDTC"] = ""

    # SDTM-ish fields
    dm.insert(1, "DOMAIN", "DM")
    dm = dm.rename(columns={
        "ARM": "ARM",
        "SITEID": "SITEID",
        "SUBJID": "SUBJID",
        "SEX": "SEX",
        "AGE": "AGE"
    })

    # Common optional SDTM vars (keep blank if not available)
    if "RACE" not in dm.columns:
        dm["RACE"] = ""
    dm["COUNTRY"] = ""
    dm["ARMCD"] = dm["ARM"]

    # Order (minimal)
    keep = [
        "STUDYID","DOMAIN","USUBJID","SUBJID","SITEID",
        "RFSTDTC","RFENDTC",
        "ARMCD","ARM",
        "SEX","AGE","RACE","COUNTRY"
    ]
    # add ECOG0 as SUPP-like info (keep in DM for convenience)
    if "ECOG0" in dm.columns:
        keep.append("ECOG0")

    return dm[[c for c in keep if c in dm.columns]]


def make_sv(sv_raw: pd.DataFrame) -> pd.DataFrame:
    sv = sv_raw.copy()
    sv.insert(1, "DOMAIN", "SV")

    # VISITNUM from VISIT order
    visit_order = (sv[["VISIT","VISITDY"]]
                   .drop_duplicates()
                   .sort_values(["VISITDY","VISIT"])
                   .reset_index(drop=True))
    visit_order["VISITNUM"] = np.arange(1, len(visit_order) + 1)
    sv = sv.merge(visit_order[["VISIT","VISITNUM"]], on="VISIT", how="left")

    sv = sv.rename(columns={
        "VISITDTC": "SVSTDTC",
        "VISITDY": "SVDY"
    })
    sv["SVENDTC"] = sv["SVSTDTC"]

    keep = ["STUDYID","DOMAIN","USUBJID","VISITNUM","VISIT","SVSTDTC","SVENDTC","SVDY"]
    return sv[keep]


def make_ae(ae_raw: pd.DataFrame) -> pd.DataFrame:
    ae = ae_raw.copy()
    ae.insert(1, "DOMAIN", "AE")

    # SDTM naming
    ae = ae.rename(columns={
        "AETERM": "AETERM",
        "AESTDTC": "AESTDTC",
        "AEENDTC": "AEENDTC",
        "AESEV": "AESEV",
        "AEREL": "AEREL",
    })

    # optional SDTM variables
    ae["AEDECOD"] = ""   # could be MedDRA decode later
    ae["AESOC"] = ""
    ae["AESER"] = ""

    keep = ["STUDYID","DOMAIN","USUBJID","AESEQ","AETERM","AEDECOD","AESOC",
            "AESTDTC","AEENDTC","AESEV","AESER","AEREL"]
    for c in keep:
        if c not in ae.columns:
            ae[c] = ""
    return ae[keep]


def make_lb(lb_raw: pd.DataFrame) -> pd.DataFrame:
    lb = lb_raw.copy()
    lb.insert(1, "DOMAIN", "LB")

    # LBTESTCD: derive from LBTEST (simple)
    lb["LBTESTCD"] = lb["LBTEST"].astype(str).str.upper().str.replace(" ", "", regex=False).str.slice(0,8)

    lb = lb.rename(columns={
        "VISITDTC": "LBDTC",
        "VISITDY": "LBDY",
        "LBORRES": "LBORRES",
        "LBORRESU": "LBORRESU",
    })

    # Keep VISIT/VISITNUM optional (if you want: merge from SV)
    keep = ["STUDYID","DOMAIN","USUBJID","LBSEQ","LBTESTCD","LBTEST","LBORRES","LBORRESU","VISIT","LBDTC","LBDY"]
    for c in keep:
        if c not in lb.columns:
            lb[c] = ""
    return lb[keep]


def make_vs(vs_raw: pd.DataFrame) -> pd.DataFrame:
    vs = vs_raw.copy()
    vs.insert(1, "DOMAIN", "VS")
    vs["VSTESTCD"] = vs["VSTEST"].astype(str).str.upper().str.replace(" ", "", regex=False).str.slice(0,8)

    vs = vs.rename(columns={
        "VISITDTC": "VSDTC",
        "VISITDY": "VSDY",
        "VSORRES": "VSORRES",
        "VSORRESU": "VSORRESU",
    })

    keep = ["STUDYID","DOMAIN","USUBJID","VSSEQ","VSTESTCD","VSTEST","VSORRES","VSORRESU","VISIT","VSDTC","VSDY"]
    for c in keep:
        if c not in vs.columns:
            vs[c] = ""
    return vs[keep]


def make_tu(tu_raw: pd.DataFrame, sv_raw: pd.DataFrame) -> pd.DataFrame:
    """
    SDTM TU typically includes a date (TUDTC) and test/result fields.
    Your TU raw has lesion identity only; we'll assign baseline date from SV (VISIT == BASE),
    and store location/type as TUORRES/TUSTRESC-style fields.
    """
    tu = tu_raw.copy()
    tu.insert(1, "DOMAIN", "TU")

    # baseline date mapping
    base = sv_raw[sv_raw["VISIT"].astype(str).str.upper().eq("BASE")][["USUBJID","VISITDTC"]].copy()
    base = base.rename(columns={"VISITDTC":"TUDTC"})
    tu = tu.merge(base, on="USUBJID", how="left")

    # Minimal TU “test/result”
    tu["TUTESTCD"] = "TUMIDENT"
    tu["TUTEST"] = "Tumor Identification"
    # put location/type into result fields
    tu["TUORRES"] = tu["TULOC"].astype(str)
    tu["TUSTRESC"] = tu["TUTYPE"].astype(str)

    keep = ["STUDYID","DOMAIN","USUBJID","TUSEQ","TULNKID","TUTESTCD","TUTEST","TUORRES","TUSTRESC","TUDTC","TULOC","TUTYPE"]
    for c in keep:
        if c not in tu.columns:
            tu[c] = ""
    return tu[keep]


def make_rs(rs_raw: pd.DataFrame) -> pd.DataFrame:
    """
    SDTM RS is a findings domain; common pattern is one record per test per visit.
    We'll output:
      - RSTESTCD=OVRLRESP, RSORRES=CR/PR/SD/PD/NE
      - RSTESTCD=NEWLIND, RSORRES=Y/N
    """
    rs = rs_raw.copy()
    rs.insert(1, "DOMAIN", "RS")

    # Build two records per assessment
    base_cols = ["STUDYID","DOMAIN","USUBJID","RSVISIT","RSVISITDY","RSDTC"]
    base = rs[base_cols].copy()
    base = base.rename(columns={
        "RSVISIT": "VISIT",
        "RSVISITDY": "RSDY"
    })

    rec1 = base.copy()
    rec1["RSTESTCD"] = "OVRLRESP"
    rec1["RSTEST"] = "Overall Response"
    rec1["RSORRES"] = rs["OVRLRESP"].astype(str)
    rec1["RSSTRESC"] = rec1["RSORRES"]

    rec2 = base.copy()
    rec2["RSTESTCD"] = "NEWLIND"
    rec2["RSTEST"] = "New Lesion Indicator"
    rec2["RSORRES"] = rs["NEWLIND"].astype(str)
    rec2["RSSTRESC"] = rec2["RSORRES"]

    out = pd.concat([rec1, rec2], ignore_index=True)
    out = out.sort_values(["USUBJID","RSDY","RSTESTCD"]).reset_index(drop=True)
    out["RSSEQ"] = out.groupby("USUBJID").cumcount() + 1

    # final rename of date variable
    out = out.rename(columns={"RSDTC": "RSDTC"})

    keep = ["STUDYID","DOMAIN","USUBJID","RSSEQ","RSTESTCD","RSTEST",
            "RSORRES","RSSTRESC","VISIT","RSDTC","RSDY"]
    return out[keep]


def make_ex(trt: pd.DataFrame, trnt: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Build minimal EX from TRT/TRNT.
    One record per subject: EXSTDTC=RFSTDTC, EXENDTC=RFSTDTC+EXPDAYS, dose assumed 200 mg.
    """
    ex = trt.copy()
    ex.insert(1, "DOMAIN", "EX")

    ex = ex.rename(columns={"TRT01A":"EXTRT", "RFSTDTC":"EXSTDTC"})
    ex["EXDOSE"] = 200
    ex["EXDOSU"] = "mg"
    ex["EXROUTE"] = "ORAL"
    ex["EXFREQ"] = "QD"

    if trnt is not None and "EXPDAYS" in trnt.columns:
        tmp = trnt[["USUBJID","EXPDAYS"]].copy()
        ex = ex.merge(tmp, on="USUBJID", how="left")
        st = parse_date(ex["EXSTDTC"])
        ex["EXENDTC"] = (st + pd.to_timedelta(ex["EXPDAYS"].fillna(0).astype(float), unit="D")).dt.strftime("%Y-%m-%d")
        ex = ex.drop(columns=["EXPDAYS"])
    else:
        ex["EXENDTC"] = ""

    ex["EXSEQ"] = ex.groupby("USUBJID").cumcount() + 1

    keep = ["STUDYID","DOMAIN","USUBJID","EXSEQ","EXTRT","EXDOSE","EXDOSU","EXROUTE","EXFREQ","EXSTDTC","EXENDTC"]
    return ex[keep]


def make_ds(ss: pd.DataFrame, trnt: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Minimal DS using STATUS from ss:
      - STATUS == DISCONTINUED -> DSDECOD=DISCONTINUED
      - else -> DSDECOD=COMPLETED/ONGOING
    DSSTDTC: use RFENDTC if available else RFSTDTC.
    """
    ds = ss.copy()
    ds.insert(1, "DOMAIN", "DS")

    ds["DSDECOD"] = np.where(ds["STATUS"].astype(str).str.upper().str.contains("DISCONT", na=False),
                             "DISCONTINUED",
                             "COMPLETED/ONGOING")

    # DSSTDTC: prefer RFENDTC from EXPDAYS
    if trnt is not None and "EXPDAYS" in trnt.columns:
        tmp = trnt[["USUBJID","EXPDAYS"]].copy()
        ds = ds.merge(tmp, on="USUBJID", how="left")
        rf = parse_date(ds["RFSTDTC"])
        ds["DSSTDTC"] = (rf + pd.to_timedelta(ds["EXPDAYS"].fillna(0).astype(float), unit="D")).dt.strftime("%Y-%m-%d")
        ds = ds.drop(columns=["EXPDAYS"])
    else:
        ds["DSSTDTC"] = ds["RFSTDTC"]

    ds["DSCAT"] = "DISPOSITION EVENT"
    ds["DSSEQ"] = ds.groupby("USUBJID").cumcount() + 1

    keep = ["STUDYID","DOMAIN","USUBJID","DSSEQ","DSCAT","DSDECOD","DSSTDTC","STATUS"]
    return ds[keep]


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Input directory containing CSV datasets")
    ap.add_argument("--out", required=True, help="Output directory for SDTM datasets")
    args = ap.parse_args()

    ds_map = list_datasets_in_dir(args.indir)

    def must(name: str) -> pd.DataFrame:
        p = ds_map.get(name)
        if not p:
            raise FileNotFoundError(f"Missing {name}.csv under directory {args.indir}")
        return pd.read_csv(p)

    # ---- read inputs from directory ----
    subject = must("subject")
    sv_raw  = must("sv")
    ae_raw  = must("ae")
    lb_raw  = must("lb")
    vs_raw  = must("vs")
    tu_raw  = must("tu")
    rs_raw  = must("rs")
    trt_raw = must("trt")
    ss_raw  = must("ss")
    trnt_raw = pd.read_csv(ds_map["trnt"]) if "trnt" in ds_map else None

    # ---- Build SDTM ----
    dm = make_dm(subject, trnt_raw)
    sv = make_sv(sv_raw)
    ae = make_ae(ae_raw)
    lb = make_lb(lb_raw)
    vs = make_vs(vs_raw)
    tu = make_tu(tu_raw, sv_raw)
    rs = make_rs(rs_raw)
    ex = make_ex(trt_raw, trnt_raw)
    ds = make_ds(ss_raw, trnt_raw)

    out_dir = args.out
    write_outputs(dm, out_dir, "dm")
    write_outputs(sv, out_dir, "sv")
    write_outputs(ae, out_dir, "ae")
    write_outputs(lb, out_dir, "lb")
    write_outputs(vs, out_dir, "vs")
    write_outputs(tu, out_dir, "tu")
    write_outputs(rs, out_dir, "rs")
    write_outputs(ex, out_dir, "ex")
    write_outputs(ds, out_dir, "ds")

    print(f"SDTM export done -> {out_dir}")
    print("Wrote: dm, sv, ae, lb, vs, tu, rs, ex, ds")
    if not (HAS_PYREADSTAT and hasattr(pyreadstat, "write_xport")):
        print("NOTE: XPT not written (pyreadstat.write_xport unavailable); CSV written only.")


if __name__ == "__main__":
    main()
