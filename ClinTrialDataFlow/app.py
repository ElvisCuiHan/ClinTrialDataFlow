# app.py
# Interactive local web app for your EDC -> SDTM -> ADaM -> TFL simulator pipeline.
# Run:
#   pip install streamlit pandas numpy matplotlib
#   streamlit run app.py
#
# What it does:
# - Lets you set output folder
# - Lets you edit/generate cfg.json fields (seed/n/etc.)
# - Runs your existing scripts (EDCSimu.py, SDTMSimu.py, ADaMSimu.py, TFLSimu.py)
# - Shows logs in the browser
# - Zips outputs for download
#
# Assumptions (adjust in the sidebar if your structure differs):
#   ProjectRoot/
#     cfg.json
#     Codes/EDCSimu.py
#     Codes/SDTMSimu.py
#     Codes/ADaMSimu.py
#     Codes/TFLSimu.py
#     Data/
#
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Config schema (lightweight)
# -----------------------------
@dataclass
class Cfg:
    seed: int = 20260210
    n: int = 200
    studyid: str = "HGR-0001"
    site_n: int = 8
    arms: Tuple[str, str] = ("PBO", "TRT")
    alloc: Tuple[int, int] = (1, 1)
    dropout_rate: float = 0.10
    form_missing_rate: float = 0.03
    item_missing_rate: float = 0.02
    query_error_rate: float = 0.01


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    _ensure_dir(p.parent)
    p.write_text(s, encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run command, capture combined stdout/stderr."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    out_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
    proc.wait()
    return proc.returncode, "".join(out_lines)


def zip_dir(dir_path: Path) -> bytes:
    """Return zip bytes of a directory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dir_path):
            for fn in files:
                fp = Path(root) / fn
                rel = fp.relative_to(dir_path)
                z.write(fp, rel.as_posix())
    return buf.getvalue()


def load_cfg_json(cfg_path: Path) -> Dict:
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(read_text(cfg_path))
    except Exception:
        return {}


def cfg_to_json_dict(cfg: Cfg) -> Dict:
    d = asdict(cfg)
    # tuples -> lists for JSON
    d["arms"] = list(cfg.arms)
    d["alloc"] = list(cfg.alloc)
    return d


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="EDC/SDTM/ADaM/TFL Simulator", layout="wide")

st.title("EDC → SDTM → ADaM → TFL (Interactive Runner)")

with st.sidebar:
    st.header("Project Paths")
    project_root = st.text_input(
        "Project root folder",
        value=str(Path.cwd()),
        help="Folder that contains cfg.json, Codes/, Data/",
    )
    project_root_p = Path(project_root).expanduser().resolve()

    codes_dir = st.text_input("Codes dir", value=str(project_root_p / "Codes"))
    data_dir = st.text_input("Data dir", value=str(project_root_p / "Data"))
    cfg_path = st.text_input("cfg.json path", value=str(project_root_p / "cfg.json"))

    st.divider()
    st.header("Scripts")
    edc_py = st.text_input("EDCSimu.py", value=str(Path(codes_dir) / "EDCSimu.py"))
    sdtm_py = st.text_input("SDTMSimu.py", value=str(Path(codes_dir) / "SDTMSimu.py"))
    adam_py = st.text_input("ADaMSimu.py", value=str(Path(codes_dir) / "ADaMSimu.py"))
    tfl_py = st.text_input("TFLSimu.py", value=str(Path(codes_dir) / "TFLSimu.py"))

    st.divider()
    st.header("Outputs")
    raw_out = st.text_input("RAW out", value=str(Path(data_dir) / "raw_out"))
    sdtm_out = st.text_input("SDTM out", value=str(Path(data_dir) / "sdtm_out"))
    adam_out = st.text_input("ADaM out", value=str(Path(data_dir) / "adam_out"))
    tfl_out = st.text_input("TFL out", value=str(Path(data_dir) / "tfl_out"))


# cfg editor
st.subheader("1) Configure cfg.json")

existing = load_cfg_json(Path(cfg_path))

colA, colB, colC = st.columns(3)
with colA:
    seed = st.number_input("seed", value=int(existing.get("seed", 20260210)), step=1)
    n = st.number_input("n", value=int(existing.get("n", 200)), step=10, min_value=10)
    studyid = st.text_input("studyid", value=str(existing.get("studyid", "HGR-0001")))
with colB:
    site_n = st.number_input("site_n", value=int(existing.get("site_n", 8)), step=1, min_value=1)
    arms = st.text_input("arms (comma-separated)", value=",".join(existing.get("arms", ["PBO", "TRT"])))
    alloc = st.text_input("alloc (comma-separated)", value=",".join(map(str, existing.get("alloc", [1, 1]))))
with colC:
    dropout_rate = st.slider("dropout_rate", 0.0, 0.6, float(existing.get("dropout_rate", 0.10)), 0.01)
    form_missing_rate = st.slider("form_missing_rate", 0.0, 0.3, float(existing.get("form_missing_rate", 0.03)), 0.005)
    item_missing_rate = st.slider("item_missing_rate", 0.0, 0.3, float(existing.get("item_missing_rate", 0.02)), 0.005)
    query_error_rate = st.slider("query_error_rate", 0.0, 0.1, float(existing.get("query_error_rate", 0.01)), 0.001)

# parse arms/alloc
arms_list = [a.strip() for a in arms.split(",") if a.strip()]
alloc_list = [int(x.strip()) for x in alloc.split(",") if x.strip().isdigit()]
if len(arms_list) < 2:
    st.warning("arms should contain at least 2 items (e.g., PBO,TRT)")
if len(alloc_list) != len(arms_list):
    st.warning("alloc length should match arms length")

cfg_obj = Cfg(
    seed=int(seed),
    n=int(n),
    studyid=studyid,
    site_n=int(site_n),
    arms=tuple(arms_list[:2]) if len(arms_list) >= 2 else ("PBO", "TRT"),
    alloc=tuple(alloc_list[:2]) if len(alloc_list) >= 2 else (1, 1),
    dropout_rate=float(dropout_rate),
    form_missing_rate=float(form_missing_rate),
    item_missing_rate=float(item_missing_rate),
    query_error_rate=float(query_error_rate),
)

cfg_json = cfg_to_json_dict(cfg_obj)

save_cfg = st.button("Save cfg.json")
if save_cfg:
    write_text(Path(cfg_path), json.dumps(cfg_json, indent=2))
    st.success(f"Saved: {cfg_path}")

st.code(json.dumps(cfg_json, indent=2), language="json")


st.subheader("2) Run pipeline")

run_cols = st.columns(4)
btn_edc = run_cols[0].button("Run EDC→RAW", use_container_width=True)
btn_sdtm = run_cols[1].button("Run RAW→SDTM", use_container_width=True)
btn_adam = run_cols[2].button("Run SDTM→ADaM", use_container_width=True)
btn_tfl = run_cols[3].button("Run ADaM→TFL", use_container_width=True)

btn_all = st.button("Run ALL (EDC→RAW→SDTM→ADaM→TFL)")

log_box = st.empty()


def show_log(title: str, text: str, code: int):
    status = "✅" if code == 0 else "❌"
    st.markdown(f"### {status} {title}")
    st.code(text or "(no output)")


def step_edc() -> Tuple[int, str]:
    _ensure_dir(Path(raw_out))
    cmd = ["python", str(edc_py), "--cfg", str(cfg_path), "--out", str(raw_out)]
    return run_cmd(cmd, cwd=project_root_p)


def step_sdtm() -> Tuple[int, str]:
    _ensure_dir(Path(sdtm_out))
    # directory mode (no zip): SDTMSimu.py should accept --indir
    cmd = ["python", str(sdtm_py), "--indir", str(raw_out), "--out", str(sdtm_out)]
    return run_cmd(cmd, cwd=project_root_p)


def step_adam() -> Tuple[int, str]:
    _ensure_dir(Path(adam_out))
    cmd = ["python", str(adam_py), "--insdtm", str(sdtm_out), "--out", str(adam_out)]
    return run_cmd(cmd, cwd=project_root_p)


def step_tfl() -> Tuple[int, str]:
    _ensure_dir(Path(tfl_out))
    cmd = ["python", str(tfl_py), "--inadam", str(adam_out), "--out", str(tfl_out)]
    return run_cmd(cmd, cwd=project_root_p)


# run logic
if btn_edc:
    rc, out = step_edc()
    show_log("EDC→RAW", out, rc)

if btn_sdtm:
    rc, out = step_sdtm()
    show_log("RAW→SDTM", out, rc)

if btn_adam:
    rc, out = step_adam()
    show_log("SDTM→ADaM", out, rc)

if btn_tfl:
    rc, out = step_tfl()
    show_log("ADaM→TFL", out, rc)

if btn_all:
    rc1, o1 = step_edc(); show_log("EDC→RAW", o1, rc1)
    if rc1 != 0:
        st.stop()
    rc2, o2 = step_sdtm(); show_log("RAW→SDTM", o2, rc2)
    if rc2 != 0:
        st.stop()
    rc3, o3 = step_adam(); show_log("SDTM→ADaM", o3, rc3)
    if rc3 != 0:
        st.stop()
    rc4, o4 = step_tfl(); show_log("ADaM→TFL", o4, rc4)


st.subheader("3) Preview outputs")

prev_cols = st.columns(4)

# show a quick peek
for label, folder, examples in [
    ("RAW", raw_out, ["subject", "sv", "tu", "rs"]),
    ("SDTM", sdtm_out, ["dm", "sv", "tu", "rs"]),
    ("ADaM", adam_out, ["adsl", "adrs", "adslrs", "adtte"]),
    ("TFL", tfl_out, ["tfl_baseline", "tfl_orr", "tfl_tte_summary"]),
]:
    with st.expander(f"{label} preview ({folder})"):
        folder_p = Path(folder)
        if not folder_p.exists():
            st.info("Folder not found yet.")
        else:
            for name in examples:
                fp = folder_p / f"{name}.csv"
                if fp.exists():
                    st.markdown(f"**{name}.csv**")
                    try:
                        d = pd.read_csv(fp)
                        st.dataframe(d.head(20), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Cannot read {fp}: {e}")


st.subheader("4) Download outputs")

dl_cols = st.columns(4)

for i, (label, folder, zipname) in enumerate([
    ("RAW", raw_out, "raw_out.zip"),
    ("SDTM", sdtm_out, "sdtm_out.zip"),
    ("ADaM", adam_out, "adam_out.zip"),
    ("TFL", tfl_out, "tfl_out.zip"),
]):
    with dl_cols[i]:
        folder_p = Path(folder)
        if folder_p.exists() and any(folder_p.glob("*.csv")):
            zbytes = zip_dir(folder_p)
            st.download_button(
                label=f"Download {label} ZIP",
                data=zbytes,
                file_name=zipname,
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.button(f"Download {label} ZIP", disabled=True, use_container_width=True)
