#!/usr/bin/env bash
set -euo pipefail

# --------- Paths (relative to project root) ----------
CFG="cfg.json"
CODES_DIR="Codes"
DATA_DIR="Data"

RAW_DIR="${DATA_DIR}/raw_out"
SDTM_DIR="${DATA_DIR}/sdtm_out"
ADAM_DIR="${DATA_DIR}/adam_out"
TFL_DIR="${DATA_DIR}/tfl_out"

RAW_ZIP="raw.zip"   # optional; only if you want to unzip or keep snapshot

# --------- Make sure output dirs exist ----------
mkdir -p "${RAW_DIR}" "${SDTM_DIR}" "${ADAM_DIR}" "${TFL_DIR}"

echo "================================================="
echo "[1/4] EDC/Raw Simulator -> RAW"
echo "================================================="
python "${CODES_DIR}/EDCSimu.py" \
  --cfg "${CFG}" \
  --out "${RAW_DIR}"

echo "================================================="
echo "[2/4] RAW -> SDTM"
echo "================================================="
python "${CODES_DIR}/SDTMSimu.py" \
  --inraw "${RAW_DIR}" \
  --out "${SDTM_DIR}"

echo "================================================="
echo "[3/4] SDTM -> ADaM"
echo "================================================="
python "${CODES_DIR}/ADaMSimu.py" \
  --insdtm "${SDTM_DIR}" \
  --out "${ADAM_DIR}"

echo "================================================="
echo "[4/4] ADaM -> TFL"
echo "================================================="
python "${CODES_DIR}/TFLSimu.py" \
  --inadam "${ADAM_DIR}" \
  --out "${TFL_DIR}"

echo "================================================="
echo "ALL DONE!"
echo "RAW : ${RAW_DIR}"
echo "SDTM: ${SDTM_DIR}"
echo "ADaM: ${ADAM_DIR}"
echo "TFL : ${TFL_DIR}"
echo "================================================="
