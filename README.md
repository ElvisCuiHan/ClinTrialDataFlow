<div align="center">

<!-- Replace with your logo once you have it -->
<!-- <img src="assets/logo.png" alt="ClinTrialDataFlow logo" width="140" /> -->

# ClinTrialDataFlow

**End-to-end clinical trial data simulation: EDC → SDTM → ADaM → TFL**  
with realistic imperfections and interactive workflows.

<br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-informational)

</div>

---

## Table of Contents

- [Why ClinTrialDataFlow](#why-clintrialdataflow)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#-quick-start)
  - [Option A: Interactive Web App (Streamlit)](#option-a-interactive-web-app-streamlit)
  - [Option B: CLI Pipeline](#option-b-cli-pipeline)
- [Configuration](#configuration)
- [Outputs](#outputs)
  - [RAW / EDC-like](#raw--edc-like)
  - [SDTM](#sdtm)
  - [ADaM](#adam)
  - [TFL](#tfl)
- [Design Notes](#design-notes)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why ClinTrialDataFlow

ClinTrialDataFlow is a reproducible simulator designed to mimic real-world
pharmaceutical clinical data workflows:

> **EDC / Raw → SDTM → ADaM → TFL**

It is intended for **education, method prototyping, QC stress-testing,
and technical demonstrations**, while ensuring that **no private or real
clinical data are used**.

---

## Features

- 🔬 **EDC / Raw data simulation**
  - Subjects, visits, exposure, laboratory data, adverse events, vitals
  - Oncology essentials: **TU** (Tumor Identification) and **RS** (Response)
- 📐 **SDTM generation**
  - DM, SV, AE, LB, VS, **TU**, **RS**, EX, DS
  - Directory-based input/output for transparency and debugging
- 📊 **ADaM derivation**
  - ADSL, ADRS, ADSLRS (**BOR / BORUNC / BORC**)
  - ADTTE with PFS / OS–ready time-to-event variables
- 📈 **TFL outputs**
  - Baseline characteristics
  - ORR (CR+PR) with confidence intervals
  - Kaplan–Meier summaries and plots (PFS / OS)
- 🧪 **Realistic data imperfections**
  - Dropout, missing forms/items, and query-like inconsistencies
- 🌐 **Interactive web application**
  - Run the full pipeline in a browser, preview datasets, and download outputs

---

## Project Structure

```text
ClinTrialDataFlow/
├── app.py                 # Streamlit web app (interactive runner)
├── cfg.json               # Simulation configuration
├── Codes/
│   ├── EDCSimu.py         # EDC / Raw data simulation
│   ├── SDTMSimu.py        # Raw → SDTM
│   ├── ADaMSimu.py        # SDTM → ADaM
│   └── TFLSimu.py         # ADaM → TFL
└── Data/
   ├── raw_out/
   ├── sdtm_out/
   ├── adam_out/
   └── tfl_out/

## 🚀 Quick Start

### Requirements

```bash
# Core dependencies
pip install pandas numpy matplotlib

# Interactive web app
pip install streamlit

# Optional: exact binomial CI for ORR (Clopper–Pearson)
pip install scipy

### Option A: Interactive Web App (Streamlit)

```bash
streamlit run app.py
