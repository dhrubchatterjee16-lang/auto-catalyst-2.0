# Auto-Catalyst 2.0: Autonomous AI Petrochemical Discovery Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![BoTorch Multi-Fidelity](https://img.shields.io/badge/Active%20Learning-BoTorch%20MF--GP-orange.svg)](https://botorch.org/)
[![Status: Frozen v2.0-LTS](https://img.shields.io/badge/Release-v2.0--LTS%20Frozen-success.svg)](ARCHITECTURAL_FREEZE_SPEC.md)

An autonomous, closed-loop discovery and optimization system for Fluid Catalytic Cracking (FCC) and Petrochemical Catalysts. **Auto-Catalyst 2.0** unifies multi-fidelity Bayesian active learning, 3D zeolite crystallography, hydrothermal dealumination aging, conserved 10-lump kinetic twins, ASTM quality gates, Opentrons robotic synthesis, online GC-MS deconvolution, and refinery techno-economic analysis into an auditable closed loop.

---

## ⚡ Master Closed-Loop Architecture

```
                  [ 1. Multi-Fidelity Bayesian Acquisition (BoTorch q-NEHVI) ]
                   Optimizes Low-Fidelity ODE (s=0.0) vs High-Fidelity ACE (s=1.0)
                                            │
                                            ▼
                  [ 2. Hydrothermal Aging to Equilibrium Catalyst (E-Cat) ]
                   ASTM D4463 Steaming (760°C) -> Locks UCS (24.24Å) & Si/Al (31.05)
                                            │
                                            ▼
                  [ 3. Intermediate ASTM Characterization Quality Gates ]
                   ASTM D3906 XRD Crystallinity (≥70%) & ASTM D3663 BET SA (≥350 m²/g)
                                            │
                                            ▼
                  [ 4. Robotic Slurry Synthesis Protocol Generation ]
                   Opentrons OT-2 API v2.14 Python Protocols with Multi-Aspirate Slurries
                                            │
                                            ▼
                  [ 5. Conserved 10-Lump Petrochemical Riser Digital Twin ]
                   Non-Isothermal Radau Stiff ODE with Ni, V & Basic-N Poisons + WGC Limits
                                            │
                                            ▼
                  [ 6. Online GC-MS Spectral Deconvolution & Refinery TEA ]
                   Continuous FID Peak Integration, ASTM PIONA Octane (RON), & CO2 Tax ($65/t)
                                            │
                                            ▼
                  [ 7. Multi-Fidelity Dataset Feedback & Posterior Updating ]
                   Dynamic Retraining of SingleTaskMultiFidelityGP with Kennedy-O'Hagan Model
```

---

## 🧪 Scientific & Engineering Invariant Guarantees

| Subsystem | Standard / Equation | Enforced Invariant | Verification Script |
| :--- | :--- | :--- | :--- |
| **Crystallography** | Breck-Flanigen Stoichiometry | $N_{\text{Al}} = 107.1 \cdot (a_0 - 24.238)$, $\text{Si/Al} = (192 - N_{\text{Al}}) / N_{\text{Al}}$ | `industrial_suite_verify.py` |
| **Zeolite Topology** | Loewenstein's Rule | Maximum $N_{\text{Al}} \le 96.0$ Al/uc (no adjacent Al-O-Al tetrahedra) | `src/models/zeolite_crystallography.py` |
| **Equilibrium Catalyst** | Hydrothermal Steaming | Dealumination shifts fresh USY ($24.30\text{--}24.50\,\text{Å}$) to E-cat ($24.24\text{--}24.26\,\text{Å}$) | `src/simulation/regenerator_aging_loop.py` |
| **Mechanical Strength** | Davison Attrition Index | ASTM D5757 attrition velocity $\le 6.5\,\text{wt\%/h}$ | `industrial_suite_verify.py` |
| **Purity Screening** | Powder XRD Gate | ASTM D3906 relative crystallinity $\ge 70.0\%$ | `src/analysis/instrument_data_ingestion.py` |
| **Textural Properties** | $\text{N}_2$ Physisorption Gate | ASTM D3663 total BET surface area $\ge 350.0\,\text{m}^2/\text{g}$ | `src/analysis/instrument_data_ingestion.py` |
| **Pilot Activity** | Small Fixed-Bed ACE Unit | ASTM D7964 mass balance closure $97.5\% \le \sum Y_i \le 102.5\%$ | `src/analysis/ace_pilot_ingestor.py` |
| **Plant Hardware** | Wet Gas Compressor (WGC) | Volumetric suction flow ceiling $\le 38,000\,\text{Nm}^3/\text{h}$ | `src/simulation/industrial_10lump_conserved.py` |
| **Carbon Taxation** | EU ETS Carbon Accounting | $\$65.00/\text{ton } \text{CO}_2$ direct debit from gross refinery crack margins | `src/analysis/techno_economics.py` |

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/dhrubchatterjee16-lang/auto-catalyst-2.0.git
cd auto-catalyst-2.0
pip install -r requirements.txt
```

### 2. Run the SOTA Closed-Loop Campaign CLI

Execute the unified 7-stage master discovery campaign across multiple cycles and feedstocks:
```bash
# Baseline campaign on Standard Arabian Light VGO
python run_sota_autonomous_campaign.py --cycles 1 --batch-size 2

# Stress-test under Severe Resid / DAO blend (4.8 ppm Ni, 9.5 ppm V, 1750 ppm Basic N)
python run_sota_autonomous_campaign.py --cycles 1 --batch-size 2 --feedstock severe_resid_dao_blend
```

### 3. Launch the Interactive 8-Tab Web Control Center

Launch the full Streamlit operations dashboard:
```bash
streamlit run src/app.py
```
- **Tab 1: Multi-Fidelity AI Discovery** (BoTorch SingleTaskMultiFidelityGP Pareto frontier)
- **Tab 2: Zeolite Crystallography** (3D unit cell rendering & pore steric exclusion)
- **Tab 3: Regenerator Aging & E-Cat** (ASTM D4463 hydrothermal dealumination curves)
- **Tab 4: Conserved 10-Lump Petrochemical Twin** (Non-isothermal stiff ODE riser dynamics)
- **Tab 5: Robotic Synthesis Planner** (Opentrons OT-2 slurry protocol generator)
- **Tab 6: Online GC-MS Spectral Octane** (FID peak integration & ASTM PIONA deconvolution)
- **Tab 7: Refinery TEA & Carbon Tax** (Scope 1 CO2 carbon taxation & EBITDA margins)
- **Tab 8: Autonomous Loop Verification** (End-to-end provenance logs & diagnostics)

### 4. Ingest Real Kayser ACE / MAT Pilot Reactor Logs

Ingest external raw pilot run logs to calibrate the Kennedy-O'Hagan discrepancy model:
```bash
python scripts/ingest_real_ace_pilot.py data/benchmarks/kayser_ace_pilot_run_sample.csv
```

### 5. Run Verification & Regression Suites

```bash
# Run full system demo suite (6 tests)
python test_system_demo.py

# Run industrial standards validation suite (5 tests)
python industrial_suite_verify.py
```

---

## 📁 Repository Structure

```
auto-catalyst-2.0/
├── ARCHITECTURAL_FREEZE_SPEC.md   # Official v2.0-LTS architectural freeze specification
├── README.md                      # Platform documentation
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT Open Source License
├── test_system_demo.py            # End-to-end regression test suite
├── industrial_suite_verify.py     # ASTM and stoichiometric verification suite
├── run_sota_autonomous_campaign.py# Master closed-loop orchestrator CLI
├── scripts/
│   └── ingest_real_ace_pilot.py   # Turnkey CLI for Kayser ACE pilot log ingestion
├── data/
│   ├── feedstock_library.json     # Commercial feed library with metal & nitrogen poisons
│   ├── benchmarks/
│   │   └── kayser_ace_pilot_run_sample.csv # ASTM D7964 pilot-scale benchmark data
│   └── multi_fidelity_active_learning_history.json # Active learning dataset
└── src/
    ├── app.py                     # Streamlit 8-Tab Web Control Center
    ├── models/                    # Active learning & crystallography
    │   ├── multi_fidelity_active_optimizer.py # BoTorch Multi-Fidelity GP
    │   ├── zeolite_crystallography.py         # 3D IZA CIF & pore geometry
    │   └── schema.py                          # Pydantic data schemas
    ├── simulation/                # Chemical reactors & aging
    │   ├── industrial_10lump_conserved.py     # Mass-conserved 10-lump twin
    │   └── regenerator_aging_loop.py          # E-cat dealumination aging
    ├── automation/                # Robotics & wet-lab
    │   └── synthesis_planner.py               # Opentrons OT-2 protocol planner
    └── analysis/                  # Analytics & economics
        ├── gcms_analyzer.py                   # Continuous GC-MS spectral engine
        ├── instrument_data_ingestion.py       # ASTM XRD & BET characterization gates
        ├── ace_pilot_ingestor.py              # ACE pilot ingestion & discrepancy
        └── techno_economics.py                # Refinery EBITDA & CO2 carbon tax
```

---

## 📜 Governance & Architectural Freeze

Auto-Catalyst 2.0 is released under the **v2.0-LTS Architectural Code Freeze**. For full technical audit logs, stoichiometric derivations, and change control policies, see [`ARCHITECTURAL_FREEZE_SPEC.md`](ARCHITECTURAL_FREEZE_SPEC.md).

---

## 📄 License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.
