# ==============================================================================
# AUTO-CATALYST 2.0: EXECUTIVE PROJECT SHOWCASE & PRESENTATION GUIDE
# Autonomous AI Discovery Platform for Cleaner, Higher-Value Petrochemicals
# ==============================================================================

## 1. The Core Pitch (30-Second Elevator Pitch)
"Auto-Catalyst 2.0 is an autonomous, closed-loop materials discovery platform that invents, validates, and evaluates next-generation petrochemical catalysts. By integrating **3D zeolite crystallography**, **10-lump non-isothermal kinetics**, **BoTorch Bayesian Active Learning**, and **real-time GC-MS deconvolution**, it breaks the historic refinery trade-off between gasoline yield and octane—delivering **+$2.57/barrel net profit lift** while tracking **Scope 1 CO2 emissions** under global ETS carbon tax mandates."

---

## 2. Key Technical Innovations (The "Why It's Different" Slide)

| Component | Industry Baseline | Auto-Catalyst 2.0 Innovation |
| :--- | :--- | :--- |
| **Active Learning AI** | Trial-and-error / Static regression | **BoTorch Log q-NEHVI:** Probabilistic Gaussian Processes with epistemic uncertainty quantification; parallel batch generation ($q=2$). |
| **Crystallography** | Black-box empirical names | **3D IZA Frameworks (FAU/MFI/BEA):** Calculates steric pore-limiting diameter (PLD) and proves size-exclusion of heavy aromatics. |
| **Framework Chemistry** | Arbitrary parameter sliders | **Breck-Flanigen Stoichiometry (ASTM D3942):** Directly calculates framework aluminum atoms ($N_{Al}$) and Hydrogen Transfer ($K_{HT} \propto N_{Al}^2$). |
| **Kinetic Digital Twin** | Academic 4-lump ODEs | **10-Lump Petrochemical Network:** Discretely separates high-value Propylene ($C_3^=$) and Butylenes ($C_4^=$) with non-isothermal endothermic cooling. |
| **Mechanical Integrity** | Ignored in software simulations | **Davison Attrition Index (DAI, ASTM D5757):** Mechanical hardness constraint ($\text{DAI} \le 6.5$) to prevent cyclone pulverization. |
| **Refinery Circuit** | Riser-only isolated model | **Regenerator Combustion & Steaming Loop:** Exothermic dense-bed combustion ($+245^\circ\text{C}$) and 50-cycle hydrothermal aging (ASTM D4463). |
| **Unit Economics** | Chemical yields only | **Full TEA & Carbon Accounting:** Direct translation into gross margins ($/bbl), annual EBITDA, and Scope 1 flue gas $\text{CO}_2$ ($kg/\text{bbl}$). |

---

## 3. Provenance & Benchmark Performance Results

- **Winning Catalyst Recipe:** $76.4\text{ wt\%}$ Dealuminated USY ($a_0 = 24.265\,\text{\AA}, N_{Al} = 8.53\,\text{Al/uc}$) $+ 23.6\text{ wt\%}$ ZSM-5 Additive.
- **Product Slate Output:**
  - Gasoline Liquid Cut: **$42.86\text{ wt\%}$** at **$98.9\text{ RON}$**
  - Petrochemical Propylene ($C_3^=$): **$7.93\text{ wt\%}$** (Polymer grade)
  - Butylenes / Alkylate Cut ($C_4^=$): **$6.95\text{ wt\%}$**
  - Coke Deposit: **$2.16\text{ wt\%}$**
  - Davison Attrition Hardness: **$\text{DAI} = 5.70$** (Passes ASTM spec $\le 6.5$)
- **Refinery Financial Impact (40,000 bpd Unit):**
  - Net Margin: **$+\$2.57\text{ / barrel}$**
  - Projected Unit EBITDA: **$\$35.96\text{ Million USD / year}$**
  - Scope 1 Carbon Intensity: **$23.93\text{ kg CO}_2\text{ / bbl}$** ($\$1.56/\text{bbl}$ ETS tax debit)

---

## 4. Live Demo Commands for Evaluators

1. **Run the One-Click Fast System Test Suite (< 15 seconds):**
   ```powershell
   python test_system_demo.py
   ```
2. **Execute an Autonomous Bayesian Active Learning Discovery Cycle:**
   ```powershell
   python run_sota_autonomous_campaign.py
   ```
3. **Launch the 7-Tab Interactive SOTA Web Dashboard:**
   ```powershell
   streamlit run src/app.py
   ```
   *(Presents interactive 3D Pareto frontier, 3D CIF crystallography, 10-lump dynamic curves, regenerator aging, FID chromatograms, TEA economics sliders, and executive report downloads).*

---
*Created for Auto-Catalyst 2.0 demonstration & defense.*
