# ArvyaX ML — Emotion & Guidance Engine

## Project Context
At ArvyaX, AI should understand human emotional states, reason under flawed signals, and decide meaningful next actions to guide users to better states. This codebase builds a completely optimized, offline-capable ML engine to interpret short, messy journal reflections and physiological signals (metadata).

**Core Pipeline Features:**
1. **Understanding:** Multi-class classification of Emotional State and ordinal regression for Intensity.
2. **Decision Layer:** Intelligent reasoning to recommend exactly `what_to_do` and `when_to_do`.
3. **Uncertainty Awareness:** Margin-based confidence tracking and flags for human-in-the-loop fallback.
4. **Bonuses Delivered:** 
    - Lightweight, offline **Supportive Message Generation**.
    - **Label Noise Filtering** during training (OOF Reweighting).

## Repository Deliverables
- `pipeline.py`: The single-script End-to-End training and inference engine.
- `predictions.csv`: The final required mappings for the 120 test inputs.
- `ERROR_ANALYSIS.md`: Detailed failure cases and insights on noisy labels vs conflicting signals.
- `EDGE_PLAN.md`: System design approach for on-device (ONNX) latency constraints.
- `error_analysis_cases.csv`: The raw validation split errors.

## Installation & Setup

Ensure you have a modern Python environment installed (Python 3.8+).

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost
```

## How to Run

1. Ensure your current directory is the root of the assignment and the `data/` folder exists.
2. **Train the Pipeline:**
```bash
python pipeline.py
```
This will train the models, run feature importance, generate the `predictions.csv`, and locally cache the model (`arvyax_model.joblib`) for the backend API.

3. **[BONUS] Launch the Local AI Dashboard (FastAPI & UI):**
```bash
# Ensure fastapi, uvicorn, jinja2, and python-multipart are installed
uvicorn app:app --reload
```
Open your browser and navigate to `http://localhost:8000`. You can interact with the live model, submit messy journal entries, toggle energy levels, and see the Decision Engine generate what to do and uncertainty flags in real-time!

## Methodology & Feature Engineering

### 1. Robustness to Messy Data
- **Text:** Processed using Scikit-Learn's `TfidfVectorizer` (ngram_range 1-2) which is far more robust to missing/vague terms while remaining mathematically fast enough for mobile deployment.
- **Metadata:** missing physiological signals (`sleep_hours`, `stress_level`) are gracefully imputed dynamically via Pipeline medians to not interrupt inference. 

### 2. Model Choice (XGBoost)
- **Part 1 (Classification):** `XGBClassifier` was selected due to its innate resilience to over-parameterization on tabular data, scaling natively to categorical inputs. It easily handles non-linear relationships, like differentiating "mixed" states from "calm" using stress levels.
- **Part 2 (Regression):** Predicting intensity (1-5) as a pure taxonomy strips the distance context. By using an `XGBRegressor`, we treat the ordinal distance mathematically. Predicting a 4 when the ground truth was a 5 incurs far less penalty than predicting a 1. The regression outputs are clipped and rounded back to standard int boundaries. 

### 3. Decision Engine Logic (What + When)
A deterministic decision tree governs actions. The rationale focuses on the principle of *gentle reset and offloading*:
- **Arousal Adjustments:** If a user is `restless` and `energy` is high (> 3), we recommend `movement` later today to burn off the energy. If energy is low (< 3), we recommend `journaling` to mentally offload the racing thoughts.
- **Triage Priority:** Severe negative states (`overwhelmed` or `restless` combined with an intensity >= 4) drastically bump the `when_to_do` action into the `now` category, prioritizing immediate intervention (`box_breathing`).
- **Productive Offloading:** If the user is `focused` but under high `stress`, we advise `light_planning` over heavy `deep_work` to prevent burnout.

### 4. Uncertainty Modeling
The flag leverages softmax probabilities. If the highest confidence probability falls below **45%** OR if the margin between the first and second prediction is critically thin (**< 15%**), `uncertain_flag` evaluates to 1. "A strong system knows when it is unsure". We flag these so the UI can gently ask for more information instead of giving blind, potentially harmful advice.

### 5. Ablation Study & Feature Importance
Our runtime pipeline proved:
- **Text** naturally dominates the global feature space (contributing roughly ~95% of splits). Important tokens include 'drained', 'organized', 'concentrate', etc.
- **Metadata** shines in disambiguating state vectors. 
- During ablation (`Text-Only Model` vs `Text + Metadata`), Accuracy slightly improved/stabilized upon adding Metadata, specifically helping separate difficult edge inputs where text was short (e.g. "ok") by observing extreme Stress or Energy values.
