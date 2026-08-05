---
title: Jet Engine Hospital
emoji: ✈️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# Jet Engine Hospital -- NASA C-MAPSS FD001 Prognostics Dashboard

Multi-task early-warning system for turbofan engine maintenance decisions, built for the
Machine Learning capstone (Shahid Beheshti University, Spring 2026).

Combines:
- **RUL regression** (Gradient Boosting on causal window features), with split-conformal
  90% prediction intervals
- **Failure-horizon classification** (h = 10, 20, 30 cycles), Platt-calibrated where it helps
- **Unsupervised anomaly detection** (Isolation Forest + LOF + One-Class SVM ensemble,
  fit only on healthy early-life data, zero failure labels)
- A traceable **CONTINUE / INSPECT / STOP** recommendation layer that shows its reasoning
  and flags disagreement between signals rather than silently resolving it

All models are pre-trained and exported to `artifacts/prognostics_bundle.pkl` from the
project notebook -- this app performs inference only, never retraining, so outputs here
match the notebook exactly for the same engine/cycle.

Select an engine and scrub through its cycle history with the slider to see how the
recommendation evolves as the engine approaches failure.
