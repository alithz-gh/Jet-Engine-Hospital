"""
Jet Engine Hospital -- FD001 prognostics dashboard.
Loads exported artifacts only; never retrains. The inference path here is
IDENTICAL to the notebook's PrognosticsSystem, so app outputs reproduce
notebook outputs exactly for the same engine/cycle.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prognostics_system import PrognosticsSystem

st.set_page_config(page_title="Jet Engine Hospital", layout="wide")

@st.cache_resource
def load_system():
    return PrognosticsSystem(bundle_path='artifacts/prognostics_bundle.pkl')

@st.cache_data
def load_demo_data():
    # A small bundled slice of FD001 validation engines for demo purposes.
    # In a full deployment this would be replaced by a live data feed per engine.
    cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv('demo_engines.csv')
    return df

sys_ = load_system()
demo_df = load_demo_data()

st.title("Jet Engine Hospital")
st.caption("NASA C-MAPSS FD001 -- multi-task early-warning dashboard (RUL + failure risk + anomaly detection)")

col_select, col_cycle = st.columns([1, 2])
with col_select:
    engine_id = st.selectbox("Select engine", sorted(demo_df.engine_id.unique()))

engine_full = demo_df[demo_df.engine_id == engine_id].sort_values('cycle').reset_index(drop=True)
max_cycle = int(engine_full['cycle'].max())

with col_cycle:
    current_cycle = st.slider("Current cycle", min_value=10, max_value=max_cycle, value=max_cycle // 2)

partial_history = engine_full[engine_full['cycle'] <= current_cycle]

# Recompute full alert history up to current_cycle for correct persistence evaluation
h10_hist, h20_hist, h30_hist, anom_hist = [], [], [], []
for t in range(10, current_cycle + 1):
    sub = engine_full[engine_full['cycle'] <= t]
    feats_t = sys_.compute_features(sub)
    last_t = feats_t.iloc[[-1]]
    risk_t = sys_.failure_risk(last_t)
    anom_t = sys_.anomaly_score(last_t)
    h10_hist.append(risk_t[10]['alert'])
    h20_hist.append(risk_t[20]['alert'])
    h30_hist.append(risk_t[30]['alert'])
    anom_hist.append(anom_t['alert'])

feats = sys_.compute_features(partial_history)
last_row = feats.iloc[[-1]]
rul_out = sys_.predict_rul(last_row)
risk_out = sys_.failure_risk(last_row)
anom_out = sys_.anomaly_score(last_row)
rec = sys_.recommend(rul_out, risk_out, anom_out, h10_hist, h20_hist, h30_hist, anom_hist)

# --- Engine timeline ---
st.subheader("Engine timeline")
fig = go.Figure()
fig.add_trace(go.Scatter(x=engine_full['cycle'], y=engine_full['sensor_11'],
                          mode='lines', name='sensor_11 (HPC outlet static pressure)', line=dict(color='gray')))
fig.add_vline(x=current_cycle, line_dash="dash", line_color="crimson", annotation_text="current cycle")
fig.update_layout(height=280, xaxis_title="cycle", yaxis_title="sensor_11 (raw)")
st.plotly_chart(fig, use_container_width=True)

# --- Cards ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("RUL estimate", f"{rul_out['point']:.0f} cycles",
               help="Point estimate from Gradient Boosting (window features)")
    st.caption(f"90% interval: [{rul_out['lower']:.0f}, {rul_out['upper']:.0f}] cycles (split conformal)")

with c2:
    st.write("**Failure risk**")
    for h in [10, 20, 30]:
        r = risk_out[h]
        flag = "🔴" if r['alert'] else "🟢"
        st.write(f"{flag} h={h}: {r['probability']:.1%} (threshold {r['threshold']:.0%})")

with c3:
    st.write("**Anomaly**")
    st.metric("Percentile", f"{anom_out['combined_percentile']:.0f}",
               delta=f"{anom_out['margin']:+.0f} vs threshold")
    st.caption("Ensemble of Isolation Forest, LOF, One-Class SVM (validation-calibrated rank, not a probability)")

with c4:
    color = {'CONTINUE': 'green', 'INSPECT': 'orange', 'STOP': 'red'}[rec['level']]
    st.write("**Recommendation**")
    st.markdown(f"### :{color}[{rec['level']}]")
    for reason in rec['reasons']:
        st.caption(f"- {reason}")
    if rec['disagreement']:
        st.warning("Signals disagree -- shown explicitly rather than resolved silently.")

# --- Model metadata ---
with st.expander("Model metadata"):
    st.json(sys_.metadata)
