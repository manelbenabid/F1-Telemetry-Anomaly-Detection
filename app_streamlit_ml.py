# streamlit run app_streamlit_ml.py
import os, yaml
import numpy as np
import pandas as pd
import streamlit as st

CFG = yaml.safe_load(open("config.yaml"))
SEASON, DRIVER = CFG["season"], CFG["driver"]
EVENTS = CFG["events"]
OUT_DIR = CFG["out_dir"]

st.set_page_config(page_title="F1 Telemetry Anomaly Detection", layout="wide")
st.title("F1 Telemetry Anomaly Detection")

event = st.selectbox("Event", EVENTS, index=min(1, len(EVENTS)-1))
ev_dir = os.path.join(OUT_DIR, f"{SEASON}_{event.replace(' ','_')}")
test_scores_p = os.path.join(ev_dir, f"{DRIVER}_if_scores_test.csv")
inj_scores_p  = os.path.join(ev_dir, f"{DRIVER}_if_scores_injected.csv")

if not (os.path.exists(test_scores_p) and os.path.exists(inj_scores_p)):
    st.warning("Run train_and_plot_ml.py first to generate IF scores.")
    st.stop()

test_scores = pd.read_csv(test_scores_p)
inj_scores  = pd.read_csv(inj_scores_p)

st.subheader("Sector anomaly scores (IF)")
c1, c2 = st.columns(2)
with c1:
    st.write("Test (no injected fault)")
    st.dataframe(test_scores.head(20))
with c2:
    st.write("Injected faults (demo)")
    st.dataframe(inj_scores.head(20))

# Compare sector means
t_means = test_scores.groupby("SectorID")["IF_sector_mean"].mean()
i_means = inj_scores.groupby("SectorID")["IF_sector_mean"].mean()
delta = (i_means - t_means).rename("Δ IF")

st.subheader("Δ IF by sector (Injected − Test)")
st.bar_chart(delta)
