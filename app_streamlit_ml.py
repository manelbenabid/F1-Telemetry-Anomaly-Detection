import os, io, yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

CFG = yaml.safe_load(open("config.yaml", "r"))
SEASON = CFG["season"]; OUT_DIR = CFG["out_dir"]; EVENTS = CFG["events"]
DEFAULT_DRIVER = CFG.get("driver", "VER")

st.set_page_config(page_title="F1 Telemetry Anomaly Detection", layout="wide")

RB_NAVY = "#001F3F"; RB_RED = "#D0021B"; RB_YELL = "#FFCC00"; RB_BLUE = "#4A90E2"
st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f2b; color: #ffffff; }
    h1,h2,h3,h4 { color: #ffffff; }
    .metric-label { color: #9aa3b2 !important; }
    .metric-value { color: #ffffff !important; }
    .stMarkdown p { color: #d9dde7; }
    </style>
    """, unsafe_allow_html=True
)
st.title("F1 Telemetry Anomaly Detection")

with st.sidebar:
    st.header("Controls")
    event = st.selectbox("Event", EVENTS, index=min(1, len(EVENTS)-1))
    driver = st.text_input("Driver code", DEFAULT_DRIVER).strip().upper()
    st.caption("Use the code used in filenames (e.g., VER, LEC, HAM).")

def ev_dir(ev: str) -> str:
    return os.path.join(OUT_DIR, f"{SEASON}_{ev.replace(' ', '_')}")

def paths(ev: str, drv: str):
    base = ev_dir(ev)
    return {
        "test_csv": os.path.join(base, f"{drv}_if_scores_test.csv"),
        "inj_csv":  os.path.join(base, f"{drv}_if_scores_injected.csv"),
        "ts_test":  os.path.join(base, f"{drv}_if_timeseries_test.parquet"),
        "ts_inj":   os.path.join(base, f"{drv}_if_timeseries_injected.parquet"),
    }

P = paths(event, driver)

def load_csv(p):  return pd.read_csv(p)  if os.path.exists(p)  else None
def load_pq(p):   return pd.read_parquet(p) if os.path.exists(p) else None

test_scores = load_csv(P["test_csv"]); inj_scores = load_csv(P["inj_csv"])
if test_scores is None or inj_scores is None:
    miss = [k for k,v in [("test",P["test_csv"]),("injected",P["inj_csv"])] if not os.path.exists(v)]
    st.warning("Missing: " + ", ".join(miss) + f". Run train_and_plot_ml.py for **{event}** / **{driver}**.")
    st.stop()

st.info(
    "**What am I comparing?**  \n"
    "- **Test** = Isolation Forest anomaly score (mean) per **Lap × Sector** from real telemetry.  \n"
    "- **Injected** = same metric after a synthetic fault on a representative lap.  \n"
    "- **Δ IF** = `Injected mean IF − Test mean IF` (how much the fault increases anomaly)."
)

# KPIs
def mean0(x): 
    try: return float(np.nanmean(x))
    except: return float("nan")
c0,c1,c2,c3 = st.columns(4)
with c0: st.metric("Rows (Test)", len(test_scores))
with c1: st.metric("Rows (Injected)", len(inj_scores))
with c2: st.metric("Avg IF (Test)", f"{mean0(test_scores.get('IF_sector_mean', pd.Series([]))):.4f}")
with c3: st.metric("Avg IF (Injected)", f"{mean0(inj_scores.get('IF_sector_mean', pd.Series([]))):.4f}")

# TABS
tab_over, tab_dist, tab_heat, tab_trend, tab_lap = st.tabs(
    ["Overview", "IF Distribution", "Lap×Sector Heatmap", "Per-sector Trends", "Lap Trace"]
)

with tab_over:
    a,b = st.columns(2)
    with a:
        st.subheader("Test (no injected fault)")
        st.dataframe(test_scores.head(300), use_container_width=True)
    with b:
        st.subheader("Injected faults (demo)")
        st.dataframe(inj_scores.head(300), use_container_width=True)

with tab_dist:
    st.subheader("Isolation Forest Score Distribution")
    fig = px.histogram(
        pd.concat([
            test_scores.assign(Set="Test"),
            inj_scores.assign(Set="Injected")
        ], ignore_index=True),
        x="IF_sector_mean", color="Set", barmode="overlay",
        color_discrete_map={"Test": RB_BLUE, "Injected": RB_RED},
        template="plotly_dark", nbins=60
    )
    fig.update_traces(opacity=0.65)
    fig.update_layout(xaxis_title="IF sector mean (↑ = more anomalous)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with tab_heat:
    st.subheader("Lap × Sector – anomaly score")
    mode = st.radio("Show", ["Test", "Injected", "Δ IF (Injected − Test)"], horizontal=True)
    t = test_scores.pivot(index="LapNumber", columns="SectorID", values="IF_sector_mean")
    i = inj_scores.pivot(index="LapNumber", columns="SectorID", values="IF_sector_mean")
    if mode == "Test":
        grid = t
        title = "Test IF (mean per Lap×Sector)"
    elif mode == "Injected":
        grid = i
        title = "Injected IF (mean per Lap×Sector)"
    else:
        grid = i - t
        title = "Δ IF (Injected − Test) per Lap×Sector"
    fig = px.imshow(
        grid.sort_index(),
        color_continuous_scale=[[0,RB_NAVY],[1,RB_RED]],
        aspect="auto", origin="lower", template="plotly_dark"
    )
    fig.update_layout(title=title, xaxis_title="SectorID", yaxis_title="LapNumber", height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab_trend:
    st.subheader("Per-sector anomaly across laps")
    # melt and label
    mtest = test_scores.melt(id_vars=["LapNumber","SectorID"], value_vars=["IF_sector_mean"], var_name="Metric", value_name="IF")
    minj  = inj_scores.melt(id_vars=["LapNumber","SectorID"], value_vars=["IF_sector_mean"], var_name="Metric", value_name="IF")
    mtest["Set"] = "Test"; minj["Set"] = "Injected"
    data = pd.concat([mtest,minj], ignore_index=True)
    fig = px.line(
        data.sort_values(["SectorID","LapNumber"]),
        x="LapNumber", y="IF", color="Set", facet_col="SectorID", facet_col_wrap=3,
        color_discrete_map={"Test": RB_BLUE, "Injected": RB_RED},
        template="plotly_dark"
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab_lap:
    st.subheader("Interactive lap trace (Speed/Throttle colored by IF)")
    ts_test = load_pq(P["ts_test"]); ts_inj = load_pq(P["ts_inj"])
    if ts_test is None or ts_inj is None:
        st.info("Time-series files not found. Re-run train_and_plot_ml.py after adding the parquet saves.")
    else:
        laps = sorted(pd.unique(ts_test["LapNumber"].dropna()))
        lap = st.select_slider("Lap (test)", options=laps, value=laps[0])
        c1, c2 = st.columns(2)
        for label, df, col in [("Test", ts_test, RB_BLUE), ("Injected", ts_inj, RB_RED)]:
            sub = df[df["LapNumber"] == lap]
            if sub.empty: 
                continue
            # Speed trace colored by IF
            fig = px.scatter(
                sub, x="t_s", y="Speed", color="IF_score", color_continuous_scale="Turbo",
                title=f"{label}: Speed vs time (colored by IF)", template="plotly_dark"
            )
            fig.update_traces(mode="lines+markers", marker=dict(size=4))
            (c1 if label=="Test" else c2).plotly_chart(fig, use_container_width=True)
        # optional throttle overlay for test
        if "Throttle01" in ts_test.columns:
            sub = ts_test[ts_test["LapNumber"] == lap]
            fig = px.line(sub, x="t_s", y="Throttle01", title="Throttle (0–1)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
