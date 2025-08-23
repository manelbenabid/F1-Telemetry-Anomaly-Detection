import os, yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

CFG = yaml.safe_load(open("config.yaml", "r"))
SEASON = CFG["season"]; OUT_DIR = CFG["out_dir"]; EVENTS = CFG["events"]
DEFAULT_DRIVER = "VER"  # Max Verstappen by default

st.set_page_config(page_title="F1 Telemetry Anomaly Detection — Max Verstappen (VER)", layout="wide")
# Make the sidebar narrow (responsive) – place near the top of app_streamlit_ml.py
st.markdown("""
<style>
/* Use a responsive width: min 220px, ~18% of viewport, cap at 300px */
:root { --sbw: clamp(220px, 18vw, 300px); }

/* Sidebar container */
[data-testid="stSidebar"] {
  width: var(--sbw) !important;
  min-width: var(--sbw) !important;
  max-width: var(--sbw) !important;
}

/* Optional: tighten inner padding a bit */
[data-testid="stSidebar"] > div {
  padding-right: 10px;
}

/* Make the main area take the rest of the space cleanly */
[data-testid="stSidebar"] + section {
  flex: 1 1 auto;
}
</style>
""", unsafe_allow_html=True)


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
st.caption("Driver: **Max Verstappen (VER)**")

with st.sidebar:
    st.header("Controls")
    event = st.selectbox("Event", EVENTS, index=min(1, len(EVENTS)-1))
    # Driver is fixed to VER for this tailored build, but we'll show it for clarity
    st.text_input("Driver code", DEFAULT_DRIVER, disabled=True, help="Tailored build for Max Verstappen (VER).")

def ev_dir(ev: str) -> str:
    return os.path.join(OUT_DIR, f"{SEASON}_{ev.replace(' ', '_')}")

def paths(ev: str):
    base = ev_dir(ev)
    drv = DEFAULT_DRIVER
    return {
        "test_csv": os.path.join(base, f"{drv}_if_scores_test.csv"),
        "inj_csv":  os.path.join(base, f"{drv}_if_scores_injected.csv"),
        "ts_test":  os.path.join(base, f"{drv}_if_timeseries_test.parquet"),
        "ts_inj":   os.path.join(base, f"{drv}_if_timeseries_injected.parquet"),
    }

P = paths(event)

def load_csv(p):  return pd.read_csv(p)  if os.path.exists(p)  else None
def load_pq(p):   return pd.read_parquet(p) if os.path.exists(p) else None

test_scores = load_csv(P["test_csv"]); inj_scores = load_csv(P["inj_csv"])
if test_scores is None or inj_scores is None:
    miss = [k for k,v in [("test",P["test_csv"]),("injected",P["inj_csv"])] if not os.path.exists(v)]
    st.warning("Missing: " + ", ".join(miss) + f". Run train_and_plot_ml.py for **{event}** / VER.")
    st.stop()

st.info(
    "**What am I comparing?**  \n"
    "- **Test** = Isolation Forest anomaly score (mean) per **Lap × Sector** from real telemetry (VER).  \n"
    "- **Injected** = same metric after a synthetic fault on a representative lap.  \n"
    "- **Δ IF** = `Injected mean IF − Test mean IF` (how much the fault increases anomaly)."
)

def mean0(x):
    try: return float(np.nanmean(x))
    except: return float("nan")
c0,c1,c2,c3 = st.columns(4)
with c0: st.metric("Rows (Test)", len(test_scores))
with c1: st.metric("Rows (Injected)", len(inj_scores))
with c2: st.metric("Avg IF (Test)", f"{mean0(test_scores.get('IF_sector_mean', pd.Series([]))):.4f}")
with c3: st.metric("Avg IF (Injected)", f"{mean0(inj_scores.get('IF_sector_mean', pd.Series([]))):.4f}")

tab_over, tab_dist, tab_heat, tab_trend, tab_lap = st.tabs(
    ["Overview", "IF Distribution", "Lap×Sector Heatmap", "Per-sector Trends", "Lap Trace"]
)

with tab_over:
    a,b = st.columns(2)
    with a:
        st.subheader("Test (no injected fault) — VER")
        st.dataframe(test_scores.head(300), use_container_width=True)
    with b:
        st.subheader("Injected faults (demo) — VER")
        st.dataframe(inj_scores.head(300), use_container_width=True)

with tab_dist:
    st.subheader("Isolation Forest Score Distribution — VER")
    fig = px.histogram(
        pd.concat([test_scores.assign(Set="Test"), inj_scores.assign(Set="Injected")], ignore_index=True),
        x="IF_sector_mean", color="Set", barmode="overlay",
        color_discrete_map={"Test": RB_BLUE, "Injected": RB_RED},
        template="plotly_dark", nbins=60
    )
    fig.update_traces(opacity=0.65)
    fig.update_layout(xaxis_title="IF sector mean (↑ = more anomalous)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with tab_heat:
    st.subheader("Lap × Sector – anomaly score — VER")
    mode = st.radio("Show", ["Test", "Injected", "Δ IF (Injected − Test)"], horizontal=True)
    t = test_scores.pivot(index="LapNumber", columns="SectorID", values="IF_sector_mean")
    i = inj_scores.pivot(index="LapNumber", columns="SectorID", values="IF_sector_mean")
    if mode == "Test":
        grid = t; title = "Test IF (mean per Lap×Sector) — VER"
    elif mode == "Injected":
        grid = i; title = "Injected IF (mean per Lap×Sector) — VER"
    else:
        grid = i - t; title = "Δ IF (Injected − Test) per Lap×Sector — VER"
    fig = px.imshow(
        grid.sort_index(), color_continuous_scale=[[0,RB_NAVY],[1,RB_RED]],
        aspect="auto", origin="lower", template="plotly_dark"
    )
    fig.update_layout(title=title, xaxis_title="SectorID", yaxis_title="LapNumber", height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab_trend:
    st.subheader("Per-sector anomaly across laps — VER")
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
    st.subheader("Interactive lap trace (Speed/Throttle colored by IF) — VER")
    ts_test = load_pq(P["ts_test"]); ts_inj = load_pq(P["ts_inj"])
    if ts_test is None or ts_inj is None:
        st.info("Time-series files not found. Re-run train_and_plot_ml.py after enabling parquet saves.")
    else:
        laps = sorted(pd.unique(ts_test["LapNumber"].dropna()))
        if laps:
            lap = st.select_slider("Lap (test) — VER", options=laps, value=laps[0])

            # Choose injected variant
            all_variants = ["All variants"]
            if "Variant" in ts_inj.columns and not ts_inj["Variant"].dropna().empty:
                all_variants += sorted(ts_inj["Variant"].dropna().unique().tolist())
            variant_pick = st.selectbox("Injected variant", all_variants, index=0)

            c1, c2 = st.columns(2)

            # TEST trace (clean)
            sub_t = ts_test[ts_test["LapNumber"] == lap].sort_values("t_s")
            fig_t = px.scatter(
                sub_t, x="t_s", y="Speed", color="IF_score",
                color_continuous_scale="Turbo", template="plotly_dark",
                title="Test: Speed vs time (colored by IF) — VER"
            )
            fig_t.update_traces(mode="lines+markers", marker=dict(size=4))
            c1.plotly_chart(fig_t, use_container_width=True)

            # INJECTED trace (choose variant or show all as markers)
            sub_i = ts_inj[ts_inj["LapNumber"] == lap].copy()
            if variant_pick != "All variants":
                sub_i = sub_i[sub_i["Variant"] == variant_pick].sort_values("t_s")
                fig_i = px.scatter(
                    sub_i, x="t_s", y="Speed", color="IF_score",
                    color_continuous_scale="Turbo", template="plotly_dark",
                    title=f"Injected ({variant_pick}): Speed vs time — VER"
                )
                fig_i.update_traces(mode="lines+markers", marker=dict(size=4))
            else:
                # avoid lines connecting different variants
                fig_i = px.scatter(
                    sub_i.sort_values(["Variant","t_s"]),
                    x="t_s", y="Speed", color="Variant",
                    template="plotly_dark",
                    title="Injected (all variants): Speed vs time — VER"
                )
                fig_i.update_traces(mode="markers", marker=dict(size=4))

            c2.plotly_chart(fig_i, use_container_width=True)

            # Optional throttle overlay (test)
            if "Throttle01" in sub_t.columns:
                fig_th = px.line(sub_t, x="t_s", y="Throttle01",
                                 template="plotly_dark", title="Throttle (0–1) — VER (Test)")
                st.plotly_chart(fig_th, use_container_width=True)
