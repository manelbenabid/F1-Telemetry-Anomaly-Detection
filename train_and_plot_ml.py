import os, sys, yaml, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from utils_ml import inject_faults

# ---------- Config ----------
CFG = yaml.safe_load(open("config.yaml", "r"))
SEASON, DRIVER = CFG["season"], CFG["driver"]
EVENTS = CFG["events"]
OUT_DIR = CFG["out_dir"]

def ev_dir(ev: str) -> str:
    return os.path.join(OUT_DIR, f"{SEASON}_{ev.replace(' ','_')}")

def load_feats(ev: str) -> pd.DataFrame:
    p = os.path.join(ev_dir(ev), f"{DRIVER}_telemetry_features.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing features for {ev}. Run preprocess_features_ml.py")
    return pd.read_parquet(p)

def savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def open_file(path: str):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess; subprocess.Popen(["open", path])
        else:
            import subprocess; subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"[warn] Could not auto-open '{path}': {e}")

if not EVENTS:
    raise SystemExit("No events in config.yaml")

# ----- 1) Fit baseline model on first event -----
baseline_ev = EVENTS[0]
print(f"[info] Baseline event: {baseline_ev}")
df_base = load_feats(baseline_ev)

FEATURES = [c for c in df_base.columns if c.endswith("_z") and c not in ("Gear_z","DRS_z","ERSDeployMode_z")]
X_base = df_base[FEATURES].fillna(0.0).values

iso = IsolationForest(
    n_estimators=300, max_samples="auto", contamination=0.02,
    random_state=42, n_jobs=-1
).fit(X_base)

df_base["IF_score"] = -iso.score_samples(X_base)  # higher = more anomalous
base_thr_98 = np.percentile(df_base["IF_score"], 98)

def agg_scores(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["LapNumber","SectorID"], dropna=False)["IF_score"].mean().reset_index()
    return g.rename(columns={"IF_score":"IF_sector_mean"})

# ----- 2) For EVERY event (including baseline): write test + injected + plots -----
for ev in EVENTS:
    print(f"[info] Processing event: {ev}")
    evd = ev_dir(ev); os.makedirs(evd, exist_ok=True)

    # 2a) Load & score event
    if ev == baseline_ev:
        df_ev = df_base.copy()
    else:
        df_ev = load_feats(ev)
        X_ev = df_ev[FEATURES].fillna(0.0).values
        df_ev["IF_score"] = -iso.score_samples(X_ev)

    # Save as "test" for this event (so Streamlit always finds it)
    agg_scores(df_ev).to_csv(os.path.join(evd, f"{DRIVER}_if_scores_test.csv"), index=False)

    # 2b) Fault injection on a representative lap
    if "LapNumber" in df_ev and not df_ev["LapNumber"].dropna().empty:
        lap_pick = int(df_ev["LapNumber"].mode().iloc[0])
        lap_df = df_ev[df_ev["LapNumber"] == lap_pick].reset_index(drop=True)
    else:
        lap_pick = None
        lap_df = df_ev.copy()

    inj1 = inject_faults(lap_df, fault="MGUK_DROP", magnitude=0.10, sector_filter={1})
    inj2 = inject_faults(lap_df, fault="THROTTLE_LAG", lag_seconds=0.2, sector_filter={2})
    inj_all = pd.concat([inj1.assign(Variant="MGUK_DROP"),
                         inj2.assign(Variant="THROTTLE_LAG")], ignore_index=True)
    X_inj = inj_all[[f for f in FEATURES if f in inj_all.columns]].fillna(0.0).values
    inj_all["IF_score"] = -iso.score_samples(X_inj)
    
    cols_keep = [c for c in ["t_s","LapNumber","SectorID","Speed","Throttle01","RPM","IF_score"] if c in df_ev.columns]
    df_ev[cols_keep].to_parquet(os.path.join(evd, f"{DRIVER}_if_timeseries_test.parquet"))
    
    cols_keep_inj = [c for c in ["t_s","LapNumber","SectorID","Speed","Throttle01","RPM","IF_score","Variant"] if c in inj_all.columns]
    inj_all[cols_keep_inj].to_parquet(os.path.join(evd, f"{DRIVER}_if_timeseries_injected.parquet"))

    agg_scores(inj_all).to_csv(os.path.join(evd, f"{DRIVER}_if_scores_injected.csv"), index=False)

    # 2c) Plots for this event (and auto-open)
    paths = []

    plt.figure(figsize=(9,5))
    plt.hist(df_base["IF_score"], bins=60, alpha=0.45, label=f"{baseline_ev} (baseline)")
    plt.hist(df_ev["IF_score"],   bins=60, alpha=0.6,  label=f"{ev} (test)")
    plt.hist(inj_all["IF_score"], bins=60, alpha=0.6,  label="Injected (demo)")
    plt.axvline(base_thr_98, color="r", linestyle="--", label="Baseline 98th pct")
    plt.xlabel("Isolation Forest anomaly score (↑ = more anomalous)")
    plt.ylabel("Count")
    plt.title(f"{DRIVER} – IF score distributions: {ev}")
    plt.legend()
    p1 = os.path.join(evd, "if_score_distributions.png")
    savefig(p1); paths.append(p1)

    pivot_t = agg_scores(df_ev).pivot(index="LapNumber", columns="SectorID", values="IF_sector_mean")
    pivot_i = agg_scores(inj_all).pivot(index="LapNumber", columns="SectorID", values="IF_sector_mean")
    common_cols = [c for c in pivot_t.columns if c in pivot_i.columns]
    if common_cols:
        delta = pivot_i[common_cols].mean(axis=0) - pivot_t[common_cols].mean(axis=0)
        plt.figure(figsize=(7,4))
        plt.bar([str(c) for c in common_cols], delta.values)
        plt.title(f"{DRIVER} – Δ IF score by Sector (Injected − Test) – {ev}")
        plt.xlabel("SectorID"); plt.ylabel("Δ mean IF score")
        plt.grid(axis='y', alpha=0.3)
        p2 = os.path.join(evd, "if_sector_delta_bar.png")
        savefig(p2); paths.append(p2)

    if lap_pick is not None and {"t_s","Speed","IF_score"}.issubset(df_ev.columns):
        sample = df_ev[df_ev["LapNumber"] == lap_pick]
        if not sample.empty:
            mask = sample["IF_score"] >= base_thr_98
            plt.figure(figsize=(10,4))
            plt.plot(sample["t_s"], sample["Speed"], label="Speed")
            plt.scatter(sample.loc[mask,"t_s"], sample.loc[mask,"Speed"], s=20,
                        edgecolors="red", facecolors="none", label="Anomaly")
            plt.xlabel("t (s)"); plt.ylabel("Speed (km/h)")
            plt.title(f"{DRIVER} – Lap {lap_pick} Speed with IF anomalies – {ev}")
            plt.grid(True, alpha=0.3); plt.legend()
            p3 = os.path.join(evd, "lap_trace_speed_if_anomalies.png")
            savefig(p3); paths.append(p3)

    for p in paths:
        open_file(p)

print("Done: IF scores and plots generated for every event (including baseline).")
