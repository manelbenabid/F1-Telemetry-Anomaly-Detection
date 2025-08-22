import os, yaml, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from utils_ml import inject_faults

warnings.filterwarnings("ignore", category=UserWarning)

CFG = yaml.safe_load(open("config.yaml"))
SEASON, DRIVER = CFG["season"], CFG["driver"]
EVENTS = CFG["events"]
OUT_DIR = CFG["out_dir"]
SHOW_BLOCK_AT_END = bool(CFG.get("show_block_at_end", True))

plt.ion()

# choose a baseline event (first one) for model fit, then test on later event + injected faults
if len(EVENTS) < 2:
    print("[info] Add at least two events in config for a nicer baseline vs test split.")
baseline_ev = EVENTS[0]
test_ev = EVENTS[min(1, len(EVENTS)-1)]

def load_feats(ev):
    ev_dir = os.path.join(OUT_DIR, f"{SEASON}_{ev.replace(' ','_')}")
    p = os.path.join(ev_dir, f"{DRIVER}_telemetry_features.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing features for {ev}. Run preprocess_features_ml.py")
    return pd.read_parquet(p)

# 1) Fit on baseline (assume healthy)
df_base = load_feats(baseline_ev)
FEATURES = [c for c in df_base.columns if c.endswith("_z")]  # standardized features
FEATURES = [c for c in FEATURES if c not in ["Gear_z","DRS_z","ERSDeployMode_z"]]  # categorical-like, drop if present
X_base = df_base[FEATURES].fillna(0.0).values

iso = IsolationForest(
    n_estimators=300, max_samples="auto", contamination=0.02,
    random_state=42, n_jobs=-1
).fit(X_base)

df_base["IF_score"] = -iso.score_samples(X_base)  # higher = more anomalous

# 2) Score a later event (test)
df_test = load_feats(test_ev)
X_test = df_test[FEATURES].fillna(0.0).values
df_test["IF_score"] = -iso.score_samples(X_test)

# 3) Create a fault-injected version of a **single lap** in the test event for demo
if "LapNumber" in df_test:
    lap_pick = int(df_test["LapNumber"].mode().iloc[0])  # a common/full lap
    lap_df = df_test[df_test["LapNumber"] == lap_pick].reset_index(drop=True)
else:
    lap_pick = None
    lap_df = df_test.copy()

# inject two kinds of faults (on sectors 1 & 2 as examples)
inj1 = inject_faults(lap_df, fault="MGUK_DROP", magnitude=0.10, sector_filter={1})
inj2 = inject_faults(lap_df, fault="THROTTLE_LAG", lag_seconds=0.2, sector_filter={2})
inj_all = pd.concat([inj1.assign(Variant="MGUK_DROP"),
                     inj2.assign(Variant="THROTTLE_LAG")], ignore_index=True)

# score injected
X_inj = inj_all[[f for f in FEATURES if f in inj_all.columns]].fillna(0.0).values
inj_all["IF_score"] = -iso.score_samples(X_inj)

# 4) Aggregate anomaly scores per lap/sector for reporting
def agg_scores(df):
    grp = df.groupby(["LapNumber","SectorID"], dropna=False)["IF_score"].mean().reset_index()
    grp.rename(columns={"IF_score":"IF_sector_mean"}, inplace=True)
    return grp

base_scores = agg_scores(df_base)
test_scores = agg_scores(df_test)
inj_scores  = agg_scores(inj_all)

# Save CSVs
ev_dir_test = os.path.join(OUT_DIR, f"{SEASON}_{test_ev.replace(' ','_')}")
os.makedirs(ev_dir_test, exist_ok=True)
base_scores.to_csv(os.path.join(ev_dir_test, f"{DRIVER}_if_scores_baseline.csv"), index=False)
test_scores.to_csv(os.path.join(ev_dir_test, f"{DRIVER}_if_scores_test.csv"), index=False)
inj_scores.to_csv(os.path.join(ev_dir_test, f"{DRIVER}_if_scores_injected.csv"), index=False)

# 5) Visuals (non-blocking)
# A) Distribution of IF scores (baseline vs test vs injected)
plt.figure(figsize=(9,5))
plt.hist(df_base["IF_score"], bins=60, alpha=0.5, label=f"{baseline_ev} (baseline)")
plt.hist(df_test["IF_score"], bins=60, alpha=0.5, label=f"{test_ev} (test)")
plt.hist(inj_all["IF_score"], bins=60, alpha=0.5, label="Injected (fault demo)")
plt.xlabel("Isolation Forest anomaly score (↑ = more anomalous)")
plt.ylabel("Count")
plt.title(f"{DRIVER} – IF score distributions")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ev_dir_test, "if_score_distributions.png"), dpi=160)
plt.show(block=False)

# B) Sector heatmap (test vs injected deltas)
pivot_t = test_scores.pivot(index="LapNumber", columns="SectorID", values="IF_sector_mean")
pivot_i = inj_scores.pivot(index="LapNumber", columns="SectorID", values="IF_sector_mean")
common_cols = [c for c in pivot_t.columns if c in pivot_i.columns]
delta = pivot_i[common_cols].mean(axis=0) - pivot_t[common_cols].mean(axis=0)

plt.figure(figsize=(7,4))
plt.bar([str(c) for c in common_cols], delta.values)
plt.title(f"{DRIVER} – Δ IF score by Sector (Injected − Test)")
plt.xlabel("SectorID"); plt.ylabel("Δ mean IF score")
plt.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ev_dir_test, "if_sector_delta_bar.png"), dpi=160)
plt.show(block=False)

# C) Lap trace (Speed) with anomalies highlighted (test event, sample lap)
if {"t_s","Speed","IF_score"}.issubset(df_test.columns):
    sample = df_test[df_test["LapNumber"] == lap_pick]
    if not sample.empty:
        thr = np.percentile(df_base["IF_score"], 98)  # anomaly threshold from baseline tail
        mask = sample["IF_score"] >= thr

        plt.figure(figsize=(10,4))
        plt.plot(sample["t_s"], sample["Speed"], label="Speed")
        plt.scatter(sample.loc[mask,"t_s"], sample.loc[mask,"Speed"], s=20, edgecolors="red", facecolors="none", label="Anomaly")
        plt.xlabel("t (s)"); plt.ylabel("Speed (km/h)")
        plt.title(f"{DRIVER} – Lap {lap_pick} Speed with IF anomalies – {test_ev}")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(ev_dir_test, "lap_trace_speed_if_anomalies.png"), dpi=160)
        plt.show(block=False)

if SHOW_BLOCK_AT_END:
    plt.ioff(); plt.show()
