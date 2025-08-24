import os, sys, yaml, warnings, json
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils_ml import (
    make_injected_dataset,     
    build_features,            
)

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
            os.startfile(path)  
        elif sys.platform == "darwin":
            import subprocess; subprocess.Popen(["open", path])
        else:
            import subprocess; subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"Warn: Could not auto-open '{path}': {e}")

def agg_scores(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["LapNumber","SectorID"], dropna=False)["IF_score"].mean().reset_index()
    return g.rename(columns={"IF_score":"IF_sector_mean"})

def refit_zscores_against_ref(target: pd.DataFrame, ref: pd.DataFrame, features_z: list, by: str = "SectorID") -> pd.DataFrame:
    """
    Return a copy of `target` with each `*_z` recomputed using per-`by` means/stds
    taken from `ref`. The matching base columns (e.g. `Speed` for `Speed_z`)
    must exist in both `target` and `ref`.
    """

    out = target.copy()
    if by not in out.columns or by not in ref.columns:
        return out

    for fz in features_z:
        if not fz.endswith("_z"):
            continue
        base = fz[:-2]
        if base not in out.columns or base not in ref.columns:
            continue

        stats = ref.groupby(by)[base].agg(mu="mean", sd="std")
        mu_map = stats["mu"].to_dict()
        sd_map = stats["sd"].replace(0, np.nan).to_dict()

        mu = out[by].map(mu_map)
        sd = out[by].map(sd_map)
        out[fz] = (out[base] - mu) / sd

    return out

if not EVENTS:
    raise SystemExit("No events in config.yaml")

# 1.Fit baseline model on first event 
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

# 2. for every event (including baseline): write test + injected + plots 
for ev in EVENTS:
    print(f"[info] Processing event: {ev}")
    evd = ev_dir(ev); os.makedirs(evd, exist_ok=True)

    # 2a. load and score TEST (real telemetry for this event)
    if ev == baseline_ev:
        df_ev = df_base.copy()
    else:
        df_ev = load_feats(ev)
        X_ev = df_ev[FEATURES].fillna(0.0).values
        df_ev["IF_score"] = -iso.score_samples(X_ev)

    # save TEST event scores (for dashboard)
    agg_scores(df_ev).to_csv(os.path.join(evd, f"{DRIVER}_if_scores_test.csv"), index=False)

    # 2b. Build an injected dataset for this event (evaluation only)
    inj_all = make_injected_dataset(
        df_ev,
        n_laps=12,
        faults=("MGUK_DROP","THROTTLE_LAG"),
        magnitudes=(0.05, 0.10, 0.15),
        lags=(0.1, 0.2, 0.3),
        sector_strategy="random",
        random_state=42
    )

    if inj_all.empty:
        print(f"[warn] No injected samples produced for {ev}. Skipping injected scoring/plots.")
        continue

    # 2c. Recompute features AFTER injection so changes matter
    inj_all = build_features(inj_all)

    # 2d. Recompute *_z features on injected data using TEST event as reference
    inj_all = refit_zscores_against_ref(inj_all, df_ev, FEATURES, by="SectorID")

    # 2e. Score injected with the trained Isolation Forest
    X_inj = inj_all[FEATURES].fillna(0.0).values
    inj_all["IF_score"] = -iso.score_samples(X_inj)

    # Save time-series Parquets for interactive lap traces
    cols_keep = [c for c in ["t_s","LapNumber","SectorID","Speed","Throttle01","RPM","IF_score"] if c in df_ev.columns]
    if cols_keep:
        df_ev[cols_keep].to_parquet(os.path.join(evd, f"{DRIVER}_if_timeseries_test.parquet"))
    cols_keep_inj = [c for c in ["t_s","LapNumber","SectorID","Speed","Throttle01","RPM","IF_score","Variant"] if c in inj_all.columns]
    if cols_keep_inj:
        inj_all[cols_keep_inj].to_parquet(os.path.join(evd, f"{DRIVER}_if_timeseries_injected.parquet"))

    # Save aggregated injected scores (for dashboard)
    agg_scores(inj_all).to_csv(os.path.join(evd, f"{DRIVER}_if_scores_injected.csv"), index=False)

    # 2f. Simple detection metric vs baseline tail
    hit_rate = float((inj_all["IF_score"] >= base_thr_98).mean() * 100.0)
    metrics = {
        "event": ev,
        "driver": DRIVER,
        "baseline_event": baseline_ev,
        "baseline_if_98pct": float(base_thr_98),
        "injected_rows": int(len(inj_all)),
        "hit_rate_pct_vs_baseline98": hit_rate
    }
    with open(os.path.join(evd, f"{DRIVER}_if_detection_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] {ev}: {hit_rate:.1f}% of injected samples exceed the baseline 98th-percentile IF.")

    # 2g. Plots for this event 
    paths = []

    # Distribution: baseline vs test vs injected + baseline 98th pct line
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

    # Δ IF by sector bar (mean over laps)
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

    # Lap trace example
    lap_pick = None
    if "LapNumber" in df_ev and not df_ev["LapNumber"].dropna().empty:
        lap_pick = int(df_ev["LapNumber"].mode().iloc[0])
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

print("Done: IF scores, injected dataset, detection metrics, and plots generated for every event (including baseline).")
