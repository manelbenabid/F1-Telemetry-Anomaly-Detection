import os, yaml
import numpy as np
import pandas as pd
from utils_ml import resample_uniform_time, sectorize_equal_distance, build_features, sectorwise_zscore

CFG = yaml.safe_load(open("config.yaml"))
SEASON, SESSION, DRIVER = CFG["season"], CFG["session"], CFG["driver"]
EVENTS = CFG["events"]
OUT_DIR = CFG["out_dir"]

DT = 0.05  # 50 ms
for ev in EVENTS:
    ev_dir = os.path.join(OUT_DIR, f"{SEASON}_{ev.replace(' ','_')}")
    tel_path = os.path.join(ev_dir, f"{DRIVER}_telemetry_raw.parquet")
    if not os.path.exists(tel_path):
        print(f"[warn] missing telemetry for {ev} — run extract_telemetry_ml.py")
        continue

    tel = pd.read_parquet(tel_path)
    # process lap-by-lap to keep memory/indices sane
    dfs = []
    for lapno, g in tel.groupby("LapNumber"):
        if g.empty: 
            continue
        # resample, sectorize (based on Distance before resample for boundaries)
        g = g.sort_values("Time")
        g_sec = sectorize_equal_distance(g, n_sectors=3)
        uni = resample_uniform_time(g_sec, dt=DT, cols=[c for c in ["Speed","RPM","Throttle","Brake","Gear","DRS","ERSDeployMode"] if c in g_sec.columns])
        # reattach sector via nearest merge on time grid
        tmp = pd.merge_asof(uni.sort_values("t_s"), g_sec[["Time","Distance","SectorID"]].assign(t_s=g_sec["Time"].dt.total_seconds()).sort_values("t_s"),
                            on="t_s", direction="nearest")
        tmp["LapNumber"] = lapno
        dfs.append(tmp)

    if not dfs: 
        print(f"[warn] all laps empty after resample for {ev}")
        continue
    tel_u = pd.concat(dfs, ignore_index=True)

    # feature engineering
    feats = build_features(tel_u)

    # sector-wise z-normalization for chosen features
    FEATURE_LIST = [c for c in ["Speed","RPM","Throttle01","Brake01","Accel","dRPM_dt","dThrottle_dt","dBrake_dt"] if c in feats.columns]
    feats_z = sectorwise_zscore(feats, FEATURE_LIST, by=["SectorID"])

    # save
    outp = os.path.join(ev_dir, f"{DRIVER}_telemetry_features.parquet")
    feats_z.to_parquet(outp)
    print(f"Saved features -> {outp}")
