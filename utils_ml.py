import numpy as np
import pandas as pd

def resample_uniform_time(df, dt=0.05, cols=None):
    """Resample telemetry to uniform time base (dt seconds) via linear interpolation."""
    if "Time" not in df.columns:
        raise ValueError("Telemetry DataFrame must include 'Time' as timedelta64[ns].")
    out = df.copy()
    out["t_s"] = out["Time"].dt.total_seconds()
    t0, t1 = float(out["t_s"].min()), float(out["t_s"].max())
    grid = np.arange(t0, t1 + 1e-9, dt)
    data = {"t_s": grid}
    cols = cols or [c for c in ["Speed","RPM","Throttle","Brake","Gear","DRS","ERSDeployMode"] if c in out.columns]
    for c in cols:
        series = out[[ "t_s", c ]].dropna()
        if len(series) < 2:  # cannot interpolate a single point
            data[c] = np.full_like(grid, np.nan, dtype=float)
            continue
        data[c] = np.interp(grid, series["t_s"].values, series[c].astype(float).values)
    return pd.DataFrame(data)

def sectorize_equal_distance(df, n_sectors=3):
    """Approx sector labels by splitting max Distance into equal segments (fallback if no official splits)."""
    if "Distance" not in df.columns:
        return df.assign(SectorID=0)
    max_d = float(df["Distance"].max()) if df["Distance"].notna().any() else 0.0
    bounds = np.linspace(0, max_d, n_sectors+1)
    # vectorized binning
    inds = np.digitize(df["Distance"].values, bounds[1:-1], right=False)
    return df.assign(SectorID=inds)

def build_features(df):
    """Create robust features for anomaly detection from uniformly-sampled telemetry."""
    out = df.copy()
    # Normalize inputs
    if "Throttle" in out: out["Throttle01"] = np.clip(out["Throttle"], 0, 100) / 100.0
    if "Brake" in out:    out["Brake01"]    = np.clip(out["Brake"], 0, 100) / 100.0

    # Finite differences (simple, stable)
    if {"t_s","Speed"}.issubset(out.columns):
        out["Accel"] = np.gradient(out["Speed"].astype(float).values, out["t_s"].values)
    if {"t_s","RPM"}.issubset(out.columns):
        out["dRPM_dt"] = np.gradient(out["RPM"].astype(float).values, out["t_s"].values)
    if {"t_s","Throttle01"}.issubset(out.columns):
        out["dThrottle_dt"] = np.gradient(out["Throttle01"].values, out["t_s"].values)
    if {"t_s","Brake01"}.issubset(out.columns):
        out["dBrake_dt"] = np.gradient(out["Brake01"].values, out["t_s"].values)

    # Rolling stats (short window) to capture micro-behavior
    for c in [c for c in ["Speed","RPM","Accel","Throttle01","Brake01"] if c in out]:
        out[c+"_rollmean"] = out[c].rolling(10, min_periods=3).mean()
        out[c+"_rollstd"]  = out[c].rolling(10, min_periods=3).std()

    return out

def sectorwise_zscore(df, features, by=["SectorID"]):
    """Standardize features within each sector (mitigate layout effects)."""
    out = df.copy()
    g = out.groupby(by)
    for c in features:
        mu = g[c].transform("mean")
        sd = g[c].transform("std").replace(0, np.nan)
        out[c+"_z"] = (out[c] - mu) / sd
    return out

def inject_faults(df, fault="MGUK_DROP", magnitude=0.10, lag_seconds=0.2, sector_filter=None):
    """Return a copy with synthetic faults for demonstration."""
    sim = df.copy()
    mask = np.ones(len(sim), dtype=bool)
    if sector_filter is not None and "SectorID" in sim:
        mask = sim["SectorID"].isin(sector_filter)

    if fault == "MGUK_DROP" and "Speed" in sim:
        sim.loc[mask, "Speed"] = sim.loc[mask, "Speed"] * (1.0 - magnitude)

    elif fault == "THROTTLE_LAG" and {"Throttle01","t_s"}.issubset(sim.columns):
        # shift throttle forward (response delayed)
        dt = np.median(np.diff(sim["t_s"].values)) if len(sim) > 1 else 0.05
        shift = max(1, int(round(lag_seconds / max(dt, 1e-3))))
        sim["Throttle01"] = sim["Throttle01"].shift(shift).bfill()   

    elif fault == "ERS_UNDERDELIVERY" and {"Speed","Accel"}.issubset(sim.columns):
        sim.loc[mask, "Accel"] = sim.loc[mask, "Accel"] * (1.0 - magnitude)

    sim["FaultTag"] = fault
    return sim
