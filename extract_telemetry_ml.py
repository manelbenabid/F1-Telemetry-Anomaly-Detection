import os, yaml
import pandas as pd
import fastf1
from utils_f1 import ensure_dirs

CFG = yaml.safe_load(open("config.yaml"))
SEASON, SESSION, DRIVER = CFG["season"], CFG["session"], CFG["driver"]
EVENTS = CFG["events"]
CACHE_DIR, OUT_DIR = CFG["cache_dir"], CFG["out_dir"]

ensure_dirs(CACHE_DIR, OUT_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

for ev in EVENTS:
    print(f"[extract] {SEASON} {ev} – {SESSION} for {DRIVER}")
    ses = fastf1.get_session(SEASON, ev, SESSION); ses.load()

    laps = ses.laps.pick_drivers([DRIVER]).copy()
    if laps.empty:
        print(f"[warn] no laps for {DRIVER} at {ev}")
        continue

    rows = []
    for _, lap in laps.iterlaps():
        try:
            car = lap.get_car_data().add_distance()
            car["LapNumber"] = lap["LapNumber"]
            car["Driver"] = DRIVER
            # keeping the common telemetry channels if available
            keep = [c for c in ["Time","Speed","RPM","Throttle","Brake","nGear","DRS","ERSDeployMode","Distance","LapNumber","Driver"] if c in car.columns or c in ["LapNumber","Driver"]]
            car = car[keep]
            car = car.rename(columns={"nGear":"Gear"})
            rows.append(car)
        except Exception:
            continue

    if not rows:
        print(f"[warn] empty telemetry for {ev}")
        continue

    df_tel = pd.concat(rows, ignore_index=True)
    ev_dir = os.path.join(OUT_DIR, f"{SEASON}_{ev.replace(' ','_')}")
    ensure_dirs(ev_dir)
    df_tel.to_parquet(os.path.join(ev_dir, f"{DRIVER}_telemetry_raw.parquet"))
    print(f"Saved -> {ev_dir}")
