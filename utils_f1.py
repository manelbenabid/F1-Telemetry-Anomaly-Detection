import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def is_green_only(ts) -> bool:
    if pd.isna(ts):
        return True
    bad = {'4','5','6'}  # SC, VSC, Red
    return not any(c in bad for c in str(ts).split())

def ensure_stint_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'Stint' not in df.columns or df['Stint'].isna().all():
        if 'TyreLife' in df.columns and not df['TyreLife'].isna().all():
            df = df.sort_values('LapNumber')
            reset = df['TyreLife'].diff().fillna(0) < 0
            df['Stint'] = reset.cumsum() + 1
        elif 'Compound' in df.columns and not df['Compound'].isna().all():
            df = df.sort_values('LapNumber')
            comp_change = (df['Compound'] != df['Compound'].shift(1)).fillna(True)
            df['Stint'] = comp_change.cumsum()
        else:
            df['Stint'] = 1
    if 'Compound' not in df.columns:
        df['Compound'] = 'UNK'
    df['Compound'] = df['Compound'].fillna('UNK')
    return df

def derive_top_speed(laps) -> list:
    tops = []
    for _, lap in laps.iterlaps():
        try:
            car = lap.get_car_data()
            tops.append(car['Speed'].max() if not car.empty else np.nan)
        except Exception:
            tops.append(np.nan)
    return tops

def weather_adjusted_normalization(session, df: pd.DataFrame, laps) -> pd.DataFrame:
    wx = session.weather_data.copy()
    keep_wx = [c for c in ['Time','TrackTemp','AirTemp','WindSpeed','Rainfall','Humidity','Pressure','WindDirection'] if c in wx.columns]
    wx = wx[keep_wx].dropna(how='all')

    if 'LapStartTime' in laps.columns and 'LapTime' in laps.columns:
        df['LapMidTime'] = laps.loc[df.index, 'LapStartTime'] + (laps.loc[df.index, 'LapTime'] / 2)
    else:
        df['LapMidTime'] = laps.loc[df.index, 'LapStartTime'] if 'LapStartTime' in laps.columns else pd.NaT

    wx_sorted = wx.sort_values('Time').copy()
    df_sorted = df.sort_values('LapMidTime').copy()

    if df_sorted['LapMidTime'].notna().any():
        merged = pd.merge_asof(df_sorted, wx_sorted, left_on='LapMidTime', right_on='Time', direction='nearest')
    else:
        merged = df_sorted.copy()

    merged['RainFlag'] = (merged.get('Rainfall', pd.Series(0, index=merged.index)).fillna(0) > 0).astype(int)

    feature_candidates = ['LapNumber','TrackTemp','AirTemp','WindSpeed','RainFlag']
    features = [c for c in feature_candidates if c in merged.columns]
    merged = merged.dropna(subset=['LapTime(s)'])

    for c in features:
        if merged[c].dtype.kind in 'fc':
            merged[c] = merged[c].fillna(merged[c].median())
        else:
            merged[c] = merged[c].fillna(0)

    X = merged[features].astype(float)
    y = merged['LapTime(s)'].astype(float)

    if len(merged) >= 10 and X.notna().all().all():
        lr = LinearRegression()
        lr.fit(X, y)
        merged['LapTime_Pred(s)'] = lr.predict(X)
        merged['NormLapTime_resid(s)'] = merged['LapTime(s)'] - merged['LapTime_Pred(s)']
    else:
        lr = LinearRegression()
        X_fallback = merged[['LapNumber']].astype(float)
        lr.fit(X_fallback, y)
        merged['LapTime_Pred(s)'] = lr.predict(X_fallback)
        merged['NormLapTime_resid(s)'] = merged['LapTime(s)'] - merged['LapTime_Pred(s)']

    return merged.sort_index()

def stint_degradation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if {'Stint','LapNumber','NormLapTime_resid(s)'}.issubset(df.columns) and not df.empty:
        for (stint_id, comp), g in df.groupby(['Stint','Compound'], dropna=False):
            g = g.dropna(subset=['NormLapTime_resid(s)','LapNumber'])
            if g.empty:
                continue
            slope = np.polyfit(g['LapNumber'], g['NormLapTime_resid(s)'], 1)[0] if len(g) >= 3 else np.nan
            rows.append({
                'Stint': int(stint_id) if pd.notna(stint_id) else -1,
                'Compound': str(comp),
                'StartLap': int(g['LapNumber'].min()),
                'EndLap': int(g['LapNumber'].max()),
                'LapsInStint': int(len(g)),
                'AvgRawLap(s)': float(g['LapTime(s)'].mean()) if 'LapTime(s)' in g.columns else np.nan,
                'AvgResid(s)': float(g['NormLapTime_resid(s)'].mean()),
                'Degradation_onResid(s_per_lap)': float(slope) if pd.notna(slope) else np.nan,
                'AvgTopSpeed(km/h)': float(g['TopSpeed'].mean()) if 'TopSpeed' in g.columns else np.nan
            })
    return pd.DataFrame(rows)
