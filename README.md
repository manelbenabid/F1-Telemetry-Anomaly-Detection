# F1 Telemetry Anomaly Detection — Max Verstappen (VER) 🏎️

A production‑style demo that detects **Power Unit (PU) anomalies** in **Max Verstappen’s** telemetry using **Isolation Forests**, with clean visualizations and an **interactive Streamlit dashboard** in a Red Bull–inspired dark theme.

---

## What this project does
- **Collects telemetry** for **Max Verstappen (VER)** from public FastF1 (Speed, Throttle, RPM, Gear, DRS, ERS mode, etc.).  
- **Resamples** signals onto a uniform time base and **sectorizes per lap** (consistent **SectorID = 1, 2, 3**).  
- **Engineers features** (acceleration, derivatives, rolling stats) for robust modeling.  
- **Trains an Isolation Forest (IF)** on a healthy baseline event and scores other events.  
- **Injects synthetic faults** (e.g., 10% MGU‑K drop, throttle lag) to show how faults shift anomaly scores.  
- **Generates plots** and launches a **dashboard** for interactive analysis tailored to **VER**.

> **IF = Isolation Forest anomaly score.** Higher IF ⇒ more anomalous versus the learned “normal”.  
> **Δ IF (Injected − Test)** = how much the synthetic fault increases anomaly for a sector or time window.

---

## Project structure (important files)
```
F1-Telemetry-Anomaly-Detection/
├── config.yaml                    # season, driver=VER, events (e.g., British GP, Italian GP), out_dir
├── run_pipeline.py                # orchestrates extraction → features → model → plots → dashboard
├── extract_telemetry_ml.py        # downloads raw telemetry per lap (with Distance)
├── preprocess_features_ml.py      # resamples, sectorizes (1..3 per lap), builds features
├── train_and_plot_ml.py           # fits IF model, scores events, injects faults, saves plots + parquet
├── app_streamlit_ml.py            # Streamlit dashboard (Plotly) in RB theme, default driver VER
├── utils_ml.py                    # helpers (resample, sectorize_per_lap, features, inject_faults)
├── requirements.txt               # Python dependencies
└── outputs/                       # generated CSVs, PNGs, Parquet by event
```

---

## How to Run

### **1. Clone the Repository**
```bash
git clone https://github.com/manelbenabid/F1-Telemetry-Anomaly-Detection.git
cd F1-Telemetry-Anomaly-Detection
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Configure Settings**
Update `config.yaml` to select:
- **Season** → e.g., `2024`
- **Driver** → e.g., `VER`
- **Events** → List of races, e.g.:
```yaml
season: 2024
driver: VER
events:
  - British Grand Prix
  - Italian Grand Prix
out_dir: data_cache
show_block_at_end: true
```

### **4. Run the Full Pipeline**
```bash
python run_pipeline.py
```

This will:
1. **Download telemetry**
2. **Preprocess features**
3. **Train anomaly detection model**
4. **Generate charts**
5. **Launch Streamlit dashboard**

### **5. Launch Dashboard (Optional)**
```bash
streamlit run app_streamlit_ml.py
```
- Browse through different events.
- Compare **healthy vs injected faults**.
- Visualize anomaly score deltas.

---

## Generated outputs for VER

For each event folder under `outputs/<season>_<EventName>/` you’ll get:

**CSV (event‑level, Lap × Sector):**
- `VER_if_scores_test.csv` — mean IF per **Lap × Sector** from real telemetry.  
- `VER_if_scores_injected.csv` — same metric after a **synthetic fault** on a representative lap.

**Figures:**
- `if_score_distributions.png` — baseline vs test vs injected IF histograms.  
- `if_sector_delta_bar.png` — Δ IF by sector (Injected − Test).  
- `lap_trace_speed_if_anomalies.png` — speed vs time for a sample lap with anomaly points highlighted.

**Time‑series (for interactive lap traces):**
- `VER_if_timeseries_test.parquet`  
- `VER_if_timeseries_injected.parquet`

---

## How to interpret the figures

### 1) IF score distribution
Overlapping histograms of IF for **Test** (real telemetry) vs **Injected** (with synthetic fault).  
A **right‑shift** in *Injected* indicates the fault makes telemetry behavior more unusual (more anomalous).

### 2) Lap × Sector heatmap
Matrix of mean IF per **LapNumber × SectorID**.  
Switch between **Test**, **Injected**, and **Δ IF (Injected − Test)**:  
- Bright/red **Δ IF** cells show **where** the fault manifests (e.g., long straights for MGU‑K drop).

### 3) Per‑sector trends across laps
Lines for Test vs Injected mean IF by sector.  
Divergence or upward drift for Injected suggests **fault impact or degradation** accumulating through the stint.

### 4) Lap trace (interactive)
Speed (and optionally Throttle) vs time **colored by IF**; spikes highlight **moments** the model deems abnormal, helping you pinpoint candidate root causes or PU delivery issues for **VER**.

---

## Notes
- Sector IDs are labeled **1, 2, 3** per lap using `sectorize_per_lap` (see `utils_ml.py`).  
- Keep event names in `config.yaml` consistent with FastF1 to download telemetry correctly.  
- Plots are saved to disk and also displayed through the Streamlit app (Plotly + RB colors).
- The dashboard defaults to **VER**; you can still change the driver code in `config.yaml` if needed.

---

## Tech Stack
- **Data Source**: [FastF1](https://theoehrly.github.io/Fast-F1/)
- **ML Models**: Isolation Forest (Scikit-learn)
- **Visualization**: Matplotlib, Streamlit
- **Language**: Python 3.12+

---

## Installation Notes
- Python ≥ 3.10 recommended.
- See `requirements.txt` for exact dependencies.
