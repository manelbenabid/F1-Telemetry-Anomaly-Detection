# F1 Telemetry Anomaly Detection 🏎️

## Overview
This project is an **F1 telemetry anomaly detection tool** designed to analyze car performance data during Formula 1 race weekends.  
It uses **FastF1** telemetry data and applies **machine learning-based anomaly detection** techniques to identify potential **Power Unit (PU) faults** and **performance degradation patterns**.

The project leverages **Isolation Forests** to detect unusual behaviors in telemetry channels like **speed, throttle, RPM, ERS deployment, and braking**, while providing visual insights and an **interactive Streamlit dashboard**.

---

## Features
- **Automatic telemetry extraction** using FastF1.
- **Feature engineering** from lap-level and sector-level telemetry.
- **Machine learning anomaly detection** using Isolation Forest.
- **Fault injection simulation** to test model sensitivity:
  - MGU-K drop simulation
  - Throttle lag simulation
- **Non-blocking visualizations** of:
  - IF score distributions
  - Sector-level anomaly deltas
  - Lap trace highlighting anomalies
- **Interactive dashboard** to explore anomaly scores.

---

## Project Structure
```
F1-Telemetry-Anomaly-Detection/
│
├── extract_telemetry_ml.py        # Downloads raw telemetry data
├── preprocess_features_ml.py      # Prepares telemetry features for ML
├── train_and_plot_ml.py           # Trains Isolation Forest + generates plots
├── app_streamlit_ml.py            # Streamlit dashboard
├── run_pipeline.py                # Orchestrates the full workflow
├── utils_ml.py                    # Utility functions for fault injection & preprocessing
├── config.yaml                    # Central configuration file
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## How to Run

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/F1-Telemetry-Anomaly-Detection.git
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

## Visualizations

### **Isolation Forest Score Distribution**
Shows how anomaly scores vary across:
- Healthy baseline laps
- Normal test laps
- Injected fault scenarios

### **Sector Anomaly Delta**
Compares average anomaly scores between injected-fault and healthy laps, per track sector.

### **Lap Trace with Anomalies**
Overlays detected anomalies on lap telemetry data, highlighting **abnormal speed or power delivery**.

---

## Dashboard Preview
The **Streamlit dashboard** lets you:
- Switch between events
- Explore test vs injected fault data
- View anomaly score deltas per sector interactively

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
