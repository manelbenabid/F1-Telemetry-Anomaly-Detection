import subprocess
import sys
import yaml
import os
from pathlib import Path

def run_script(script_name):
    print(f"\nRunning: {script_name}\n" + "-"*60)
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"Finished: {script_name}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error in {script_name}. Aborting pipeline.")
        sys.exit(e.returncode)

def launch_streamlit(app_path="app_streamlit_ml.py"):
    print("\nStarting Streamlit dashboard...\n" + "-"*60)
    try:
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])
        print("Streamlit launched. If a browser didn't open automatically, visit the URL shown in the console.")
        return proc
    except FileNotFoundError:
        print("Streamlit not found. Install it with: pip install streamlit")
        return None

def main():
    cfg_path = "config.yaml"
    if not os.path.exists(cfg_path):
        print("Missing config.yaml. Cannot run pipeline.")
        sys.exit(1)

    cfg = yaml.safe_load(open(cfg_path))
    season, events, driver = cfg["season"], cfg["events"], cfg["driver"]

    print("\nPipeline context")
    print(f"   Season   : {season}")
    print(f"   Events   : {', '.join(events)}")
    print(f"   Driver   : {driver}")
    print("-"*60)

    # Extract telemetry
    run_script("extract_telemetry_ml.py")

    # Preprocess & feature engineering
    run_script("preprocess_features_ml.py")

    # Train & plot
    run_script("train_and_plot_ml.py")

    # Launch Streamlit dashboard 
    launch_streamlit("app_streamlit_ml.py")

    print("\nPipeline complete!")
    print("Plots have been opened in image viewer.")
    print("Dashboard is running")

if __name__ == "__main__":
    main()
