import subprocess
import sys
import yaml
import os

def run_script(script_name):
    """Run a Python script and stream output live."""
    print(f"\nRunning: {script_name}\n" + "-"*60)
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"Finished: {script_name}")
        return result
    except subprocess.CalledProcessError:
        print(f"Error in {script_name}. Aborting pipeline.")
        sys.exit(1)

def main():
    # Load config to get context
    cfg_path = "config.yaml"
    if not os.path.exists(cfg_path):
        print(f"Missing config.yaml! Cannot run pipeline.")
        sys.exit(1)

    cfg = yaml.safe_load(open(cfg_path))
    season, events, driver = cfg["season"], cfg["events"], cfg["driver"]
    print("\nPipeline context")
    print(f"   Season   : {season}")
    print(f"   Events   : {', '.join(events)}")
    print(f"   Driver   : {driver}")
    print("-"*60)

    # Stage 1: Extract telemetry
    run_script("extract_telemetry_ml.py")

    # Stage 2: Preprocess & feature engineering
    run_script("preprocess_features_ml.py")

    # Stage 3: Train & plot
    run_script("train_and_plot_ml.py")

    # Stage 4: Optionally suggest dashboard launch
    print("\nIf you want an interactive dashboard, run:")
    print("   streamlit run app_streamlit_ml.py\n")

    print("Pipeline complete!")

if __name__ == "__main__":
    main()
