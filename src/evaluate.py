import pandas as pd
import hydra
from omegaconf import DictConfig
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import sys
import os

@hydra.main(config_path="../config", config_name="main", version_base=None)
def evaluate_drift(cfg: DictConfig):
    # Load Reference (Training Data) and Current (New Data)
    # In a real scenario, 'current' would be the new batch of data fetched today
    reference_data = pd.read_csv(cfg.data.reference_path)
    current_data = pd.read_csv(cfg.data.processed_path) # Assuming this was just updated
    
    # Generate Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save Report
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/drift_report.html")
    
    # Check for Drift (Fail pipeline if drift detected)
    results = report.as_dict()
    drift_detected = results['metrics'][0]['result']['dataset_drift']
    
    if drift_detected:
        print("CRITICAL: Data Drift Detected! Retraining recommended.")
        # In a real pipeline, you might exit with 0 to trigger retraining, 
        # or exit with 1 to block deployment if validation fails.
        # For this demo, we warn but allow proceed.
    else:
        print("Data is stable. No drift detected.")

if __name__ == "__main__":
    evaluate_drift()