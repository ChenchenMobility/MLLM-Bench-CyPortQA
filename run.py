#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py
---------------------------------
Master runner that sequentially executes multiple model run scripts
based on the configuration file (config.json).
"""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

# -------------------------------
# Utility functions
# -------------------------------

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_model_script(script_path, args_dict):
    """Run a single model script as a subprocess."""
    cmd = ["python", str(script_path)]
    for k, v in args_dict.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    print(f"\n🚀 Running {script_path.name} with args: {args_dict}")
    print("----------------------------------------------------")

    # Capture logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{script_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        process.wait()

    if process.returncode == 0:
        print(f"✅ Finished {script_path.name}, logs saved at {log_file}")
    else:
        print(f"❌ Error running {script_path.name}, see logs at {log_file}")

def main():
    config = load_config("config.json")
    models_to_run = config.get("models_to_run", [])
    args_dict = config.get("args", {})

    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("❌ 'models/' folder not found. Make sure run_*.py scripts are inside it.")

    for model_script in models_to_run:
        script_path = models_dir / model_script
        if not script_path.exists():
            print(f"⚠️  Skipping {model_script}: file not found in models/")
            continue
        run_model_script(script_path, args_dict)

    print("\n🎯 All selected model runs completed.")

if __name__ == "__main__":
    main()