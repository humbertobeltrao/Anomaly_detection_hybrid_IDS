import os
import re
import shutil
import yaml

from training.train import run_training
from evaluation.evaluate import (evaluate_model, save_evaluation_results)

def get_next_experiment_dir(results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    experiment_numbers = []

    for name in os.listdir(results_dir):
        match = re.fullmatch(r"experiment_(\d+)", name)

        if match:
            experiment_numbers.append(int(match.group(1)))

    next_number = max(experiment_numbers, default=0) + 1

    return os.path.join(results_dir, f"experiment_{next_number:03d}")

# ============================================================
# CONFIG
# ============================================================

with open("config/dataset.yaml") as file:
    dataset_config = yaml.safe_load(file)

# CHANGE THIS ARCHIVE WHEN TESTING
with open("config/model_tuned.yaml") as file:
    model_config = yaml.safe_load(file)

# ============================================================
# EXPERIMENT
# ============================================================

experiment_dir = get_next_experiment_dir()

os.makedirs(experiment_dir, exist_ok=True)

# Save configuration used in this experiment
shutil.copy(
    "config/model_tuned.yaml",
    f"{experiment_dir}/model_config.yaml"
)

shutil.copy(
    "config/dataset.yaml",
    f"{experiment_dir}/dataset_config.yaml"
)

# ============================================================
# TRAINING
# ============================================================

model, history, id_to_index, index_to_id, X_val, y_val = run_training(
    dataset_config,
    model_config,
    experiment_dir
)

# ============================================================
# EVALUATION
# ============================================================

results = evaluate_model(
    model,
    dataset_config,
    model_config,
    id_to_index,
    X_val,
    y_val
)

save_evaluation_results(
    results,
    experiment_dir
)

