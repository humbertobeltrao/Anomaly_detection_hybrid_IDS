import os
import pandas as pd
import numpy as np

from evaluation.predict import predict_file_with_timestamps
from evaluation.metrics import (
    calculate_statistics,
    calculate_thresholds,
    calculate_anomaly_metrics,
    calculate_temporal_statistics,
    calculate_classification_metrics,
)
from preprocessing.load_data import (
    get_file_paths,
    load_attack_metadata,
)

def save_evaluation_results(results, output_dir):
    """
    Save evaluation results to CSV files.

    Parameters
    ----------
    results : dict
        Results returned by evaluate_model().

    output_dir : str
        Directory where the results will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    validation_metrics = []

    for metric, value in results["validation_statistics"].items():
        validation_metrics.append({
            "metric": metric,
            "value": value
        })

    for threshold_name, threshold in results["thresholds"].items():
        validation_metrics.append({
            "metric": f"threshold_{threshold_name}",
            "value": threshold
        })

    validation_df = pd.DataFrame(validation_metrics)

    validation_df.to_csv(
        os.path.join(
            output_dir,
            "validation_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # ATTACK METRICS
    # ========================================================

    evaluation_df = pd.DataFrame(
        results["evaluation"]
    )

    evaluation_df.to_csv(
        os.path.join(
            output_dir,
            "attack_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # TEMPORAL METRICS
    # ========================================================

    temporal_df = pd.DataFrame(
        results["temporal"]
    )

    temporal_df.to_csv(
        os.path.join(
            output_dir,
            "temporal_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    classification_df = pd.DataFrame(
        results["classification"]
    )

    classification_df.to_csv(
        os.path.join(
            output_dir,
            "classification_metrics.csv"
        ),
        index=False
    )

def evaluate_model(model, dataset_config, model_config, id_to_index, X_val, y_val):
    # ========================================================
    # THRESHOLD CALIBRATION
    # ========================================================

    predictions = model.predict(X_val)

    validation_scores = predictions[
        np.arange(len(y_val)),
        y_val
    ]

    validation_statistics = calculate_statistics(
        validation_scores
    )

    thresholds = calculate_thresholds(
        validation_scores
    )

    # ========================================================
    # ATTACK FILES
    # ========================================================

    attack_files = dataset_config["dataset"]["test"]["files"]

    attack_metadata = load_attack_metadata(
        dataset_config["dataset"]["path"]
    )

    test_file_paths = get_file_paths(
        dataset_config["dataset"]["path"],
        dataset_config["dataset"]["test"]["directory"],
        attack_files
    )

    evaluation_results = []
    temporal_results = []
    classification_results = []

    # ========================================================
    # EVALUATION
    # ========================================================

    for test_file in test_file_paths:

        timestamps, y, probabilities = (
            predict_file_with_timestamps(
                model,
                test_file,
                id_to_index,
                model_config["sliding_window"]["n"]
            )
        )

        metadata = attack_metadata[test_file.stem]
        injection_interval = metadata["injection_interval"]

        # ----------------------------------------------------
        # Threshold evaluation
        # ----------------------------------------------------

        for threshold_name, threshold in thresholds.items():

            metrics = calculate_anomaly_metrics(
                probabilities,
                threshold
            )

            evaluation_results.append({
                "file": test_file.name,
                "threshold_name": threshold_name,
                "threshold": threshold,
                **metrics,
            })

            # Classification metrics
            if injection_interval is not None:
                classification = calculate_classification_metrics(
                    timestamps,
                    probabilities,
                    injection_interval,
                    threshold
                )

                classification_results.append({
                    "file": test_file.name,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    **classification,
                })

        # ----------------------------------------------------
        # Temporal analysis
        # ----------------------------------------------------

        temporal = calculate_temporal_statistics(
            timestamps,
            probabilities,
            injection_interval
        )

        if temporal is None:
            continue

        for phase in ["BEFORE", "DURING", "AFTER"]:

            temporal_results.append({
                "file": test_file.name,
                "phase": phase,
                "injection_start": temporal["injection_start"],
                "injection_end": temporal["injection_end"],
                **temporal[phase],
            })

    return {
        "validation_statistics": validation_statistics,
        "thresholds": thresholds,
        "evaluation": evaluation_results,
        "temporal": temporal_results,
        "classification": classification_results,
    }
