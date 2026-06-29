"""
Evaluation metrics for Fine-Tuning-PS-LLM

This module provides reusable evaluation functions for
baseline, SFT and DPO experiments.
"""

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)


# -------------------------------------------------------
# Basic Metrics
# -------------------------------------------------------

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)


def compute_macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


def compute_weighted_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )


# -------------------------------------------------------
# Reports
# -------------------------------------------------------

def classification_results(y_true, y_pred):

    return classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )


def compute_confusion_matrix(y_true, y_pred):

    return confusion_matrix(
        y_true,
        y_pred,
    )


# -------------------------------------------------------
# Complete Evaluation
# -------------------------------------------------------

def evaluate_predictions(y_true, y_pred):

    return {

        "accuracy": float(compute_accuracy(y_true, y_pred)),

        "cohen_kappa": float(compute_kappa(y_true, y_pred)),

        "macro_f1": float(compute_macro_f1(y_true, y_pred)),

        "weighted_f1": float(compute_weighted_f1(y_true, y_pred)),

    }


# -------------------------------------------------------
# Save Metrics
# -------------------------------------------------------

def save_metrics(metrics, output_file):

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_file, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(f"Metrics saved to {output_file}")