"""
Per-model checkpoint utilities for long-running training jobs.

Saves/loads model evaluation results as JSON so that training can resume
after a SLURM wall-time kill without re-training completed models.
Uses atomic write (tmp + os.replace) to prevent corruption from signals.
"""

import json
import os

import numpy as np


def load_checkpoint(path, fresh=False):
    """
    Load previously completed model results from checkpoint file.

    Parameters
    ----------
    path : str
        Path to the checkpoint JSON file.
    fresh : bool
        If True, ignore existing checkpoint and return empty dict.

    Returns
    -------
    dict
        Model name -> metrics dict. Empty if no checkpoint or fresh=True.
    """
    if fresh or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    # Convert confusion_matrix lists back to numpy arrays
    for m in data.values():
        m["confusion_matrix"] = np.array(m["confusion_matrix"])
    return data


def save_checkpoint(all_results, path):
    """
    Write all completed model results to checkpoint (atomic overwrite).

    Parameters
    ----------
    all_results : dict
        Model name -> metrics dict (with numpy confusion_matrix).
    path : str
        Path to the checkpoint JSON file.
    """
    serializable = {}
    for name, m in all_results.items():
        serializable[name] = {
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "confusion_matrix": m["confusion_matrix"].tolist(),
            "per_class": m["per_class"],
            "train_time_s": m["train_time_s"],
            "predict_time_s": m["predict_time_s"],
            "params": m["params"],
        }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(serializable, f, indent=2)
    os.replace(tmp_path, path)
