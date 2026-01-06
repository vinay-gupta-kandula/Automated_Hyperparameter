import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import json
import os
import optuna


def save_results(
    study,
    test_rmse,
    test_r2,
    optimization_time_seconds,
    output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )

    pruned_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )

    results = {
        "n_trials_completed": completed_trials,
        "n_trials_pruned": pruned_trials,
        "best_cv_rmse": float(np.sqrt(-study.best_value)),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "best_params": study.best_params,
        "optimization_time_seconds": float(optimization_time_seconds)
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
