import random
import numpy as np
import time
import os

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import mlflow
import mlflow.xgboost

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_and_split_data
from objective import objective
from evaluate import save_results

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def main():
    start_time = time.time()

    # MODIFICATION 1: Ensure directory exists at the very start using absolute path
    os.makedirs("/app/outputs", exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    X_train, X_test, y_train, y_test = load_and_split_data(
        test_size=0.2,
        random_state=42
    )

    # -------------------------
    # MLflow experiment
    # -------------------------
    mlflow.set_experiment("optuna-xgboost-optimization")

    # -------------------------
    # Optuna study configuration
    # -------------------------
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5
    )

    # MODIFICATION 2: Use absolute path with four slashes for SQLite
    # This ensures the database file is created inside the mounted volume correctly.
    study = optuna.create_study(
        study_name="xgboost-housing-optimization",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:////app/outputs/optuna_study.db",
        load_if_exists=True
    )

    # -------------------------
    # Optimization
    # -------------------------
    def wrapped_objective(trial):
        with mlflow.start_run(nested=True):
            try:
                value = objective(trial, X_train, y_train)
                mlflow.set_tag("trial_state", "COMPLETE")
                return value
            except optuna.TrialPruned:
                mlflow.set_tag("trial_state", "PRUNED")
                raise
            except Exception:
                mlflow.set_tag("trial_state", "FAIL")
                raise

    study.optimize(
        wrapped_objective,
        n_trials=100,
        n_jobs=2
    )

    # -------------------------
    # Train best model
    # -------------------------
    best_params = study.best_params.copy()
    best_params.update({
        "random_state": 42,
        "objective": "reg:squarederror",
        "n_jobs": 1
    })

    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # -------------------------
    # Test evaluation
    # -------------------------
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred)

    # -------------------------
    # Optuna visualizations & File Setup
    # -------------------------
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances
    )
    
    # Using absolute paths for outputs
    history_path = "/app/outputs/optimization_history.png"
    importance_path = "/app/outputs/param_importance.png"
    
    plot_optimization_history(study).write_image(history_path)
    plot_param_importances(study).write_image(importance_path)

    # -------------------------
    # Final MLflow run
    # -------------------------
    with mlflow.start_run(run_name="best_model"):
        mlflow.set_tag("best_model", "true")

        mlflow.log_params(best_params)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        mlflow.log_artifact(history_path)
        mlflow.log_artifact(importance_path)

        mlflow.xgboost.log_model(
            best_model,
            artifact_path="model"
        )

    # -------------------------
    # Save final results (delegate to evaluate.py)
    # -------------------------
    save_results(
        study=study,
        test_rmse=test_rmse,
        test_r2=test_r2,
        optimization_time_seconds=time.time() - start_time,
        output_dir="/app/outputs"
    )

if __name__ == "__main__":
    main()