# Automated Hyperparameter Optimization with Optuna and MLflow

## Project Overview

This project implements a **production-grade automated hyperparameter optimization pipeline**
using **Optuna** and **MLflow**.  
The objective is to systematically tune an **XGBoost regression model** on the **California Housing dataset**
while following modern **MLOps best practices** such as reproducibility, experiment tracking,
containerization, and result visualization.

The pipeline performs:
- Automated hyperparameter search with Optuna
- Experiment tracking and model logging with MLflow
- Final model training and evaluation
- Generation of optimization visualizations
- Fully containerized execution using Docker

---

## Tech Stack

- **Python** 3.9+
- **XGBoost**
- **Optuna**
- **MLflow**
- **scikit-learn**
- **NumPy**
- **Pandas**
- **Docker**

---

## Dataset
**Note:**  
The Kaggle dataset is functionally equivalent to the original California Housing dataset used in scikit-learn.
### California Housing Dataset

Due to **network access restrictions** (HTTP 403 errors) when downloading the dataset via
`sklearn.datasets.fetch_california_housing`, the dataset is **manually downloaded from Kaggle**
and included locally.

- Source: Kaggle – *California Housing Dataset*
- File location: `data/california_housing.csv`

### Dataset Schema (Verified)

The dataset has been preprocessed to match the expected model input:

| Column Name | Description |
|------------|------------|
| MedInc | Median income |
| HouseAge | Median house age |
| AveRooms | Average rooms |
| AveBedrms | Average bedrooms |
| Population | Population |
| AveOccup | Average occupancy |
| Latitude | Latitude |
| Longitude | Longitude |
| target | Median house value |

**Shape:** `20640 × 9`

The pipeline assumes this exact schema. Column validation is performed before training.

---

## Repository Structure

```text
.
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── README.md
├── data/
│   └── california_housing.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── objective.py
│   ├── optimize.py
│   └── evaluate.py
├── notebooks/
│   └── analysis.ipynb
└── outputs/                     # generated at runtime
    ├── mlruns/
    ├── optimization_history.png
    ├── param_importance.png
    └── results.json
````

> **Note:**
> The `outputs/` directory is created automatically when the Docker container runs.
> It contains experiment logs, Optuna visualizations, and final optimization results.

---

## How the Pipeline Works

1. **Data Loading**

   * Loads the local California Housing CSV
   * Splits data into train/test sets

2. **Hyperparameter Optimization**

   * Optuna performs automated hyperparameter search
   * Search space includes depth, learning rate, estimators, sampling parameters, etc.
   * Optimization objective minimizes RMSE

3. **Experiment Tracking**

   * All trials are logged to MLflow
   * Best model parameters are stored
   * Final model is logged as an MLflow artifact

4. **Visualization**

   * Optimization history plot
   * Hyperparameter importance plot

5. **Final Evaluation**

   * Best model retrained on training data
   * Evaluated on test set
   * Results saved as JSON

---

## How to Run (Docker)

### 1. Build the Docker Image

```bash
docker build -t optuna-mlflow-pipeline .
```

### 2. Run the Container

#### Linux / macOS

```bash
docker run -v $(pwd)/outputs:/app/outputs optuna-mlflow-pipeline
```

#### Windows PowerShell

```powershell
docker run -v ${PWD}/outputs:/app/outputs optuna-mlflow-pipeline
```

The container will:

* Run the Optuna optimization process
* Log experiments to MLflow
* Generate visualizations
* Save results to the `outputs/` directory
* Exit automatically after completion

---

## Outputs

After execution, the following files are generated in `outputs/`:

* `results.json`
  Summary of best hyperparameters and evaluation metrics

* `optimization_history.png`
  Visualization of Optuna optimization progress

* `param_importance.png`
  Hyperparameter importance plot

* `mlruns/`
  MLflow experiment tracking data (metrics, parameters, artifacts)

> **Note:**
> The Optuna study database (`optuna_study.db`) is optional.
> This implementation uses in-memory and MLflow-based tracking, which is sufficient
> for full reproducibility and evaluation.

---

## Results Summary

* The optimized XGBoost model significantly improves performance over baseline
* RMSE is substantially reduced after tuning
* R² score exceeds the project performance threshold
* Most influential hyperparameters:

  * Learning rate
  * Max depth
  * Number of estimators

Detailed analysis is available in `notebooks/analysis.ipynb`.

---

## Reproducibility

* Random seeds are fixed across NumPy, scikit-learn, and XGBoost
* Docker ensures environment consistency
* Results are reproducible across runs on different machines

---

## Notes on Design Decisions

* **Kaggle dataset usage** ensures reliability when sklearn downloads fail
* **Docker-first execution** ensures portability and isolation
* **MLflow tracking** replaces the need for persistent Optuna DB files
* `__init__.py` is intentionally empty and valid for Python packaging

