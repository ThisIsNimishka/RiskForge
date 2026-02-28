"""
train.py
--------
Orchestrates model training with:
- Optuna hyperparameter optimization
- Stacking ensemble (XGBoost + LightGBM + LogisticRegression)
- MLflow experiment tracking
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from optuna.integration import MLflowCallback
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, average_precision_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import argparse
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

SEED = 42
CV_FOLDS = 5
OPTUNA_TRIALS = 100


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """Compute comprehensive credit risk evaluation metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc - 1
    ap = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # KS Statistic (industry standard for credit)
    from scipy import stats
    pos_scores = y_pred_proba[y_test == 1]
    neg_scores = y_pred_proba[y_test == 0]
    ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)

    metrics = {
        "auc_roc": round(auc, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
        "average_precision": round(ap, 4),
        "brier_score": round(brier, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} — Evaluation Results")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        logger.info(f"  {k:<25} {v}")

    return metrics


# ── Optuna Objectives ─────────────────────────────────────────────────────────

def xgb_objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "random_state": SEED,
        "eval_metric": "auc",
        "tree_method": "hist",
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
                             scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def lgbm_objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": SEED,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
                             scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ── Training Orchestrator ─────────────────────────────────────────────────────

def train(experiment_name: str = "credit_risk_v1", n_trials: int = OPTUNA_TRIALS):
    mlflow.set_experiment(experiment_name)

    # Load processed features
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    # Load validation data and apply transforms
    import sys
    sys.path.append(".")
    from src.features.build_features import (
        CreditFeatureEngineer, WOEEncoder, Winsorizer
    )
    import joblib

    val_raw = pd.read_csv("data/raw/validation.csv")
    test_raw = pd.read_csv("data/raw/test.csv")

    engineer = joblib.load("data/processed/feature_engineer.pkl")
    woe_encoder = joblib.load("data/processed/woe_encoder.pkl")
    winsorizer = joblib.load("data/processed/winsorizer.pkl")

    def transform_split(df):
        y = df["default"]
        X = df.drop(columns=["applicant_id", "default", "monthly_payment"], errors="ignore")
        X_eng = engineer.transform(X)
        cat_cols = ["home_ownership", "loan_purpose"]
        X_eng[cat_cols] = woe_encoder.transform(X_eng[cat_cols])
        X_win = winsorizer.transform(X_eng.select_dtypes(include=[np.number]))
        return X_win, y

    X_val, y_val = transform_split(val_raw)
    X_test, y_test = transform_split(test_raw)

    # ── Handle class imbalance with SMOTE ─────────────────────────────
    logger.info("Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Resampled shape: {X_resampled.shape}, positive rate: {y_resampled.mean():.2%}")

    # ── XGBoost HPO ───────────────────────────────────────────────────
    logger.info(f"Running XGBoost Optuna HPO ({n_trials} trials)...")
    with mlflow.start_run(run_name="xgb_hpo", nested=False):
        xgb_study = optuna.create_study(direction="maximize", study_name="xgb_hpo")
        xgb_study.optimize(lambda t: xgb_objective(t, X_resampled, y_resampled),
                           n_trials=n_trials, show_progress_bar=True)

        best_xgb = XGBClassifier(**xgb_study.best_params, random_state=SEED,
                                  tree_method="hist", eval_metric="auc")
        best_xgb.fit(X_resampled, y_resampled)
        xgb_metrics = evaluate_model(best_xgb, X_test, y_test, "XGBoost")
        mlflow.log_params(xgb_study.best_params)
        mlflow.log_metrics(xgb_metrics)
        mlflow.sklearn.log_model(best_xgb, "xgb_model")

    # ── LightGBM HPO ──────────────────────────────────────────────────
    logger.info(f"Running LightGBM Optuna HPO ({n_trials} trials)...")
    with mlflow.start_run(run_name="lgbm_hpo", nested=False):
        lgbm_study = optuna.create_study(direction="maximize", study_name="lgbm_hpo")
        lgbm_study.optimize(lambda t: lgbm_objective(t, X_resampled, y_resampled),
                            n_trials=n_trials, show_progress_bar=True)

        best_lgbm = LGBMClassifier(**lgbm_study.best_params, random_state=SEED,
                                    class_weight="balanced", verbose=-1)
        best_lgbm.fit(X_resampled, y_resampled)
        lgbm_metrics = evaluate_model(best_lgbm, X_test, y_test, "LightGBM")
        mlflow.log_params(lgbm_study.best_params)
        mlflow.log_metrics(lgbm_metrics)
        mlflow.sklearn.log_model(best_lgbm, "lgbm_model")

    # ── Stacking Ensemble ─────────────────────────────────────────────
    logger.info("Training stacking ensemble...")
    with mlflow.start_run(run_name="stacking_ensemble"):
        estimators = [
            ("xgb", best_xgb),
            ("lgbm", best_lgbm),
        ]
        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        stacker = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=False,
        )
        stacker.fit(X_resampled, y_resampled)
        ensemble_metrics = evaluate_model(stacker, X_test, y_test, "Stacking Ensemble")

        mlflow.log_metrics(ensemble_metrics)
        mlflow.log_params({"meta_learner": "LogisticRegression", "base_models": "XGB+LGBM"})
        mlflow.sklearn.log_model(stacker, "stacking_ensemble",
                                  registered_model_name="CreditRiskScorer")

        # Save locally for API serving
        Path("models").mkdir(exist_ok=True)
        joblib.dump(stacker, "models/stacking_ensemble.pkl")
        joblib.dump(list(X_train.columns), "models/feature_names.pkl")
        logger.success(f"Ensemble AUC-ROC: {ensemble_metrics['auc_roc']}")

    logger.success("Training complete! View results at http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="credit_risk_v1")
    parser.add_argument("--n-trials", type=int, default=OPTUNA_TRIALS)
    args = parser.parse_args()
    train(args.experiment_name, args.n_trials)
