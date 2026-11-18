"""
PathSnap: Bayesian Hyperparameter Optimization
Author: Teammate 3 (Sequence Modeling & Bayesian Lead)
Description:
    Performs Bayesian hyperparameter optimization (Optuna TPE) on the champion
    classifier using 5-fold GroupKFold cross-validation on the 80 training trips.
"""

import os
import sys
import time
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

from features.feature_extractor import FEATURE_COLUMNS
from evaluation.metrics import evaluate_model_predictions
from models.train_all_models import load_test_arcs_dict

def run_bayesian_optimization(
    train_path: str = "artifacts/train_features.pkl",
    test_path: str = "artifacts/student_test_features.pkl",
    n_trials: int = 20,
    results_dir: str = "results"
):
    os.makedirs(f"{results_dir}/tables", exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    print("[Bayesian Opt] Loading feature matrices...")
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)
    
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df['is_match'].values
    groups = train_df['trip_id'].values
    
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df['is_match'].values
    arcs_dict = load_test_arcs_dict()
    
    # 5-fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_train, y_train, groups=groups))
    
    trials_record = []
    
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.25, log=True)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 63)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 80)
        l2_reg = trial.suggest_float("l2_regularization", 1e-4, 5.0, log=True)
        max_iter = trial.suggest_int("max_iter", 60, 140, step=20)
        
        cv_scores = []
        t_start = time.time()
        
        for trn_idx, val_idx in splits:
            clf = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_reg,
                max_iter=max_iter,
                random_state=42
            )
            clf.fit(X_train[trn_idx], y_train[trn_idx])
            val_probs = clf.predict_proba(X_train[val_idx])[:, 1]
            val_metrics = evaluate_model_predictions(train_df.iloc[val_idx], val_probs)
            cv_scores.append(val_metrics['top1_accuracy'])
            
        mean_acc = float(np.mean(cv_scores))
        trial_duration = time.time() - t_start
        
        trials_record.append({
            "trial_id": trial.number,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_reg,
            "max_iter": max_iter,
            "cv_top1_accuracy": mean_acc,
            "duration_s": trial_duration
        })
        return mean_acc

    # Suppress verbose Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    print(f"[Bayesian Opt] Commencing {n_trials} Bayesian Optimization trials with TPE...")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_cv_score = study.best_value
    print(f"\n[Bayesian Opt] Best Trial Found:")
    print(f"  CV Top-1 Accuracy: {best_cv_score*100:.2f}%")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    # Save trials dataframe
    trials_df = pd.DataFrame(trials_record)
    trials_df.to_csv(f"{results_dir}/tables/bayesian_trials.csv", index=False)
    
    # Train Untuned vs Tuned Model on full train set & test on student-test set
    print("\n[Bayesian Opt] Refitting Baseline (Untuned) vs Tuned Champion Model...")
    
    # Untuned baseline
    untuned_clf = HistGradientBoostingClassifier(random_state=42)
    untuned_clf.fit(X_train, y_train)
    t0 = time.time()
    untuned_preds = untuned_clf.predict_proba(X_test)[:, 1]
    untuned_infer_time = (time.time() - t0) * 1000.0 / len(test_df)
    untuned_metrics = evaluate_model_predictions(test_df, untuned_preds, arcs_dict=arcs_dict)
    
    # Tuned champion
    tuned_clf = HistGradientBoostingClassifier(**best_params, random_state=42)
    tuned_clf.fit(X_train, y_train)
    t1 = time.time()
    tuned_preds = tuned_clf.predict_proba(X_test)[:, 1]
    tuned_infer_time = (time.time() - t1) * 1000.0 / len(test_df)
    tuned_metrics = evaluate_model_predictions(test_df, tuned_preds, arcs_dict=arcs_dict)
    
    # Comparison table
    tuning_comparison = pd.DataFrame([
        {
            "Configuration": "Untuned Baseline (Default HGB)",
            "Top-1 Accuracy": round(untuned_metrics['top1_accuracy'], 4),
            "Macro F1-score": round(untuned_metrics['macro_f1'], 4),
            "Route Continuity": round(untuned_metrics['route_continuity'], 4),
            "Log Loss": round(untuned_metrics['log_loss'], 4),
            "Inference Time (ms/cand)": round(untuned_infer_time, 4)
        },
        {
            "Configuration": "Bayesian Tuned (Best Optuna TPE)",
            "Top-1 Accuracy": round(tuned_metrics['top1_accuracy'], 4),
            "Macro F1-score": round(tuned_metrics['macro_f1'], 4),
            "Route Continuity": round(tuned_metrics['route_continuity'], 4),
            "Log Loss": round(tuned_metrics['log_loss'], 4),
            "Inference Time (ms/cand)": round(tuned_infer_time, 4)
        }
    ])
    tuning_comparison.to_csv(f"{results_dir}/tables/tuning_comparison.csv", index=False)
    print("\n[Tuning Comparison Table]")
    print(tuning_comparison.to_string())
    
    # Save the final tuned classifier artifact
    joblib.dump(tuned_clf, "artifacts/tuned_champion_model.joblib")
    
    # Plot Bayesian optimization convergence
    plt.figure(figsize=(8, 4.5))
    running_max = trials_df['cv_top1_accuracy'].cummax() * 100
    trial_scores = trials_df['cv_top1_accuracy'] * 100
    
    plt.plot(trials_df['trial_id'], trial_scores, 'o--', color='#94a3b8', alpha=0.7, label='Trial Score')
    plt.plot(trials_df['trial_id'], running_max, 's-', color='#2563eb', linewidth=2, label='Best Running Score')
    plt.xlabel('Optimization Trial', fontsize=11)
    plt.ylabel('CV Top-1 Accuracy (%)', fontsize=11)
    plt.title('Bayesian Hyperparameter Optimization Convergence (Optuna TPE)', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/bayesian_optimization_convergence.png", dpi=300)
    plt.close()
    
    return tuning_comparison, best_params

if __name__ == '__main__':
    run_bayesian_optimization()
