"""
PathSnap: Multi-Model Training and Benchmark Evaluation Pipeline
Author: Teammate 2 (ML Architect)
Description:
    Trains 5 distinct machine learning algorithm families using GroupKFold
    cross-validation, records all performance metrics, and logs results.
"""

import os
import sys
import time
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from features.feature_extractor import FEATURE_COLUMNS
from evaluation.metrics import evaluate_model_predictions

def load_test_arcs_dict() -> dict[str, pd.DataFrame]:
    """Loads arc topology for student-test trips to evaluate route continuity."""
    arcs_dict = {}
    test_dirs = glob.glob("data/student_test/*")
    for td in test_dirs:
        tid = os.path.basename(td)
        arcs_file = os.path.join(td, f"{tid}.arcs")
        if os.path.exists(arcs_file):
            arcs_dict[tid] = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['u', 'v'])
    return arcs_dict

def get_models_suite() -> dict[str, dict]:
    """
    Returns 5 distinct algorithm families spanning:
    1. Linear/Probabilistic, 2. Instance-Based, 3. Tree-Based, 4. Ensemble, 5. Boosting
    """
    return {
        "Logistic_Regression": {
            "family": "Linear & Probabilistic",
            "config": "C=2.0, penalty=l2, max_iter=1000",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=2.0, penalty='l2', max_iter=1000, random_state=42))
            ]),
            "subsample_train": None
        },
        "KNN_Distance_Weighted": {
            "family": "Instance-Based",
            "config": "k=15, weights=distance, metric=euclidean",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1))
            ]),
            "subsample_train": 50000  # Subsample for KNN scaling efficiency
        },
        "Decision_Tree": {
            "family": "Tree-Based",
            "config": "criterion=gini, max_depth=12, min_samples_leaf=15",
            "model": DecisionTreeClassifier(max_depth=12, min_samples_leaf=15, random_state=42),
            "subsample_train": None
        },
        "Random_Forest": {
            "family": "Ensemble",
            "config": "n_estimators=100, max_depth=15, min_samples_leaf=4",
            "model": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=4, n_jobs=-1, random_state=42),
            "subsample_train": 180000
        },
        "Hist_Gradient_Boosting": {
            "family": "Boosting",
            "config": "max_iter=150, learning_rate=0.08, max_leaf_nodes=63",
            "model": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=25, l2_regularization=0.2, random_state=42),
            "subsample_train": None
        }
    }

def train_and_benchmark_models(
    train_path: str = "artifacts/train_features.pkl",
    test_path: str = "artifacts/student_test_features.pkl",
    results_dir: str = "results"
):
    os.makedirs(f"{results_dir}/tables", exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    print("[Pipeline] Loading precomputed feature datasets...")
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)
    
    X_train_full = train_df[FEATURE_COLUMNS].values
    y_train_full = train_df['is_match'].values
    groups_full = train_df['trip_id'].values
    
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df['is_match'].values
    
    arcs_dict = load_test_arcs_dict()
    models = get_models_suite()
    
    benchmark_results = []
    trained_estimators = {}
    test_predictions_dict = {}
    
    # Setup 5-fold GroupKFold cross-validation
    gkf = GroupKFold(n_splits=5)
    
    for name, spec in models.items():
        print(f"\n=======================================================")
        print(f"  Training Model: {name} ({spec['family']})")
        print(f"  Config: {spec['config']}")
        print(f"=======================================================")
        
        # Prepare training data (with subsampling if required by algorithm complexity)
        if spec['subsample_train'] is not None and len(train_df) > spec['subsample_train']:
            idx = np.random.RandomState(42).choice(len(train_df), spec['subsample_train'], replace=False)
            X_train = X_train_full[idx]
            y_train = y_train_full[idx]
            groups = groups_full[idx]
        else:
            X_train = X_train_full
            y_train = y_train_full
            groups = groups_full
            
        # 1. Group-aware Cross Validation
        cv_macro_f1s = []
        cv_top1_accs = []
        
        print("  Running 5-Fold GroupKFold Cross-Validation...")
        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
            fold_model = spec['model']
            # Re-instantiate or clone model for fold
            from sklearn.base import clone
            fold_clf = clone(fold_model)
            
            fold_clf.fit(X_train[trn_idx], y_train[trn_idx])
            val_probs = fold_clf.predict_proba(X_train[val_idx])[:, 1]
            
            val_sub_df = train_df.iloc[val_idx] if spec['subsample_train'] is None else train_df.iloc[idx[val_idx]]
            fold_metrics = evaluate_model_predictions(val_sub_df, val_probs)
            cv_macro_f1s.append(fold_metrics['macro_f1'])
            cv_top1_accs.append(fold_metrics['top1_accuracy'])
            
        mean_cv_f1 = float(np.mean(cv_macro_f1s))
        mean_cv_top1 = float(np.mean(cv_top1_accs))
        print(f"  -> Mean CV Macro F1: {mean_cv_f1:.4f} | Mean CV Top-1: {mean_cv_top1:.4f}")
        
        # 2. Fit on full training partition
        t_start = time.time()
        final_model = spec['model']
        final_model.fit(X_train, y_train)
        train_time = time.time() - t_start
        trained_estimators[name] = final_model
        
        # 3. Evaluate on Student-Test Partition
        t_infer = time.time()
        test_probs = final_model.predict_proba(X_test)[:, 1]
        inference_time = (time.time() - t_infer) * 1000.0 / len(test_df) # ms per candidate
        test_predictions_dict[name] = test_probs
        
        test_metrics = evaluate_model_predictions(test_df, test_probs, arcs_dict=arcs_dict)
        print(f"  -> Student-Test Classification Acc: {test_metrics['candidate_accuracy']*100:.2f}%")
        print(f"  -> Student-Test Weighted F1:       {test_metrics['weighted_f1']:.4f}")
        print(f"  -> Student-Test Macro F1:          {test_metrics['macro_f1']:.4f}")
        print(f"  -> Student-Test Top-1 Accuracy:    {test_metrics['top1_accuracy']*100:.2f}%")
        print(f"  -> Student-Test Route Contin:      {test_metrics['route_continuity']*100:.2f}%")
        print(f"  -> Student-Test Log Loss:          {test_metrics['log_loss']:.4f}")
        print(f"  -> Training Time:                  {train_time:.2f}s")
        print(f"  -> Latency per Candidate:          {inference_time:.4f}ms")
        
        benchmark_results.append({
            "Model": name,
            "Algorithm Family": spec['family'],
            "Key Configuration": spec['config'],
            "CV Macro F1-score": round(mean_cv_f1, 4),
            "Classification Accuracy": round(test_metrics['candidate_accuracy'], 4),
            "F1-Score (Weighted)": round(test_metrics['weighted_f1'], 4),
            "Top-1 Accuracy": round(test_metrics['top1_accuracy'], 4),
            "Route continuity": round(test_metrics['route_continuity'], 4),
            "Student-Test Macro F1-score": round(test_metrics['macro_f1'], 4),
            "Log Loss": round(test_metrics['log_loss'], 4),
            "Inference Time (ms/cand)": round(inference_time, 4),
            "Training Time (s)": round(train_time, 2)
        })
        
    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df.to_csv(f"{results_dir}/tables/model_comparison.csv", index=False)
    print("\n[Benchmark Summary Table]")
    print(benchmark_df.to_string())
    
    # Save champion model (Hist_Gradient_Boosting / Random_Forest)
    best_row = benchmark_df.sort_values(by="Top-1 Accuracy", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    print(f"\n[Model Selection] Champion Model: {best_model_name} (Top-1: {best_row['Top-1 Accuracy']*100:.2f}%)")
    
    joblib.dump(trained_estimators[best_model_name], "artifacts/best_classifier.joblib")
    joblib.dump(trained_estimators[best_model_name], "artifacts/tuned_champion_model.joblib")
    joblib.dump(trained_estimators, "artifacts/all_trained_models.joblib")
    
    # Generate Diagnostic Visualizations
    generate_comparison_plots(benchmark_df, test_df, test_predictions_dict, results_dir)
    return benchmark_df

def generate_comparison_plots(benchmark_df: pd.DataFrame, test_df: pd.DataFrame, test_preds: dict, results_dir: str):
    """
    Renders and saves all required model comparison and diagnostic figures.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    models = benchmark_df['Model'].values
    cv_f1 = benchmark_df['CV Macro F1-score'].values
    test_acc = benchmark_df['Top-1 Accuracy'].values * 100
    test_f1 = benchmark_df['Student-Test Macro F1-score'].values
    log_loss = benchmark_df['Log Loss'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    # 1. Top-1 Accuracy and F1 Comparison Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    rects1 = ax1.bar(x - width/2, test_acc, width, label='Top-1 Accuracy (%)', color='#2563eb')
    ax1.set_ylabel('Accuracy (%)', color='#2563eb', fontsize=12)
    ax1.set_ylim(0, 105)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, test_f1, width, label='Macro F1', color='#10b981')
    ax2.set_ylabel('Macro F1-Score', color='#10b981', fontsize=12)
    ax2.set_ylim(0, 1.05)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
    plt.title('PathSnap: Model Benchmark Comparison on Student-Testing Partition', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{results_dir}/plots/model_accuracy_f1_comparison.png", dpi=300)
    plt.close()
    
    # 2. Cross-Validation Macro F1 vs Student-Test F1 Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, cv_f1, width, label='5-Fold CV Macro F1', color='#8b5cf6')
    ax.bar(x + width/2, test_f1, width, label='Student-Test Macro F1', color='#06b6d4')
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    plt.title('Cross-Validation vs Student-Test Generalization Performance', fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{results_dir}/plots/cv_vs_test_f1_comparison.png", dpi=300)
    plt.close()
    
    # 3. Log Loss Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, log_loss, color='#ef4444', width=0.5)
    ax.set_ylabel('Binary Cross-Entropy (Log Loss)', fontsize=12)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)
    plt.title('Student-Testing Log Loss Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{results_dir}/plots/log_loss_comparison.png", dpi=300)
    plt.close()
    
    # 4. Confusion Matrices for Champion and Baseline
    y_true = test_df['is_match'].values
    for m_name in ['Hist_Gradient_Boosting', 'Logistic_Regression']:
        if m_name in test_preds:
            y_pred_bin = (test_preds[m_name] >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred_bin)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Distractor (0)', 'True Arc (1)'])
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            plt.title(f'Diagnostic Confusion Matrix: {m_name}', fontsize=12, fontweight='bold')
            fig.tight_layout()
            plt.savefig(f"{results_dir}/plots/confusion_matrix_{m_name}.png", dpi=300)
            plt.close()

if __name__ == '__main__':
    train_and_benchmark_models()
