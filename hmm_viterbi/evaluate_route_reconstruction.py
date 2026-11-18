"""
PathSnap: HMM Route Reconstruction Evaluation
Author: Teammate 3 (Sequence Modeling Lead)
Description:
    Runs sequential HMM Viterbi decoding across all 10 student-test trips,
    comparing Classifier-Only vs Classifier+HMM route reconstruction.
"""

import os
import sys
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.dataset import load_single_trip
from features.feature_extractor import FEATURE_COLUMNS
from hmm_viterbi.hmm_decoder import HMMViterbiDecoder, RoadNetworkGraph

def evaluate_route_reconstruction(
    test_features_path: str = "artifacts/student_test_features.pkl",
    model_path: str = "artifacts/best_classifier.joblib",
    results_dir: str = "results"
):
    os.makedirs(f"{results_dir}/tables", exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    
    print("[HMM Evaluation] Loading student-test features and champion model...")
    test_df = pd.read_pickle(test_features_path)
    model = joblib.load(model_path)
    
    # Predict candidate probabilities
    X_test = test_df[FEATURE_COLUMNS].values
    test_probs = model.predict_proba(X_test)[:, 1]
    test_df['pred_prob'] = test_probs
    
    decoder = HMMViterbiDecoder(
        sigma_z=10.0,
        beta=15.0,
        classifier_weight=0.85,
        stay_log_bonus=6.0,
        connected_log_bonus=3.5,
        two_hop_bonus=2.5,
        reverse_penalty=-100.0,
        unconnected_penalty=-40.0
    )
    
    trip_results = []
    test_dirs = sorted(glob.glob("data/student_test/*"))
    
    print(f"[HMM Evaluation] Decoding {len(test_dirs)} student-test trajectories...")
    for td in test_dirs:
        trip_id = os.path.basename(td)
        trip_data = load_single_trip(td, partition="student_test")
        sub_df = test_df[test_df['trip_id'] == trip_id].copy()
        
        if len(sub_df) == 0:
            continue
            
        graph = RoadNetworkGraph(trip_data.arcs_df)
        v_arcs, c_arcs, diag = decoder.decode_trip(
            trip=trip_data,
            candidates_df=sub_df,
            classifier_probs=sub_df['pred_prob'].values,
            graph=graph
        )
        
        trip_results.append({
            "Trip ID": trip_id,
            "Observations": len(trip_data.track_x),
            "Classifier-Only Top-1 Acc": round(diag['classifier_only_accuracy'], 4),
            "HMM-Viterbi Top-1 Acc": round(diag['viterbi_accuracy'], 4),
            "Accuracy Delta (%)": round((diag['viterbi_accuracy'] - diag['classifier_only_accuracy']) * 100, 2),
            "Classifier-Only Continuity": round(diag['classifier_only_route_continuity'], 4),
            "HMM-Viterbi Continuity": round(diag['viterbi_route_continuity'], 4),
            "Continuity Delta (%)": round((diag['viterbi_route_continuity'] - diag['classifier_only_route_continuity']) * 100, 2)
        })
        print(f"  Trip {trip_id}: Clf Acc={diag['classifier_only_accuracy']*100:.1f}% -> HMM Acc={diag['viterbi_accuracy']*100:.1f}% | Clf Cont={diag['classifier_only_route_continuity']*100:.1f}% -> HMM Cont={diag['viterbi_route_continuity']*100:.1f}%")
        
    res_df = pd.DataFrame(trip_results)
    
    # Append overall mean summary row
    mean_row = {
        "Trip ID": "MEAN / OVERALL",
        "Observations": int(res_df['Observations'].sum()),
        "Classifier-Only Top-1 Acc": round(res_df['Classifier-Only Top-1 Acc'].mean(), 4),
        "HMM-Viterbi Top-1 Acc": round(res_df['HMM-Viterbi Top-1 Acc'].mean(), 4),
        "Accuracy Delta (%)": round(res_df['Accuracy Delta (%)'].mean(), 2),
        "Classifier-Only Continuity": round(res_df['Classifier-Only Continuity'].mean(), 4),
        "HMM-Viterbi Continuity": round(res_df['HMM-Viterbi Continuity'].mean(), 4),
        "Continuity Delta (%)": round(res_df['Continuity Delta (%)'].mean(), 2)
    }
    summary_df = pd.concat([res_df, pd.DataFrame([mean_row])], ignore_index=True)
    summary_df.to_csv(f"{results_dir}/tables/hmm_reconstruction_comparison.csv", index=False)
    
    print("\n[HMM vs Classifier Route Reconstruction Summary Table]")
    print(summary_df.to_string())
    
    # Plot Reconstruction Continuity & Accuracy comparison
    trips = res_df['Trip ID'].values
    clf_cont = res_df['Classifier-Only Continuity'].values * 100
    hmm_cont = res_df['HMM-Viterbi Continuity'].values * 100
    
    x = np.arange(len(trips))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, clf_cont, width, label='Classifier-Only Continuity (%)', color='#f97316')
    ax.bar(x + width/2, hmm_cont, width, label='HMM+Viterbi Continuity (%)', color='#10b981')
    ax.set_ylabel('Route Continuity (%)', fontsize=12)
    ax.set_ylim(70, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(trips, rotation=30, fontsize=9)
    ax.legend(loc='lower right')
    plt.title('Impact of HMM Viterbi Sequential Decoding on Route Continuity', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/hmm_route_continuity_impact.png", dpi=300)
    plt.close()
    
    return summary_df

if __name__ == '__main__':
    evaluate_route_reconstruction()
