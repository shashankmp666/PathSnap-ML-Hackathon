"""
PathSnap: Comprehensive Evaluator Test Benchmark
Author: PathSnap Team (Teammate 1, 2, 3, 4)
Description:
    Runs full end-to-end map-matching evaluation on the official evaluator_test dataset:
    - Candidate extraction and 11D feature extraction
    - Champion Classifier scoring
    - Pointwise Candidate Classification Accuracy and F1-score
    - Topological HMM Viterbi Sequential Decoding
    - Observation-level Top-1 Accuracy and Route Continuity
    - Interactive Folium map generation
"""

import os
import sys
import time
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_processing.dataset import load_single_trip
from features.feature_extractor import extract_candidates_and_features, FEATURE_COLUMNS
from hmm_viterbi.hmm_decoder import HMMViterbiDecoder, RoadNetworkGraph
from visualization.map_visualizer import create_folium_trajectory_map

def run_evaluator_benchmark(
    evaluator_dir: str = "data/evaluator_test",
    model_path: str = "artifacts/tuned_champion_model.joblib",
    fallback_model_path: str = "artifacts/best_classifier.joblib",
    output_table_path: str = "results/tables/evaluator_test_benchmark.csv",
    maps_dir: str = "results/maps"
):
    os.makedirs(os.path.dirname(output_table_path), exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    # 1. Load model
    selected_model_path = model_path if os.path.exists(model_path) else fallback_model_path
    print(f"\n[Evaluator Benchmark] Loading champion classifier model: {selected_model_path}...")
    model = joblib.load(selected_model_path)

    # 2. Initialize HMM Decoder
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

    trip_dirs = sorted(glob.glob(os.path.join(evaluator_dir, "*")))
    trip_dirs = [d for d in trip_dirs if os.path.isdir(d)]
    print(f"[Evaluator Benchmark] Found {len(trip_dirs)} trips in '{evaluator_dir}'")

    results = []

    all_y_true = []
    all_y_pred = []

    for td in trip_dirs:
        trip_id = os.path.basename(td)
        t0 = time.time()
        trip_data = load_single_trip(td, partition="evaluator_test")
        load_time = time.time() - t0

        t1 = time.time()
        cand_df, stats = extract_candidates_and_features(trip_data, k_candidates=10)
        feat_time = time.time() - t1

        t2 = time.time()
        X = cand_df[FEATURE_COLUMNS].values
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        cand_df['pred_prob'] = probs
        cand_df['pred_label'] = preds
        infer_time = time.time() - t2

        # Pointwise candidate metrics
        has_gt = 'is_match' in cand_df.columns and (cand_df['is_match'] >= 0).any() and (trip_data.route_arcs is not None)
        if has_gt:
            y_true = cand_df['is_match'].values
            cand_acc = accuracy_score(y_true, preds)
            cand_f1_weighted = f1_score(y_true, preds, average='weighted', zero_division=0)
            cand_f1_macro = f1_score(y_true, preds, average='macro', zero_division=0)
            all_y_true.extend(y_true)
            all_y_pred.extend(preds)
        else:
            cand_acc, cand_f1_weighted, cand_f1_macro = np.nan, np.nan, np.nan

        # HMM Viterbi Sequential Decoding
        t3 = time.time()
        graph = RoadNetworkGraph(trip_data.arcs_df)
        v_arcs, c_arcs, diag = decoder.decode_trip(
            trip=trip_data,
            candidates_df=cand_df,
            classifier_probs=probs,
            graph=graph
        )
        hmm_time = time.time() - t3
        total_time_ms = (time.time() - t0) * 1000.0

        # Unique traversed arcs
        unique_v_arcs = []
        for a in v_arcs:
            if not unique_v_arcs or unique_v_arcs[-1] != int(a):
                unique_v_arcs.append(int(a))

        # Generate Folium map
        map_path = os.path.join(maps_dir, f"evaluator_trip_{trip_id}_map.html")
        create_folium_trajectory_map(trip_data, predicted_arcs=v_arcs, output_html_path=map_path)

        res_entry = {
            "Trip ID": trip_id,
            "Observations": len(trip_data.track_x),
            "Traversed Arcs": len(unique_v_arcs),
            "Cand Recall (%)": round((1.0 - stats['candidate_miss_rate']) * 100, 2),
            "Cand Miss Rate (%)": round(stats['candidate_miss_rate'] * 100, 2),
            "Classification Acc (%)": round(cand_acc * 100, 2) if not np.isnan(cand_acc) else None,
            "Weighted F1 (%)": round(cand_f1_weighted * 100, 2) if not np.isnan(cand_f1_weighted) else None,
            "Top-1 Acc (Clf) (%)": round(diag['classifier_only_accuracy'] * 100, 2),
            "Top-1 Acc (HMM) (%)": round(diag['viterbi_accuracy'] * 100, 2),
            "Continuity (Clf) (%)": round(diag['classifier_only_route_continuity'] * 100, 2),
            "Continuity (HMM) (%)": round(diag['viterbi_route_continuity'] * 100, 2),
            "Continuity Gain (%)": round((diag['viterbi_route_continuity'] - diag['classifier_only_route_continuity']) * 100, 2),
            "Total Latency (ms)": round(total_time_ms, 1)
        }
        results.append(res_entry)

        print(f"  Trip {trip_id} [{len(trip_data.track_x)} pts]: "
              f"CandAcc={res_entry['Classification Acc (%)']}% | "
              f"WeightedF1={res_entry['Weighted F1 (%)']}% | "
              f"Top-1(HMM)={res_entry['Top-1 Acc (HMM) (%)']}% | "
              f"Continuity(HMM)={res_entry['Continuity (HMM) (%)']}% | "
              f"Latency={total_time_ms:.1f}ms")

    df = pd.DataFrame(results)

    # Compute overall summary
    summary_row = {
        "Trip ID": "MEAN / OVERALL",
        "Observations": int(df['Observations'].sum()),
        "Traversed Arcs": int(df['Traversed Arcs'].sum()),
        "Cand Recall (%)": round(df['Cand Recall (%)'].mean(), 2),
        "Cand Miss Rate (%)": round(df['Cand Miss Rate (%)'].mean(), 2),
        "Classification Acc (%)": round(df['Classification Acc (%)'].mean(), 2),
        "Weighted F1 (%)": round(df['Weighted F1 (%)'].mean(), 2),
        "Top-1 Acc (Clf) (%)": round(df['Top-1 Acc (Clf) (%)'].mean(), 2),
        "Top-1 Acc (HMM) (%)": round(df['Top-1 Acc (HMM) (%)'].mean(), 2),
        "Continuity (Clf) (%)": round(df['Continuity (Clf) (%)'].mean(), 2),
        "Continuity (HMM) (%)": round(df['Continuity (HMM) (%)'].mean(), 2),
        "Continuity Gain (%)": round(df['Continuity Gain (%)'].mean(), 2),
        "Total Latency (ms)": round(df['Total Latency (ms)'].mean(), 1)
    }

    if len(all_y_true) > 0:
        overall_cand_acc = accuracy_score(all_y_true, all_y_pred) * 100
        overall_cand_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0) * 100
        summary_row["Classification Acc (%)"] = round(overall_cand_acc, 2)
        summary_row["Weighted F1 (%)"] = round(overall_cand_f1, 2)

    df_full = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    df_full.to_csv(output_table_path, index=False)

    print("\n" + "="*95)
    print("                     EVALUATOR TEST BENCHMARK AUDIT REPORT")
    print("="*95)
    print(df_full.to_string(index=False))
    print("="*95)
    print(f"[Done] Benchmark table saved to: {output_table_path}")
    print(f"[Done] Interactive maps saved to: {maps_dir}/evaluator_trip_<id>_map.html\n")
    return df_full

if __name__ == '__main__':
    run_evaluator_benchmark()
