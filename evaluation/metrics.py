"""
PathSnap: Evaluation Metrics and Benchmark Scoring
Author: Teammate 2 (ML Architect) & Teammate 3 (Sequence Modeling)
Description:
    Computes observation-level Top-1 accuracy, candidate Macro F1, Log Loss,
    candidate miss rate, inference latency, and sequential route continuity.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

def compute_observation_top1_accuracy(df: pd.DataFrame, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    """
    Computes Observation-level Top-1 Accuracy:
    For each GPS observation, picks the candidate with the highest predicted probability.
    Returns:
        accuracy: Fraction of observations where the top-scoring candidate is the true arc.
        top1_df: DataFrame of top-1 chosen candidates per observation.
    """
    df_eval = df[['trip_id', 'obs_id', 'arc_id', 'is_match']].copy()
    df_eval['score'] = scores
    
    # Sort descending by score, then pick the first candidate for each (trip_id, obs_id)
    df_sorted = df_eval.sort_values(by=['trip_id', 'obs_id', 'score'], ascending=[True, True, False])
    top1 = df_sorted.drop_duplicates(subset=['trip_id', 'obs_id'], keep='first')
    
    acc = top1['is_match'].mean()
    return float(acc), top1

def compute_route_continuity(top1_df: pd.DataFrame, arcs_dict: dict[str, pd.DataFrame]) -> float:
    """
    Computes sequence-level route continuity:
    Percentage of consecutive observation transitions where the vehicle either:
      1. Stays on the identical road segment (arc_t == arc_{t+1})
      2. Moves onto an adjacent connected segment (to_node(arc_t) == from_node(arc_{t+1}))
    """
    total_transitions = 0
    connected_transitions = 0
    
    for trip_id, group in top1_df.groupby('trip_id'):
        ordered_arcs = group.sort_values('obs_id')['arc_id'].values
        if len(ordered_arcs) <= 1:
            continue
            
        arcs_df = arcs_dict.get(trip_id)
        if arcs_df is None:
            continue
            
        u_nodes = arcs_df['u'].values
        v_nodes = arcs_df['v'].values
        
        for t in range(len(ordered_arcs) - 1):
            a1 = ordered_arcs[t]
            a2 = ordered_arcs[t + 1]
            total_transitions += 1
            
            if a1 == a2:
                connected_transitions += 1
            else:
                # Check topological link
                if v_nodes[a1] == u_nodes[a2]:
                    connected_transitions += 1
                    
    return connected_transitions / total_transitions if total_transitions > 0 else 0.0

def evaluate_model_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    arcs_dict: dict[str, pd.DataFrame] | None = None
) -> dict[str, float]:
    """
    Consolidated evaluation covering all required metrics in the hackathon syllabus.
    """
    y_true = df['is_match'].values
    y_pred_binary = (probs >= 0.5).astype(int)
    
    # Clip probs to prevent log loss explosion
    clipped_probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    loss = float(log_loss(y_true, clipped_probs))
    
    macro_f1 = float(f1_score(y_true, y_pred_binary, average='macro', zero_division=0))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    
    top1_acc, top1_df = compute_observation_top1_accuracy(df, probs)
    
    continuity = 0.0
    if arcs_dict is not None:
        continuity = compute_route_continuity(top1_df, arcs_dict)
        
    return {
        'top1_accuracy': top1_acc,
        'macro_f1': macro_f1,
        'precision': prec,
        'recall': rec,
        'log_loss': loss,
        'route_continuity': continuity
    }
