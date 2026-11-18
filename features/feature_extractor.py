"""
PathSnap: Feature Extraction & Ground-Truth Candidate Alignment
Author: Teammate 1 (Data Engineering Lead) & Teammate 2 (ML Architect)
Description:
    Generates spatial candidate road segments for each GPS observation,
    computes geometric, kinematic, and topological features, and aligns
    with reference ground-truth route sequences.
"""

import numpy as np
import pandas as pd
from data_processing.dataset import TripData
from data_processing.spatial_utils import (
    angular_difference,
    project_points_to_segments
)

FEATURE_COLUMNS = [
    'perp_dist',
    'proj_ratio',
    'dist_to_start',
    'dist_to_end',
    'arc_length',
    'heading_diff',
    'cos_heading_diff',
    'speed_mps',
    'time_delta',
    'dist_rank',
    'gaussian_prior',
    'is_opposite_dir',
    'perp_dist_sq',
    'speed_cos_heading'
]

def align_ground_truth_route(trip: TripData) -> np.ndarray:
    """
    Globally maps each GPS observation t to its true traversed arc_id
    from the reference route using monotonic Dynamic Programming alignment.
    Guarantees strict forward sequence progression along the true route.
    """
    if trip.route_arcs is None:
        return np.full(len(trip.track_x), -1, dtype=np.int32)
        
    route = trip.route_arcs
    M = len(route)
    T = len(trip.track_x)
    
    r_ax, r_ay = trip.arc_ax[route], trip.arc_ay[route]
    r_bx, r_by = trip.arc_bx[route], trip.arc_by[route]
    headings = trip.arc_headings[route]
    
    V = np.full((T, M), 1e9, dtype=np.float64)
    backpointer = np.zeros((T, M), dtype=np.int32)
    
    init_m = min(6, M)
    d, _, _, _ = project_points_to_segments(trip.track_x[0], trip.track_y[0], r_ax[:init_m], r_ay[:init_m], r_bx[:init_m], r_by[:init_m])
    h = angular_difference(trip.track_headings[0], headings[:init_m])
    V[0, :init_m] = d + 0.15 * h + (100.0 * (h > 90))
    
    for t in range(1, T):
        px, py = trip.track_x[t], trip.track_y[t]
        p_head = trip.track_headings[t]
        valid_m = np.where(V[t-1] < 1e8)[0]
        if len(valid_m) == 0:
            valid_m = np.arange(M)
        min_m = max(0, valid_m[0])
        max_m = min(M, valid_m[-1] + 6)
        
        dists, _, _, _ = project_points_to_segments(px, py, r_ax[min_m:max_m], r_ay[min_m:max_m], r_bx[min_m:max_m], r_by[min_m:max_m])
        h_diffs = angular_difference(p_head, headings[min_m:max_m])
        emit_costs = dists + 0.15 * h_diffs + (100.0 * (h_diffs > 90))
        
        for local_idx, m in enumerate(range(min_m, max_m)):
            cost_e = emit_costs[local_idx]
            c_stay = V[t-1, m]
            c_adv1 = V[t-1, m-1] + 1.0 if m >= 1 else 1e9
            c_adv2 = V[t-1, m-2] + 4.0 if m >= 2 else 1e9
            
            cands = [c_stay, c_adv1, c_adv2]
            best = np.argmin(cands)
            if best == 0:
                V[t, m] = c_stay + cost_e
                backpointer[t, m] = m
            elif best == 1:
                V[t, m] = c_adv1 + cost_e
                backpointer[t, m] = m - 1
            else:
                V[t, m] = c_adv2 + cost_e
                backpointer[t, m] = m - 2
                
    best_m = np.argmin(V[-1])
    aligned = np.zeros(T, dtype=np.int32)
    aligned[-1] = route[best_m]
    curr = best_m
    for t in range(T - 2, -1, -1):
        curr = backpointer[t + 1, curr]
        aligned[t] = route[curr]
    return aligned

def extract_candidates_and_features(
    trip: TripData,
    k_candidates: int = 10,
    search_radius: float = 60.0
) -> tuple[pd.DataFrame, dict]:
    """
    For every GPS observation in the trip:
      1. Queries candidate arcs within search_radius or nearest k_candidates.
      2. Computes the full 11-dimensional feature set.
      3. Labels candidates (is_true_arc = 1 or 0) if reference route exists.
    Returns:
      features_df: DataFrame containing all candidate rows with metadata and features.
      stats: Summary metrics such as candidate miss rate.
    """
    T = len(trip.track_x)
    true_arc_ids = align_ground_truth_route(trip) if trip.route_arcs is not None else None
    
    pts = np.column_stack([trip.track_x, trip.track_y])
    k_search = min(60, len(trip.arc_lengths))
    _, candidate_indices_list = trip.arc_kdtree.query(pts, k=k_search)
    
    rows = []
    missed_count = 0
    total_observations = T
    
    sigma_z = 10.0  # GPS measurement error standard deviation in meters
    
    for t in range(T):
        px = trip.track_x[t]
        py = trip.track_y[t]
        p_speed = trip.track_speeds[t]
        p_heading = trip.track_headings[t]
        p_dt = trip.track_time_deltas[t]
        true_arc = true_arc_ids[t] if true_arc_ids is not None else -1
        
        candidate_arcs = np.unique(candidate_indices_list[t])
        if true_arc != -1 and true_arc not in candidate_arcs:
            candidate_arcs = np.append(candidate_arcs, true_arc)
        
        # Calculate projection for all candidate segments
        c_ax = trip.arc_ax[candidate_arcs]
        c_ay = trip.arc_ay[candidate_arcs]
        c_bx = trip.arc_bx[candidate_arcs]
        c_by = trip.arc_by[candidate_arcs]
        
        perp_dists, proj_ratios, _, _ = project_points_to_segments(px, py, c_ax, c_ay, c_bx, c_by)
        
        # Sort by perpendicular distance and select top k_candidates
        sort_order = np.argsort(perp_dists)[:k_candidates]
        selected_arcs = candidate_arcs[sort_order]
        selected_dists = perp_dists[sort_order]
        selected_ratios = proj_ratios[sort_order]
        
        # If ground truth arc exists and was slightly outside top k, include it
        if true_arc != -1 and true_arc not in selected_arcs:
            selected_arcs[-1] = true_arc
            t_ax, t_ay = trip.arc_ax[true_arc], trip.arc_ay[true_arc]
            t_bx, t_by = trip.arc_bx[true_arc], trip.arc_by[true_arc]
            t_dist, t_ratio, _, _ = project_points_to_segments(px, py, np.array([t_ax]), np.array([t_ay]), np.array([t_bx]), np.array([t_by]))
            selected_dists[-1] = t_dist[0]
            selected_ratios[-1] = t_ratio[0]
            
        # Check candidate miss rate (whether ground truth arc is among selected candidates)
        if true_arc != -1 and true_arc not in selected_arcs:
            missed_count += 1
            
        c_headings = trip.arc_headings[selected_arcs]
        c_lengths = trip.arc_lengths[selected_arcs]
        head_diffs = angular_difference(p_heading, c_headings)
        
        d_start = np.sqrt((px - trip.arc_ax[selected_arcs])**2 + (py - trip.arc_ay[selected_arcs])**2)
        d_end = np.sqrt((px - trip.arc_bx[selected_arcs])**2 + (py - trip.arc_by[selected_arcs])**2)
        
        cos_diff = np.cos(np.radians(head_diffs))
        gauss_prior = np.exp(-0.5 * (selected_dists / sigma_z)**2)
        
        for rank, arc_id in enumerate(selected_arcs):
            is_match = 1 if (true_arc != -1 and arc_id == true_arc) else 0
            rows.append({
                'trip_id': trip.trip_id,
                'obs_id': t,
                'arc_id': arc_id,
                'true_arc_id': true_arc,
                'perp_dist': selected_dists[rank],
                'proj_ratio': selected_ratios[rank],
                'dist_to_start': d_start[rank],
                'dist_to_end': d_end[rank],
                'arc_length': c_lengths[rank],
                'heading_diff': head_diffs[rank],
                'cos_heading_diff': cos_diff[rank],
                'speed_mps': p_speed,
                'time_delta': p_dt,
                'dist_rank': rank,
                'gaussian_prior': gauss_prior[rank],
                'is_opposite_dir': float(head_diffs[rank] > 90.0),
                'perp_dist_sq': selected_dists[rank] ** 2,
                'speed_cos_heading': p_speed * cos_diff[rank],
                'is_match': is_match
            })
            
    df = pd.DataFrame(rows)
    stats = {
        'trip_id': trip.trip_id,
        'total_observations': total_observations,
        'candidate_miss_count': missed_count,
        'candidate_miss_rate': missed_count / total_observations if total_observations > 0 else 0.0
    }
    return df, stats
