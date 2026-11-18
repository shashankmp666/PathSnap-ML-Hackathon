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
    'gaussian_prior'
]

def align_ground_truth_route(trip: TripData) -> np.ndarray:
    """
    Monotonically maps each GPS observation t to its true traversed arc_id
    from the reference route sequence.
    """
    if trip.route_arcs is None:
        return np.full(len(trip.track_x), -1, dtype=np.int32)
        
    route = trip.route_arcs
    M = len(route)
    T = len(trip.track_x)
    
    # Pre-extract geometry for route arcs
    r_ax = trip.arc_ax[route]
    r_ay = trip.arc_ay[route]
    r_bx = trip.arc_bx[route]
    r_by = trip.arc_by[route]
    
    true_arc_ids = np.zeros(T, dtype=np.int32)
    curr_idx = 0
    
    for t in range(T):
        px = trip.track_x[t]
        py = trip.track_y[t]
        
        # Look ahead window from current route index (avoiding backwards jumps)
        window_end = min(curr_idx + 8, M)
        sub_ax = r_ax[curr_idx:window_end]
        sub_ay = r_ay[curr_idx:window_end]
        sub_bx = r_bx[curr_idx:window_end]
        sub_by = r_by[curr_idx:window_end]
        
        dists, _, _, _ = project_points_to_segments(px, py, sub_ax, sub_ay, sub_bx, sub_by)
        best_local_idx = int(np.argmin(dists))
        curr_idx = curr_idx + best_local_idx
        true_arc_ids[t] = route[curr_idx]
        
    return true_arc_ids

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
