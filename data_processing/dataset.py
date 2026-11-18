"""
PathSnap: Trip Dataset Ingestion and Road Network Graph
Author: Teammate 1 (Data Engineering Lead)
Description:
    Loads trip track points, node coordinate tables, directed arc lists,
    and reference ground-truth routes with validation and graph representation.
"""

import os
import glob
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from data_processing.spatial_utils import (
    to_metric_coordinates,
    compute_azimuth_heading,
    project_points_to_segments
)

@dataclass
class TripData:
    trip_id: str
    partition: str
    track_df: pd.DataFrame
    nodes_df: pd.DataFrame
    arcs_df: pd.DataFrame
    route_arcs: np.ndarray | None  # None if evaluator partition without route
    
    # Metric Cartesian coordinates
    track_x: np.ndarray
    track_y: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    
    # Kinematics
    track_speeds: np.ndarray
    track_headings: np.ndarray
    track_time_deltas: np.ndarray
    
    # Arc geometry
    arc_ax: np.ndarray
    arc_ay: np.ndarray
    arc_bx: np.ndarray
    arc_by: np.ndarray
    arc_lengths: np.ndarray
    arc_headings: np.ndarray
    arc_mid_x: np.ndarray
    arc_mid_y: np.ndarray
    
    # Spatial index
    arc_kdtree: cKDTree

def load_single_trip(trip_dir: str, partition: str = "train") -> TripData:
    """
    Parses and geometrically prepares all raw files for a given trip folder.
    """
    trip_id = os.path.basename(trip_dir)
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    track_df = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'], dtype=np.float64)
    nodes_df = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'], dtype=np.float64)
    arcs_df = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['u', 'v'], dtype=np.int32)
    
    route_arcs = None
    if os.path.exists(route_path):
        route_df = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'], dtype=np.int32)
        route_arcs = route_df['arc_id'].values
        
    ref_lat = track_df['lat'].mean()
    track_x, track_y = to_metric_coordinates(track_df['lon'].values, track_df['lat'].values, ref_lat)
    node_x, node_y = to_metric_coordinates(nodes_df['lon'].values, nodes_df['lat'].values, ref_lat)
    
    # GPS Kinematics
    dx = np.diff(track_x, prepend=track_x[0])
    dy = np.diff(track_y, prepend=track_y[0])
    dt = np.diff(track_df['time'].values, prepend=track_df['time'].values[0])
    dt[dt <= 0] = 1.0  # safe guard for duplicate timestamps
    
    dist_steps = np.sqrt(dx**2 + dy**2)
    track_speeds = dist_steps / dt
    track_headings = compute_azimuth_heading(dx, dy)
    
    # If first step has zero displacement, use second step heading
    if len(track_headings) > 1 and np.isclose(dist_steps[0], 0.0):
        track_headings[0] = track_headings[1]
        
    # Arc geometry
    u_idx = arcs_df['u'].values
    v_idx = arcs_df['v'].values
    arc_ax = node_x[u_idx]
    arc_ay = node_y[u_idx]
    arc_bx = node_x[v_idx]
    arc_by = node_y[v_idx]
    
    arc_dx = arc_bx - arc_ax
    arc_dy = arc_by - arc_ay
    arc_lengths = np.sqrt(arc_dx**2 + arc_dy**2)
    arc_headings = compute_azimuth_heading(arc_dx, arc_dy)
    
    arc_mid_x = (arc_ax + arc_bx) / 2.0
    arc_mid_y = (arc_ay + arc_by) / 2.0
    
    # Build 2D spatial tree on arc midpoints
    arc_kdtree = cKDTree(np.column_stack([arc_mid_x, arc_mid_y]))
    
    return TripData(
        trip_id=trip_id,
        partition=partition,
        track_df=track_df,
        nodes_df=nodes_df,
        arcs_df=arcs_df,
        route_arcs=route_arcs,
        track_x=track_x,
        track_y=track_y,
        node_x=node_x,
        node_y=node_y,
        track_speeds=track_speeds,
        track_headings=track_headings,
        track_time_deltas=dt,
        arc_ax=arc_ax,
        arc_ay=arc_ay,
        arc_bx=arc_bx,
        arc_by=arc_by,
        arc_lengths=arc_lengths,
        arc_headings=arc_headings,
        arc_mid_x=arc_mid_x,
        arc_mid_y=arc_mid_y,
        arc_kdtree=arc_kdtree
    )

def get_all_trip_ids(base_data_dir: str = "data") -> dict[str, list[str]]:
    """
    Returns sorted lists of trip IDs for train and student_test partitions.
    """
    train_paths = sorted(glob.glob(os.path.join(base_data_dir, "train", "*")))
    test_paths = sorted(glob.glob(os.path.join(base_data_dir, "student_test", "*")))
    return {
        "train": [os.path.basename(p) for p in train_paths if os.path.isdir(p)],
        "student_test": [os.path.basename(p) for p in test_paths if os.path.isdir(p)]
    }
