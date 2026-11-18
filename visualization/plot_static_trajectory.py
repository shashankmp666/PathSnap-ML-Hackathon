"""
PathSnap: Static Trajectory Geographical Plotter
Author: Teammate 4 (Visualization Lead)
Description:
    Generates high-resolution spatial trajectory plots for embedding
    in technical reports and presentation documents.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.dataset import load_single_trip

def plot_static_trajectory_map(
    trip_id: str = "00000007",
    predicted_arcs_path: str | None = None,
    output_png_path: str = "results/plots/geographical_map_trip.png"
):
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    trip_dir = f"data/student_test/{trip_id}"
    if not os.path.exists(trip_dir):
        trip_dir = f"data/train/{trip_id}"
        
    trip = load_single_trip(trip_dir, partition="eval")
    
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    
    # Bounding box
    min_lon, max_lon = trip.track_df['lon'].min() - 0.003, trip.track_df['lon'].max() + 0.003
    min_lat, max_lat = trip.track_df['lat'].min() - 0.003, trip.track_df['lat'].max() + 0.003
    
    node_lons = trip.nodes_df['lon'].values
    node_lats = trip.nodes_df['lat'].values
    u_idx = trip.arcs_df['u'].values
    v_idx = trip.arcs_df['v'].values
    
    # 1. Plot background road network
    mask = (
        (node_lons[u_idx] >= min_lon) & (node_lons[u_idx] <= max_lon) &
        (node_lats[u_idx] >= min_lat) & (node_lats[u_idx] <= max_lat)
    )
    relevant_arcs = np.where(mask)[0]
    if len(relevant_arcs) > 1500:
        relevant_arcs = relevant_arcs[:1500]
        
    for idx, arc in enumerate(relevant_arcs):
        u, v = u_idx[arc], v_idx[arc]
        ax.plot(
            [node_lons[u], node_lons[v]],
            [node_lats[u], node_lats[v]],
            color='#cbd5e1',
            linewidth=0.9,
            alpha=0.6,
            label='Road Network' if idx == 0 else ""
        )
        
    # 2. Ground Truth Route
    if trip.route_arcs is not None:
        gt_lons, gt_lats = [], []
        for arc in trip.route_arcs:
            u, v = u_idx[arc], v_idx[arc]
            gt_lons.extend([node_lons[u], node_lons[v]])
            gt_lats.extend([node_lats[u], node_lats[v]])
        ax.plot(gt_lons, gt_lats, color='#10b981', linewidth=4.0, alpha=0.85, label='Ground Truth Route')
        
    # 3. Raw GPS Trajectory
    ax.plot(
        trip.track_df['lon'].values,
        trip.track_df['lat'].values,
        color='#3b82f6',
        linestyle='--',
        linewidth=1.2,
        alpha=0.6,
        label='Raw GPS Track'
    )
    ax.scatter(
        trip.track_df['lon'].values[::5],
        trip.track_df['lat'].values[::5],
        color='#1d4ed8',
        s=12,
        alpha=0.8,
        zorder=5,
        label='GPS Observations'
    )
    
    # Formatting
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
    ax.set_title(f'PathSnap Geographical Map Matching Visualisation (Trip {trip_id})', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close()
    print(f"[Plotter] Geographical map saved to: {output_png_path}")

if __name__ == '__main__':
    plot_static_trajectory_map("00000007", output_png_path="results/plots/geographical_map_trip_00000007.png")
