"""
PathSnap: Exploratory Data Analysis (EDA) Plotter
Author: Teammate 1 (Data Engineering Lead)
Description:
    Generates exploratory distributions of sampling intervals, vehicle speeds,
    road segment lengths, and perpendicular distance distributions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.dataset import load_single_trip

def generate_eda_figures(output_dir: str = "results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load representative trip
    trip = load_single_trip("data/train/00000001")
    train_features = pd.read_pickle("artifacts/train_features.pkl")
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)
    
    # 1. Sampling Interval Delta t
    dt = trip.track_time_deltas
    axes[0, 0].hist(dt, bins=np.linspace(0.5, 3.5, 31), color='#3b82f6', edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('GPS Sampling Interval Distribution (Nominal 1 Hz)', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Time Delta Δt (seconds)', fontsize=10)
    axes[0, 0].set_ylabel('Observation Frequency', fontsize=10)
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    axes[0, 0].axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Nominal 1.0s')
    axes[0, 0].legend()
    
    # 2. Vehicle Speeds
    speeds_kmh = trip.track_speeds * 3.6
    axes[0, 1].hist(speeds_kmh, bins=30, color='#10b981', edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Instantaneous Vehicle Speed Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Speed (km/h)', fontsize=10)
    axes[0, 1].set_ylabel('Observation Frequency', fontsize=10)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    # 3. Road Segment Lengths
    arc_lens = trip.arc_lengths
    axes[1, 0].hist(arc_lens[arc_lens < 300], bins=35, color='#8b5cf6', edgecolor='black', alpha=0.8)
    axes[1, 0].set_title('Road Segment Length Distribution (Sub-300m)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Arc Length (meters)', fontsize=10)
    axes[1, 0].set_ylabel('Road Segment Count', fontsize=10)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # 4. Candidate Perpendicular Distance
    perp_dists = train_features['perp_dist'].values
    axes[1, 1].hist(perp_dists[perp_dists < 60], bins=35, color='#f59e0b', edgecolor='black', alpha=0.8)
    axes[1, 1].set_title('GPS-to-Segment Perpendicular Distance (k=10)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Orthogonal Distance (meters)', fontsize=10)
    axes[1, 1].set_ylabel('Candidate Pairs', fontsize=10)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "eda_gps_distributions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[EDA] Figures saved to: {plot_path}")

if __name__ == '__main__':
    generate_eda_figures()
