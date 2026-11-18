"""
PathSnap: Feature Dataset Construction Pipeline
Author: Teammate 1 (Data Engineering Lead)
Description:
    Processes raw GPS trajectories and road network graphs across partitions,
    extracts spatial-kinematic candidate features, and compiles structured
    training and evaluation datasets.
"""

import os
import sys
import glob
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data_processing.dataset import load_single_trip
from features.feature_extractor import extract_candidates_and_features

def build_datasets(
    train_dir: str = "data/train",
    test_dir: str = "data/student_test",
    output_dir: str = "artifacts",
    max_train_trips: int = 40,
    max_test_trips: int = 10,
    k_candidates: int = 10
):
    os.makedirs(output_dir, exist_ok=True)
    train_subdirs = sorted(glob.glob(os.path.join(train_dir, "*")))
    test_subdirs = sorted(glob.glob(os.path.join(test_dir, "*")))
    
    if max_train_trips > 0:
        train_subdirs = train_subdirs[:max_train_trips]
    if max_test_trips > 0:
        test_subdirs = test_subdirs[:max_test_trips]
        
    print(f"[Dataset Build] Processing {len(train_subdirs)} training trips...")
    train_frames = []
    train_stats = []
    
    t0 = time.time()
    for idx, td in enumerate(train_subdirs):
        trip_id = os.path.basename(td)
        trip_data = load_single_trip(td, partition="train")
        df, stats = extract_candidates_and_features(trip_data, k_candidates=k_candidates)
        train_frames.append(df)
        train_stats.append(stats)
        if (idx + 1) % 5 == 0 or (idx + 1) == len(train_subdirs):
            print(f"  Processed {idx + 1}/{len(train_subdirs)} train trips (current: {trip_id}, miss rate: {stats['candidate_miss_rate']*100:.1f}%)")
            
    train_df = pd.concat(train_frames, ignore_index=True)
    train_pkl_path = os.path.join(output_dir, "train_features.pkl")
    train_df.to_pickle(train_pkl_path)
    
    train_stats_df = pd.DataFrame(train_stats)
    train_stats_df.to_csv(os.path.join(output_dir, "train_candidate_stats.csv"), index=False)
    
    print(f"[Dataset Build] Train features saved: {train_df.shape} in {time.time() - t0:.1f}s")
    print(f"  Overall Candidate Miss Rate on Train: {train_stats_df['candidate_miss_count'].sum() / train_stats_df['total_observations'].sum() * 100:.2f}%")
    
    print(f"\n[Dataset Build] Processing {len(test_subdirs)} student-test trips...")
    test_frames = []
    test_stats = []
    
    t0_test = time.time()
    for idx, td in enumerate(test_subdirs):
        trip_id = os.path.basename(td)
        trip_data = load_single_trip(td, partition="student_test")
        df, stats = extract_candidates_and_features(trip_data, k_candidates=k_candidates)
        test_frames.append(df)
        test_stats.append(stats)
        print(f"  Processed test trip {trip_id} (obs: {stats['total_observations']}, miss rate: {stats['candidate_miss_rate']*100:.1f}%)")
        
    test_df = pd.concat(test_frames, ignore_index=True)
    test_pkl_path = os.path.join(output_dir, "student_test_features.pkl")
    test_df.to_pickle(test_pkl_path)
    
    test_stats_df = pd.DataFrame(test_stats)
    test_stats_df.to_csv(os.path.join(output_dir, "student_test_candidate_stats.csv"), index=False)
    
    print(f"[Dataset Build] Test features saved: {test_df.shape} in {time.time() - t0_test:.1f}s")
    print(f"  Overall Candidate Miss Rate on Student-Test: {test_stats_df['candidate_miss_count'].sum() / test_stats_df['total_observations'].sum() * 100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-trips", type=int, default=25)
    parser.add_argument("--max-test-trips", type=int, default=10)
    parser.add_argument("--k-candidates", type=int, default=10)
    args = parser.parse_args()
    
    build_datasets(
        max_train_trips=args.max_train_trips,
        max_test_trips=args.max_test_trips,
        k_candidates=args.k_candidates
    )
