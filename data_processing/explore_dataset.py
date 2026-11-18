import os
import glob
import numpy as np
import pandas as pd

def explore():
    train_dirs = sorted(glob.glob("data/train/*"))
    test_dirs = sorted(glob.glob("data/student_test/*"))
    
    print(f"Total train trips: {len(train_dirs)}")
    print(f"Total student_test trips: {len(test_dirs)}")
    
    # Inspect first train trip
    t1 = train_dirs[0]
    trip_id = os.path.basename(t1)
    
    track_file = os.path.join(t1, f"{trip_id}.track")
    nodes_file = os.path.join(t1, f"{trip_id}.nodes")
    arcs_file = os.path.join(t1, f"{trip_id}.arcs")
    route_file = os.path.join(t1, f"{trip_id}.route")
    
    # Track: lon, lat, timestamp
    track_df = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes_df = pd.read_csv(nodes_file, sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs_df = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['from_node', 'to_node'])
    route_df = pd.read_csv(route_file, sep=r'\s+', header=None, names=['arc_id'])
    
    print(f"\nTrip {trip_id}:")
    print(f"  Track shape: {track_df.shape}, Time duration: {track_df['time'].iloc[-1] - track_df['time'].iloc[0]:.1f}s")
    print(f"  Nodes count: {len(nodes_df)}, Arcs count: {len(arcs_df)}, Route steps: {len(route_df)}")
    
    # Time diffs
    time_diffs = np.diff(track_df['time'])
    exact_1s = np.mean(np.isclose(time_diffs, 1.0)) * 100
    print(f"  Sampling rate ~1Hz percentage: {exact_1s:.2f}% (median={np.median(time_diffs):.2f}s, max={np.max(time_diffs):.2f}s)")
    
    # Check route continuity in graph
    route_arcs = route_df['arc_id'].values
    route_edges = arcs_df.iloc[route_arcs]
    consecutive_connected = 0
    for i in range(len(route_arcs) - 1):
        curr_to = route_edges.iloc[i]['to_node']
        next_from = route_edges.iloc[i+1]['from_node']
        if curr_to == next_from:
            consecutive_connected += 1
            
    print(f"  Ground truth route consecutive connectivity: {consecutive_connected}/{len(route_arcs)-1} ({consecutive_connected/(len(route_arcs)-1)*100:.1f}%)")
    
    # Check overall dataset statistics across all train trips
    track_lens = []
    route_lens = []
    missing_vals = 0
    
    for td in train_dirs:
        tid = os.path.basename(td)
        tr = pd.read_csv(os.path.join(td, f"{tid}.track"), sep=r'\s+', header=None)
        ro = pd.read_csv(os.path.join(td, f"{tid}.route"), sep=r'\s+', header=None)
        track_lens.append(len(tr))
        route_lens.append(len(ro))
        missing_vals += tr.isna().sum().sum()
        
    print(f"\nAcross all 80 train trips:")
    print(f"  Total GPS points: {sum(track_lens)}")
    print(f"  Track lengths: min={min(track_lens)}, max={max(track_lens)}, mean={np.mean(track_lens):.1f}")
    print(f"  Route lengths: min={min(route_lens)}, max={max(route_lens)}, mean={np.mean(route_lens):.1f}")
    print(f"  Missing values in GPS tracks: {missing_vals}")

if __name__ == '__main__':
    explore()
