import os
import numpy as np
import pandas as pd

def test_route_projection():
    td = "data/train/00000001"
    track = pd.read_csv(f"{td}/00000001.track", sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(f"{td}/00000001.nodes", sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(f"{td}/00000001.arcs", sep=r'\s+', header=None, names=['u', 'v'])
    route = pd.read_csv(f"{td}/00000001.route", sep=r'\s+', header=None, names=['arc_id'])['arc_id'].values
    
    # Let's project in metric space
    ref_lat = track['lat'].mean()
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(ref_lat))
    
    # Convert nodes to meters
    node_x = nodes['lon'].values * m_per_deg_lon
    node_y = nodes['lat'].values * m_per_deg_lat
    
    # Convert track to meters
    track_x = track['lon'].values * m_per_deg_lon
    track_y = track['lat'].values * m_per_deg_lat
    
    # Route segments
    route_u = arcs.iloc[route]['u'].values
    route_v = arcs.iloc[route]['v'].values
    
    ax = node_x[route_u]
    ay = node_y[route_u]
    bx = node_x[route_v]
    by = node_y[route_v]
    
    # For each GPS point, find the closest arc among the route arcs!
    # Because ground truth is the route, every GPS point comes from one of the route arcs!
    # Let's verify how close GPS points are to the route arcs.
    min_dists = []
    matched_route_indices = []
    
    for i in range(min(50, len(track_x))):
        px, py = track_x[i], track_y[i]
        # Distance to all route segments
        vx = bx - ax
        vy = by - ay
        seg_len_sq = vx**2 + vy**2
        seg_len_sq[seg_len_sq == 0] = 1e-9
        
        t = ((px - ax)*vx + (py - ay)*vy) / seg_len_sq
        t_clamped = np.clip(t, 0.0, 1.0)
        
        proj_x = ax + t_clamped * vx
        proj_y = ay + t_clamped * vy
        
        dists = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        best_idx = np.argmin(dists)
        min_dists.append(dists[best_idx])
        matched_route_indices.append(best_idx)
        
    print(f"Sample 50 GPS points distance to route:")
    print(f"  Mean dist: {np.mean(min_dists):.2f}m, Max dist: {np.max(min_dists):.2f}m, Min dist: {np.min(min_dists):.2f}m")
    print(f"  Matched route indices progress: {matched_route_indices[:20]}")

if __name__ == '__main__':
    test_route_projection()
