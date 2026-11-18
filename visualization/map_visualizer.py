"""
PathSnap: Interactive Geographical Map Visualizer (Folium)
Author: Teammate 4 (Distributed Systems & Visualization Lead)
Description:
    Generates multi-layer interactive Folium maps comparing original GPS
    observations, road-network geometry, ground-truth reference routes,
    and predicted route reconstructions with full LayerControl and legends.
"""

import os
import sys
import folium
from folium import plugins
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.dataset import TripData

def create_folium_trajectory_map(
    trip: TripData,
    predicted_arcs: np.ndarray | None = None,
    output_html_path: str | None = None
) -> folium.Map:
    """
    Renders an interactive multi-layer Folium map for the specified trip.
    Layers:
      1. Road Network Geometry (Subtle gray lines)
      2. Ground Truth Route (Emerald Green polyline)
      3. Raw GPS Trajectory (Blue circle markers + dashed track)
      4. Predicted Reconstructed Route (Crimson polyline)
    """
    center_lat = float(trip.track_df['lat'].mean())
    center_lon = float(trip.track_df['lon'].mean())
    
    # Initialize Map with standard OpenStreetMap tiles (no API key required)
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Layer 1: Road Network Geometry (Candidate / Localized arcs)
    # To keep the interactive HTML lightweight and responsive, render arcs within trajectory bbox
    min_lat, max_lat = trip.track_df['lat'].min() - 0.005, trip.track_df['lat'].max() + 0.005
    min_lon, max_lon = trip.track_df['lon'].min() - 0.005, trip.track_df['lon'].max() + 0.005
    
    node_lats = trip.nodes_df['lat'].values
    node_lons = trip.nodes_df['lon'].values
    u_idx = trip.arcs_df['u'].values
    v_idx = trip.arcs_df['v'].values
    
    # Filter arcs within bounding box
    mask = (
        (node_lats[u_idx] >= min_lat) & (node_lats[u_idx] <= max_lat) &
        (node_lons[u_idx] >= min_lon) & (node_lons[u_idx] <= max_lon)
    )
    relevant_arcs = np.where(mask)[0]
    if len(relevant_arcs) > 2000:
        relevant_arcs = relevant_arcs[:2000]  # Cap to prevent browser DOM lag
        
    road_layer = folium.FeatureGroup(name="1. Road Network Geometry", show=True)
    for arc_id in relevant_arcs:
        u = u_idx[arc_id]
        v = v_idx[arc_id]
        p1 = [float(node_lats[u]), float(node_lons[u])]
        p2 = [float(node_lats[v]), float(node_lons[v])]
        folium.PolyLine(
            locations=[p1, p2],
            color="#94a3b8",
            weight=1.5,
            opacity=0.6,
            tooltip=f"Arc ID: {arc_id}"
        ).add_to(road_layer)
    road_layer.add_to(fmap)
    
    # Layer 2: Ground Truth Route (if available)
    if trip.route_arcs is not None:
        gt_layer = folium.FeatureGroup(name="2. Ground Truth Reference Route", show=True)
        gt_coords = []
        for arc_id in trip.route_arcs:
            u = u_idx[arc_id]
            v = v_idx[arc_id]
            gt_coords.append([float(node_lats[u]), float(node_lons[u])])
            gt_coords.append([float(node_lats[v]), float(node_lons[v])])
            
        folium.PolyLine(
            locations=gt_coords,
            color="#10b981",
            weight=5.0,
            opacity=0.85,
            tooltip=f"Ground Truth Route ({len(trip.route_arcs)} segments)"
        ).add_to(gt_layer)
        gt_layer.add_to(fmap)
        
    # Layer 3: Raw GPS Observations
    gps_layer = folium.FeatureGroup(name="3. Raw GPS Trajectory (Noisy)", show=True)
    track_coords = list(zip(trip.track_df['lat'].values, trip.track_df['lon'].values))
    
    # Path line
    folium.PolyLine(
        locations=track_coords,
        color="#3b82f6",
        weight=2.0,
        dash_array="4, 4",
        opacity=0.7
    ).add_to(gps_layer)
    
    # Point markers (subsample for visual clarity)
    step = max(1, len(track_coords) // 200)
    for i in range(0, len(track_coords), step):
        lat, lon = track_coords[i]
        t = trip.track_df['time'].iloc[i]
        speed = trip.track_speeds[i] * 3.6  # km/h
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#1d4ed8",
            fill=True,
            fill_color="#60a5fa",
            fill_opacity=0.9,
            popup=f"Obs #{i}<br>Time: {t:.1f}s<br>Speed: {speed:.1f} km/h"
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)
    
    # Layer 4: Predicted / Reconstructed Route
    if predicted_arcs is not None:
        pred_layer = folium.FeatureGroup(name="4. PathSnap Predicted Route", show=True)
        pred_coords = []
        for arc_id in predicted_arcs:
            u = u_idx[arc_id]
            v = v_idx[arc_id]
            pred_coords.append([float(node_lats[u]), float(node_lons[u])])
            pred_coords.append([float(node_lats[v]), float(node_lons[v])])
            
        folium.PolyLine(
            locations=pred_coords,
            color="#ef4444",
            weight=4.0,
            opacity=0.9,
            dash_array="2, 6" if trip.route_arcs is not None else None,
            tooltip=f"Predicted Reconstructed Route"
        ).add_to(pred_layer)
        pred_layer.add_to(fmap)
        
    # Add Layer Control to easily toggle views
    folium.LayerControl(position='topright', collapsed=False).add_to(fmap)
    
    # Floating Map Legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 25px; left: 25px; width: 230px; 
                background-color: white; z-index:9999; font-size:12px;
                border:2px solid #cbd5e1; border-radius: 8px; padding: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        <b style="font-size: 13px; color: #1e293b;">PathSnap Map Legend</b><br>
        <span style="font-size: 11px; color: #64748b;">Trip ID: {trip.trip_id}</span><hr style="margin: 6px 0;">
        <i style="background: #3b82f6; width: 12px; height: 12px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> Raw GPS Observations<br>
        <i style="background: #94a3b8; width: 16px; height: 3px; display: inline-block; margin-right: 6px;"></i> Road Network Segments<br>
        <i style="background: #10b981; width: 16px; height: 4px; display: inline-block; margin-right: 6px;"></i> Ground Truth Route<br>
        <i style="background: #ef4444; width: 16px; height: 4px; display: inline-block; margin-right: 6px;"></i> Predicted Reconstructed<br>
    </div>
    '''
    fmap.get_root().html.add_child(folium.Element(legend_html))
    
    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        fmap.save(output_html_path)
        print(f"[Visualizer] Folium map written to: {output_html_path}")
        
    return fmap

if __name__ == '__main__':
    # Test generation for sample trip
    sample_trip = load_single_trip("data/train/00000001")
    create_folium_trajectory_map(
        sample_trip,
        predicted_arcs=sample_trip.route_arcs,
        output_html_path="results/maps/trip_00000001_map.html"
    )
