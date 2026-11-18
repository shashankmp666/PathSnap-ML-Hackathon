"""
PathSnap: Spatial & Geometric Utilities
Author: Teammate 1 (Data Engineering Lead)
Description:
    Vectorized coordinate conversions, orthogonal projection onto road segments,
    bearing calculations, and geodesic distance formulations in metric space.
"""

import numpy as np

# Equirectangular approximation constants for the Darmstadt/Frankfurt region (~49.85 deg N)
METERS_PER_DEGREE_LAT = 111320.0

def to_metric_coordinates(lon: np.ndarray, lat: np.ndarray, ref_lat: float = 49.85) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts angular WGS84 degrees into Cartesian metric coordinates (meters)
    relative to a local reference latitude.
    """
    scale_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(ref_lat))
    x = lon * scale_lon
    y = lat * METERS_PER_DEGREE_LAT
    return x, y

def to_wgs84_coordinates(x: np.ndarray, y: np.ndarray, ref_lat: float = 49.85) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts Cartesian metric coordinates (meters) back to angular WGS84 degrees.
    """
    scale_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(ref_lat))
    lon = x / scale_lon
    lat = y / METERS_PER_DEGREE_LAT
    return lon, lat

def compute_azimuth_heading(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Computes compass bearing / heading in degrees [0, 360) clockwise from North.
    dx = easting difference, dy = northing difference.
    """
    angles = np.degrees(np.arctan2(dx, dy))
    return (angles + 360.0) % 360.0

def angular_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Calculates the absolute minimum angular difference in [0, 180] degrees between two headings.
    """
    diff = np.abs(angle1 - angle2) % 360.0
    return np.minimum(diff, 360.0 - diff)

def project_points_to_segments(
    px: np.ndarray, py: np.ndarray,
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthogonally projects points (px, py) onto directed segments A(ax, ay) -> B(bx, by).
    Returns:
        perp_dist: Euclidean perpendicular distance (meters)
        proj_ratio: Clamped projection fraction along segment [0.0, 1.0]
        proj_x: Projected point x (meters)
        proj_y: Projected point y (meters)
    """
    vx = bx - ax
    vy = by - ay
    seg_len_sq = vx**2 + vy**2
    # Prevent division by zero for degenerate segments
    seg_len_sq = np.maximum(seg_len_sq, 1e-9)
    
    t = ((px - ax) * vx + (py - ay) * vy) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    
    proj_x = ax + t_clamped * vx
    proj_y = ay + t_clamped * vy
    
    perp_dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    return perp_dist, t_clamped, proj_x, proj_y
