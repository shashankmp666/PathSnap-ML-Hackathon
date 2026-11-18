"""
PathSnap: API Request and Response Schemas
Author: Teammate 4 (Distributed Systems Lead)
Description:
    Pydantic schemas for REST API endpoints ensuring type safety and isolation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    service: str = "PathSnap Distributed Map-Matching API"
    model_family: str
    classifier_loaded: bool
    evaluator_ready: bool

class TripItem(BaseModel):
    trip_id: str
    partition: str
    total_observations: int
    duration_seconds: float
    bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat]

class TripsListResponse(BaseModel):
    total_trips: int
    trips: List[TripItem]

class PredictionResponse(BaseModel):
    trip_id: str
    total_observations: int
    predicted_route_arcs: List[int]
    total_reconstructed_arcs: int
    inference_latency_ms: float
    map_url: str

class EvaluationResponse(BaseModel):
    trip_id: str
    total_observations: int
    observation_top1_accuracy: float = Field(..., description="Top-1 observation accuracy against ground truth")
    route_continuity: float = Field(..., description="Topological continuity of the reconstructed route")
    candidate_miss_rate: float
    inference_latency_ms: float
    total_reconstructed_arcs: int
    map_url: str
    status: str = "EVALUATION_SUCCESS"
