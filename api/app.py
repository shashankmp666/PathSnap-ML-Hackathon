"""
PathSnap: Distributed FastAPI Backend (Laptop 1)
Author: Teammate 4 (Distributed Systems Lead)
Description:
    Production FastAPI backend server hosting the machine learning inference pipeline,
    HMM Viterbi route decoder, Folium map generator, and server-side evaluation.
"""

import os
import sys
import time
import glob
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.dataset import load_single_trip
from features.feature_extractor import extract_candidates_and_features, FEATURE_COLUMNS
from hmm_viterbi.hmm_decoder import HMMViterbiDecoder, RoadNetworkGraph
from visualization.map_visualizer import create_folium_trajectory_map
from api.schemas import HealthResponse, TripsListResponse, TripItem, PredictionResponse, EvaluationResponse

# Configure evaluator-visible CMD logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap-API] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pathsnap")

app = FastAPI(
    title="PathSnap: Distributed Map-Matching & Route Reconstruction API",
    description="Backend API for Laptop 1 in Two-Laptop Evaluation Architecture",
    version="1.0.0"
)

# Enable CORS for Laptop 2 evaluator interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        response = Response(status_code=204)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# Global model state
MODEL = None
DECODER = None
DATA_DIR = "data"
MAPS_DIR = "results/maps"
os.makedirs(MAPS_DIR, exist_ok=True)

@app.on_event("startup")
def load_artifacts():
    global MODEL, DECODER
    logger.info("Initializing PathSnap backend services...")
    
    # Priority: tuned model -> champion model
    model_paths = [
        "artifacts/tuned_champion_model.joblib",
        "artifacts/best_classifier.joblib"
    ]
    for p in model_paths:
        if os.path.exists(p):
            MODEL = joblib.load(p)
            logger.info(f"Loaded classifier artifact from: {p}")
            break
            
    if MODEL is None:
        logger.warning("No saved classifier artifact found in artifacts/. Running in fallback heuristic mode.")
        
    DECODER = HMMViterbiDecoder(
        sigma_z=10.0,
        beta=15.0,
        classifier_weight=0.85,
        stay_log_bonus=3.0,
        connected_log_bonus=3.5,
        unconnected_penalty=-15.0
    )
    logger.info("HMM Viterbi Decoder initialized and calibrated.")
    logger.info("Laptop 1 API Backend is LIVE and ready for incoming evaluator requests.")

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_family="Gradient Boosted Decision Ensembles (Optuna Tuned)",
        classifier_loaded=(MODEL is not None),
        evaluator_ready=True
    )

@app.get("/trips", response_model=TripsListResponse)
def list_evaluation_trips():
    """
    Returns metadata for evaluation trips found in evaluator_test, student_test, or train partitions.
    Prioritizes evaluator_test if present.
    """
    items = []
    
    # Priority 1: evaluator_test
    eval_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "evaluator_test", "*")))
    for td in eval_dirs:
        tid = os.path.basename(td)
        tr_file = os.path.join(td, f"{tid}.track")
        if os.path.exists(tr_file):
            tr_df = pd.read_csv(tr_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
            duration = float(tr_df['time'].iloc[-1] - tr_df['time'].iloc[0])
            bbox = [
                float(tr_df['lon'].min()), float(tr_df['lat'].min()),
                float(tr_df['lon'].max()), float(tr_df['lat'].max())
            ]
            items.append(TripItem(
                trip_id=tid,
                partition="evaluator_test",
                total_observations=len(tr_df),
                duration_seconds=duration,
                bbox=bbox
            ))
            
    # Priority 2: student_test (if evaluator_test is empty or to complement)
    if not items:
        test_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "student_test", "*")))
        for td in test_dirs:
            tid = os.path.basename(td)
            tr_file = os.path.join(td, f"{tid}.track")
            if os.path.exists(tr_file):
                tr_df = pd.read_csv(tr_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
                duration = float(tr_df['time'].iloc[-1] - tr_df['time'].iloc[0])
                bbox = [
                    float(tr_df['lon'].min()), float(tr_df['lat'].min()),
                    float(tr_df['lon'].max()), float(tr_df['lat'].max())
                ]
                items.append(TripItem(
                    trip_id=tid,
                    partition="student_test",
                    total_observations=len(tr_df),
                    duration_seconds=duration,
                    bbox=bbox
                ))
            
    return TripsListResponse(total_trips=len(items), trips=items)

@app.post("/predict/{trip_id}", response_model=PredictionResponse)
def predict_trip(trip_id: str, request: Request):
    """
    Executes full map-matching prediction pipeline:
    1. Loads GPS track and localized road network.
    2. Generates candidate segments and computes spatial-kinematic features.
    3. Scores candidates using trained classifier.
    4. Reconstructs sequence via Viterbi DP.
    5. Exports interactive Folium map.
    """
    t0 = time.time()
    trip_dir = os.path.join(DATA_DIR, "evaluator_test", trip_id)
    if not os.path.exists(trip_dir):
        trip_dir = os.path.join(DATA_DIR, "student_test", trip_id)
        if not os.path.exists(trip_dir):
            trip_dir = os.path.join(DATA_DIR, "train", trip_id)
            if not os.path.exists(trip_dir):
                raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found.")
            
    client_ip = (request.client.host if (request and request.client) else "127.0.0.1")
    logger.info(f"Incoming PREDICT request from {client_ip} for Trip ID: {trip_id}")
    
    trip_data = load_single_trip(trip_dir, partition="inference")
    cand_df, stats = extract_candidates_and_features(trip_data, k_candidates=10)
    
    # Model inference
    if MODEL is not None:
        X = cand_df[FEATURE_COLUMNS].values
        probs = MODEL.predict_proba(X)[:, 1]
    else:
        probs = cand_df['gaussian_prior'].values
        
    graph = RoadNetworkGraph(trip_data.arcs_df)
    viterbi_arcs, _, _ = DECODER.decode_trip(
        trip=trip_data,
        candidates_df=cand_df,
        classifier_probs=probs,
        graph=graph
    )
    
    # Save Folium Map
    map_filename = f"trip_{trip_id}_map.html"
    map_path = os.path.join(MAPS_DIR, map_filename)
    create_folium_trajectory_map(trip_data, predicted_arcs=viterbi_arcs, output_html_path=map_path)
    
    total_latency_ms = (time.time() - t0) * 1000.0
    logger.info(f"PREDICT Complete: {len(viterbi_arcs)} observations matched in {total_latency_ms:.1f}ms")
    
    # Unique sequence of traversed arcs
    unique_arcs = []
    for arc in viterbi_arcs:
        if not unique_arcs or unique_arcs[-1] != int(arc):
            unique_arcs.append(int(arc))
            
    return PredictionResponse(
        trip_id=trip_id,
        total_observations=len(trip_data.track_x),
        predicted_route_arcs=unique_arcs,
        total_reconstructed_arcs=len(unique_arcs),
        inference_latency_ms=round(total_latency_ms, 2),
        map_url=f"/maps/{trip_id}"
    )

@app.post("/evaluate/{trip_id}", response_model=EvaluationResponse)
def evaluate_trip(trip_id: str, request: Request):
    """
    Server-side Evaluation Endpoint for evaluators:
    Executes prediction and computes ground truth metrics (Top-1 Accuracy, Route Continuity)
    strictly on the server side without leaking reference routes to the client.
    """
    t0 = time.time()
    trip_dir = os.path.join(DATA_DIR, "evaluator_test", trip_id)
    if not os.path.exists(trip_dir):
        trip_dir = os.path.join(DATA_DIR, "student_test", trip_id)
        if not os.path.exists(trip_dir):
            trip_dir = os.path.join(DATA_DIR, "train", trip_id)
            if not os.path.exists(trip_dir):
                raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found.")
            
    client_ip = (request.client.host if (request and request.client) else "127.0.0.1")
    logger.info(f"Incoming EVALUATE request from {client_ip} for Trip ID: {trip_id}")
    
    trip_data = load_single_trip(trip_dir, partition="eval")
    cand_df, stats = extract_candidates_and_features(trip_data, k_candidates=10)
    
    if MODEL is not None:
        X = cand_df[FEATURE_COLUMNS].values
        probs = MODEL.predict_proba(X)[:, 1]
    else:
        probs = cand_df['gaussian_prior'].values
        
    graph = RoadNetworkGraph(trip_data.arcs_df)
    viterbi_arcs, _, diag = DECODER.decode_trip(
        trip=trip_data,
        candidates_df=cand_df,
        classifier_probs=probs,
        graph=graph
    )
    
    # Generate Map with both ground truth and predicted route
    map_filename = f"trip_{trip_id}_map.html"
    map_path = os.path.join(MAPS_DIR, map_filename)
    create_folium_trajectory_map(trip_data, predicted_arcs=viterbi_arcs, output_html_path=map_path)
    
    total_latency_ms = (time.time() - t0) * 1000.0
    
    unique_arcs = []
    for arc in viterbi_arcs:
        if not unique_arcs or unique_arcs[-1] != int(arc):
            unique_arcs.append(int(arc))
            
    top1_acc = diag['viterbi_accuracy'] * 100.0
    continuity = diag['viterbi_route_continuity'] * 100.0
    miss_rate = stats['candidate_miss_rate'] * 100.0
    
    # Print formatted evaluator logs on Laptop 1 terminal
    print("\n" + "="*65)
    print(f" [EVALUATION AUDIT LOG] Trip ID: {trip_id}")
    print(f" Client IP:                {client_ip}")
    print(f" Observations Matched:     {len(trip_data.track_x)}")
    print(f" Traversed Arcs Decoded:   {len(unique_arcs)}")
    print(f" Observation Top-1 Acc:    {top1_acc:.2f}%")
    print(f" Route Continuity:         {continuity:.2f}%")
    print(f" Candidate Miss Rate:      {miss_rate:.2f}%")
    print(f" Total Inference Latency:  {total_latency_ms:.1f}ms")
    print("="*65 + "\n")
    
    return EvaluationResponse(
        trip_id=trip_id,
        total_observations=len(trip_data.track_x),
        observation_top1_accuracy=round(top1_acc, 2),
        route_continuity=round(continuity, 2),
        candidate_miss_rate=round(miss_rate, 2),
        inference_latency_ms=round(total_latency_ms, 2),
        total_reconstructed_arcs=len(unique_arcs),
        map_url=f"/maps/{trip_id}",
        status="EVALUATION_SUCCESS"
    )

@app.get("/maps/{trip_id}", response_class=HTMLResponse)
def get_trip_map(trip_id: str):
    map_path = os.path.join(MAPS_DIR, f"trip_{trip_id}_map.html")
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="Map not yet generated. Run /predict or /evaluate first.")
    with open(map_path, "r", encoding="utf-8") as f:
        return f.read()

# Serve UI static files
ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

@app.get("/")
def root_redirect():
    return FileResponse(os.path.join(ui_dir, "index.html"))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
