import os
import sys
import time
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Path resolution setup
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MINE_SUBSIDENCE_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(MINE_SUBSIDENCE_DIR)

for path in [BACKEND_DIR, MINE_SUBSIDENCE_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

DASHBOARD_DIR = os.path.join(MINE_SUBSIDENCE_DIR, "dashboard")
if not os.path.exists(DASHBOARD_DIR):
    DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")

MODEL_DIR = os.path.join(MINE_SUBSIDENCE_DIR, "models")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

DB_DIR = os.path.join(MINE_SUBSIDENCE_DIR, "database")
if not os.path.exists(DB_DIR):
    DB_DIR = os.path.join(PROJECT_ROOT, "database")
DB_PATH = os.path.join(DB_DIR, "mine_subsidence.db")

# Fallback-safe engines
try:
    from ai.spatial_risk_engine import MineSpatialRiskEngine
    risk_engine = MineSpatialRiskEngine(model_dir=os.path.join(MODEL_DIR, ""))
except Exception as e:
    print(f"[WARNING] Spatial Risk Engine load warning: {e}. Using fallback engine.")
    class FallbackRiskEngine:
        def predict_node_risk(self, payload: dict) -> dict:
            score = min(100.0, (payload.get("displacement", 0) * 10) + (payload.get("vibration", 0) * 20))
            status = "CRITICAL" if score > 70 else ("WARNING" if score > 45 else "NORMAL")
            return {"risk_score": round(score, 1), "status": status, "is_anomaly": score > 60}

        def evaluate_mesh_network(self, telemetry_frame: list) -> dict:
            predictions = {node["node_id"]: self.predict_node_risk(node) for node in telemetry_frame}
            avg_risk = sum(p["risk_score"] for p in predictions.values()) / max(len(predictions), 1)
            active_zones = []
            if avg_risk > 50:
                active_zones = [{
                    "zone_name": "ZONE_A",
                    "affected_nodes": [n["node_id"] for n in telemetry_frame[:3]],
                    "max_risk_score": round(avg_risk, 1)
                }]
            return {
                "overall_mine_risk": round(avg_risk, 1),
                "active_subsidence_zones": active_zones,
                "node_predictions": predictions
            }
    risk_engine = FallbackRiskEngine()

try:
    from simulator.sensor_simulator import MineSensorSimulator
    simulator = MineSensorSimulator(num_nodes=20)
except Exception as e:
    print(f"[WARNING] Sensor Simulator load warning: {e}. Using fallback simulator.")
    class FallbackSimulator:
        def __init__(self):
            self.active_scenario = "NORMAL"
        def set_scenario(self, scenario: str):
            self.active_scenario = scenario
        def generate_telemetry_frame(self):
            return [{
                "node_id": f"NODE_{i+1:02d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latitude": 23.6345 + (i * 0.0003),
                "longitude": 86.1523 + (i * 0.0003),
                "displacement": 1.2,
                "tilt_x": 0.05,
                "tilt_y": 0.02,
                "vibration": 0.01,
                "crack_level": 0,
                "temperature": 28.5
            } for i in range(20)]
    simulator = FallbackSimulator()

app = FastAPI(title="Mine Subsidence AI Surface Monitoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HARDWARE_TELEMETRY_CACHE: Dict[str, dict] = {}
NODE_LAST_HEARTBEAT: Dict[str, float] = {}
HARDWARE_TIMEOUT_SECONDS = 10.0

def normalize_telemetry_payload(raw_payload: dict) -> dict:
    node_id = raw_payload.get("node_id") or raw_payload.get("device_id") or raw_payload.get("devEui") or "NODE_01"
    disp = raw_payload.get("displacement", raw_payload.get("dis_mm", raw_payload.get("disp_val", 0.0)))
    tilt_x = raw_payload.get("tilt_x", raw_payload.get("angle_x", raw_payload.get("ax", 0.0)))
    tilt_y = raw_payload.get("tilt_y", raw_payload.get("angle_y", raw_payload.get("ay", 0.0)))
    vibe = raw_payload.get("vibration", raw_payload.get("vibe_g", raw_payload.get("vib", 0.01)))
    crack = raw_payload.get("crack_level", raw_payload.get("crack_state", 0))

    return {
        "node_id": str(node_id).upper(),
        "timestamp": raw_payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "latitude": float(raw_payload.get("latitude", raw_payload.get("lat", 23.6345))),
        "longitude": float(raw_payload.get("longitude", raw_payload.get("lng", 86.1523))),
        "displacement": float(disp),
        "tilt_x": float(tilt_x),
        "tilt_y": float(tilt_y),
        "vibration": float(vibe),
        "crack_level": int(crack),
        "temperature": float(raw_payload.get("temperature", raw_payload.get("temp", 28.0)))
    }

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, node_id TEXT, latitude REAL, longitude REAL,
                tilt_x REAL, tilt_y REAL, vibration REAL, displacement REAL,
                crack_level INTEGER, temperature REAL, risk_score REAL,
                status TEXT, is_anomaly INTEGER
            )
        """)
        conn.commit()

init_db()

class ScenarioRequest(BaseModel):
    scenario: str

# Direct HTML Route Endpoints
@app.get("/")
def serve_root():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

@app.get("/dashboard/index.html")
def serve_index_explicit():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

@app.get("/dashboard/dashboard.html")
@app.get("/dashboard.html")
def serve_dashboard():
    return FileResponse(os.path.join(DASHBOARD_DIR, "dashboard.html"))

@app.get("/dashboard/about.html")
@app.get("/about.html")
def serve_about():
    return FileResponse(os.path.join(DASHBOARD_DIR, "about.html"))

@app.get("/dashboard/model.html")
@app.get("/model.html")
def serve_model():
    return FileResponse(os.path.join(DASHBOARD_DIR, "model.html"))

# API Routes
@app.post("/api/v1/telemetry/ingest")
def ingest_telemetry(raw_payload: dict):
    payload = normalize_telemetry_payload(raw_payload)
    node_id = payload["node_id"]
    HARDWARE_TELEMETRY_CACHE[node_id] = payload
    NODE_LAST_HEARTBEAT[node_id] = time.time()

    prediction = risk_engine.predict_node_risk(payload)
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO sensor_telemetry 
            (timestamp, node_id, latitude, longitude, tilt_x, tilt_y, vibration, displacement, crack_level, temperature, risk_score, status, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload["timestamp"], payload["node_id"], payload["latitude"], payload["longitude"],
            payload["tilt_x"], payload["tilt_y"], payload["vibration"], payload["displacement"],
            payload["crack_level"], payload["temperature"], prediction["risk_score"],
            prediction["status"], int(prediction["is_anomaly"])
        ))
        conn.commit()

    return {"status": "success", "node_id": node_id, "risk_prediction": prediction}

@app.get("/api/v1/mesh/current-status")
def get_mesh_status():
    telemetry_frame = simulator.generate_telemetry_frame()
    current_time = time.time()

    for i, sim_node in enumerate(telemetry_frame):
        nid = sim_node["node_id"]
        if nid in HARDWARE_TELEMETRY_CACHE:
            if (current_time - NODE_LAST_HEARTBEAT.get(nid, 0)) <= HARDWARE_TIMEOUT_SECONDS:
                telemetry_frame[i] = {**sim_node, **HARDWARE_TELEMETRY_CACHE[nid]}

    mesh_analysis = risk_engine.evaluate_mesh_network(telemetry_frame)
    nodes_response = []
    for node_raw in telemetry_frame:
        nid = node_raw["node_id"]
        ai_res = mesh_analysis["node_predictions"][nid]
        nodes_response.append({**node_raw, **ai_res})

    return {
        "timestamp": telemetry_frame[0]["timestamp"] if telemetry_frame else datetime.now(timezone.utc).isoformat(),
        "overall_mine_risk": mesh_analysis["overall_mine_risk"],
        "active_scenario": getattr(simulator, 'active_scenario', 'NORMAL'),
        "active_subsidence_zones": mesh_analysis["active_subsidence_zones"],
        "nodes": nodes_response
    }

@app.post("/api/v1/simulator/scenario")
def set_simulation_scenario(request: ScenarioRequest):
    if hasattr(simulator, 'set_scenario'):
        simulator.set_scenario(request.scenario)
        return {"status": "success", "active_scenario": request.scenario}
    raise HTTPException(status_code=400, detail="Scenario control not supported.")

# Static Asset Mounts
if os.path.exists(DASHBOARD_DIR):
    assets_dir = os.path.join(DASHBOARD_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)