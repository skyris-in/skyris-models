from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from api.inference import (
    predict_image_logic,
    predict_tabular_logic,
    get_latest
)

from api.mqtt_listener import start_mqtt_client, stop_mqtt_client
from api.init_router import router as init_router
from api.sensor_registry import clear_sensors, get_all_sensors, remove_sensor
from api.mqtt_listener import clear_retained_metadata
from api.xai import check_and_update_explanation, get_cached_explanation


logger = logging.getLogger("skyris.config")


def _load_environment() -> None:
    # Resolve .env robustly for common launch locations: backend/, backend/api/, repo root.
    api_dir = Path(__file__).resolve().parent
    backend_dir = api_dir.parent
    repo_dir = backend_dir.parent

    candidate_env_files = (
        backend_dir / ".env",
        api_dir / ".env",
        repo_dir / ".env",
    )

    for env_path in candidate_env_files:
        if env_path.exists():
            loaded = load_dotenv(dotenv_path=env_path, override=False)
            if loaded:
                logger.info("Loaded environment variables from %s", env_path)
            break
    else:
        logger.warning("No .env file found in expected locations; using process environment only")


_load_environment()

# -----------------------------
# WebSocket Manager
# -----------------------------
class ConnectionManager:
    def __init__(self):
        # device_id -> set of WebSocket connections
        self._connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, device_id: str):
        await websocket.accept()
        self._connections.setdefault(device_id, set()).add(websocket)

    def disconnect(self, websocket: WebSocket, device_id: str):
        bucket = self._connections.get(device_id, set())
        bucket.discard(websocket)

    async def broadcast_to_sensor(self, device_id: str, message: dict):
        """Send message to clients subscribed to this sensor AND catch-all listeners."""
        targets = (
            list(self._connections.get(device_id, set())) +
            list(self._connections.get("__all__", set()))
        )
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws, device_id)
                
    async def broadcast_explanation(self, explanation: dict):
        """Send explanation message to all explanation listeners."""
        targets = list(self._connections.get("__explanation__", set()))
        for ws in targets:
            try:
                await ws.send_json(explanation)
            except Exception:
                self.disconnect(ws, "__explanation__")

    async def broadcast_all(self, message: dict):
        """Fallback: send to every connected client."""
        for bucket in list(self._connections.values()):
            for ws in list(bucket):
                try:
                    await ws.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()
loop = None

def dispatch_prediction(prediction_data: dict):
    device_id = prediction_data.get("device_id", "__all__")
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_to_sensor(device_id, prediction_data), loop
        )
        # Spawn the async XAI task (SHAP features already in prediction_data -> LLM -> explanation socket)
        asyncio.run_coroutine_threadsafe(
            check_and_update_explanation(prediction_data, manager.broadcast_explanation), loop
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global loop
    loop = asyncio.get_running_loop()

    # Start MQTT and pass it the callback
    client = start_mqtt_client(dispatch_prediction)
    yield
    stop_mqtt_client(client)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init route (POST /init) — called once when user opens the app
app.include_router(init_router)

# -----------------------------
# Schema
# -----------------------------


class CloudArray(BaseModel):
    data: List[float]


class SensorResetRequest(BaseModel):
    sensor_ids: Optional[List[str]] = None
    clear_retained: bool = True
    clear_registry: bool = True

# -----------------------------
# Routes
# -----------------------------


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>Skyris Models API is running</h1><p><a href='/docs'>Swagger Docs</a> | <a href='/health'>Health</a></p>")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/image/")
async def predict_image(file: UploadFile = File(...)):
    import base64
    from datetime import datetime
    
    contents = await file.read()
    device_id = "vision-node-01"
    vision_preds = predict_image_logic(contents, device_id=device_id)
    
    result = {
        "vision_predictions": vision_preds,
        "device_id": device_id,
        "image": base64.b64encode(contents).decode('utf-8'),
        "ts": int(datetime.utcnow().timestamp() * 1000)
    }
    dispatch_prediction(result)
    return result


@app.post("/data/")
async def predict_data(cloud: Optional[CloudArray] = None):
    if cloud is None or not cloud.data:
        return {"error": "No input"}

    result = predict_tabular_logic(cloud.data)
    dispatch_prediction(result)
    return result


@app.get("/latest-result/")
def latest():
    return get_latest()


@app.get("/debug/sensors")
def debug_sensors():
    sensors = get_all_sensors()
    return {
        "count": len(sensors),
        "sensors": sensors,
    }


@app.post("/debug/reset-sensors")
def debug_reset_sensors(body: SensorResetRequest):
    sensors_before = get_all_sensors()
    sensor_ids = body.sensor_ids or [
        (s.get("id") or s.get("device_id"))
        for s in sensors_before
        if (s.get("id") or s.get("device_id"))
    ]

    sensor_ids = [str(sid) for sid in sensor_ids]

    retained_result = {"cleared": [], "failed": []}
    if body.clear_retained and sensor_ids:
        retained_result = clear_retained_metadata(sensor_ids)

    if body.clear_registry:
        if body.sensor_ids:
            for sensor_id in sensor_ids:
                remove_sensor(sensor_id)
        else:
            clear_sensors()

    sensors_after = get_all_sensors()

    return {
        "before_count": len(sensors_before),
        "after_count": len(sensors_after),
        "targets": sensor_ids,
        "retained": retained_result,
    }

@app.websocket("/ws/sensor/{device_id}/")
async def websocket_sensor_endpoint(websocket: WebSocket, device_id: str):
    """Per-sensor WebSocket: client only receives predictions from device_id."""
    await manager.connect(websocket, device_id)
    try:
        current_data = get_latest(device_id)
        if current_data:
            await websocket.send_json(current_data)
        while True:
            await websocket.receive_text()  # keepalive
    except WebSocketDisconnect:
        manager.disconnect(websocket, device_id)


@app.websocket("/ws/latest-result/")
async def websocket_latest_endpoint(websocket: WebSocket):
    """Legacy catch-all: receives predictions from every sensor."""
    await manager.connect(websocket, "__all__")
    try:
        current_data = get_latest()
        if current_data:
            await websocket.send_json(current_data)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, "__all__")

@app.websocket("/ws/explanation/")
async def websocket_explanation_endpoint(websocket: WebSocket):
    """Receives natural language explanations driven by SHAP and LLMs."""
    await manager.connect(websocket, "__explanation__")
    try:
        current_exp = get_cached_explanation()
        if current_exp:
            await websocket.send_json(current_exp)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, "__explanation__")
