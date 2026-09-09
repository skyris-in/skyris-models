"""
init_router.py

POST /init — called once when the user opens the app.

Flow:
    client sends { latitude, longitude }
        → fetch weather from OpenWeatherMap
        → find closest sensor from registry
        → return { weather, sensors (all), closest_sensor }

The WebSocket URL is also returned so the app can connect to /ws/latest-result/
which is already in main.py and broadcasts ML predictions in real-time.
"""

import os
import json
import logging
import urllib.request
import urllib.parse

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from api.sensor_registry import get_all_sensors, get_closest_sensor

logger = logging.getLogger(__name__)
router = APIRouter()

OWM_BASE = "https://api.openweathermap.org/data/2.5/weather"


class InitRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@router.post("/init")
def init_session(body: InitRequest):
    user_lat = body.latitude
    user_lon = body.longitude
    print(f"Received init request with coordinates: lat={user_lat}, lon={user_lon}")

    # ── 1. OpenWeatherMap ─────────────────────────────────────────────────────
    owm_key = os.getenv("OWM_API_KEY", "")
    weather = None

    if owm_key and user_lat is not None and user_lon is not None:
        try:
            params = urllib.parse.urlencode({
                "lat": user_lat,
                "lon": user_lon,
                "appid": owm_key,
                "units": "metric",
            })
            req = urllib.request.urlopen(f"{OWM_BASE}?{params}", timeout=5)
            raw = json.loads(req.read().decode())
            weather = _parse_owm(raw)
        except Exception as e:
            logger.error(f"OWM request failed: {e}")
    else:
        if not owm_key:
            logger.warning("OWM_API_KEY not set — skipping weather fetch")

    # ── 2. Sensor registry ────────────────────────────────────────────────────
    all_sensors = get_all_sensors()
    closest = (
        get_closest_sensor(user_lat, user_lon)
        if user_lat is not None and user_lon is not None
        else (all_sensors[0] if all_sensors else None)
    )

    # Skip OWM if no coordinates provided
    if weather and (user_lat is None or user_lon is None):
        weather = None

    return JSONResponse({
        "weather": weather,
        "sensors": all_sensors,
        "closest_sensor": closest,
        # Prefer the sensor-specific stream so each user receives the nearest location's data.
        "socket_path": (
            f"/ws/sensor/{closest.get('id') or closest.get('device_id')}/"
            if closest and (closest.get("id") or closest.get("device_id"))
            else "/ws/latest-result/"
        ),
    })


def _parse_owm(raw: dict) -> dict:
    """Flatten the OWM response to match the frontend WeatherData type."""
    main = raw.get("main", {})
    wind = raw.get("wind", {})
    clouds = raw.get("clouds", {})
    weather_arr = raw.get("weather", [{}])
    w = weather_arr[0] if weather_arr else {}
    sys = raw.get("sys", {})

    return {
        "temp": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "temp_min": main.get("temp_min"),
        "temp_max": main.get("temp_max"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "visibility": raw.get("visibility"),
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "wind_gust": wind.get("gust"),
        "clouds": clouds.get("all"),
        "description": w.get("description", ""),
        "icon": w.get("icon", ""),
        "sunrise": sys.get("sunrise"),
        "sunset": sys.get("sunset"),
        "city": raw.get("name", ""),
    }
