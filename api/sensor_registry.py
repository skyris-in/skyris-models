"""
sensor_registry.py

In-memory store for discovered sensor nodes.
Populated by the MQTT listener when it receives retained metadata.
Read by the init endpoint to find sensors near a user.

Do NOT import anything from main.py here — this module is imported by both
mqtt_listener.py and the route handlers, so it must stay dependency-free.
"""

import math
from typing import Optional

# sensor_id -> { id, latitude, longitude, label, source, connected_at }
_sensors: dict[str, dict] = {}


def upsert_sensor(sensor_id: str, metadata: dict) -> None:
    """Insert or update a sensor's metadata with normalized id/device_id keys."""
    normalized_id = (
        metadata.get("id")
        or metadata.get("device_id")
        or sensor_id
    )

    record = {
        **metadata,
        "id": normalized_id,
        "device_id": metadata.get("device_id") or normalized_id,
        "latitude": metadata.get("latitude"),
        "longitude": metadata.get("longitude"),
    }

    _sensors[normalized_id] = record


def remove_sensor(sensor_id: str) -> None:
    """Remove a sensor from the registry if it no longer exists."""
    _sensors.pop(sensor_id, None)

    # Also remove entries where device_id/id differs from dict key.
    stale_keys = [
        key
        for key, value in _sensors.items()
        if value.get("device_id") == sensor_id or value.get("id") == sensor_id
    ]
    for key in stale_keys:
        _sensors.pop(key, None)


def clear_sensors() -> None:
    """Remove every sensor from the registry."""
    _sensors.clear()


def get_all_sensors() -> list[dict]:
    """Return all known sensors."""
    return list(_sensors.values())


def get_closest_sensor(user_lat: float, user_lon: float) -> Optional[dict]:
    """Return the sensor with the smallest haversine distance to the user."""
    if not _sensors:
        return None

    best = None
    best_dist = float("inf")

    for sensor in _sensors.values():
        d = haversine_km(user_lat, user_lon, sensor["latitude"], sensor["longitude"])
        if d < best_dist:
            best_dist = d
            best = sensor

    return best


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
