#!/usr/bin/env python3
"""Frontend-style smoke test for the live dashboard flow.

It sends the same initial location request the app uses, prints the init
response, then connects to the websocket path returned by /init and logs
messages until interrupted.

Usage:
  python backend/scripts/smoke_frontend_client.py --base-url http://localhost:8000 --lat 28.6139 --lon 77.2090
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import sys
from typing import TypedDict, cast
from urllib.parse import urlparse

import websockets


class InitPayload(TypedDict):
    weather: dict[str, object] | None
    sensors: list[dict[str, object]]
    closest_sensor: dict[str, object] | None
    socket_path: str


@dataclass(frozen=True)
class CliArgs:
    base_url: str
    lat: float
    lon: float


@dataclass(frozen=True)
class InitResult:
    weather: dict[str, object] | None
    sensors: list[dict[str, object]]
    closest_sensor: dict[str, object] | None
    socket_path: str


def parse_args(argv: list[str]) -> CliArgs:
    base_url = "http://localhost:8000"
    lat_text: str | None = None
    lon_text: str | None = None

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--base-url" and i + 1 < len(argv):
            base_url = argv[i + 1]
            i += 2
            continue
        if arg == "--lat" and i + 1 < len(argv):
            lat_text = argv[i + 1]
            i += 2
            continue
        if arg == "--lon" and i + 1 < len(argv):
            lon_text = argv[i + 1]
            i += 2
            continue
        if arg in {"-h", "--help"}:
            raise SystemExit(
                "Usage: smoke_frontend_client.py --base-url http://localhost:8000 --lat <float> --lon <float>"
            )
        raise SystemExit(f"Unknown argument: {arg}")

    if lat_text is None or lon_text is None:
        raise SystemExit(
            "Missing required arguments: --lat <float> --lon <float>"
        )

    return CliArgs(base_url=base_url, lat=float(lat_text), lon=float(lon_text))


def post_init(base_url: str, lat: float, lon: float) -> InitResult:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"Invalid base URL: {base_url}")

    connection: http.client.HTTPConnection | http.client.HTTPSConnection
    if parsed.scheme == "https":
        connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=10)
    else:
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=10)

    path = f"{parsed.path.rstrip('/')}/init" if parsed.path else "/init"
    payload = json.dumps({"latitude": lat, "longitude": lon})

    try:
        connection.request(
            "POST",
            path,
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response = connection.getresponse()
        raw_body = response.read().decode("utf-8")
    except Exception as exc:
        raise SystemExit(f"/init request failed: {exc}") from exc
    finally:
        connection.close()

    if response.status >= 400:
        raise SystemExit(f"/init failed: {response.status} {raw_body}")

    data = cast(InitPayload, json.loads(raw_body))
    return InitResult(
        weather=data["weather"],
        sensors=data["sensors"],
        closest_sensor=data["closest_sensor"],
        socket_path=data["socket_path"],
    )


def to_ws_url(base_url: str, socket_path: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"Invalid base URL: {base_url}")

    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    port = f":{parsed.port}" if parsed.port else ""
    base = f"{ws_scheme}://{parsed.hostname}{port}{parsed.path.rstrip('/')}"
    path = socket_path if socket_path.startswith("/") else f"/{socket_path}"
    return f"{base}{path}"


async def run_socket(url: str) -> None:
    print(f"WS connect -> {url}")
    async with websockets.connect(url) as ws:
        print("WS connected")
        async for message in ws:
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                print(f"WS text: {message}")
            else:
                print("WS json:", json.dumps(parsed, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args(sys.argv)
    init_result = post_init(args.base_url, args.lat, args.lon)

    print("INIT weather:", json.dumps(init_result.weather, indent=2, sort_keys=True))
    print("INIT sensors:", json.dumps(init_result.sensors, indent=2, sort_keys=True))
    print(
        "INIT closest_sensor:",
        json.dumps(init_result.closest_sensor, indent=2, sort_keys=True),
    )
    print("INIT socket_path:", init_result.socket_path)

    ws_url = to_ws_url(args.base_url, init_result.socket_path)

    try:
        import asyncio

        asyncio.run(run_socket(ws_url))
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as exc:
        raise SystemExit(f"WebSocket failed: {exc}") from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())