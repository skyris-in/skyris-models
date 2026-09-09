import json
import logging
import paho.mqtt.client as mqtt
import paho.mqtt.publish as mqtt_publish
from api.inference import predict_tabular_logic
from api.sensor_registry import remove_sensor, upsert_sensor

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "skyris/sensor/lora_node"
MQTT_META_TOPIC = "skyris/sensor/lora_node/meta/+"  # retained device metadata

_prediction_callback = None


def on_connect(client, userdata, flags, reason_code, properties):
    logger.info(
        f"MQTT connected to {MQTT_BROKER} with result code {reason_code}")
    client.subscribe(MQTT_TOPIC)
    logger.info(f"Subscribed to topic: {MQTT_TOPIC}")
    client.subscribe("skyris/sensor/vision_node")
    logger.info("Subscribed to topic: skyris/sensor/vision_node")
    # Also subscribe to retained metadata messages from all devices
    client.subscribe(MQTT_META_TOPIC)
    logger.info(f"Subscribed to metadata topic: {MQTT_META_TOPIC}")


def on_message(client, userdata, msg):
    try:
        # Route metadata messages to sensor registry
        if msg.topic.startswith("skyris/sensor/lora_node/meta/"):
            raw_payload = msg.payload.decode("utf-8").strip()
            sensor_id = msg.topic.split("/")[-1]

            if not raw_payload:
                remove_sensor(sensor_id)
                logger.info(f"Sensor removed: {sensor_id}")
                return

            meta = json.loads(raw_payload)
            sensor_id = meta.get("device_id") or sensor_id
            upsert_sensor(sensor_id, meta)
            logger.info(f"Sensor registered/updated: {sensor_id} @ "
                        f"{meta.get('latitude')},{meta.get('longitude')}")
            return

        # ── existing data-topic handling below — unchanged ──────────────────
        # 1. Parse incoming message as JSON
        payload = json.loads(msg.payload.decode("utf-8"))
        
        log_payload = payload.copy()
        if "image" in log_payload:
            log_payload["image"] = f"<base64_string_length_{len(log_payload['image'])}>"
            
        logger.info(f"Received via MQTT on {msg.topic}: {log_payload}")

        if msg.topic == "skyris/sensor/vision_node":
            device_id = payload.get("device_id", "vision-node-01")
            
            if "image" in payload:
                import base64
                from api.inference import predict_image_logic
                
                # Decode image base64 bytes
                img_bytes = base64.b64decode(payload["image"])
                vision_preds = predict_image_logic(img_bytes, device_id=device_id)
                logger.info(f"Vision Inference Results [{device_id}]: {vision_preds}")
                
                # Construct result payload for frontend
                result = {
                    "vision_predictions": vision_preds,
                    "device_id": device_id,
                    "image": payload["image"],
                    "ts": payload.get("ts")
                }
                
                if _prediction_callback:
                    _prediction_callback(result)
            return

        # 2. Extract "data" array and pass to inference
        device_id = payload.get("device_id", "unknown")
        if "data" in payload and isinstance(payload["data"], list):
            # Pass device_id so each sensor gets its own sliding window
            predictions = predict_tabular_logic(payload["data"], device_id=device_id)
            logger.info(f"Inference Results [{device_id}]: {predictions}")

            # 3. Fire the callback to push data to WebSocket clients
            if _prediction_callback:
                _prediction_callback(predictions)
        else:
            logger.warning(
                f"Invalid payload format, expected 'data' list: {payload}")

    except Exception as e:
        logger.error(f"Error parsing MQTT message: {e}")


def start_mqtt_client(prediction_callback=None):
    global _prediction_callback
    _prediction_callback = prediction_callback

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    logger.info(f"Starting MQTT client, connecting to {MQTT_BROKER}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # loop_start() creates a background thread to handle network I/O
    client.loop_start()
    return client


def stop_mqtt_client(client):
    if client:
        client.loop_stop()
        client.disconnect()
        logger.info("MQTT client disconnected and stopped.")


def clear_retained_metadata(sensor_ids: list[str]) -> dict[str, list[str]]:
    """Clear retained metadata for provided sensor IDs by publishing tombstones."""
    cleared: list[str] = []
    failed: list[str] = []

    for sensor_id in sensor_ids:
        topic = f"skyris/sensor/lora_node/meta/{sensor_id}"
        try:
            mqtt_publish.single(
                topic=topic,
                payload="",
                qos=1,
                retain=True,
                hostname=MQTT_BROKER,
                port=MQTT_PORT,
            )
            cleared.append(sensor_id)
        except Exception as exc:
            logger.error("Failed to clear retained metadata for %s: %s", sensor_id, exc)
            failed.append(sensor_id)

    return {"cleared": cleared, "failed": failed}
