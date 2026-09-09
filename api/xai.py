from openai import AsyncOpenAI
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

# Fallback values are optional since main.py loads .env
_client = None
_cached_explanation = {
    "reason": "Waiting for significant data...",
    "prob_threshold": 0.0,
    "last_factors": [],
    "device_id": None
}

def get_openai_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
    return _client

async def generate_explanation(prob: float, top_factors: list, sensor_data: list, device_id: str) -> str:
    """Generate a 1-2 sentence explanation using LLM based on SHAP output."""
    client = get_openai_client()
    
    cols = ["cloud_top", "distance", "light", "rain", "humidity", "temperature", "pressure"]
    data_dict = dict(zip(cols, sensor_data))
    
    prompt = (
        f"You are an expert meteorologist. A cloudburst risk may happen."
        f"The main triggers driving this risk are {', '.join(top_factors)}. "
        f"Current sensor readings: {data_dict}. "
        "In exactly 1 or 2 concise professional sentences, explain why these readings indicate a cloudburst risk."
    )
    
    try:
        response = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM API Error: {e}")
        return "An error occurred while generating the atmospheric explanation."


async def check_and_update_explanation(prediction_data: dict, broadcast_callback):
    """
    Checks if the probability or factors changed significantly enough to warrant an LLM call.
    If so, fetches a new explanation and broadcasts it.
    """
    global _cached_explanation
    
    prob = prediction_data.get("final_prediction_prob")
    if prob is None:
        return
        
    top_factors = prediction_data.get("top_factors", [])
    device_id = prediction_data.get("device_id", "__global__")
    
    # Only fetch a new explanation if there is a significant jump (> 10% diff) 
    # OR if the top driving factors changed and the risk is > 40%.
    current_threshold = _cached_explanation["prob_threshold"]
    factors_changed = set(top_factors) != set(_cached_explanation["last_factors"])
    
    needs_update = False
    
    if abs(prob - current_threshold) >= 0.10:
        needs_update = True
    elif factors_changed and prob > 0.40:
        needs_update = True
        
    if needs_update and top_factors:
        new_reason = await generate_explanation(
            prob, 
            top_factors, 
            prediction_data.get("last_input", []), 
            device_id
        )
        
        _cached_explanation = {
            "reason": new_reason,
            "prob_threshold": prob,
            "last_factors": top_factors,
            "device_id": device_id
        }
        
        # Broadcast the new explanation immediately
        await broadcast_callback(_cached_explanation)
        
def get_cached_explanation():
    return _cached_explanation
