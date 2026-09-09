import joblib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
from typing import Optional

from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from api.prediction_window import SlidingWindowManager

# -----------------------------
# Load Dataset & Fit Scaler
# -----------------------------
try:
    df = pd.read_csv("data/sensor/cloudburst_dataset.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv("backend/data/sensor/cloudburst_dataset.csv")
    except FileNotFoundError:
        # Fallback synthetic frame if dataset is completely missing from repo
        print("WARNING: Dataset not found. Creating mock dataset for local scaler.")
        import numpy as np
        df = pd.DataFrame({
            "cloud_top": np.random.choice([0, 1], 100),
            "distance": np.random.uniform(10, 500, 100),
            "light": np.random.uniform(100, 4000, 100),
            "rain": np.random.randint(0, 1024, 100),
            "humidity": np.random.uniform(20, 100, 100),
            "temperature": np.random.uniform(-5, 50, 100),
            "pressure": np.random.uniform(900, 1100, 100),
            "cloud_burst": np.random.choice([0, 1], 100),
            "source_type": ["mock"] * 100
        })

X = df.drop(columns=["cloud_burst", "source_type"])
y = df["cloud_burst"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# -----------------------------
# Load models
# -----------------------------
xgb_model = joblib.load("models/xgboost_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
svc_model = joblib.load("models/svc_model.pkl")

cnn_model = tf.keras.models.load_model("models/cnn_binary_classification_model.keras")
dn_model = tf.keras.models.load_model("models/DenseNet_model.keras")

models = {
    "xgboost_tabular": xgb_model,
    "rf_tabular": rf_model,
    "svc_tabular": svc_model,
}

# -----------------------------
# Initialize SHAP for the Ensemble
# -----------------------------
def _ensemble_predict_proba(X_features):
    # Models were fitted on numpy arrays (scaled data), so pass numpy arrays to avoid warnings
    X_np = X_features.values if isinstance(X_features, pd.DataFrame) else X_features
    
    p_xgb = xgb_model.predict_proba(X_np)[:, 1]
    p_rf = rf_model.predict_proba(X_np)[:, 1]
    p_svc = svc_model.predict_proba(X_np)[:, 1]
    
    # Return exactly the same average probability that the sliding window calculates
    return (p_xgb + p_rf + p_svc) / 3.0

# Generate a small background dataset from the scaled training data (K-Means is extremely fast)
_bg_data = shap.kmeans(X_train, 10)
ensemble_explainer = shap.KernelExplainer(_ensemble_predict_proba, _bg_data)

# -----------------------------
# GLOBAL STATE
# -----------------------------
WINDOW_SIZE = 5
_window_manager = SlidingWindowManager(WINDOW_SIZE, list(models.keys()))

# Global latest per device
_latest_results: dict[str, dict] = {}
_latest_overall: Optional[dict] = None


# -----------------------------
# CORE FUNCTIONS
# -----------------------------
def get_prediction(model, input_data: list) -> tuple[int, float]:
    """Run a single inference on one reading."""
    features_df = pd.DataFrame([input_data], columns=X.columns)
    features_scaled = scaler.transform(features_df)

    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    return int(pred), float(prob)


def preprocess_image(contents: bytes):
    from io import BytesIO
    from PIL import Image

    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((128, 128))

    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    return arr


def predict_image_logic(contents: bytes, device_id: str = "__global__"):
    img = preprocess_image(contents)

    cnn_prob = float(cnn_model.predict(img)[0][0])
    dn_prob = float(dn_model.predict(img)[0][0])

    vision_preds = {
        "cnn1": {"predicted_class": int(cnn_prob > 0.5), "probability": cnn_prob},
        "cnn2": {"predicted_class": int(dn_prob > 0.5), "probability": dn_prob},
    }
    _window_manager.set_vision_predictions(device_id, vision_preds)

    return vision_preds


def predict_tabular_logic(data: list, device_id: str = "__global__") -> dict:
    """
    Sliding-window inference.

    - Each call appends one reading to the per-device deque (maxlen=WINDOW_SIZE).
    - Once the deque is full (>= WINDOW_SIZE readings) a prediction is emitted
      on *every* call — this is the sliding-window behaviour.
    - final_prediction / final_prediction_prob are None until we have enough data.
    """
    global _latest_overall

    # Translate binary cloud top indicator to realistic altitude metrics
    if data[0] == 1.0 or data[0] == 1:
        data[0] = float(random.choice([14000, 15000, 16000, 17000, 18000]))
    elif data[0] == 0.0 or data[0] == 0:
        data[0] = float(random.choice([1000, 1200, 1400, 1600, 1800]))

    # Append this reading to each model's window
    for name, model in models.items():
        pred, prob = get_prediction(model, data.copy())
        _window_manager.add_tabular_result(device_id, name, pred, prob)

    final_prediction, final_prediction_prob = _window_manager.compute_prediction(device_id)

    top_factors = []
    if final_prediction_prob is not None:
        try:
            # We skip pandas wrapping here because KernelExplainer takes the numpy array directly
            features_df = pd.DataFrame([data], columns=X.columns)
            features_scaled = scaler.transform(features_df)
            
            # The KernelExplainer will give us the SHAP values representing the *true ensemble* voting output
            shap_vs = ensemble_explainer.shap_values(features_scaled, silent=True)
            
            # shape mapping depending on shap version (sometimes returns list, sometimes array)
            sv = shap_vs[0] if isinstance(shap_vs, list) else shap_vs[0]
            
            impacts = list(zip(X.columns, sv))
            impacts.sort(key=lambda x: x[1], reverse=True)
            top_factors = [feat for feat, score in impacts[:2] if score > 0]
        except Exception as e:
            print(f"SHAP error: {e}")

    result = {
        "final_prediction": final_prediction,
        "final_prediction_prob": final_prediction_prob,
        "top_factors": top_factors,
        "last_input": data,
        "device_id": device_id,
        "window_size": _window_manager.current_window_size(device_id),
    }

    _latest_results[device_id] = result
    _latest_overall = result
    return result


def get_latest(device_id: Optional[str] = None) -> Optional[dict]:
    if device_id:
        return _latest_results.get(device_id)
    return _latest_overall