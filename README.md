---
title: Skyris Models API
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# 🛰️ Skyris Models API

Real-time ML inference engine and WebSocket streaming server for the Skyris wildfire & agricultural risk monitoring platform.

---

## 📌 Features

- **Multimodal AI Inferences**:
  - Image classification using Convolutional Neural Networks & DenseNet.
  - Tabular sensor data risk classification with XGBoost, Random Forest, & SVM.
  - Explainable AI (XAI) feature importance and natural language summaries via SHAP & LLM.
- **Real-Time Communication**:
  - MQTT broker integration for edge sensor data collection.
  - High-throughput WebSockets for streaming live sensor metrics & risk scores to frontend clients.
- **Containerized Deployment**:
  - Fast, multi-stage builds using `uv` package manager and Python 3.12.
  - Ready for Hugging Face Spaces (Docker SDK) and container orchestrators.

---

## 🚀 Local Docker Setup

### 1. Build the Docker Image
From the repository root:
```bash
docker build -t skyris-models .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 --env-file .env skyris-models
```
*(If you do not have a `.env` file yet, you can run without `--env-file .env`)*:
```bash
docker run -p 8000:8000 skyris-models
```

### 3. Verify Local Deployment
- **API Status**: [http://localhost:8000/](http://localhost:8000/)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Endpoints

### REST Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status check |
| `GET` | `/health` | Healthcheck endpoint |
| `POST` | `/init` | Device & sensor initialization handshake |
| `POST` | `/image/` | Upload camera image for visual risk prediction |
| `POST` | `/data/` | Send array of environmental sensor readings for risk evaluation |
| `GET` | `/latest-result/` | Fetch the latest inference results |
| `GET` | `/debug/sensors` | List active registered sensors |
| `POST` | `/debug/reset-sensors` | Reset retained sensor state |

### WebSocket Endpoints
| Protocol | Endpoint | Description |
| :--- | :--- | :--- |
| `WS` | `/ws/sensor/{device_id}/` | Stream live updates and predictions for a specific sensor node |
| `WS` | `/ws/latest-result/` | Stream broadcast predictions across all sensor nodes |
| `WS` | `/ws/explanation/` | Stream SHAP / LLM natural language risk explanations |

---

## 🔄 Hugging Face Space Sync (GitHub Actions)

This repository includes automated CI/CD synchronization with Hugging Face Spaces via GitHub Actions.

### Setup Instructions
1. Create a Hugging Face Space with the **Docker SDK**.
2. Generate a Hugging Face User Access Token with **Write** permissions under **Settings -> Access Tokens**.
3. In your GitHub repository, go to **Settings -> Secrets and variables -> Actions** and add:
   - `HF_TOKEN`: Your Hugging Face write access token.
   - *(Optional)* `HF_SPACE_REPO`: The target Space repository (e.g. `skyris-in/skyris-models` or `your-username/space-name`). If omitted, it defaults to `skyris-in/skyris-models`.
4. Any push to `master` or `main` will automatically deploy changes directly to the Hugging Face Space.
