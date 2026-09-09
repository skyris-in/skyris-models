FROM python:3.12-slim

# Install uv from the official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /backend

# 1. Install system dependencies (Crucial for XGBoost/SHAP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy only dependency files
COPY pyproject.toml README.md ./

# 3. FIXED UV INSTALL COMMAND
# We removed --prefer-binary. UV automatically picks binaries by default.
RUN uv pip install --system --no-cache .

# 4. Copy code
COPY . .

ENV PYTHONPATH=/backend
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]