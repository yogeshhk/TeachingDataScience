"""
Session 19 (MLOps): Step 3 of the "Deploy Your Own Model" section: call the service.

This script tests the API WITHOUT starting a server (FastAPI's TestClient calls the app in
process). It shows a good request (HTTP 200 and the prediction) and a bad request (a text value
for engine_size: HTTP 422, rejected by the Car type).

To try it against a real server instead:
    uvicorn app:app --port 8000
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"engine_size": 2.0, "horsepower": 140, "weight": 2800}'

Run: conda activate mlcoep && python train_model.py && python test_api.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient  # noqa: E402
import app as service  # noqa: E402

client = TestClient(service.app)

good = client.post("/predict", json={"engine_size": 2.0, "horsepower": 140, "weight": 2800})
print("good input -> HTTP", good.status_code, good.text)

bad = client.post("/predict", json={"engine_size": "big", "horsepower": 140, "weight": 2800})
print("bad input  -> HTTP", bad.status_code)
