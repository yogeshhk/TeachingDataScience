# Dockerfile: Step 4 of the "Deploy Your Own Model" section (LaTeX/ml_mlops_workflow.tex)
# Build:  python train_model.py            (creates mpg_model.joblib)
#         docker build -t mpg-service .
# Run:    docker run -p 8000:8000 mpg-service
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py mpg_model.joblib ./
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
