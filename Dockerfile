FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY models ./models

ENV PPO_MODEL_PATH=/app/models/ppo_aircraft_model.zip
ENV CORS_ORIGINS=http://localhost:5173

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
