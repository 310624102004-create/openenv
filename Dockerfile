# CodeDebug OpenEnv — Server Dockerfile
# Build:  docker build -t codedebug-env .
# Run:    docker run -p 7860:7860 codedebug-env

FROM python:3.11-slim

LABEL maintainer="Giridharan"
LABEL description="CodeDebug OpenEnv environment server"
LABEL version="1.0.0"

RUN useradd --create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY manifest.json .
COPY openenv.yaml .
COPY app.py .

USER appuser

EXPOSE 7860

ENV PORT=7860 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/v1/health')"

CMD ["python", "app.py"]
