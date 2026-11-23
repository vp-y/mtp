# Breast ultrasound analysis web app (CPU). Build from the repository root, with the exported
# model bundle in ./deploy_bundle:
#   docker build -t bus-app .
#   docker run -p 7860:7860 bus-app        -> http://localhost:7860
# Port 7860 is what Hugging Face Spaces expects (see docs/DEPLOYMENT.md).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# non-root user (required by Hugging Face Spaces, good practice elsewhere)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-app.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-app.txt

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user deploy_bundle ./deploy_bundle

USER user
ENV PYTHONPATH=/home/user/app/src BUS_BUNDLE=/home/user/app/deploy_bundle BUS_DEVICE=cpu \
    OMP_NUM_THREADS=2
EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
