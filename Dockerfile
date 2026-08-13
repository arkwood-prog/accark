# Deployable image for the dashboard.
#
#   docker build -t bettingedge .
#   docker run -p 8000:8000 -e BETTINGEDGE_TOKEN=pick-something bettingedge
#
# History is downloaded on start-up and cached in the container, so the first
# boot takes a little longer than later restarts.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BETTINGEDGE_CACHE=/data/cache \
    BETTINGEDGE_LEAGUE=EC \
    BETTINGEDGE_SEASONS=6 \
    PORT=8000

WORKDIR /app
COPY pyproject.toml README.md ./
COPY bettingedge ./bettingedge
RUN pip install --no-cache-dir -e . && mkdir -p /data/cache

EXPOSE 8000
# 0.0.0.0 so the container is reachable; set BETTINGEDGE_TOKEN to lock it down.
CMD ["sh", "-c", "bettingedge serve --host 0.0.0.0 --port ${PORT} \
     --league ${BETTINGEDGE_LEAGUE} --seasons ${BETTINGEDGE_SEASONS}"]
