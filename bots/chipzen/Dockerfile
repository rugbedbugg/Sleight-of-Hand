# Adapted from ChipZen's reference-bot container contract. Build the staged
# context produced by scripts/build_chipzen.py, not the repository root.
FROM python:3.13.15-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /bot
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY bot.py ./
COPY sleight_of_hand/ ./sleight_of_hand/
COPY LICENSE ./
USER 10001:10001
ENTRYPOINT ["python", "-u", "bot.py"]
