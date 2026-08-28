FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live')"]
CMD ["uvicorn", "evidencebench.api:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
