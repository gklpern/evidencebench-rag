from prometheus_client import Counter, Histogram

REQUESTS = Counter("evidencebench_requests_total", "API requests", ["route", "status"])
LATENCY = Histogram(
    "evidencebench_request_duration_seconds",
    "API request latency",
    ["route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
ABSTENTIONS = Counter("evidencebench_abstentions_total", "Abstained answers", ["reason"])
INGESTED_CHUNKS = Counter("evidencebench_ingested_chunks_total", "Indexed chunks")
