# Observability and SLO panel

## Unified metrics
- **QPS**: `rate(kb_api_requests_total[5m])`
- **p95 latency**: `histogram_quantile(0.95, sum(rate(kb_api_request_duration_seconds_bucket[5m])) by (le))`
- **Error rate**: `sum(rate(kb_api_requests_total{status="error"}[5m])) / sum(rate(kb_api_requests_total[5m]))`
- **Queue lag**: `kb_ingest_queue_lag`
- **Ingestion throughput**: `rate(kb_ingestion_throughput_docs_total{status="success"}[5m])`
- **Token usage**: `sum(rate(kb_llm_token_usage_total[5m])) by (direction, provider)`

## Health checks
- `/api/v1/ops/health/liveness` — process is alive.
- `/api/v1/ops/health/readiness` — app can accept traffic, includes degradation flags.
- `/api/v1/ops/health/dependencies` — external/internal dependencies status.

## Suggested alert rules
- `p95_latency_high`: p95 > target for 10m.
- `error_rate_high`: error rate > threshold for 5m.
- `queue_lag_high`: queue lag above SLO limit for 10m.
- `ingest_throughput_drop`: throughput is near zero during business hours.
- `llm_token_spike`: sudden token usage growth (possible runaway prompts).
