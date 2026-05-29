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

## Alert: retrieval degraded

`kb_retrieval_degraded{reason,severity}` is `1` while a retrieval path is running in a quality-compromised mode and `0` otherwise. Reasons: `vector_backend_down` (Qdrant/FAISS down → grep fallback), `hashing_embedder` (no real embedder configured), `embedding_dim_mismatch` (index incoherent with the active embedder — reindex needed), `search_truncated` (corpus exceeds the scan cap).

Recommended Prometheus rule — page when any critical degradation persists for 5 minutes:

```yaml
- alert: RetrievalDegradedCritical
  expr: max_over_time(kb_retrieval_degraded{severity="critical"}[5m]) == 1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Retrieval degraded ({{ $labels.reason }})"
    description: "Answers may be irrelevant or empty. Check embedder config, reindex state, and vector backend health."
```
