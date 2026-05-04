# Runbook: indexing incidents and LLM degradation

## 1) Indexing backlog / stalled ingestion
1. Check `/api/v1/ops/health/readiness` and `/api/v1/ops/health/dependencies`.
2. Validate queue lag metric `kb_ingest_queue_lag` and worker logs (filter by `task_id=ingest-worker`).
3. If scheduler degraded, restart worker process.
4. Re-run stuck tasks via maintenance/retry procedures.
5. Confirm recovery by observing ingestion throughput and error-rate normalization.

## 2) Elevated latency / p95 breach
1. Confirm p95 breach on SLO panel.
2. Correlate with `request_id` from API logs and downstream worker logs.
3. Identify hottest route and tenant impacted.
4. Apply temporary mitigation: reduce concurrency pressure, disable expensive reranking.
5. Post-incident: tune capacity and update thresholds.

## 3) LLM degraded quality or instability
1. Track `kb_llm_token_usage_total` for spikes and potential prompt loops.
2. Review error logs with request/task correlation context.
3. Switch to fallback model/provider if available.
4. Trigger warmup endpoint and verify dependencies.
5. Record incident timeline and corrective actions.

## Logging context checklist
Every incident log search should include:
- `request_id`
- `tenant_id`
- `user_id`
- `task_id`


## 4) Reindex incident rollback
1. Confirm the active alias still points to the last known-good collection (for example `kb_prod -> kb_vNN`).
2. If alias switch happened before failure detection, switch alias back to previous collection immediately.
3. Delete the temporary reindex collection (`*_tmp_*`) only after alias rollback is validated.
4. Re-run `GET /api/v1/ingest/jobs/{job_id}` and verify the failed job error details were captured for RCA.
5. Re-run reindex only after fixing root cause and validating on a canary tenant/document.
