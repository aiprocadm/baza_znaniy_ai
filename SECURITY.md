# Security Policy

## Reporting a vulnerability

Email security reports to **aiproc.adm@gmail.com** with subject prefix `[KB.AI security]`.
Please include:

- Affected version (commit hash or release tag).
- Steps to reproduce.
- Impact assessment (data exposure, RCE, DoS, etc.).
- Suggested mitigation, if any.

You will receive an acknowledgement within **7 days**. A coordinated disclosure
timeline (typically 30 days) will be proposed.

## What we treat as security issues

- Authentication / authorization bypasses on protected endpoints.
- Path traversal, SSRF, command injection.
- SQL injection (we use parameterized queries; report any deviation).
- Sensitive data leakage (API keys, document contents) to unauthorized callers.
- CSRF on mutating endpoints.

## What we do NOT treat as security issues

- Lack of rate limiting on individual endpoints — by design, single-tenant.
- LLM hallucinations or factual errors in answers — known RAG limitation, mitigate at the prompt level.
- Lack of multi-tenant isolation — `/api/kb/*` is single-tenant by design.
- Reliance on `KB_API_KEY` for all mutations — single-shared-key model is intentional for MVP.

## Disclosure preferences

- **No bug bounty.** This is a side-project; we cannot pay.
- **Credit happily given** in release notes if requested.
- **30-day grace period** before public disclosure, extendable by mutual agreement.
- **No SLA** for response time beyond the 7-day acknowledgement target.

## Known limitations (not bugs)

- Single-tenant: one `KB_API_KEY` for the whole installation.
- No document-level RBAC: anyone with the key sees all documents.
- SQLite for the MVP store: not crash-tolerant under heavy concurrent writes.
- No audit-log retention policy: `audit_log` table grows unbounded — operator's responsibility to prune.

## Supported versions

Only the latest released tag receives security patches. Older tags are best-effort.
