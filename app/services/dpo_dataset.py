"""Compose a synthetic DPO preference dataset.

Pure-logic core of Workstream 4 (DPO post-training / preference
learning) in the Pack B++ ML strengthening plan. Builds on W1's
:mod:`app.services.synthetic_qa` for seed Q&A and W3's
:mod:`app.services.rag_dataset` (Hamilton apportionment +
citation stripping helpers).

The module is intentionally I/O free: teacher provider and the
seed iterator are injected so the logic is deterministic in tests.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)
