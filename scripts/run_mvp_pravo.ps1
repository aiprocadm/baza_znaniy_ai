# Launch the MVP server against the full RF legal corpus with keyless, full-coverage RAG.
#
# The corpus (6141 articles / 14231 chunks) is ingested in var/data/pravo_public.sqlite
# with the SentenceTransformer e5-small embedder (st:384, real semantic vectors). Three
# settings make the product serve it correctly:
#
#   KB_MVP_DB_PATH        point the MVP store at the pravo corpus DB
#   ST_EMBED_MODEL        MUST be e5-small to match the index — explicit KB_EMBEDDINGS_BACKEND=st
#                         resolves to bge-m3 (st:1024) and the embedder-signature guard rejects it
#   KB_SEARCH_HARD_LIMIT  the corpus has 14231 chunks; the default brute-force cap is 10000, which
#                         leaves ~30% unsearched. 20000 covers all chunks. (Brute-force cosine over
#                         14k x 384-dim vectors is milliseconds; FAISS only pays off at ~100k+.)
#
# These are read via raw os.environ (not pydantic .env), so they must be real env vars at launch.
# Leave KB_API_KEY unset for open local dev, or set it to require an X-API-Key header.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$env:KB_MVP_DB_PATH       = "var/data/pravo_public.sqlite"
$env:ST_EMBED_MODEL       = "intfloat/multilingual-e5-small"
$env:KB_SEARCH_HARD_LIMIT = "20000"

py -3.13 -m uvicorn scripts.dev_server_mvp:app --reload --port 8001
