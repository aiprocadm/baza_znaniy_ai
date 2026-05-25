# Screenshots

These three PNGs are referenced from the top-level `README.md`:

| File | What it should show |
|---|---|
| `chat-with-citations.png` | «Вопрос-ответ» tab: an LLM answer with clickable `[файл.pdf, стр. N]` citation buttons. |
| `pdf-viewer-modal.png` | The PDF.js modal opened from a citation, with the highlighted snippet visible. |
| `upload-flow.png` | «Документы» tab during a drag-and-drop upload (progress visible). |

**Current state:** all three files are 1×1 transparent placeholders so the README
renders without broken-image icons. **Replace them with real screenshots before
tagging v1.0.0** — the release checklist (`docs/release_checklist.md`, Sprint 3.8)
includes the capture procedure.

## Capture procedure (Linux/macOS/Windows)

```bash
# 1. Boot the MVP dev server
py -m uvicorn scripts.dev_server_mvp:app --port 8001

# 2. Open http://127.0.0.1:8001/ in a browser
# 3. Upload a real PDF with text (any short regulation/manual)
# 4. Wait for indexing → switch to «Вопрос-ответ» tab
# 5. Ask a question; capture the answer with citations → chat-with-citations.png
# 6. Click a citation; capture the PDF modal → pdf-viewer-modal.png
# 7. Switch to «Документы»; drag a file onto upload area → upload-flow.png
```

Optimise each PNG to ≤500 KB (e.g. `pngquant --quality 65-80 --output FILE --force FILE`).
