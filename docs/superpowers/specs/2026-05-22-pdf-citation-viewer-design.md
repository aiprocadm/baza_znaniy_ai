# PDF Citation Viewer — Design

**Date:** 2026-05-22
**Branch:** `feat/kb-mvp-corporate-rag`
**Scope:** Phase 1.2 из vision-roadmap (`2026-05-22-project-vision-design.md`): точные цитаты `[файл.pdf, стр. 12]` со встроенным PDF-viewer в modal overlay. Только PDF; non-PDF и legacy документы получают graceful fallback на text snippet.
**Status:** Validated design. Source of truth для имплементации.

---

## 1. Background and motivation

В текущем MVP-слое `/api/kb/*` ответы `/api/kb/ask` возвращают массив `sources` с плоскими text snippets каждого чанка. Источник определяется только по `document_id` + `chunk_index` — без указания страницы и без возможности проверить цитату в исходном документе.

Vision-документ (`docs/superpowers/specs/2026-05-22-project-vision-design.md`, раздел 6, фаза 1.2) ставит задачу: цитаты должны быть в формате `[документ.pdf, стр. 12, раздел 3.2]` с кликабельным переходом в PDF-вьюер с подсветкой фрагмента. Это **wedge-усиление** — фича, которая отличает KB.AI от Notion AI / MS Copilot / Onyx и идёт в demo-видео и landing page.

Текущее состояние блокирует фичу в трёх местах:

1. **Page-number теряется при ingestion** — `_parse_file_bytes` (`app/api/kb_mvp.py:494`) склеивает Docling per-page результат в один `\n\n`-text перед chunking. Сам Docling возвращает `list[tuple[int, str]]`, данные есть, но мы их не доносим.
2. **Оригинал не сохраняется** — `upload_document` (`app/api/kb_mvp.py:558`) парсит файл в текст и выбрасывает raw bytes. Для PDF-вьюера нужен исходный PDF.
3. **`SearchHit` (`app/services/kb_store.py:83-92`) не имеет поля `page`** — даже если бы chunking сохранял страницу, она бы не дошла до API response.

Этот документ закрывает все три блокера.

## 2. Goals / Non-goals

### Goals

- Каждая цитата в `/api/kb/ask` (sync + stream) и `/api/kb/search` ссылается на конкретную страницу с указанием имени файла
- Клик по цитате в UI открывает modal overlay с PDF.js, показывающим страницу и подсвечивающим текст чанка через PDF.js find API
- Обратная совместимость: документы без оригинала (legacy + не-PDF) продолжают работать через text snippet
- Auth: blob-эндпоинт оригинала уважает существующий `KB_API_KEY` через `protected` router
- Zero breakage: старые клиенты (которые не парсят новые поля response) продолжают работать без правок

### Non-goals

- ❌ DOCX/PPTX/XLSX native rendering — fallback к text snippet (отдельный sprint после первого пилот-запроса)
- ❌ Точные bbox-highlight rectangles — только text-search через PDF.js find API
- ❌ Retrofit старых документов — clean break (Подход A), схема forward-compatible для будущего апгрейда
- ❌ Поддержка отсканированных PDF без text layer — работает только для extractable text PDF (включая OCR-выход Docling, но не raw scan без OCR)
- ❌ Document-level RBAC на blob endpoint — endpoint защищён общим `KB_API_KEY`, без per-source permissions
- ❌ E2E browser-tests на PDF.js — manual smoke checklist + unit tests на surface

## 3. Architecture

### 3.1 Data flow

```
Upload PDF
   │
   ▼
[1] Сохранить raw PDF в var/data/kb_files/<doc_id>.pdf
   │
   ▼
[2] Парсинг Docling → list[(page_number, page_text)]
   │
   ▼
[3] Chunking PER PAGE: каждый чанк помнит свой page_number
   │
   ▼
[4] INSERT INTO kb_chunks (..., page_number)
       INSERT INTO kb_documents (..., has_original_file=TRUE, file_relpath)

──────────────────────────────────────────────

User asks question
   │
   ▼
[5] /api/kb/ask → search → rerank → выдаёт чанки + page_number
   │
   ▼
[6] sources[i] = {document_id, document_title, filename, page,
                  snippet, has_original, chunk_index, score}
   │
   ▼
[7] UI рендерит цитату:
      [регламент.pdf, стр.12 ▸]  (если has_original=true и .pdf)
   │
   ▼
[8] Клик → openCitationModal({doc_id, page, snippet})
   │
   ▼
[9] Modal загружает /api/kb/documents/<doc_id>/file
   ▼  PDF.js рендерит page=12, dispatch find(snippet[:30 words])
   ▼
[10] Подсветка фрагмента, скролл к нему
```

### 3.2 Изменения по слоям

| Слой | Файлы | Что меняется |
|------|---|---|
| **DB schema** | `alembic/versions/20260523_01_pdf_citation.py` (new) | +`kb_chunks.page_number INT NULL`<br>+`kb_documents.has_original_file BOOL DEFAULT FALSE`<br>+`kb_documents.file_relpath TEXT NULL` |
| **Storage** | `var/data/kb_files/` (dir, gitignored) | Filesystem-стор PDF-блобов |
| **Ingestion** | `app/api/kb_mvp.py` (upload_document), `app/services/kb_store.py` (add_document) | Сохранение blob; chunking per-page |
| **Search** | `app/services/kb_store.py` (SearchHit, search) | Поле `page` в SearchHit, propagate в API |
| **API** | `app/api/kb_mvp.py` (+endpoint, response model) | `GET /api/kb/documents/<id>/file`; `HitOut` +`page`+`has_original` |
| **Frontend** | `data/www/index.html`, `data/www/js/kb-auth.js` (new, extract), `data/www/js/pdf-viewer.js` (new), `data/www/i18n/ru.json` (+keys), `data/www/vendor/pdfjs/` (new vendored) | Modal viewer, citation rendering, PDF.js bundle, shared auth-helpers |
| **Tests** | `tests/test_kb_citation_*.py` (new) | Unit + integration + JS smoke |

### 3.3 Design rationale

**Filesystem, а не SQLite BLOB:** SQLite на BLOB-ах размера 1-10MB пишет приемлемо, но при чтении тащит весь blob в память. PDF может быть 50-100MB (тех-документация, договоры). Filesystem — стандартный путь; для бэкапа (задача 1.4 из vision) `tar.gz` гребёт оба места одинаково.

**Chunking per-page, а не глобально с offset-маппингом:** альтернатива — чанковать весь текст одной строкой и матчить чанки обратно на pages по символьным смещениям. Это требует хранить offset map и пересчитывать после reranker'а. Per-page chunking даёт ровный `page_number` на каждом чанке без post-processing. Trade-off: иногда чанки получаются меньше `chunk_size` (если страница маленькая), что **уже** происходит на coarse-grained PDF.

**Save blob ДО парсинга, а не после:** если Docling падает на странном PDF, blob уже на диске и его можно изучить для багов парсера. Альтернатива (save after parse) даёт чище rollback на ошибке, но теряет диагностическую ценность. Trade-off в пользу operability.

## 4. DB schema and file storage

### 4.1 Миграция Alembic

Новая миграция `alembic/versions/20260523_01_pdf_citation.py`:

```python
revision = "20260523_01_pdf_citation"
down_revision = "20260522_01_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("kb_chunks", sa.Column("page_number", sa.Integer, nullable=True))
    op.add_column(
        "kb_documents",
        sa.Column("has_original_file", sa.Boolean, nullable=False, server_default=sa.text("0")),
    )
    op.add_column("kb_documents", sa.Column("file_relpath", sa.String(512), nullable=True))
    op.create_index("idx_kb_chunks_doc_page", "kb_chunks", ["document_id", "page_number"])


def downgrade() -> None:
    op.drop_index("idx_kb_chunks_doc_page", table_name="kb_chunks")
    op.drop_column("kb_documents", "file_relpath")
    op.drop_column("kb_documents", "has_original_file")
    op.drop_column("kb_chunks", "page_number")
```

Существующие строки после миграции:
- `kb_chunks.page_number = NULL` для всех старых чанков
- `kb_documents.has_original_file = FALSE`, `file_relpath = NULL` для всех старых документов

`server_default=sa.text("0")` нужен потому, что SQLAlchemy `default=False` применяется только при INSERT новых строк. Без `server_default` миграция упадёт на NOT NULL колонке для существующих строк.

### 4.2 Filesystem layout

```
var/data/
├── kb_mvp.sqlite              (уже есть)
└── kb_files/                  (новое — gitignored)
    ├── 1.pdf                  (doc_id = 1)
    ├── 2.pdf                  (doc_id = 2)
    ├── 17.pdf                 (doc_id = 17)
    └── ...
```

**Соглашение:**
- Имя файла = `<doc_id>.<ext>`, где ext всегда `pdf` в текущей итерации
- `file_relpath` в БД хранит относительный путь `kb_files/<doc_id>.pdf` (не абсолютный — переносится между машинами)
- Путь резолвится через `Settings.data_dir / file_relpath`
- При `DELETE /api/kb/documents/<id>` сначала удаляется файл, потом запись из БД. Если файл не удалился — лог warning, БД-запись всё равно удаляется

### 4.3 Конфигурация

Новых env-vars не вводится. `data_dir` уже есть в Settings, `kb_files/` — поддиректория.

`.gitignore` уже покрывает `var/` (после Task A.3 из foundation-cleanup плана).

### 4.4 Backwards-compatibility таблица

| Документ | `has_original_file` | `file_relpath` | Чанки имеют `page_number` | Поведение в UI |
|---|---|---|---|---|
| Загружен **до** миграции | `false` | `null` | `null` для всех | Цитата → modal с text snippet only |
| Текстовый POST `/documents` после миграции | `false` | `null` | `null` (нет страниц) | То же — text snippet only |
| PDF upload после миграции | `true` | `kb_files/<id>.pdf` | целое для каждого | Цитата → PDF.js viewer на page=N |
| DOCX/PPTX/XLSX upload после миграции | `false` | `null` | `null` (оригинал не сохраняем) | Text snippet only |

### 4.5 Edge cases схемы

- **`page_number` overflow:** SQLite Integer 64-bit, до 9 квинтиллионов — недостижимо. OK
- **`file_relpath` коллизии:** doc_id — PK autoincrement, уникален. OK
- **Орфанные файлы в `kb_files/`:** если БД-запись удалена, но файл остался (race condition, ручной cleanup пошёл не так) — нормально, копится как мусор. Опциональный cleanup можно добавить в backup CLI (задача 1.4), но в этой spec не делается
- **Симлинки и path traversal:** запрещены — `file_relpath` должен начинаться с `kb_files/` и не содержать `..`. Защита проверяется в blob-эндпоинте (см. секция 5.4)

## 5. Backend changes

### 5.1 Ingestion pipeline

В `app/api/kb_mvp.py:upload_document` три изменения:

```python
# (1) ПОСЛЕ валидации размера, ДО парсинга — сохраняем blob:
file_relpath: str | None = None
tmp_blob: Path | None = None
if ext == "pdf":
    settings = get_settings()
    kb_files_dir = Path(settings.data_dir) / "kb_files"
    kb_files_dir.mkdir(parents=True, exist_ok=True)
    tmp_blob = kb_files_dir / f".tmp-{uuid.uuid4().hex}.pdf"
    tmp_blob.write_bytes(data)

# (2) парсинг возвращает уже pages — НЕ склеиваем
pages, mime_type = _parse_file_bytes_with_pages(filename, data)
# pages: list[tuple[int, str]] — то что Docling уже даёт

# (3) add_document получает pages вместо текста
doc = store.add_document(
    title=title or filename,
    pages=pages,
    source="file",
    filename=filename,
    mime_type=mime_type,
)

# (4) ПОСЛЕ INSERT — переименовываем tmp в финальное имя
if tmp_blob is not None and tmp_blob.exists():
    final_blob = kb_files_dir / f"{doc.id}.pdf"
    tmp_blob.rename(final_blob)
    store.update_file_metadata(doc.id, file_relpath=f"kb_files/{doc.id}.pdf")
    file_relpath = f"kb_files/{doc.id}.pdf"
```

Новый helper `_parse_file_bytes_with_pages` параллельно к существующему `_parse_file_bytes` — возвращает `(list[tuple[int, str]], mime_type)`. Старый можно либо удалить (если других callers нет), либо оставить как тонкую обёртку.

**Atomic rollback:** если `store.add_document` бросает exception после write tmp_blob, нужен `try/finally` для unlink tmp-файла. Иначе orphan tmp накопится.

### 5.2 `KnowledgeBaseStore.add_document` — новая сигнатура

```python
def add_document(
    self,
    title: str,
    pages: Sequence[tuple[int, str]],     # БЫЛО text: str
    *,
    source: str = "text",
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> Document:
    full_text = "\n\n".join(t for _, t in pages if t)
    self._validate_text(full_text)  # MAX_TEXT_LEN
    
    # Per-page chunking — используем существующий split_text() из этого же модуля
    chunks: list[tuple[int, str]] = []
    for page_number, page_text in pages:
        for chunk_text in split_text(page_text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
            chunks.append((page_number, chunk_text))
    
    chunk_texts = [t for _, t in chunks]
    embedded_blobs, embedder_name, dim = self._embed_chunks(chunk_texts)
    
    with self._connect() as conn:
        cur = conn.execute(
            "INSERT INTO kb_documents(title, text, created_at, source, filename, mime_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (title, full_text, _now_iso(), source, filename, mime_type),
        )
        doc_id = cur.lastrowid
        for idx, ((page_number, text), blob) in enumerate(zip(chunks, embedded_blobs)):
            conn.execute(
                "INSERT INTO kb_chunks(document_id, chunk_index, text, embedding, "
                "embedder, dim, page_number) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, idx, text, blob, embedder_name, dim, page_number),
            )
        conn.commit()
    
    return self._get_document(doc_id)
```

**Совместимость с текстовым POST `/documents`:** там нет страниц → call site вызывает `add_document(title, pages=[(1, text)], source="text")`. `page_number=1` для всех чанков, но `has_original_file=false` — UI всё равно покажет text snippet, не PDF viewer.

Новый метод `update_file_metadata`:

```python
def update_file_metadata(self, doc_id: int, *, file_relpath: str) -> None:
    with self._connect() as conn:
        conn.execute(
            "UPDATE kb_documents SET has_original_file=1, file_relpath=? WHERE id=?",
            (file_relpath, doc_id),
        )
        conn.commit()
```

### 5.3 `SearchHit` и propagation

```python
@dataclass(frozen=True)
class SearchHit:
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float
    source: str = "text"
    filename: Optional[str] = None
    page: Optional[int] = None              # NEW
    has_original: bool = False              # NEW
```

`search()` SELECT расширяется: `SELECT ..., c.page_number, d.has_original_file FROM kb_chunks c JOIN kb_documents d ...`. Существующий cosine-similarity сортинг не меняется.

### 5.4 API контракт

**`HitOut` (response model):**

```python
class HitOut(BaseModel):
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float
    source: str = "text"
    filename: Optional[str] = None
    page: Optional[int] = None              # NEW — null для legacy/non-PDF
    has_original: bool = False              # NEW
```

Существующие эндпоинты `/api/kb/search`, `/api/kb/ask`, `/api/kb/ask/stream` — никаких новых обязательных полей в request. Response расширяется опционально (старые клиенты игнорируют). Zero-breakage.

**Новый эндпоинт:**

```python
@protected.get("/documents/{doc_id}/file")
def get_document_file(doc_id: int, request: Request) -> Response:
    """Stream the original blob for documents with has_original_file=true."""
    store = _store_for(request)
    doc = store.get_document(doc_id)  # raises 404 if missing
    
    if not doc.has_original_file or not doc.file_relpath:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="NO_ORIGINAL_FILE")
    
    settings = get_settings()
    data_dir = Path(settings.data_dir).resolve()
    absolute = (data_dir / doc.file_relpath).resolve()
    expected_root = (data_dir / "kb_files").resolve()
    try:
        absolute.relative_to(expected_root)
    except ValueError:
        LOGGER.error("Path traversal attempted for doc %d: %s", doc_id, doc.file_relpath)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="STORAGE_ERROR")
    
    if not absolute.is_file():
        LOGGER.warning("Original file missing for doc %d: %s", doc_id, absolute)
        raise HTTPException(status.HTTP_410_GONE, detail="FILE_DELETED")
    
    return FileResponse(
        absolute,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{doc.filename or doc_id}.pdf"'},
    )
```

`Path.resolve().relative_to()` — стандартная защита от path traversal на Python 3.9+. Альтернатива со строковым prefix check ломается на симлинках и unicode normalization.

### 5.5 Auth поведение

- Endpoint в `protected` роутере → подчиняется существующему `kb_auth.require_api_key`
- Если `KB_API_KEY` не задан → endpoint открыт (как остальные mutating endpoints)
- Если задан → требует `X-API-Key` header; UI делает это автоматически через `kbAuth.fetch`

### 5.6 Error handling

| Кейс | HTTP | Detail | UI поведение |
|---|---|---|---|
| `doc_id` не существует | 404 | `DOCUMENT_NOT_FOUND` | Modal: "Документ не найден" |
| Документ есть, `has_original_file=false` | 404 | `NO_ORIGINAL_FILE` | UI не должен дёргать endpoint — кнопка viewer не показана |
| Документ есть, файл на диске удалён | 410 | `FILE_DELETED` | Modal: "Оригинал недоступен" + text snippet fallback |
| Path traversal в `file_relpath` (DB corruption) | 500 | `STORAGE_ERROR` | Generic error, log warning |
| Нет ключа при включённом auth | 401 | `API_KEY_REQUIRED` | UI просит сохранить ключ |

## 6. Frontend

### 6.1 PDF.js vendoring

Используем PDF.js от Mozilla (Apache-2.0), последний stable v4.x (на момент написания v4.10.38), **legacy build** для совместимости со старыми корпоративными браузерами.

Размещение:

```
data/www/vendor/pdfjs/
├── build/
│   ├── pdf.mjs              (core library)
│   ├── pdf.worker.mjs       (web worker)
└── LICENSE                  (Apache-2.0)
```

Размер: ~2.5MB после gzip. Грузится **лениво при первом клике на цитату**, не входит в bundle главной страницы.

Loader:

```javascript
async function loadPdfJs() {
  if (window._pdfjsLib) return window._pdfjsLib;
  const lib = await import("/vendor/pdfjs/build/pdf.mjs");
  lib.GlobalWorkerOptions.workerSrc = "/vendor/pdfjs/build/pdf.worker.mjs";
  window._pdfjsLib = lib;
  return lib;
}
```

### 6.2 Citation rendering

Новый рендер каждой цитаты — кнопка с data-атрибутами:

```html
<button class="kb-citation"
        data-doc-id="17"
        data-page="12"
        data-snippet="Сотрудник имеет право на 28 календарных дней..."
        data-has-original="true">
  <span class="kb-citation-icon" aria-hidden="true">PDF</span>
  <span class="kb-citation-text">регламент.pdf, стр. 12</span>
  <span class="kb-citation-chevron" aria-hidden="true">&rsaquo;</span>
</button>
```

Поведение по `has_original`:
- `true` + `.pdf`: кнопка с chevron, клик → modal PDF viewer
- `false` или не-PDF: кнопка без chevron, клик → modal с text snippet only

Текстовая форма:
- С page: `{filename}, стр. {page}` → "регламент.pdf, стр. 12"
- Без page (legacy/non-PDF): `{filename}` → "регламент.pdf"
- Без filename (text doc): `{document_title}` → "Регламент отпусков"

### 6.3 Modal viewer структура

Один `<dialog>` элемент (native HTML5, без polyfill):

```html
<dialog id="kb-pdf-modal" class="kb-modal">
  <header class="kb-modal-header">
    <h2 class="kb-modal-title" data-i18n="modal.viewer_title">Просмотр документа</h2>
    <button class="kb-modal-close"
            data-i18n-attr="aria-label" data-i18n="action.close">&times;</button>
  </header>
  <div class="kb-modal-toolbar">
    <span class="kb-modal-filename"></span>
    <span class="kb-modal-pages">
      <span data-i18n="viewer.page">Стр.</span>
      <input type="number" class="kb-modal-page-input" min="1" />
      / <span class="kb-modal-page-total">&mdash;</span>
    </span>
    <button class="kb-modal-prev" data-i18n="viewer.prev">&larr;</button>
    <button class="kb-modal-next" data-i18n="viewer.next">&rarr;</button>
  </div>
  <div class="kb-modal-body">
    <div class="kb-modal-loading" data-i18n="status.loading">Загрузка...</div>
    <div class="kb-modal-error" hidden></div>
    <div class="kb-modal-canvas-host"></div>
    <div class="kb-modal-text-fallback" hidden></div>
  </div>
</dialog>
```

Native `<dialog>` сам управляет focus trap, Esc-close, backdrop click, aria-modal.

### 6.4 `pdf-viewer.js` контроллер

Новый файл `data/www/js/pdf-viewer.js`. Auth-coordination: текущий `data/www/index.html` имеет inline-скрипт с `getApiKey()`, `withAuthHeaders()`, `rawApi()` (см. `index.html:394-416`). Чтобы `pdf-viewer.js` мог переиспользовать ту же логику без дублирования, выносим auth-helpers в отдельный модуль `data/www/js/kb-auth.js`, экспортирующий:

```javascript
window.kbAuth = {
  getApiKey() { /* ...из localStorage по ключу "kb_mvp_api_key" */ },
  withAuthHeaders(headers) { /* добавляет X-API-Key если ключ есть */ },
  fetch(path, opts) { /* fetch(`/api/kb${path}`, {...opts, headers: withAuthHeaders(...)}) */ },
};
```

В `index.html` inline-скрипт переключается на `window.kbAuth.fetch` вместо собственного `rawApi` (минимальный рефактор, ~15 строк замены). Тесты `test_kb_mvp_*.py` не трогаются — это исключительно frontend.

`pdf-viewer.js` API:

```javascript
window.kbPdfViewer = {
  async openCitation({ docId, page, snippet, hasOriginal, filename, fallbackTitle }) {
    const modal = document.getElementById("kb-pdf-modal");
    const titleEl = modal.querySelector(".kb-modal-filename");
    titleEl.textContent = filename || fallbackTitle || `Документ #${docId}`;
    
    modal.showModal();
    
    if (!hasOriginal) {
      this._renderTextFallback(snippet, t("viewer.fallback.text_only"));
      return;
    }
    
    try {
      const pdfBytes = await this._fetchPdfBlob(docId);
      const pdfjs = await loadPdfJs();
      const doc = await pdfjs.getDocument({ data: pdfBytes }).promise;
      await this._renderPage(doc, page || 1, snippet);
    } catch (err) {
      console.error("PDF viewer error", err);
      this._renderTextFallback(snippet, t("viewer.error.load_failed") + ": " + err.message);
    }
  },
  
  async _fetchPdfBlob(docId) {
    const resp = await window.kbAuth.fetch(`/documents/${docId}/file`);
    if (resp.status === 410) throw new Error(t("viewer.error.file_deleted"));
    if (resp.status === 404) throw new Error(t("viewer.error.not_available"));
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.arrayBuffer();
  },
  
  async _renderPage(pdfDoc, pageNum, snippet) { /* PDF.js canvas render */ },
  _renderTextFallback(snippet, reason) { /* показывает .kb-modal-text-fallback */ },
  _triggerFind(snippet) { /* dispatch find event с первыми ~30 словами */ },
};
```

### 6.5 Text-search highlight

После рендера страницы:

```javascript
const findController = new pdfjs.PDFFindController({
  linkService: this._linkService,
  eventBus: this._eventBus,
});
findController.setDocument(pdfDoc);

const query = snippet.split(/\s+/).slice(0, 30).join(" ");

this._eventBus.dispatch("find", {
  query: query,
  caseSensitive: false,
  phraseSearch: true,
  highlightAll: true,
  findPrevious: false,
});
```

`phraseSearch: true` обязателен — без него PDF.js токенизирует query и подсвечивает случайные слова на странице.

### 6.6 i18n-keys

Добавляются в `data/www/i18n/ru.json`:

```json
{
  "citation.with_page": "{filename}, стр. {page}",
  "citation.no_page": "{filename}",
  "citation.text_doc": "{title}",
  "modal.viewer_title": "Просмотр документа",
  "action.close": "Закрыть",
  "viewer.page": "Стр.",
  "viewer.prev": "Предыдущая",
  "viewer.next": "Следующая",
  "viewer.error.not_available": "Оригинал документа недоступен",
  "viewer.error.file_deleted": "Файл удалён на сервере",
  "viewer.error.load_failed": "Не удалось открыть PDF",
  "viewer.fallback.text_only": "Показан только текст фрагмента — оригинальный документ недоступен"
}
```

Существующий `_loader.js` поддерживает `data-i18n` + `data-i18n-attr`. Interpolation `{filename}` / `{page}` нужно добавить в loader — мелкое расширение (~10 строк JS).

### 6.7 Edge cases UX

| Сценарий | Поведение |
|---|---|
| Esc / клик вне диалога | Modal закрывается, focus возвращается на кнопку цитаты |
| PDF >50MB | Прогресс-бар через `pdfjs.getDocument().onProgress` |
| Скан PDF без text layer | PDF рендерится, find ничего не находит → баннер "Не удалось подсветить фрагмент, прокрутите вручную" |
| Цитата на стр.12, в PDF 8 страниц | Открываем последнюю, баннер "Указанная страница 12 за пределами документа" |
| Кэш blob'ов | После закрытия modal `URL.revokeObjectURL` + сброс из памяти; нет global cache |

## 7. Testing strategy

### 7.1 Backend unit tests

| Файл | Что проверяет |
|---|---|
| `tests/test_migration_pdf_citation.py` | Alembic upgrade добавляет колонки, downgrade удаляет; существующие строки получают `has_original_file=false` |
| `tests/test_kb_store_pages.py` | `add_document(pages=[(1, "..."), (2, "...")])` сохраняет per-page chunks; `SearchHit` имеет `page`; legacy text path работает |
| `tests/test_kb_mvp_upload_blob.py` | PDF upload сохраняет blob в `var/data/kb_files/<id>.pdf`; не-PDF не сохраняет blob; orphan tmp-файл удаляется на ошибке парсера |
| `tests/test_kb_mvp_file_endpoint.py` | `GET /documents/<id>/file` возвращает 200 для PDF, 404 для no-original, 410 для удалённого файла, 500 на path traversal, 401 без ключа (когда auth включён) |
| `tests/test_kb_mvp_search_response.py` | `/search` и `/ask` возвращают `page` и `has_original` в каждом source; legacy chunks дают `page=null, has_original=false` |

### 7.2 Frontend smoke tests

| Файл | Что проверяет |
|---|---|
| `tests/test_pdf_viewer_js.py` | Python-тест читает `pdf-viewer.js`, проверяет наличие `kbPdfViewer.openCitation`, корректный fetch URL, обработку 404/410 |
| `tests/test_citation_rendering.py` | Проверяет, что `data/www/index.html` имеет template для citation button (через regex/grep) и не содержит хардкода для legacy `<a href>` |
| `tests/test_i18n_loader.py` (extends existing) | `ru.json` имеет все новые ключи (`citation.*`, `modal.*`, `viewer.*`) |

### 7.3 Что НЕ тестируем

- Реальный PDF.js рендеринг — это код Mozilla, доверяем
- Реальная подсветка через find API — same reason
- E2E через Playwright/Cypress — не оправдывает 5-10 минут на CI прогон для текущей стадии

### 7.4 Manual smoke checklist

Часть финального шага плана:

1. Загрузить тестовый PDF (5-10 страниц с разной структурой — таблицы, headings, обычный текст)
2. Задать вопрос, чьё ожидаемое цитирование на стр. 5
3. Кликнуть цитату → modal открывается → PDF на стр. 5 → snippet подсвечен
4. Esc закрывает modal, focus на кнопке цитаты
5. Удалить файл `var/data/kb_files/<id>.pdf` через FS → клик показывает "Файл удалён"
6. Включить `KB_API_KEY`, перезалить, проверить что modal работает (kbAuth добавляет header)
7. Открыть DevTools, network tab — убедиться что PDF.js bundle грузится лениво при первом клике, а не на initial paint

## 8. Build phasing

4 коммита, каждый ревёртабельный:

| Phase | Что | DoD | ~часов |
|---|---|---|---|
| **1** | Schema + migration + `SearchHit.page` propagation | `test_migration_pdf_citation` зелёный; `test_kb_store_pages` зелёный; legacy `/documents` POST path работает; существующий `test_kb_mvp` зелёный | 4 |
| **2** | Blob upload + file endpoint | `test_kb_mvp_upload_blob` + `test_kb_mvp_file_endpoint` зелёные; manual curl test PDF download | 6 |
| **3** | Frontend: kb-auth.js extract + PDF.js vendor + citation button + modal + viewer controller | `test_pdf_viewer_js` + `test_citation_rendering` + i18n test зелёные; `index.html` использует `window.kbAuth.fetch` вместо inline `rawApi`; manual smoke в браузере (`scripts/dev_server_mvp:app`) | 8 |
| **4** | Polish: error states, edge cases (out-of-range page, scan без text-layer), focus management, prefers-color-scheme dark mode | Manual smoke checklist 1-7 пройден; visually OK в темной и светлой теме | 2 |

**Total: ~20 часов** = соответствует vision-doc estimate.

Backend без frontend (phase 1+2 без UI) — валидный intermediate state. API расширен, старый UI не использует новые поля. Можно задеплоить phase 1+2 в проде и работать со старым UI, ждать phase 3.

## 9. Risks

| Риск | Митигация |
|------|-----------|
| PDF.js не работает в старом Edge у пилот-клиента | Используем legacy build; manual smoke на Chrome 120+, Firefox 115+, Safari 17+. Документируем "минимальные браузеры" в README |
| Скан-PDF без text-layer — подсветка не работает | Очевидное поведение: PDF открыт на странице, баннер "text-search недоступен". Long-term: включить Docling OCR при ingestion (Phase 1 extension) |
| Очень большие PDF (>100MB) ломают memory | `pdfjs.getDocument()` использует worker, не main thread. Progress bar. Реальный risk низкий для корп-доков (<50MB norm) |
| Path traversal в `file_relpath` (DB corruption или ручное editing SQLite) | Защита через `Path.resolve().relative_to()`, unit-test покрывает попытку `../` |
| PDF в file storage и SQLite расходятся (FS crash после INSERT) | Acceptable: orphan blob — мусор; запись без файла — 410 GONE с понятным сообщением. Cross-FS atomic не оправдывает MVP |
| PDF.js bundle ломает CSP | Если в `nginx.conf` задан Content-Security-Policy, нужно `worker-src 'self'` и `script-src 'self'`. Текущий `data/nginx.conf` CSP не задаёт — в текущей итерации не блокер. Документируем |
| Storage растёт неограниченно | `kb-cli backup` (задача 1.4 из vision) делает tar.gz всего `var/data/`. Cleanup orphan-блобов — отдельный job, в этой spec не делается |

## 10. Open questions

Нет открытых вопросов на момент финального дизайна. Все архитектурные выборы зафиксированы предыдущими ответами в брейншторме (формат: PDF-only; подсветка: page + PDF.js find; UX: modal overlay; retrofit: clean break).

## 11. Document lifecycle

- Spec живёт в `docs/superpowers/specs/2026-05-22-pdf-citation-viewer-design.md`
- После реализации добавить в spec секцию "Validation results" с реальными метриками: время рендера на тестовом корпусе, размер blobs пилот-корпуса
- Если пилот скажет "highlight кривой" — апгрейд до bbox-overlay (Подход C из брейншторма); схема forward-compatible, миграция добавит `bbox_json` колонку без потери данных
- Если пилот скажет "нужен DOCX viewer" — отдельный sprint с docx-preview / mammoth.js (Подход B из брейншторма про форматы)
