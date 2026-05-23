/* PDF citation viewer controller.
 *
 * Exposes window.kbPdfViewer.openCitation({docId, page, snippet,
 * hasOriginal, filename, fallbackTitle}). Lazy-imports PDF.js from
 * /vendor/pdfjs on first use. Uses window.kbAuth.fetch for the
 * /api/kb/documents/{id}/file blob request.
 *
 * Highlight strategy: PDF.js text layer + find API with phraseSearch
 * on the first ~30 words of the chunk snippet. Falls back to a plain
 * text view in the modal when the document has no original or the PDF
 * has no text layer.
 */
(function () {
  "use strict";

  const PDFJS_URL = "/vendor/pdfjs/build/pdf.mjs";
  const WORKER_URL = "/vendor/pdfjs/build/pdf.worker.mjs";
  let _pdfjsLib = null;
  let _state = null;

  async function loadPdfJs() {
    if (_pdfjsLib) return _pdfjsLib;
    const lib = await import(PDFJS_URL);
    lib.GlobalWorkerOptions.workerSrc = WORKER_URL;
    _pdfjsLib = lib;
    return lib;
  }

  function modalEls() {
    const modal = document.getElementById("kb-pdf-modal");
    return {
      modal,
      filename: modal.querySelector(".kb-modal-filename"),
      pageInput: modal.querySelector(".kb-modal-page-input"),
      pageTotal: modal.querySelector(".kb-modal-page-total"),
      prevBtn: modal.querySelector(".kb-modal-prev"),
      nextBtn: modal.querySelector(".kb-modal-next"),
      closeBtn: modal.querySelector(".kb-modal-close"),
      loading: modal.querySelector(".kb-modal-loading"),
      error: modal.querySelector(".kb-modal-error"),
      canvasHost: modal.querySelector(".kb-modal-canvas-host"),
      textFallback: modal.querySelector(".kb-modal-text-fallback"),
    };
  }

  function tr(key, fallback, vars) {
    if (typeof window.t === "function") return window.t(key, fallback, vars);
    return fallback || key;
  }

  function reset(els) {
    els.loading.hidden = false;
    els.error.hidden = true;
    els.error.textContent = "";
    els.canvasHost.innerHTML = "";
    els.textFallback.hidden = true;
    els.textFallback.innerHTML = "";
    els.pageInput.value = "1";
    els.pageTotal.textContent = "—";
  }

  function showError(els, message) {
    els.loading.hidden = true;
    els.error.hidden = false;
    els.error.textContent = message;
  }

  function showTextFallback(els, snippet, reason) {
    els.loading.hidden = true;
    els.canvasHost.innerHTML = "";
    els.textFallback.hidden = false;
    const reasonEl = document.createElement("span");
    reasonEl.className = "kb-fallback-reason";
    reasonEl.textContent = reason;
    const snippetEl = document.createElement("div");
    snippetEl.textContent = snippet || "";
    els.textFallback.replaceChildren(reasonEl, snippetEl);
  }

  async function fetchPdfBlob(docId) {
    const resp = await window.kbAuth.fetch(`/documents/${docId}/file`);
    if (resp.status === 410) {
      throw Object.assign(new Error(tr("viewer.error.file_deleted")), { code: 410 });
    }
    if (resp.status === 404) {
      throw Object.assign(new Error(tr("viewer.error.not_available")), { code: 404 });
    }
    if (resp.status === 401) {
      throw Object.assign(new Error("API key required"), { code: 401 });
    }
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    return await resp.arrayBuffer();
  }

  async function renderPage(state, pageNum) {
    const { pdfDoc, els, scale } = state;
    const clamped = Math.max(1, Math.min(pdfDoc.numPages, pageNum | 0));
    state.currentPage = clamped;
    els.pageInput.value = String(clamped);
    els.pageTotal.textContent = String(pdfDoc.numPages);

    const page = await pdfDoc.getPage(clamped);
    const viewport = page.getViewport({ scale });

    const wrap = document.createElement("div");
    wrap.className = "kb-canvas-wrap";

    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    canvas.style.width = viewport.width + "px";
    canvas.style.height = viewport.height + "px";
    wrap.appendChild(canvas);

    const textLayerDiv = document.createElement("div");
    textLayerDiv.className = "kb-text-layer";
    textLayerDiv.style.width = viewport.width + "px";
    textLayerDiv.style.height = viewport.height + "px";
    wrap.appendChild(textLayerDiv);

    els.canvasHost.innerHTML = "";
    els.canvasHost.appendChild(wrap);
    els.loading.hidden = true;

    const ctx = canvas.getContext("2d");
    await page.render({ canvasContext: ctx, viewport: viewport }).promise;

    const textContent = await page.getTextContent();
    const pdfjs = state.pdfjsLib;
    if (pdfjs && pdfjs.renderTextLayer) {
      await pdfjs.renderTextLayer({
        textContentSource: textContent,
        container: textLayerDiv,
        viewport: viewport,
      }).promise;
    }
  }

  function buildSearchQuery(snippet) {
    if (!snippet) return "";
    return String(snippet).split(/\s+/).slice(0, 30).join(" ").trim();
  }

  function triggerFind(state) {
    const { pdfjsLib, pdfDoc, snippet } = state;
    const query = buildSearchQuery(snippet);
    if (!query) return;

    try {
      const eventBus = new pdfjsLib.EventBus();
      const linkService = new pdfjsLib.PDFLinkService({ eventBus: eventBus });
      linkService.setDocument(pdfDoc, null);
      const findController = new pdfjsLib.PDFFindController({
        eventBus: eventBus,
        linkService: linkService,
      });
      findController.setDocument(pdfDoc);
      eventBus.dispatch("find", {
        source: window,
        type: "",
        query: query,
        caseSensitive: false,
        entireWord: false,
        phraseSearch: true,
        highlightAll: true,
        findPrevious: false,
      });
      state.findController = findController;
    } catch (err) {
      console.warn("PDF.js find dispatch failed:", err);
    }
  }

  function wireToolbar(state) {
    const { els } = state;
    els.prevBtn.onclick = () => renderPage(state, state.currentPage - 1).then(() => triggerFind(state));
    els.nextBtn.onclick = () => renderPage(state, state.currentPage + 1).then(() => triggerFind(state));
    els.pageInput.onchange = () => {
      const n = parseInt(els.pageInput.value, 10);
      if (Number.isFinite(n)) renderPage(state, n).then(() => triggerFind(state));
    };
    els.closeBtn.onclick = () => els.modal.close();
  }

  async function openCitation(opts) {
    const { docId, page, snippet, hasOriginal, filename, fallbackTitle } = opts || {};
    const els = modalEls();
    reset(els);
    els.filename.textContent = filename || fallbackTitle || `Документ #${docId}`;
    els.modal.showModal();

    if (!hasOriginal) {
      showTextFallback(els, snippet, tr("viewer.fallback.text_only"));
      return;
    }

    try {
      const pdfBytes = await fetchPdfBlob(docId);
      const pdfjsLib = await loadPdfJs();
      const pdfDoc = await pdfjsLib.getDocument({ data: pdfBytes }).promise;

      let initialPage = page || 1;
      if (initialPage > pdfDoc.numPages) {
        showError(els, tr(
          "viewer.fallback.page_out_of_range",
          "Страница вне диапазона",
          { page: initialPage, total: pdfDoc.numPages },
        ));
        initialPage = pdfDoc.numPages;
      }

      _state = {
        pdfDoc: pdfDoc,
        pdfjsLib: pdfjsLib,
        els: els,
        scale: 1.2,
        snippet: snippet,
        currentPage: initialPage,
        findController: null,
      };
      wireToolbar(_state);
      await renderPage(_state, initialPage);
      triggerFind(_state);
    } catch (err) {
      console.error("PDF viewer error:", err);
      showTextFallback(els, snippet, tr("viewer.error.load_failed") + ": " + (err.message || err));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("kb-pdf-modal");
    if (!modal) return;
    modal.addEventListener("close", () => {
      if (_state && _state.pdfDoc && typeof _state.pdfDoc.destroy === "function") {
        try { _state.pdfDoc.destroy(); } catch (_) { /* ignore */ }
      }
      _state = null;
    });
  });

  window.kbPdfViewer = { openCitation: openCitation };
})();
