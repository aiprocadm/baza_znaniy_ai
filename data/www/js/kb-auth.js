/* Shared auth helpers for the MVP UI.
 *
 * Exposes:
 *   window.kbAuth.getApiKey()       — read API key from localStorage
 *   window.kbAuth.withAuthHeaders(h) — clone h and add X-API-Key when set
 *   window.kbAuth.fetch(path, opts) — fetch `/api/kb${path}` with auth headers
 *   window.kbAuth.setApiKey(value)  — persist or clear the API key
 *
 * The previous implementation was inline in index.html. This file is
 * also consumed by pdf-viewer.js for the original-blob endpoint.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "kb_mvp_api_key";
  const API_BASE = "/api/kb";

  function getApiKey() {
    try {
      return localStorage.getItem(STORAGE_KEY) || "";
    } catch (_) {
      return "";
    }
  }

  function setApiKey(value) {
    try {
      if (value) localStorage.setItem(STORAGE_KEY, value);
      else localStorage.removeItem(STORAGE_KEY);
    } catch (_) {
      /* private mode — silently degrade */
    }
  }

  function withAuthHeaders(headers) {
    const out = Object.assign({}, headers || {});
    const key = getApiKey();
    if (key) out["X-API-Key"] = key;
    return out;
  }

  function authFetch(path, opts) {
    const options = Object.assign({}, opts || {});
    options.headers = withAuthHeaders(options.headers);
    return fetch(API_BASE + path, options);
  }

  window.kbAuth = {
    storageKey: STORAGE_KEY,
    apiBase: API_BASE,
    getApiKey: getApiKey,
    setApiKey: setApiKey,
    withAuthHeaders: withAuthHeaders,
    fetch: authFetch,
  };
})();
