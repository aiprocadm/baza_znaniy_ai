/* Minimal i18n loader for KB.AI UI.
 *
 * Usage in HTML:
 *   <span data-i18n="action.upload">Загрузить</span>
 *
 * The fallback content (between the tags) is the Russian default,
 * shown if i18n loading fails. When _loader.js runs, it fetches
 * /i18n/{lang}.json and replaces textContent for each [data-i18n] element.
 *
 * Default language: ru. Can be overridden by setting localStorage.kbLang
 * to a supported language code before page load.
 */
(function () {
  "use strict";

  const DEFAULT_LANG = "ru";
  const SUPPORTED = ["ru"];

  function pickLang() {
    const stored = localStorage.getItem("kbLang");
    if (stored && SUPPORTED.includes(stored)) return stored;
    return DEFAULT_LANG;
  }

  async function loadDict(lang) {
    try {
      const resp = await fetch(`/i18n/${lang}.json`, { cache: "no-cache" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();
    } catch (err) {
      console.warn("i18n load failed, falling back to inline text:", err);
      return null;
    }
  }

  function applyDict(dict) {
    if (!dict) return;
    const nodes = document.querySelectorAll("[data-i18n]");
    nodes.forEach((node) => {
      const key = node.getAttribute("data-i18n");
      if (key && Object.prototype.hasOwnProperty.call(dict, key)) {
        const attr = node.getAttribute("data-i18n-attr");
        if (attr) {
          node.setAttribute(attr, dict[key]);
        } else {
          node.textContent = dict[key];
        }
      }
    });
  }

  window.t = function (key, fallback) {
    if (window._kbDict && Object.prototype.hasOwnProperty.call(window._kbDict, key)) {
      return window._kbDict[key];
    }
    return fallback || key;
  };

  document.addEventListener("DOMContentLoaded", async () => {
    const lang = pickLang();
    const dict = await loadDict(lang);
    window._kbDict = dict;
    applyDict(dict);
  });
})();
