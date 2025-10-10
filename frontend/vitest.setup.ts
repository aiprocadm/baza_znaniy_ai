import '@testing-library/jest-dom/vitest';

window.matchMedia =
  window.matchMedia ||
  function matchMedia(query: string): MediaQueryList {
    return {
      matches: false,
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false
    } as unknown as MediaQueryList;
  };

if (!('ResizeObserver' in globalThis)) {
  class ResizeObserverStub {
    observe() {
      return undefined;
    }
    unobserve() {
      return undefined;
    }
    disconnect() {
      return undefined;
    }
  }
  // @ts-expect-error polyfill for tests
  globalThis.ResizeObserver = ResizeObserverStub;
}
