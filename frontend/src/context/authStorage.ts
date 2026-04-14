const STORAGE_KEY = 'operations-console.session';

type StoredSession = Record<string, unknown>;

type TokenPair = {
  accessToken: string | null;
  refreshToken: string | null;
};

let unauthorizedHandler: (() => void) | null = null;

const readSession = (): StoredSession | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const rawValue = window.localStorage.getItem(STORAGE_KEY);
    return rawValue ? (JSON.parse(rawValue) as StoredSession) : null;
  } catch (error) {
    console.error('Failed to parse session from storage', error);
    return null;
  }
};

export const getStoredSession = <T>() => readSession() as T | null;

export const setStoredSession = (value: StoredSession) => {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
};

export const clearStoredSession = () => {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.removeItem(STORAGE_KEY);
};

export const getStoredTokens = (): TokenPair => {
  const session = readSession();
  return {
    accessToken: typeof session?.access_token === 'string' ? session.access_token : null,
    refreshToken: typeof session?.refresh_token === 'string' ? session.refresh_token : null
  };
};

export const updateStoredTokens = (tokens: { access_token: string; refresh_token: string }) => {
  const currentSession = readSession() ?? {};
  setStoredSession({
    ...currentSession,
    access_token: tokens.access_token,
    refresh_token: tokens.refresh_token
  });
};

export const registerUnauthorizedHandler = (handler: () => void) => {
  unauthorizedHandler = handler;
  return () => {
    if (unauthorizedHandler === handler) {
      unauthorizedHandler = null;
    }
  };
};

export const forceLogout = () => {
  clearStoredSession();
  unauthorizedHandler?.();

  if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
    window.history.pushState({}, '', '/login');
  }
};
