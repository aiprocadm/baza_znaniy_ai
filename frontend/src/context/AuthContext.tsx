import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { apiClient } from '../api/client';
import {
  clearStoredSession,
  getStoredSession,
  getStoredTokens,
  registerUnauthorizedHandler,
  setStoredSession
} from './authStorage';
import { fetchAuthSession, type AuthSession, type Role, type TokenResponse } from '../api';

/**
 * AuthContext encapsulates token management and role-based access control.
 */
export type Role = 'user' | 'admin';

export type Session = {
  user_id: string;
  email: string;
  name: string;
  roles: Role[];
  token_expires_at: string;
  access_token?: string;
  refresh_token?: string;
};

type StoredTokens = TokenResponse & {
  issued_at: number;
};

type AuthContextValue = {
  tokens: StoredTokens | null;
  user: AuthSession | null;
  isAuthenticated: boolean;
  hasRole: (role: Role) => boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
};

type StoredAuthState = {
  tokens: StoredTokens;
  user: AuthSession | null;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const STORAGE_KEY = 'operations-console.auth';

const decodeJwtPayload = (token: string): Record<string, unknown> | null => {
  const payload = token.split('.')[1];
  if (!payload) {
    return null;
  }

  try {
    const base64 = payload.replace(/-/g, '+').replace(/_/g, '/');
    const normalized = `${base64}${'='.repeat((4 - (base64.length % 4)) % 4)}`;
    const json = window.atob(normalized);
    return JSON.parse(json) as Record<string, unknown>;
  } catch (error) {
    console.error('Failed to decode JWT payload', error);
    return null;
  }
};

const isRole = (value: unknown): value is Role => value === 'user' || value === 'admin';

const mapRoles = (claims: Record<string, unknown>): Role[] => {
  if (Array.isArray(claims.roles)) {
    return claims.roles.filter(isRole);
  }

  if (isRole(claims.role)) {
    return [claims.role];
  }

  return [];
};

const userFromClaims = (claims: Record<string, unknown>): AuthSession => ({
  user_id: typeof claims.sub === 'string' ? claims.sub : '',
  email: typeof claims.email === 'string' ? claims.email : undefined,
  name: typeof claims.name === 'string' ? claims.name : undefined,
  roles: mapRoles(claims)
});

const isTokenExpired = (tokens: StoredTokens): boolean =>
  Date.now() >= tokens.issued_at + tokens.expires_in * 1000;

const getStoredAuthState = (): StoredAuthState | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    if (!value) {
      return null;
    }

    const parsed = JSON.parse(value) as StoredAuthState;
    if (!parsed.tokens || isTokenExpired(parsed.tokens)) {
      window.localStorage.removeItem(STORAGE_KEY);
      return null;
    }

    return parsed;
  } catch (error) {
    console.error('Failed to parse auth state from storage', error);
    return null;
  }
};

const buildTokens = (tokenResponse: TokenResponse): StoredTokens => ({
  ...tokenResponse,
  token_type: tokenResponse.token_type ?? 'bearer',
  issued_at: Date.now()
});

const resolveUserSession = async (accessToken: string): Promise<AuthSession | null> => {
  try {
    const response = await fetchAuthSession();
    return response.data;
  } catch {
    const claims = decodeJwtPayload(accessToken);
    return claims ? userFromClaims(claims) : null;
  }
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [session, setSession] = useState<Session | null>(() => getStoredSession<Session>());

  useEffect(() => {
    if (!session) {
      clearStoredSession();
      return;
    }

    setStoredSession(session);
  }, [session]);
  const [authState, setAuthState] = useState<StoredAuthState | null>(getStoredAuthState);

  useEffect(() => {
    const token = authState?.tokens.access_token;
    if (token) {
      apiClient.defaults.headers.common.Authorization = `Bearer ${token}`;
    } else {
      delete apiClient.defaults.headers.common.Authorization;
    }
  }, [authState?.tokens.access_token]);

  useEffect(() => {
    if (!authState) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(authState));
  }, [authState]);

  useEffect(() => {
    return registerUnauthorizedHandler(() => {
      setSession(null);
    });
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const response = await apiClient.post<TokenResponse>('/auth/login', { email, password });
    const tokens = buildTokens(response.data);
    const user = await resolveUserSession(tokens.access_token);
    setAuthState({ tokens, user });
  }, []);

  const logout = useCallback(() => {
    const { refreshToken } = getStoredTokens();

    clearStoredSession();
    setSession(null);

    if (!refreshToken) {
      return;
    }

    void apiClient.post('/auth/logout', { refresh_token: refreshToken }).catch(() => undefined);
    setAuthState(null);
  }, []);

  const hasRole = useCallback((role: Role) => authState?.user?.roles.includes(role) ?? false, [authState]);

  const isAuthenticated = useMemo(() => {
    if (!authState) {
      return false;
    }
    return !isTokenExpired(authState.tokens);
  }, [authState]);

  const value = useMemo(
    () => ({
      tokens: authState?.tokens ?? null,
      user: authState?.user ?? null,
      isAuthenticated,
      hasRole,
      login,
      logout
    }),
    [authState, isAuthenticated, hasRole, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
