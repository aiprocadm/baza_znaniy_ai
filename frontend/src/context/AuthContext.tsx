import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { login as loginRequest, logout as logoutRequest, type TokenResponse } from '../api';

/**
 * AuthContext encapsulates session management and role-based access control.
 */
export type Role = 'admin' | 'manager' | 'member';

export type Session = {
  user_id: number;
  email: string;
  name: string | null;
  role: Role;
  access_token: string;
  refresh_token: string;
  token_expires_at: string;
};

type AuthContextValue = {
  session: Session | null;
  isAuthenticated: boolean;
  hasRole: (role: Role) => boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const STORAGE_KEY = 'operations-console.session';

type JwtPayload = {
  sub?: string;
  role?: Role;
  exp?: number;
};

const decodeJwtPayload = (token: string): JwtPayload => {
  const base64Payload = token.split('.')[1];
  if (!base64Payload) return {};
  const normalized = base64Payload.replace(/-/g, '+').replace(/_/g, '/');
  const decoded = window.atob(normalized);
  return JSON.parse(decoded) as JwtPayload;
};

const toSession = (email: string, tokens: TokenResponse): Session => {
  const payload = decodeJwtPayload(tokens.access_token);
  return {
    user_id: Number(payload.sub ?? 0),
    email,
    name: email,
    role: payload.role ?? 'member',
    access_token: tokens.access_token,
    refresh_token: tokens.refresh_token,
    token_expires_at: new Date((payload.exp ?? 0) * 1000).toISOString()
  };
};

const getStoredSession = (): Session | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value ? (JSON.parse(value) as Session) : null;
  } catch (error) {
    console.error('Failed to parse session from storage', error);
    return null;
  }
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [session, setSession] = useState<Session | null>(getStoredSession);

  useEffect(() => {
    if (!session) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }, [session]);

  const login = useCallback(async (email: string, password: string) => {
    const response = await loginRequest(email, password);
    setSession(toSession(email, response.data));
  }, []);

  const logout = useCallback(() => {
    const refreshToken = session?.refresh_token;
    if (refreshToken) {
      void logoutRequest({ refresh_token: refreshToken }).catch(() => undefined);
    }
    setSession(null);
  }, [session]);

  const hasRole = useCallback((role: Role) => session?.role === role, [session]);

  const value = useMemo(
    () => ({
      session,
      isAuthenticated: Boolean(session),
      hasRole,
      login,
      logout
    }),
    [session, hasRole, login, logout]
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
