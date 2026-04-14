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

/**
 * AuthContext encapsulates session management and role-based access control.
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

type AuthContextValue = {
  session: Session | null;
  isAuthenticated: boolean;
  hasRole: (role: Role) => boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);


export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [session, setSession] = useState<Session | null>(() => getStoredSession<Session>());

  useEffect(() => {
    if (!session) {
      clearStoredSession();
      return;
    }

    setStoredSession(session);
  }, [session]);

  useEffect(() => {
    return registerUnauthorizedHandler(() => {
      setSession(null);
    });
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const response = await apiClient.post<Session>('/auth/login', { email, password });
    setSession(response.data);
  }, []);

  const logout = useCallback(() => {
    const { refreshToken } = getStoredTokens();

    clearStoredSession();
    setSession(null);

    if (!refreshToken) {
      return;
    }

    void apiClient.post('/auth/logout', { refresh_token: refreshToken }).catch(() => undefined);
  }, []);

  const hasRole = useCallback((role: Role) => session?.roles.includes(role) ?? false, [session]);

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
