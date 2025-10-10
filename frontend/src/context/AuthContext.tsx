import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { apiClient } from '../api/client';

/**
 * AuthContext encapsulates session management and role-based access control.
 */
export type Role = 'user' | 'admin';

export type Session = {
  id: string;
  email: string;
  name: string;
  roles: Role[];
  token: string;
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
    const response = await apiClient.post<Session>('/auth/login', { email, password });
    setSession(response.data);
  }, []);

  const logout = useCallback(() => {
    setSession(null);
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
