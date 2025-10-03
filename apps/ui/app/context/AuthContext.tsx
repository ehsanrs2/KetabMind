'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

type User = {
  id: string;
  name: string;
  email?: string;
  language?: string;
};

type AuthContextType = {
  user: User | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  logout: () => Promise<void>;
};

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

async function fetchMe() {
  const response = await fetch('/api/me', {
    credentials: 'include',
  });

  if (!response.ok) {
    if (response.status === 401) {
      return null;
    }
    throw new Error('Failed to load user profile');
  }

  return (await response.json()) as User;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const applyResult = useCallback((nextUser: User | null, nextError: string | null) => {
    if (!mountedRef.current) {
      return;
    }
    setUser(nextUser);
    setError(nextError);
  }, []);

  const loadUser = useCallback(async () => {
    try {
      const data = await fetchMe();
      applyResult(data, null);
    } catch (err) {
      applyResult(null, err instanceof Error ? err.message : 'Unknown error');
    }
  }, [applyResult]);

  useEffect(() => {
    (async () => {
      setLoading(true);
      await loadUser();
      if (mountedRef.current) {
        setLoading(false);
      }
    })();
  }, [loadUser]);

  const refresh = useCallback(async () => {
    setLoading(true);
    await loadUser();
    if (mountedRef.current) {
      setLoading(false);
    }
  }, [loadUser]);

  const logout = useCallback(async () => {
    try {
      await fetch('/api/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } catch (err) {
      console.warn('Logout failed', err);
    } finally {
      if (mountedRef.current) {
        setUser(null);
      }
    }
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      error,
      refresh,
      logout,
    }),
    [error, loading, logout, refresh, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }

  return context;
}
