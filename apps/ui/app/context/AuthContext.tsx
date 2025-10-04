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
  csrfToken: string | null;
};

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

const CSRF_STORAGE_KEY = 'ketabmind.csrf-token';

function readStoredCsrfToken(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return sessionStorage.getItem(CSRF_STORAGE_KEY);
  } catch (error) {
    console.warn('Unable to read stored CSRF token', error);
    return null;
  }
}

function persistCsrfToken(value: string | null) {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    if (value) {
      sessionStorage.setItem(CSRF_STORAGE_KEY, value);
    } else {
      sessionStorage.removeItem(CSRF_STORAGE_KEY);
    }
  } catch (error) {
    console.warn('Unable to persist CSRF token', error);
  }
}

async function fetchMe() {
  const response = await fetch('/api/me', {
    credentials: 'include',
  });

  const token = response.headers.get('x-csrf-token');

  if (!response.ok) {
    if (response.status === 401) {
      return { user: null, token: null };
    }
    throw new Error('Failed to load user profile');
  }

  const user = (await response.json()) as User;
  return { user, token: token ?? null };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [csrfToken, setCsrfToken] = useState<string | null>(() => readStoredCsrfToken());
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const applyResult = useCallback(
    (nextUser: User | null, nextError: string | null, nextToken?: string | null) => {
      if (!mountedRef.current) {
        return;
      }
      setUser(nextUser);
      setError(nextError);
      if (nextToken !== undefined) {
        setCsrfToken(nextToken);
        persistCsrfToken(nextToken);
      }
    },
    [],
  );

  const loadUser = useCallback(async () => {
    if (!mountedRef.current) {
      return;
    }
    try {
      const { user: nextUser, token } = await fetchMe();
      applyResult(nextUser, null, token);
    } catch (err) {
      applyResult(null, err instanceof Error ? err.message : 'Unknown error', null);
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
        headers: csrfToken ? { 'x-csrf-token': csrfToken } : undefined,
      });
    } catch (err) {
      console.warn('Logout failed', err);
    } finally {
      applyResult(null, null, null);
    }
  }, [applyResult, csrfToken]);

  const value = useMemo(
    () => ({
      user,
      loading,
      error,
      refresh,
      logout,
      csrfToken,
    }),
    [csrfToken, error, loading, logout, refresh, user],
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
