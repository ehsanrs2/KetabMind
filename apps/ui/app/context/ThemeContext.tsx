'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';

type Theme = 'light' | 'dark';

type ThemeContextValue = {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
};


const STORAGE_KEY = 'ketabmind.theme';

function getEnvDefaultTheme(): Theme {
  const raw =
    (process.env.NEXT_PUBLIC_THEME_DEFAULT ?? process.env.THEME_DEFAULT ?? 'light').toString();
  return normaliseTheme(raw) ?? 'light';
}

function normaliseTheme(value: unknown): Theme | null {
  if (typeof value !== 'string') {
    return null;
  }
  const lower = value.toLowerCase();
  return lower === 'dark' || lower === 'light' ? (lower as Theme) : null;
}

function readStoredTheme(): Theme | null {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return normaliseTheme(stored);
  } catch (error) {
    console.warn('Unable to read stored theme preference', error);
    return null;
  }
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export function ThemeProvider({
  children,
  initialTheme,
}: {
  children: React.ReactNode;
  initialTheme?: Theme;
}) {
  const defaultTheme = useMemo(() => initialTheme ?? getEnvDefaultTheme(), [initialTheme]);
  const [theme, setTheme] = useState<Theme>(() => {
    if (initialTheme) {
      return initialTheme;
    }
    return readStoredTheme() ?? defaultTheme;
  });

  useEffect(() => {
    if (initialTheme) {
      setTheme(initialTheme);
    }
  }, [initialTheme]);

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.body.dataset.theme = theme;
    }
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem(STORAGE_KEY, theme);
      } catch (error) {
        console.warn('Unable to persist theme preference', error);
      }
    }
  }, [theme]);

  useEffect(() => {
    if (typeof window === 'undefined' || initialTheme) {
      return;
    }
    const handleStorage = (event: StorageEvent) => {
      if (event.key !== STORAGE_KEY) {
        return;
      }
      const nextTheme = normaliseTheme(event.newValue);
      if (nextTheme) {
        setTheme((current) => (current === nextTheme ? current : nextTheme));
      }
    };
    const stored = readStoredTheme();
    if (stored) {
      setTheme((current) => (current === stored ? current : stored));
    }
    window.addEventListener('storage', handleStorage);
    return () => {
      window.removeEventListener('storage', handleStorage);
    };
  }, [initialTheme]);

  const toggleTheme = useCallback(() => {
    setTheme((current) => (current === 'light' ? 'dark' : 'light'));
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({
      theme,
      setTheme,
      toggleTheme,
    }),
    [theme, toggleTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

