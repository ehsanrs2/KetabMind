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

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export function ThemeProvider({
  children,
  initialTheme,
}: {
  children: React.ReactNode;
  initialTheme?: Theme;
}) {
  const defaultTheme = useMemo(() => initialTheme ?? getEnvDefaultTheme(), [initialTheme]);
  const [theme, setTheme] = useState<Theme>(defaultTheme);

  useEffect(() => {
    if (typeof window === 'undefined' || initialTheme) {
      return;
    }
    const stored = window.localStorage.getItem(STORAGE_KEY);
    const resolved = normaliseTheme(stored) ?? defaultTheme;
    setTheme(resolved);
  }, [defaultTheme, initialTheme]);

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.body.dataset.theme = theme;
    }
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, theme);
    }
  }, [theme]);

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

