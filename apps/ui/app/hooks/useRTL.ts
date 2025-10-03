'use client';

import { useEffect } from 'react';

const RTL_LANGS = new Set(['ar', 'fa', 'he', 'ur', 'ps']);

export function useRTL(languageCode?: string | null) {
  useEffect(() => {
    if (typeof document === 'undefined') {
      return;
    }

    const isRTL = Boolean(languageCode && RTL_LANGS.has(languageCode.toLowerCase()));
    const dir = isRTL ? 'rtl' : 'ltr';

    document.body.setAttribute('dir', dir);
    const root = document.documentElement;
    if (root) {
      root.setAttribute('dir', dir);
    }

    document.body.classList.toggle('rtl', isRTL);

    return () => {
      if (typeof document === 'undefined') {
        return;
      }
      document.body.classList.remove('rtl');
      document.body.removeAttribute('dir');
      document.documentElement?.removeAttribute('dir');
    };
  }, [languageCode]);
}
