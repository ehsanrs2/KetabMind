import { useEffect } from 'react';
import { isRTLLanguage } from '../i18n';

export function useRTLSync(locale) {
  useEffect(() => {
    if (typeof document === 'undefined') {
      return;
    }
    document.documentElement.dir = isRTLLanguage(locale) ? 'rtl' : 'ltr';
  }, [locale]);
}
