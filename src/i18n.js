import { createContext, useContext, useEffect, useMemo, useState } from 'react';

const DEFAULT_LOCALE = 'en';
const RTL_LANGS = new Set(['fa', 'ar', 'he', 'ur']);

const translations = {
  en: {
    uploadTitle: 'Upload a document',
    uploadHint: 'Choose a book or document to add to KetabMind.',
    selectFile: 'Select file',
    uploading: 'Uploading…',
    uploadSuccess: 'Upload complete!',
    uploadError: 'Upload failed, please try again.',
    chatTitle: 'Ask your library',
    chatPlaceholder: 'Type your question…',
    send: 'Send',
    assistantLabel: 'Assistant',
    userLabel: 'You',
    citationsHeading: 'Citations',
    emptyResponse: 'Waiting for response…',
    progressLabel: 'Upload progress',
  },
  fa: {
    uploadTitle: 'بارگذاری سند',
    uploadHint: 'کتاب یا سند خود را برای افزودن به کتاب‌مایند انتخاب کنید.',
    selectFile: 'انتخاب فایل',
    uploading: 'در حال بارگذاری…',
    uploadSuccess: 'بارگذاری کامل شد!',
    uploadError: 'بارگذاری انجام نشد، دوباره تلاش کنید.',
    chatTitle: 'از کتابخانهٔ خود بپرسید',
    chatPlaceholder: 'سؤال خود را بنویسید…',
    send: 'ارسال',
    assistantLabel: 'دستیار',
    userLabel: 'شما',
    citationsHeading: 'استنادها',
    emptyResponse: 'در انتظار پاسخ…',
    progressLabel: 'پیشرفت بارگذاری',
  },
};

const I18nContext = createContext({
  locale: DEFAULT_LOCALE,
  direction: 'ltr',
  setLocale: () => {},
  t: (key) => translations[DEFAULT_LOCALE][key] ?? key,
});

export function isRTLLanguage(locale) {
  if (!locale) {
    return false;
  }
  const base = locale.toLowerCase().split('-')[0];
  return RTL_LANGS.has(base);
}

export function I18nProvider({ children, initialLocale }) {
  const detectLocale = () => {
    if (initialLocale) {
      return initialLocale;
    }
    if (typeof navigator !== 'undefined' && navigator.language) {
      return navigator.language.split('-')[0];
    }
    return DEFAULT_LOCALE;
  };

  const [locale, setLocale] = useState(detectLocale);

  useEffect(() => {
    const dir = isRTLLanguage(locale) ? 'rtl' : 'ltr';
    if (typeof document !== 'undefined') {
      document.documentElement.lang = locale;
      document.documentElement.dir = dir;
    }
  }, [locale]);

  const value = useMemo(() => {
    const messages = translations[locale] ?? translations[DEFAULT_LOCALE];
    const fallbackMessages = translations[DEFAULT_LOCALE];
    const direction = isRTLLanguage(locale) ? 'rtl' : 'ltr';

    return {
      locale,
      direction,
      setLocale,
      t: (key) => messages[key] ?? fallbackMessages[key] ?? key,
    };
  }, [locale, setLocale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return useContext(I18nContext);
}
