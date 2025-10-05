import { useState } from 'react';
import { Chat } from './pages/Chat';
import { Upload } from './pages/Upload';
import { I18nProvider, useI18n } from './i18n';

function LanguageSwitcher() {
  const { locale, setLocale } = useI18n();
  const [value, setValue] = useState(locale);

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        setLocale(value);
      }}
      style={{ marginBottom: '1rem' }}
    >
      <label>
        Locale:
        <input value={value} onChange={(event) => setValue(event.target.value)} style={{ marginInlineStart: '0.5rem' }} />
      </label>
      <button type="submit" style={{ marginInlineStart: '0.5rem' }}>
        Apply
      </button>
    </form>
  );
}

function AppShell() {
  return (
    <div style={{ display: 'grid', gap: '2rem', padding: '2rem' }}>
      <LanguageSwitcher />
      <Upload />
      <Chat />
    </div>
  );
}

export default function App({ initialLocale }) {
  return (
    <I18nProvider initialLocale={initialLocale}>
      <AppShell />
    </I18nProvider>
  );
}
