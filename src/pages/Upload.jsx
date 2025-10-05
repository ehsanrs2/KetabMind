import { useCallback, useMemo, useState } from 'react';
import { useI18n, isRTLLanguage } from '../i18n';

const API_BASE = process.env.REACT_APP_API_BASE ?? 'http://127.0.0.1:8000';

function statusMessage(status, t) {
  switch (status) {
    case 'uploading':
      return t('uploading');
    case 'success':
      return t('uploadSuccess');
    case 'error':
      return t('uploadError');
    default:
      return '';
  }
}

export function Upload() {
  const { t, locale } = useI18n();
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  const [selectedFile, setSelectedFile] = useState(null);
  const [errorDetail, setErrorDetail] = useState(null);

  const direction = useMemo(() => (isRTLLanguage(locale) ? 'rtl' : 'ltr'), [locale]);

  const handleUpload = useCallback(
    (file) => {
      if (!file) {
        return;
      }
      setStatus('uploading');
      setErrorDetail(null);
      setProgress(0);

      const formData = new FormData();
      formData.append('file', file);

      const request = new XMLHttpRequest();
      request.open('POST', `${API_BASE}/upload`, true);

      request.upload.onprogress = (event) => {
        if (!event.lengthComputable) {
          return;
        }
        const value = Math.round((event.loaded / event.total) * 100);
        setProgress(value);
      };

      request.onload = () => {
        if (request.status >= 200 && request.status < 300) {
          setProgress(100);
          setStatus('success');
        } else {
          setStatus('error');
          setErrorDetail(request.responseText || request.statusText);
        }
      };

      request.onerror = () => {
        setStatus('error');
        setErrorDetail('network-error');
      };

      request.send(formData);
    },
    [],
  );

  const handleChange = useCallback(
    (event) => {
      const file = event.target.files?.[0];
      setSelectedFile(file ?? null);
      if (file) {
        handleUpload(file);
      }
    },
    [handleUpload],
  );

  return (
    <section style={{ direction }}>
      <h1>{t('uploadTitle')}</h1>
      <p>{t('uploadHint')}</p>
      <label style={{ display: 'inline-block', margin: '1rem 0' }}>
        <span>{t('selectFile')}</span>
        <input
          type="file"
          onChange={handleChange}
          style={{ display: 'block', marginTop: '0.5rem' }}
        />
      </label>
      {selectedFile ? <p>{selectedFile.name}</p> : null}
      {status !== 'idle' ? <p>{statusMessage(status, t)}</p> : null}
      {status === 'uploading' ? (
        <progress value={progress} max="100" aria-label={t('progressLabel')} />
      ) : null}
      {status === 'error' && errorDetail ? <pre>{errorDetail}</pre> : null}
    </section>
  );
}

export default Upload;
