'use client';

import type { CSSProperties } from 'react';
import { ChangeEvent, FormEvent, useCallback, useMemo, useState } from 'react';
import { useAuth } from '../../context/AuthContext';

type UploadMetadata = {
  author?: string;
  year?: number | string;
  subject?: string;
  title?: string;
};

type UploadResponse = {
  book_id?: string;
  version?: number | string;
  file_hash?: string;
  path?: string;
  meta?: UploadMetadata;
  indexed_chunks?: number;
  message?: string;
  detail?: string;
  already_indexed?: boolean;
  deduplicated?: boolean;
  duplicate?: boolean;
  [key: string]: unknown;
};

type StatusState = {
  type: 'idle' | 'success' | 'error';
  message: string | null;
};

type IndexStatusState = {
  type: 'success' | 'error';
  message: string;
} | null;

const INITIAL_FORM = {
  title: '',
  author: '',
  year: '',
  subject: '',
};

function parseXhrResponse(xhr: XMLHttpRequest): UploadResponse | null {
  const raw =
    typeof xhr.response === 'string' && xhr.response
      ? xhr.response
      : typeof xhr.responseText === 'string'
        ? xhr.responseText
        : null;
  if (!raw || raw.trim().length === 0) {
    return null;
  }

  try {
    return JSON.parse(raw) as UploadResponse;
  } catch (error) {
    console.warn('Failed to parse upload response', error);
    return null;
  }
}

function extractMessage(payload: UploadResponse | null, fallback: string): string {
  if (payload) {
    if (typeof payload.detail === 'string' && payload.detail.trim().length > 0) {
      return payload.detail;
    }

    if (typeof payload.message === 'string' && payload.message.trim().length > 0) {
      return payload.message;
    }
  }

  return fallback;
}

function isAlreadyIndexed(payload: UploadResponse | null): boolean {
  if (!payload) {
    return false;
  }

  if (payload.already_indexed || payload.deduplicated || payload.duplicate) {
    return true;
  }

  const detail = typeof payload.detail === 'string' ? payload.detail.toLowerCase() : '';
  const message = typeof payload.message === 'string' ? payload.message.toLowerCase() : '';

  return detail.includes('already indexed') || message.includes('already indexed');
}

export default function UploadPage() {
  const { csrfToken } = useAuth();
  const [formValues, setFormValues] = useState(INITIAL_FORM);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [status, setStatus] = useState<StatusState>({ type: 'idle', message: null });
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [dedupMessage, setDedupMessage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState<IndexStatusState>(null);

  const handleInputChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setFormValues((previous) => ({ ...previous, [name]: value }));
  }, []);

  const resetStateForNewUpload = useCallback(() => {
    setStatus({ type: 'idle', message: null });
    setUploadResponse(null);
    setDedupMessage(null);
    setIndexStatus(null);
  }, []);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!file) {
      setStatus({ type: 'error', message: 'Please choose a file to upload.' });
      return;
    }

    setIsUploading(true);
    setProgress(0);
    resetStateForNewUpload();

    const formData = new FormData();
    formData.append('title', formValues.title);
    formData.append('author', formValues.author);
    formData.append('year', formValues.year);
    formData.append('subject', formValues.subject);
    formData.append('file', file);

    try {
      const payload = await new Promise<UploadResponse>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload');
        xhr.responseType = 'text';
        xhr.withCredentials = true;

        if (csrfToken) {
          xhr.setRequestHeader('x-csrf-token', csrfToken);
        }

        xhr.upload.onprogress = (progressEvent) => {
          if (progressEvent.lengthComputable) {
            const percentage = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            setProgress(percentage);
          }
        };

        xhr.onload = () => {
          const responsePayload = parseXhrResponse(xhr);
          if (xhr.status >= 200 && xhr.status < 300) {
            if (responsePayload) {
              resolve(responsePayload);
            } else {
              reject(new Error('Upload succeeded but response payload was empty.'));
            }
          } else {
            reject(new Error(extractMessage(responsePayload, `Upload failed with status ${xhr.status}`)));
          }
        };

        xhr.onerror = () => {
          reject(new Error('Network error while uploading.'));
        };

        xhr.send(formData);
      });

      setUploadResponse(payload);
      setStatus({
        type: 'success',
        message: extractMessage(payload, 'Upload completed successfully.'),
      });
      setDedupMessage(
        isAlreadyIndexed(payload)
          ? 'This file was already indexed. You can trigger indexing again if needed.'
          : null,
      );
      setFormValues(INITIAL_FORM);
      setFile(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Upload failed.';
      setStatus({ type: 'error', message });
      setUploadResponse(null);
    } finally {
      setIsUploading(false);
      setProgress(null);
    }
  };

  const handleFileChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] ?? null);
  }, []);

  const handleIndexNow = useCallback(async () => {
    if (!uploadResponse) {
      return;
    }

    if (!uploadResponse.path) {
      setIndexStatus({ type: 'error', message: 'Unable to locate uploaded file for indexing. Please upload again.' });
      return;
    }

    setIsIndexing(true);
    setIndexStatus(null);

    try {
      const indexPayload: Record<string, unknown> = {
        path: uploadResponse.path,
      };
      const metaSource: UploadMetadata | undefined = uploadResponse.meta
        ? { ...uploadResponse.meta }
        : undefined;
      const maybeMeta: UploadMetadata = metaSource ?? ({ ...formValues } as UploadMetadata);
      (['author', 'year', 'subject', 'title'] as const).forEach((key) => {
        const value = maybeMeta[key];
        if (value !== undefined && value !== null && value !== '') {
          indexPayload[key] = value;
        }
      });

      const response = await fetch('/index', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(csrfToken ? { 'x-csrf-token': csrfToken } : {}),
        },
        credentials: 'include',
        body: JSON.stringify(indexPayload),
      });

      const text = await response.text();
      let json: UploadResponse | null = null;
      if (text) {
        try {
          json = JSON.parse(text) as UploadResponse;
        } catch (error) {
          console.warn('Failed to parse index response', error);
        }
      }

      if (!response.ok) {
        throw new Error(extractMessage(json, `Indexing failed with status ${response.status}`));
      }

      setIndexStatus({
        type: 'success',
        message: extractMessage(json, 'Indexing started.'),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Indexing failed.';
      setIndexStatus({ type: 'error', message });
    } finally {
      setIsIndexing(false);
    }
  }, [csrfToken, formValues, uploadResponse]);

  const clampedProgress = progress === null ? 0 : Math.min(100, Math.max(0, progress));
  const progressFillStyle = useMemo(
    () => ({ '--progress': `${clampedProgress}%` }) as CSSProperties,
    [clampedProgress],
  );

  const bookId = useMemo(() => uploadResponse?.book_id ?? '—', [uploadResponse]);
  const version = useMemo(() => (uploadResponse?.version ?? '—').toString(), [uploadResponse]);
  const fileHash = useMemo(() => uploadResponse?.file_hash ?? '—', [uploadResponse]);

  return (
    <div className="page-card">
      <h1>Upload a book</h1>
      <form className="form" onSubmit={handleSubmit} noValidate>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="title">Title</label>
            <input
              id="title"
              name="title"
              type="text"
              value={formValues.title}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-field">
            <label htmlFor="author">Author</label>
            <input
              id="author"
              name="author"
              type="text"
              value={formValues.author}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-field">
            <label htmlFor="year">Year</label>
            <input
              id="year"
              name="year"
              type="number"
              min="0"
              inputMode="numeric"
              value={formValues.year}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-field">
            <label htmlFor="subject">Subject</label>
            <input
              id="subject"
              name="subject"
              type="text"
              value={formValues.subject}
              onChange={handleInputChange}
              required
            />
          </div>
        </div>
        <div className="form-field">
          <label htmlFor="file">File</label>
          <input
            id="file"
            name="file"
            type="file"
            accept="application/pdf,.pdf,.epub,application/epub+zip"
            onChange={handleFileChange}
            required
          />
        </div>
        <button type="submit" disabled={isUploading}>
          {isUploading ? 'Uploading…' : 'Upload'}
        </button>
      </form>

      {isUploading ? (
        <div className="upload-progress" aria-live="polite">
          <span className="upload-progress__label">Upload progress: {clampedProgress}%</span>
          <div
            className="upload-progress__bar skeleton"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={clampedProgress}
          >
            <div className="upload-progress__fill" style={progressFillStyle} />
          </div>
        </div>
      ) : null}

      {status.message && (
        <p
          role="status"
          className={status.type === 'error' ? 'text-danger mt-2' : 'text-success mt-2'}
        >
          {status.message}
        </p>
      )}

      {dedupMessage && (
        <p role="status" className="mt-2">
          {dedupMessage}
        </p>
      )}

      {uploadResponse && (
        <div className="result-card mt-4">
          <h2>Upload details</h2>
          <dl>
            <div>
              <dt>Book ID</dt>
              <dd data-testid="upload-result-book-id">{bookId}</dd>
            </div>
            <div>
              <dt>Version</dt>
              <dd data-testid="upload-result-version">{version}</dd>
            </div>
            <div>
              <dt>File hash</dt>
              <dd data-testid="upload-result-file-hash">{fileHash}</dd>
            </div>
            {uploadResponse?.path ? (
              <div>
                <dt>Stored path</dt>
                <dd data-testid="upload-result-path">{uploadResponse.path}</dd>
              </div>
            ) : null}
          </dl>
          <button type="button" onClick={handleIndexNow} disabled={isIndexing || !uploadResponse?.path}>
            {isIndexing ? 'Indexing…' : 'Index now'}
          </button>
          {indexStatus && (
            <p
              role="status"
              className={indexStatus.type === 'error' ? 'text-danger mt-2' : 'text-success mt-2'}
            >
              {indexStatus.message}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
