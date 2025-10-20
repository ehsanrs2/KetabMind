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
  detail?: unknown;
  already_indexed?: boolean;
  deduplicated?: boolean;
  duplicate?: boolean;
  error?: unknown;
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
  const rawResponse = (xhr.response ?? xhr.responseText) as unknown;

  if (rawResponse == null) {
    return null;
  }

  if (typeof rawResponse === 'object') {
    return rawResponse as UploadResponse;
  }

  const text = String(rawResponse);
  if (!text.trim()) {
    return null;
  }

  try {
    return JSON.parse(raw) as UploadResponse;
  } catch (error) {
    console.warn('Failed to parse upload response', error);
    const fallback = text.trim();
    return fallback ? { message: fallback } : null;
  }
}

function isLikelyHtml(text: string): boolean {
  const trimmed = text.trim();
  return /^<!DOCTYPE html/i.test(trimmed) || /^<html/i.test(trimmed);
}

function normalizeMessage(text: string | null | undefined, fallback: string): string {
  if (!text) {
    return fallback;
  }

  const trimmed = text.trim();
  if (!trimmed || isLikelyHtml(trimmed)) {
    return fallback;
  }

  return trimmed;
}

function extractFromDetail(detail: unknown): string | null {
  if (typeof detail === 'string') {
    const trimmed = detail.trim();
    return trimmed ? trimmed : null;
  }

  if (Array.isArray(detail)) {
    for (const item of detail) {
      const candidate = extractFromDetail(
        typeof item === 'object' && item !== null
          ? (item as Record<string, unknown>).msg ??
            (item as Record<string, unknown>).message ??
            (item as Record<string, unknown>).detail ??
            item
          : item,
      );
      if (candidate) {
        return candidate;
      }
    }
    return null;
  }

  if (typeof detail === 'object' && detail !== null) {
    const record = detail as Record<string, unknown>;
    return (
      extractFromDetail(record.msg) ??
      extractFromDetail(record.message) ??
      extractFromDetail(record.detail)
    );
  }

  return null;
}

function extractMessage(payload: UploadResponse | null, fallback: string): string {
  if (payload) {
    const detailMessage = extractFromDetail(payload.detail);
    if (detailMessage) {
      return normalizeMessage(detailMessage, fallback);
    }

    if (typeof payload.message === 'string') {
      return normalizeMessage(payload.message, fallback);
    }

    if (
      typeof payload.error === 'object' &&
      payload.error !== null &&
      'message' in (payload.error as Record<string, unknown>)
    ) {
      return normalizeMessage(
        String((payload.error as Record<string, unknown>).message ?? ''),
        fallback,
      );
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
        xhr.open('POST', '/api/upload');
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

    if (!uploadResponse.file_hash) {
      setIndexStatus({
        type: 'error',
        message: 'Unable to trigger indexing: missing file hash.',
      });
      return;
    }

    setIsIndexing(true);
    setIndexStatus(null);

    try {
      const response = await fetch('/api/index', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(csrfToken ? { 'x-csrf-token': csrfToken } : {}),
        },
        credentials: 'include',
        body: JSON.stringify(indexPayload),
      });

      const text = await response.text();
      let json: Record<string, unknown> | null = null;
      if (text) {
        try {
          json = JSON.parse(text) as Record<string, unknown>;
        } catch (error) {
          console.warn('Failed to parse index response', error);
        }
      }

      if (!response.ok) {
        throw new Error(
          extractMessage(json as UploadResponse | null, `Indexing failed with status ${response.status}`),
        );
      }

      let indexMessage = 'Indexing started.';
      if (json && typeof json === 'object') {
        const indexedField = json.indexed;
        if (typeof indexedField === 'boolean') {
          indexMessage = indexedField ? 'Indexing completed.' : 'Indexing failed.';
        } else {
          const newCount = json.new;
          const skippedCount = json.skipped;
          if (typeof newCount === 'number' || typeof skippedCount === 'number') {
            const parts: string[] = [];
            if (typeof newCount === 'number') {
              parts.push(`${newCount} new`);
            }
            if (typeof skippedCount === 'number') {
              parts.push(`${skippedCount} skipped`);
            }
            indexMessage = `Indexing completed: ${parts.join(', ')}.`;
          } else {
            indexMessage = extractMessage(json as UploadResponse | null, 'Indexing started.');
          }
        }
      } else {
        indexMessage = extractMessage(json as UploadResponse | null, 'Indexing started.');
      }

      setIndexStatus({
        type: 'success',
        message: indexMessage,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Indexing failed.';
      setIndexStatus({ type: 'error', message });
    } finally {
      setIsIndexing(false);
    }
  }, [csrfToken, uploadResponse]);

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
