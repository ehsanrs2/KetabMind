'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Worker, Viewer } from '@react-pdf-viewer/core';
import { pageNavigationPlugin } from '@react-pdf-viewer/page-navigation';
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/page-navigation/lib/styles/index.css';
import { useSearchParams } from 'next/navigation';

const DEFAULT_PAGE = 1;
const PDF_WORKER_URL = 'https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js';

function parsePageNumber(hash: string | null): number {
  if (!hash) {
    return DEFAULT_PAGE;
  }
  const match = hash.match(/page=(\d+)/i);
  if (!match) {
    return DEFAULT_PAGE;
  }
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) && value > 0 ? value : DEFAULT_PAGE;
}

export default function ViewerPage(): JSX.Element {
  const searchParams = useSearchParams();
  const bookId = searchParams.get('book');
  const initialPage = typeof window !== 'undefined' ? parsePageNumber(window.location.hash) : DEFAULT_PAGE;

  const [page, setPage] = useState<number>(initialPage);
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [documentLoaded, setDocumentLoaded] = useState<boolean>(false);

  const pageNavigationPluginInstance = useMemo(() => pageNavigationPlugin(), []);
  const { jumpToPage } = pageNavigationPluginInstance;

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const handleHashChange = () => {
      setPage(parsePageNumber(window.location.hash));
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => {
      window.removeEventListener('hashchange', handleHashChange);
    };
  }, []);

  useEffect(() => {
    if (!bookId) {
      setError('Book not specified.');
      setViewerUrl(null);
      return;
    }
    const controller = new AbortController();

    const loadSignedUrl = async () => {
      setLoading(true);
      setError(null);
      setDocumentLoaded(false);
      try {
        const response = await fetch(`/book/${bookId}/page/${page}/view`, {
          method: 'GET',
          credentials: 'include',
          signal: controller.signal,
        });
        if (response.status === 403) {
          setError('You do not have access to this book.');
          setViewerUrl(null);
          return;
        }
        if (!response.ok) {
          throw new Error(`Failed to load viewer (${response.status})`);
        }
        const payload = (await response.json()) as { url?: string };
        if (!payload?.url) {
          throw new Error('Viewer response missing signed URL.');
        }
        setViewerUrl(payload.url);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Unknown error.');
          setViewerUrl(null);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    };

    void loadSignedUrl();

    return () => {
      controller.abort();
    };
  }, [bookId, page]);

  useEffect(() => {
    if (documentLoaded) {
      jumpToPage(Math.max(page - 1, 0));
    }
  }, [documentLoaded, jumpToPage, page]);

  const handleDocumentLoad = useCallback(() => {
    setDocumentLoaded(true);
  }, []);

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold">Book Viewer</h1>
        <p className="text-sm text-gray-600">Displaying book pages inside the application viewer.</p>
      </header>

      {loading && <p role="status">Loading book…</p>}
      {error && <p role="alert" className="text-red-600">{error}</p>}

      {viewerUrl && !error ? (
        <div className="flex-1 overflow-hidden rounded border border-gray-200 shadow-sm">
          <Worker workerUrl={PDF_WORKER_URL}>
            <Viewer
              fileUrl={viewerUrl}
              initialPage={Math.max(page - 1, 0)}
              onDocumentLoad={handleDocumentLoad}
              plugins={[pageNavigationPluginInstance]}
            />
          </Worker>
        </div>
      ) : null}
    </div>
  );
}
