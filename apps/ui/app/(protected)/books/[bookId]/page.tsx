'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getBook, type BookMetadata, type BookRecord } from '../api';

function formatDate(value: string | null | undefined): string {
  if (!value) {
    return '—';
  }
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return '—';
    }
    return date.toLocaleString();
  } catch (error) {
    return '—';
  }
}

function normalizeMetadata(metadata: BookMetadata): Array<[string, string]> {
  if (!metadata || typeof metadata !== 'object') {
    return [];
  }
  const entries: Array<[string, string]> = [];
  Object.entries(metadata).forEach(([key, value]) => {
    if (value == null) {
      return;
    }
    const stringValue = typeof value === 'string' ? value : JSON.stringify(value);
    entries.push([key, stringValue]);
  });
  return entries;
}

type BookDetailsPageProps = {
  params: { bookId: string };
};

export default function BookDetailsPage({ params }: BookDetailsPageProps): JSX.Element {
  const { bookId } = params;
  const [book, setBook] = useState<BookRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    const loadBook = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await getBook(bookId);
        if (mounted) {
          setBook(result);
        }
      } catch (err) {
        if (mounted) {
          setBook(null);
          setError(err instanceof Error ? err.message : 'Unable to load book.');
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    void loadBook();

    return () => {
      mounted = false;
    };
  }, [bookId]);

  const metadataEntries = normalizeMetadata(book?.metadata);

  return (
    <main className="book-details">
      <nav className="book-details__breadcrumbs">
        <Link href="/books" className="button button--ghost">
          ← Back to books
        </Link>
      </nav>

      {loading ? <p className="book-details__loading">Loading book…</p> : null}
      {error ? (
        <div role="alert" className="book-details__error">
          <p>{error}</p>
        </div>
      ) : null}

      {book ? (
        <section className="book-details__card">
          <header>
            <h1>{book.title || book.id}</h1>
            <p className="book-details__meta">Book ID: {book.id}</p>
          </header>

          <dl className="book-details__grid">
            <div>
              <dt>Status</dt>
              <dd>{book.status ?? 'unknown'}</dd>
            </div>
            <div>
              <dt>Created</dt>
              <dd>{formatDate(book.created_at)}</dd>
            </div>
            <div>
              <dt>Updated</dt>
              <dd>{formatDate(book.updated_at)}</dd>
            </div>
            <div>
              <dt>Version</dt>
              <dd>{book.version ?? '—'}</dd>
            </div>
            <div>
              <dt>Indexed chunks</dt>
              <dd>{book.indexed_chunks ?? '—'}</dd>
            </div>
            <div>
              <dt>File hash</dt>
              <dd>{book.file_hash ?? '—'}</dd>
            </div>
          </dl>

          {book.description ? (
            <section className="book-details__section">
              <h2>Description</h2>
              <p>{book.description}</p>
            </section>
          ) : null}

          <section className="book-details__section">
            <h2>Metadata</h2>
            {metadataEntries.length > 0 ? (
              <dl className="book-details__metadata">
                {metadataEntries.map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="book-details__empty">No additional metadata available.</p>
            )}
          </section>
        </section>
      ) : null}
    </main>
  );
}
