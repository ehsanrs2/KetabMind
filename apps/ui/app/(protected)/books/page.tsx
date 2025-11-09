'use client';

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import Link from 'next/link';
import { deleteBook, listBooks, renameBook, type BookMetadata, type BookRecord } from './api';
import { useAuth } from '../../context/AuthContext';

const PAGE_SIZE = 10;

type BannerState = {
  type: 'success' | 'error';
  message: string;
} | null;

type RenameDialogState = {
  book: BookRecord | null;
  title: string;
  description: string;
  loading: boolean;
  error: string | null;
};

type DeleteDialogState = {
  book: BookRecord | null;
  loading: boolean;
  error: string | null;
};

function extractAuthor(metadata: BookMetadata): string | null {
  if (!metadata || typeof metadata !== 'object') {
    return null;
  }
  const record = metadata as Record<string, unknown>;
  const authorKeys = ['author', 'Author', 'writer', 'Writer', 'creator', 'Creator'];
  for (const key of authorKeys) {
    const value = record[key];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

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

type DialogProps = {
  isOpen: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
};

function Dialog({ isOpen, title, onClose, children, footer }: DialogProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);
  const titleId = useMemo(() => `dialog-${Math.random().toString(36).slice(2)}`, []);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }
    previousActiveElement.current = document.activeElement as HTMLElement | null;
    const focusable = containerRef.current?.querySelector<HTMLElement>(
      'button, [href], input, textarea, select, [tabindex]:not([tabindex="-1"])',
    );
    focusable?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      previousActiveElement.current?.focus();
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <div className="dialog-backdrop" role="presentation" onClick={onClose}>
      <div
        ref={containerRef}
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="dialog__header">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="dialog__close" onClick={onClose} aria-label="Close dialog">
            ×
          </button>
        </header>
        <div className="dialog__body">{children}</div>
        {footer ? <footer className="dialog__footer">{footer}</footer> : null}
      </div>
    </div>
  );
}

export default function BooksPage(): JSX.Element {
  const { csrfToken } = useAuth();
  const [books, setBooks] = useState<BookRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [banner, setBanner] = useState<BannerState>(null);
  const [renameState, setRenameState] = useState<RenameDialogState>({
    book: null,
    title: '',
    description: '',
    loading: false,
    error: null,
  });
  const [deleteState, setDeleteState] = useState<DeleteDialogState>({ book: null, loading: false, error: null });

  const loadBooks = useCallback(
    async (requestedOffset: number = offset) => {
      setLoading(true);
      setError(null);
      let nextOffset = Math.max(requestedOffset, 0);
      try {
        for (let attempt = 0; attempt < 2; attempt += 1) {
          const { books: nextBooks, total: nextTotal, offset: responseOffset, limit } = await listBooks({
            limit: PAGE_SIZE,
            offset: nextOffset,
          });
          if (nextBooks.length === 0 && nextTotal > 0 && nextOffset > 0) {
            const maxOffset = nextTotal > 0 ? Math.floor((nextTotal - 1) / PAGE_SIZE) * PAGE_SIZE : 0;
            const candidateOffset = Math.max(0, Math.min(nextOffset - PAGE_SIZE, maxOffset));
            if (candidateOffset !== nextOffset) {
              nextOffset = candidateOffset;
              continue;
            }
          }
          setBooks(nextBooks);
          setTotal(nextTotal);
          setOffset(typeof responseOffset === 'number' ? responseOffset : nextOffset);
          if (typeof limit === 'number' && limit !== PAGE_SIZE) {
            console.debug('Books API returned unexpected page size', limit);
          }
          return;
        }
        setBooks([]);
        setTotal(0);
        setOffset(0);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load books.');
        setBooks([]);
        setTotal(0);
      } finally {
        setLoading(false);
      }
    },
    [offset],
  );

  useEffect(() => {
    void loadBooks(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openRenameDialog = useCallback((book: BookRecord) => {
    setRenameState({
      book,
      title: book.title ?? '',
      description: book.description ?? '',
      loading: false,
      error: null,
    });
  }, []);

  const closeRenameDialog = useCallback(() => {
    setRenameState((previous) => ({ ...previous, book: null, error: null, loading: false }));
  }, []);

  const openDeleteDialog = useCallback((book: BookRecord) => {
    setDeleteState({ book, loading: false, error: null });
  }, []);

  const closeDeleteDialog = useCallback(() => {
    setDeleteState({ book: null, loading: false, error: null });
  }, []);

  const handleRenameSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!renameState.book) {
        return;
      }
      const trimmedTitle = renameState.title.trim();
      if (!trimmedTitle) {
        setRenameState((previous) => ({ ...previous, error: 'Title is required.' }));
        return;
      }
      setRenameState((previous) => ({ ...previous, loading: true, error: null }));
      try {
        const updated = await renameBook(
          renameState.book.id,
          { title: trimmedTitle, description: renameState.description.trim() || null },
          csrfToken ?? null,
        );
        setBooks((previous) => previous.map((book) => (book.id === updated.id ? { ...book, ...updated } : book)));
        setBanner({ type: 'success', message: 'Book renamed successfully.' });
        setRenameState({ book: null, title: '', description: '', loading: false, error: null });
      } catch (err) {
        setRenameState((previous) => ({
          ...previous,
          loading: false,
          error: err instanceof Error ? err.message : 'Unable to rename book.',
        }));
      }
    },
    [csrfToken, renameState.book, renameState.description, renameState.title],
  );

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteState.book) {
      return;
    }
    setDeleteState((previous) => ({ ...previous, loading: true, error: null }));
    try {
      await deleteBook(deleteState.book.id, csrfToken ?? null);
      const nextTotal = Math.max(total - 1, 0);
      const maxOffset = nextTotal > 0 ? Math.floor((nextTotal - 1) / PAGE_SIZE) * PAGE_SIZE : 0;
      const nextOffset = Math.min(offset, maxOffset);
      setBanner({ type: 'success', message: 'Book deleted successfully.' });
      setDeleteState({ book: null, loading: false, error: null });
      await loadBooks(nextOffset);
    } catch (err) {
      setDeleteState((previous) => ({
        ...previous,
        loading: false,
        error: err instanceof Error ? err.message : 'Unable to delete book.',
      }));
    }
  }, [csrfToken, deleteState.book, loadBooks, offset, total]);

  const handlePreviousPage = useCallback(() => {
    const nextOffset = Math.max(offset - PAGE_SIZE, 0);
    if (nextOffset !== offset) {
      void loadBooks(nextOffset);
    }
  }, [loadBooks, offset]);

  const handleNextPage = useCallback(() => {
    if (offset + PAGE_SIZE >= total) {
      return;
    }
    const nextOffset = offset + PAGE_SIZE;
    void loadBooks(nextOffset);
  }, [loadBooks, offset, total]);

  const clearBanner = useCallback(() => {
    setBanner(null);
  }, []);

  const hasBooks = books.length > 0;
  const showingStart = hasBooks ? offset + 1 : 0;
  const showingEnd = hasBooks ? offset + books.length : 0;

  return (
    <main className="books-page">
      <header className="books-page__header">
        <div>
          <h1>Books</h1>
          <p>Manage indexed books in your library.</p>
        </div>
      </header>

      {banner ? (
        <div className={`books-page__banner books-page__banner--${banner.type}`} role="status">
          <span>{banner.message}</span>
          <button type="button" onClick={clearBanner} aria-label="Dismiss message">
            ×
          </button>
        </div>
      ) : null}

      {error ? (
        <div role="alert" className="books-page__error">
          <p>{error}</p>
          <button type="button" className="button" onClick={() => void loadBooks(offset)} disabled={loading}>
            Retry
          </button>
        </div>
      ) : null}

      {loading && !hasBooks ? <p className="books-page__loading">Loading books…</p> : null}

      {!loading && !error && !hasBooks ? <p className="books-page__empty">No books found.</p> : null}

      {hasBooks ? (
        <div className="books-page__table-wrapper">
          <table className="books-page__table">
            <thead>
              <tr>
                <th scope="col">Title</th>
                <th scope="col">Author</th>
                <th scope="col">Indexed</th>
                <th scope="col">Status</th>
                <th scope="col">Actions</th>
              </tr>
            </thead>
            <tbody>
              {books.map((book) => {
                const author = extractAuthor(book.metadata);
                return (
                  <tr key={book.id}>
                    <th scope="row">
                      <div className="books-page__title">
                        <span>{book.title || book.id}</span>
                        <span className="books-page__meta">ID: {book.id}</span>
                      </div>
                    </th>
                    <td>{author ?? '—'}</td>
                    <td>{formatDate(book.created_at)}</td>
                    <td>
                      <span
                        className={`books-page__status books-page__status--${
                          (book.status ?? 'unknown').toLowerCase()
                        }`}
                      >
                        {book.status ?? 'unknown'}
                      </span>
                    </td>
                    <td className="books-page__actions">
                      <Link href={`/books/${encodeURIComponent(book.id)}`} className="button button--ghost">
                        Details
                      </Link>
                      <button type="button" className="button" onClick={() => openRenameDialog(book)}>
                        Rename
                      </button>
                      <button
                        type="button"
                        className="button button--danger"
                        onClick={() => openDeleteDialog(book)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      {hasBooks ? (
        <div className="books-page__pagination" aria-live="polite">
          <button
            type="button"
            className="button"
            onClick={handlePreviousPage}
            disabled={offset === 0 || loading}
          >
            Previous
          </button>
          <span>
            Showing {showingStart}–{showingEnd} of {total}
          </span>
          <button
            type="button"
            className="button"
            onClick={handleNextPage}
            disabled={offset + PAGE_SIZE >= total || loading}
          >
            Next
          </button>
        </div>
      ) : null}

      <Dialog
        isOpen={renameState.book !== null}
        title="Rename book"
        onClose={renameState.loading ? () => undefined : closeRenameDialog}
        footer={
          <>
            <button type="button" className="button button--ghost" onClick={closeRenameDialog} disabled={renameState.loading}>
              Cancel
            </button>
            <button type="submit" form="rename-form" className="button button--primary" disabled={renameState.loading}>
              {renameState.loading ? 'Saving…' : 'Save changes'}
            </button>
          </>
        }
      >
        <form id="rename-form" className="books-page__form" onSubmit={handleRenameSubmit}>
          <label className="books-page__form-field">
            <span>Title</span>
            <input
              type="text"
              name="title"
              value={renameState.title}
              onChange={(event) =>
                setRenameState((previous) => ({ ...previous, title: event.target.value, error: null }))
              }
              disabled={renameState.loading}
              required
            />
          </label>
          <label className="books-page__form-field">
            <span>Description (optional)</span>
            <textarea
              name="description"
              value={renameState.description}
              onChange={(event) =>
                setRenameState((previous) => ({ ...previous, description: event.target.value, error: null }))
              }
              disabled={renameState.loading}
              rows={3}
            />
          </label>
          {renameState.error ? (
            <p role="alert" className="books-page__form-error">
              {renameState.error}
            </p>
          ) : null}
        </form>
      </Dialog>

      <Dialog
        isOpen={deleteState.book !== null}
        title="Delete book"
        onClose={deleteState.loading ? () => undefined : closeDeleteDialog}
        footer={
          <>
            <button type="button" className="button button--ghost" onClick={closeDeleteDialog} disabled={deleteState.loading}>
              Cancel
            </button>
            <button
              type="button"
              className="button button--danger"
              onClick={handleDeleteConfirm}
              disabled={deleteState.loading}
            >
              {deleteState.loading ? 'Deleting…' : 'Delete book'}
            </button>
          </>
        }
      >
        <p>
          Are you sure you want to delete{' '}
          <strong>{deleteState.book?.title ?? deleteState.book?.id ?? 'this book'}</strong>? This action cannot be undone.
        </p>
        {deleteState.error ? (
          <p role="alert" className="books-page__form-error">
            {deleteState.error}
          </p>
        ) : null}
      </Dialog>
    </main>
  );
}
