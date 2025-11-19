'use client';

import Link from 'next/link';
import {
  Edit3,
  Loader2,
  MessageSquare,
  MoreVertical,
  Search,
  Trash2,
  X,
} from 'lucide-react';
import type { FormEvent } from 'react';
import { useEffect, useId, useMemo, useState } from 'react';

import type { BookRecord } from '../../(protected)/chat/types';

export type BookMutation = { id: string; type: 'delete' | 'rename' } | null;

export type BookSidebarFeedback = {
  type: 'success' | 'error';
  message: string;
};

type BookSidebarProps = {
  books: BookRecord[];
  selectedBookIds: string[];
  onToggleBook: (bookId: string) => void;
  onClearSelection: () => void;
  isLoading: boolean;
  error: string | null;
  onRetry?: () => void;
  onDeleteBook: (bookId: string) => Promise<void>;
  onRenameBook: (bookId: string, title: string) => Promise<void>;
  pendingAction: BookMutation;
  feedback?: BookSidebarFeedback | null;
  onDismissFeedback?: () => void;
};

const SKELETON_ITEMS = Array.from({ length: 4 });

function formatCount(value: number | null | undefined): string | null {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null;
  }
  return `${value.toLocaleString()} indexed chunks`;
}

export function BookSidebar({
  books,
  selectedBookIds,
  onToggleBook,
  onClearSelection,
  isLoading,
  error,
  onRetry,
  onDeleteBook,
  onRenameBook,
  pendingAction,
  feedback,
  onDismissFeedback,
}: BookSidebarProps) {
  const searchInputId = useId();
  const renameHeadingId = useId();
  const deleteHeadingId = useId();

  const [searchTerm, setSearchTerm] = useState('');
  const [activeMenuId, setActiveMenuId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renameModal, setRenameModal] = useState<{ id: string; title: string } | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; title: string } | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    if (renameModal) {
      setRenameDraft(renameModal.title);
      setRenameError(null);
    }
  }, [renameModal]);

  useEffect(() => {
    if (deleteTarget) {
      setDeleteError(null);
    }
  }, [deleteTarget]);

  useEffect(() => {
    if (!feedback || !onDismissFeedback) {
      return;
    }
    const timeout = window.setTimeout(() => {
      onDismissFeedback();
    }, 4000);
    return () => window.clearTimeout(timeout);
  }, [feedback, onDismissFeedback]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setActiveMenuId(null);
        setRenameModal(null);
        setDeleteTarget(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const filteredBooks = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    if (!query) {
      return books;
    }
    return books.filter((book) => book.title.toLowerCase().includes(query));
  }, [books, searchTerm]);

  const selectedCount = selectedBookIds.length;
  const showSkeleton = isLoading && books.length === 0;
  const isRefreshing = isLoading && books.length > 0;

  const handleRenameSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!renameModal) {
      return;
    }
    const trimmed = renameDraft.trim();
    if (!trimmed) {
      setRenameError('Title cannot be empty.');
      return;
    }
    try {
      await onRenameBook(renameModal.id, trimmed);
      setRenameModal(null);
      setRenameDraft('');
      setRenameError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unable to rename book. Please try again.';
      setRenameError(message);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) {
      return;
    }
    setDeleteError(null);
    try {
      await onDeleteBook(deleteTarget.id);
      setDeleteTarget(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unable to delete book. Please try again.';
      setDeleteError(message);
    }
  };

  const renderEmptyState = () => {
    if (books.length === 0 && !isLoading) {
      return (
        <div className="mt-6 flex flex-col items-center gap-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-400">
          <MessageSquare className="h-6 w-6 text-slate-400" aria-hidden="true" />
          <p>No books found. Upload a book to start chatting with your library.</p>
          <Link
            href="/upload"
            className="rounded-full border border-slate-900 px-4 py-1.5 text-sm font-semibold text-slate-900 transition hover:bg-slate-900 hover:text-white dark:border-slate-200 dark:text-slate-200 dark:hover:bg-slate-200 dark:hover:text-slate-900"
          >
            Upload a book
          </Link>
        </div>
      );
    }

    if (filteredBooks.length === 0 && books.length > 0 && !isLoading) {
      return (
        <div className="mt-6 flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <MessageSquare className="h-5 w-5 text-slate-400" aria-hidden="true" />
          <p>No books match “{searchTerm.trim()}”.</p>
        </div>
      );
    }

    return null;
  };

  const renderFeedback = () => {
    if (!feedback) {
      return null;
    }
    const styles =
      feedback.type === 'success'
        ? 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-400/30 dark:bg-emerald-500/10 dark:text-emerald-100'
        : 'border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-100';
    return (
      <div
        className={`mt-4 flex items-start gap-3 rounded-xl border px-4 py-3 text-sm shadow-sm ${styles}`}
        role="status"
      >
        <p className="flex-1">{feedback.message}</p>
        {onDismissFeedback ? (
          <button
            type="button"
            onClick={onDismissFeedback}
            className="rounded-full p-1 text-current transition hover:bg-black/5 dark:hover:bg-white/10"
            aria-label="Dismiss notification"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        ) : null}
      </div>
    );
  };

  const renderBooks = () => {
    if (showSkeleton) {
      return (
        <div className="mt-4 space-y-3" aria-live="polite" aria-busy="true">
          {SKELETON_ITEMS.map((_, index) => (
            <div
              key={index}
              className="h-16 w-full animate-pulse rounded-2xl border border-slate-100 bg-slate-100/70 dark:border-slate-800 dark:bg-slate-800/70"
            />
          ))}
        </div>
      );
    }

    if (filteredBooks.length === 0) {
      return null;
    }

    return (
      <div className="mt-4 flex-1 overflow-y-auto pr-1">
        <ul className="space-y-3">
          {filteredBooks.map((book) => {
            const isSelected = selectedBookIds.includes(book.bookId);
            const isMutating = pendingAction?.id === book.bookId;
            const author =
              book.metadata && typeof book.metadata.author === 'string'
                ? (book.metadata.author as string)
                : null;
            const indexed = formatCount(book.indexedChunks ?? null);
            return (
              <li
                key={book.bookId}
                className={`rounded-2xl border p-3 transition focus-within:ring-2 focus-within:ring-slate-400 dark:border-slate-800 ${
                  isSelected
                    ? 'border-slate-900 bg-slate-50 shadow-sm dark:border-slate-400 dark:bg-slate-800'
                    : 'border-slate-200 bg-white hover:border-slate-300 dark:bg-slate-900'
                }`}
              >
                <div className="flex items-start gap-2">
                  <button
                    type="button"
                    onClick={() => onToggleBook(book.bookId)}
                    className="flex flex-1 items-start gap-3 text-left"
                    aria-pressed={isSelected}
                  >
                    <div
                      className={`mt-0.5 flex h-5 w-5 items-center justify-center rounded-full border text-xs font-semibold ${
                        isSelected
                          ? 'border-slate-900 bg-slate-900 text-white'
                          : 'border-slate-300 text-slate-500'
                      }`}
                    >
                      {isSelected ? '✓' : ''}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{book.title}</p>
                      {author ? (
                        <p className="text-xs text-slate-500 dark:text-slate-400">{author}</p>
                      ) : null}
                      {indexed ? (
                        <p className="text-xs text-slate-400 dark:text-slate-500">{indexed}</p>
                      ) : null}
                    </div>
                  </button>
                  <div className="relative">
                    <button
                      type="button"
                      className="rounded-full p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 dark:hover:bg-slate-800"
                      aria-label={`Book actions for ${book.title}`}
                      onClick={() =>
                        setActiveMenuId((current) => (current === book.bookId ? null : book.bookId))
                      }
                      disabled={Boolean(pendingAction)}
                    >
                      <MoreVertical className="h-4 w-4" aria-hidden="true" />
                    </button>
                    {activeMenuId === book.bookId ? (
                      <div className="absolute right-0 z-10 mt-2 w-44 rounded-xl border border-slate-200 bg-white p-1 text-sm shadow-xl dark:border-slate-800 dark:bg-slate-900">
                        <button
                          type="button"
                          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-slate-700 transition hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800"
                          onClick={() => {
                            setRenameModal({ id: book.bookId, title: book.title });
                            setActiveMenuId(null);
                          }}
                        >
                          <Edit3 className="h-4 w-4" aria-hidden="true" /> Rename
                        </button>
                        <button
                          type="button"
                          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-red-600 transition hover:bg-red-50 dark:text-red-300 dark:hover:bg-red-500/10"
                          onClick={() => {
                            setDeleteTarget({ id: book.bookId, title: book.title });
                            setActiveMenuId(null);
                          }}
                        >
                          <Trash2 className="h-4 w-4" aria-hidden="true" /> Delete
                        </button>
                      </div>
                    ) : null}
                  </div>
                </div>
                {isMutating ? (
                  <p className="mt-3 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                    {pendingAction?.type === 'delete' ? 'Deleting…' : 'Renaming…'}
                  </p>
                ) : null}
              </li>
            );
          })}
        </ul>
      </div>
    );
  };

  return (
    <aside className="flex w-full flex-col rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-950">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">Books</p>
          {selectedCount > 0 ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">{selectedCount} selected</p>
          ) : (
            <p className="text-xs text-slate-500 dark:text-slate-400">Filter search scope</p>
          )}
        </div>
        {selectedCount > 0 ? (
          <button
            type="button"
            className="text-xs font-semibold text-slate-600 transition hover:text-slate-900 dark:text-slate-300 dark:hover:text-white"
            onClick={onClearSelection}
          >
            Clear
          </button>
        ) : null}
      </div>

      <label
        className="mt-4 flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-500 focus-within:border-slate-400 focus-within:bg-white dark:border-slate-800 dark:bg-slate-900"
        htmlFor={searchInputId}
      >
        <Search className="h-4 w-4" aria-hidden="true" />
        <input
          id={searchInputId}
          type="search"
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.target.value)}
          placeholder="Search books"
          className="h-7 flex-1 border-0 bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none dark:text-slate-100"
        />
      </label>

      {renderFeedback()}

      {error ? (
        <div
          className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-100"
          role="alert"
        >
          <p>{error}</p>
          {onRetry ? (
            <button
              type="button"
              className="mt-2 text-xs font-semibold underline"
              onClick={onRetry}
            >
              Try again
            </button>
          ) : null}
        </div>
      ) : null}

      {isRefreshing ? (
        <p className="mt-4 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> Refreshing library…
        </p>
      ) : null}

      {renderBooks()}
      {renderEmptyState()}

      {selectedCount > 1 ? (
        <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
          Searches will include every selected book.
        </p>
      ) : null}

      {renameModal ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/30 px-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby={renameHeadingId}
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              setRenameModal(null);
            }
          }}
        >
          <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl dark:border-slate-800 dark:bg-slate-900">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id={renameHeadingId} className="text-base font-semibold text-slate-900 dark:text-slate-100">
                  Rename book
                </h2>
                <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                  Update the title for “{renameModal.title}”.
                </p>
              </div>
              <button
                type="button"
                className="rounded-full p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800"
                onClick={() => setRenameModal(null)}
                aria-label="Close rename dialog"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
            <form className="mt-4 space-y-3" onSubmit={handleRenameSubmit}>
              <input
                type="text"
                value={renameDraft}
                onChange={(event) => setRenameDraft(event.target.value)}
                className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-slate-500 focus:outline-none dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                autoFocus
              />
              {renameError ? (
                <p className="text-xs text-rose-600 dark:text-rose-300" role="alert">
                  {renameError}
                </p>
              ) : null}
              <div className="flex justify-end gap-3 text-sm">
                <button
                  type="button"
                  className="rounded-full px-4 py-2 text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
                  onClick={() => setRenameModal(null)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-4 py-2 font-semibold text-white transition hover:bg-slate-800 disabled:opacity-60 dark:bg-slate-100 dark:text-slate-900"
                  disabled={pendingAction?.id === renameModal.id && pendingAction?.type === 'rename'}
                >
                  {pendingAction?.id === renameModal.id && pendingAction?.type === 'rename' ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Saving…
                    </>
                  ) : (
                    'Save'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : null}

      {deleteTarget ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/30 px-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby={deleteHeadingId}
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              setDeleteTarget(null);
            }
          }}
        >
          <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl dark:border-slate-800 dark:bg-slate-900">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id={deleteHeadingId} className="text-base font-semibold text-slate-900 dark:text-slate-100">
                  Delete book
                </h2>
                <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                  This will permanently delete “{deleteTarget.title}” and its indexed pages.
                </p>
              </div>
              <button
                type="button"
                className="rounded-full p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800"
                onClick={() => setDeleteTarget(null)}
                aria-label="Close delete dialog"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
            {deleteError ? (
              <p className="mt-3 text-xs text-rose-600 dark:text-rose-300" role="alert">
                {deleteError}
              </p>
            ) : null}
            <div className="mt-6 flex justify-end gap-3 text-sm">
              <button
                type="button"
                className="rounded-full px-4 py-2 text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
                onClick={() => setDeleteTarget(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-full bg-rose-600 px-4 py-2 font-semibold text-white transition hover:bg-rose-700 disabled:opacity-70"
                onClick={handleDeleteConfirm}
                disabled={pendingAction?.id === deleteTarget.id && pendingAction?.type === 'delete'}
              >
                {pendingAction?.id === deleteTarget.id && pendingAction?.type === 'delete' ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Deleting…
                  </>
                ) : (
                  'Delete'
                )}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </aside>
  );
}
