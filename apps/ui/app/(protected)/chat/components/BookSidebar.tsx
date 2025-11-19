'use client';

import { Check, Edit3, Loader2, MoreVertical, Search, Trash2, X } from 'lucide-react';
import { FormEvent, useMemo, useState } from 'react';

import type { BookRecord } from '../types';

export type BookMutation = { id: string; type: 'delete' | 'rename' } | null;

const EMPTY_STATE_MESSAGE = 'Upload a book to enable filtering.';

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
};

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
}: BookSidebarProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeMenuId, setActiveMenuId] = useState<string | null>(null);
  const [editingBookId, setEditingBookId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [renameError, setRenameError] = useState<string | null>(null);

  const filteredBooks = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    if (!query) {
      return books;
    }
    return books.filter((book) => book.title.toLowerCase().includes(query));
  }, [books, searchTerm]);

  const selectedCount = selectedBookIds.length;

  const handleRenameSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!editingBookId) {
      return;
    }
    const trimmed = renameDraft.trim();
    if (!trimmed) {
      setRenameError('Title cannot be empty.');
      return;
    }
    try {
      await onRenameBook(editingBookId, trimmed);
      setEditingBookId(null);
      setRenameDraft('');
      setRenameError(null);
    } catch (err) {
      console.warn('Failed to rename book', err);
      setRenameError('Unable to rename book. Please try again.');
    }
  };

  const handleDeleteBook = async (bookId: string, bookTitle: string) => {
    setActiveMenuId(null);
    const confirmed =
      typeof window === 'undefined'
        ? true
        : window.confirm(`Delete “${bookTitle}”? This cannot be undone.`);
    if (!confirmed) {
      return;
    }
    try {
      await onDeleteBook(bookId);
    } catch (err) {
      console.warn('Failed to delete book', err);
    }
  };

  const startRename = (book: BookRecord) => {
    setActiveMenuId(null);
    setEditingBookId(book.bookId);
    setRenameDraft(book.title);
    setRenameError(null);
  };

  return (
    <aside className="w-full rounded-2xl border border-slate-200 bg-white p-4 shadow-sm lg:max-w-xs xl:max-w-sm">
      <div className="flex items-center justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-slate-900">Indexed books</p>
          {selectedCount > 0 ? (
            <p className="text-xs text-slate-500">{selectedCount} selected</p>
          ) : null}
        </div>
        {selectedCount > 0 ? (
          <button
            type="button"
            className="text-sm font-medium text-slate-600 hover:text-slate-900"
            onClick={onClearSelection}
          >
            Clear
          </button>
        ) : null}
      </div>

      <label className="mt-4 flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-500 focus-within:border-slate-400">
        <Search className="h-4 w-4" aria-hidden="true" />
        <input
          type="search"
          placeholder="Search books"
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.target.value)}
          className="h-6 flex-1 border-0 bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
        />
      </label>

      {error ? (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700" role="alert">
          <p>{error}</p>
          {onRetry ? (
            <button
              type="button"
              className="mt-2 text-sm font-medium text-red-700 underline hover:text-red-900"
              onClick={onRetry}
            >
              Try again
            </button>
          ) : null}
        </div>
      ) : null}

      {isLoading ? (
        <div className="mt-4 flex items-center gap-2 text-sm text-slate-500">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          Loading books…
        </div>
      ) : null}

      {!isLoading && books.length === 0 ? (
        <p className="mt-4 text-sm text-slate-500">{EMPTY_STATE_MESSAGE}</p>
      ) : null}

      {!isLoading && books.length > 0 && filteredBooks.length === 0 ? (
        <p className="mt-4 text-sm text-slate-500">No books match “{searchTerm.trim()}”.</p>
      ) : null}

      <ul className="mt-4 space-y-3">
        {filteredBooks.map((book) => {
          const isSelected = selectedBookIds.includes(book.bookId);
          const isRenaming = pendingAction?.id === book.bookId && pendingAction.type === 'rename';
          const isDeleting = pendingAction?.id === book.bookId && pendingAction.type === 'delete';
          const author = book.metadata?.author;
          return (
            <li
              key={book.bookId}
              className={`rounded-xl border p-3 ${isSelected ? 'border-slate-900 bg-slate-50' : 'border-slate-200 bg-white'}`}
            >
              <div className="flex items-start gap-2">
                <button
                  type="button"
                  onClick={() => onToggleBook(book.bookId)}
                  className="flex flex-1 items-start gap-2 text-left"
                >
                  <div className="flex h-5 w-5 items-center justify-center rounded-full border border-slate-300">
                    {isSelected ? <Check className="h-4 w-4 text-slate-900" aria-hidden="true" /> : null}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-slate-900">{book.title}</p>
                    {typeof author === 'string' ? (
                      <p className="text-xs text-slate-500">{author}</p>
                    ) : null}
                    {typeof book.indexedChunks === 'number' ? (
                      <p className="text-xs text-slate-400">{book.indexedChunks} indexed chunks</p>
                    ) : null}
                  </div>
                </button>
                <div className="relative">
                  <button
                    type="button"
                    className="rounded-md p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-900"
                    aria-label={`Book actions for ${book.title}`}
                    onClick={() => setActiveMenuId((current) => (current === book.bookId ? null : book.bookId))}
                  >
                    <MoreVertical className="h-4 w-4" aria-hidden="true" />
                  </button>
                  {activeMenuId === book.bookId ? (
                    <div className="absolute right-0 z-10 mt-2 w-40 rounded-lg border border-slate-200 bg-white p-1 text-sm shadow-lg">
                      <button
                        type="button"
                        className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-slate-700 hover:bg-slate-100"
                        onClick={() => startRename(book)}
                      >
                        <Edit3 className="h-4 w-4" aria-hidden="true" /> Rename
                      </button>
                      <button
                        type="button"
                        className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-red-600 hover:bg-red-50"
                        onClick={() => handleDeleteBook(book.bookId, book.title)}
                      >
                        <Trash2 className="h-4 w-4" aria-hidden="true" /> Delete
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>

              {editingBookId === book.bookId ? (
                <form className="mt-3 space-y-2" onSubmit={handleRenameSubmit}>
                  <input
                    type="text"
                    value={renameDraft}
                    onChange={(event) => setRenameDraft(event.target.value)}
                    className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
                    disabled={isRenaming}
                  />
                  {renameError ? (
                    <p className="text-xs text-red-600" role="alert">
                      {renameError}
                    </p>
                  ) : null}
                  <div className="flex items-center gap-2">
                    <button
                      type="submit"
                      className="inline-flex items-center justify-center rounded-md border border-slate-200 px-3 py-1 text-sm font-medium transition hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-slate-400"
                      disabled={isRenaming}
                    >
                      {isRenaming ? (
                        <span className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Saving
                        </span>
                      ) : (
                        'Save'
                      )}
                    </button>
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 rounded-md px-3 py-1 text-sm text-slate-600 hover:text-slate-900"
                      onClick={() => {
                        setEditingBookId(null);
                        setRenameDraft('');
                        setRenameError(null);
                      }}
                      disabled={isRenaming}
                    >
                      <X className="h-4 w-4" aria-hidden="true" /> Cancel
                    </button>
                  </div>
                </form>
              ) : null}

              {isDeleting ? (
                <p className="mt-3 text-xs text-red-500">Deleting…</p>
              ) : null}
            </li>
          );
        })}
      </ul>

      {books.length > 0 && selectedCount > 1 ? (
        <p className="mt-4 text-xs text-slate-500">Searches will include all selected books.</p>
      ) : null}
    </aside>
  );
}
