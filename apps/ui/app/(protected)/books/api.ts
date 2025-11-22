export type BookMetadata = Record<string, unknown> | null | undefined;

export type BookRecord = {
  id: string;
  db_id?: number | string | null;
  vector_id?: string | null;
  is_indexed?: boolean | null;
  title: string;
  description?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  status?: string | null;
  version?: number | string | null;
  file_hash?: string | null;
  indexed_chunks?: number | string | null;
  metadata?: BookMetadata;
};

export type BookListResponse = {
  total: number;
  limit: number;
  offset: number;
  books: BookRecord[];
};

export type BookResponse = {
  book: BookRecord;
};

const BOOKS_API_BASE_PATH = '/api/books';

function buildQuery(params: Record<string, number | string | undefined>): string {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined) {
      return;
    }
    query.set(key, String(value));
  });
  const serialized = query.toString();
  return serialized ? `?${serialized}` : '';
}

async function parseError(response: Response, fallback: string): Promise<Error> {
  try {
    const payload = await response.json();
    if (payload && typeof payload === 'object') {
      const detail = (payload as { detail?: unknown }).detail;
      if (typeof detail === 'string' && detail.trim()) {
        return new Error(detail.trim());
      }
      if ('message' in (payload as Record<string, unknown>)) {
        const message = (payload as Record<string, unknown>).message;
        if (typeof message === 'string' && message.trim()) {
          return new Error(message.trim());
        }
      }
    }
  } catch (error) {
    // Ignore JSON parsing issues and fall back to text parsing below.
  }

  try {
    const text = await response.text();
    if (text && text.trim()) {
      return new Error(text.trim());
    }
  } catch (error) {
    // Ignore secondary parsing errors and return fallback.
  }

  return new Error(fallback);
}

export async function listBooks({
  limit,
  offset,
}: {
  limit: number;
  offset: number;
}): Promise<BookListResponse> {
  const response = await fetch(`${BOOKS_API_BASE_PATH}${buildQuery({ limit, offset })}`, {
    method: 'GET',
    credentials: 'include',
  });

  if (!response.ok) {
    throw await parseError(response, 'Failed to load books.');
  }

  const payload = (await response.json()) as Partial<BookListResponse>;
  return {
    total: typeof payload.total === 'number' ? payload.total : 0,
    limit: typeof payload.limit === 'number' ? payload.limit : limit,
    offset: typeof payload.offset === 'number' ? payload.offset : offset,
    books: Array.isArray(payload.books) ? (payload.books as BookRecord[]) : [],
  };
}

export async function getBook(bookId: string): Promise<BookRecord> {
  const response = await fetch(`${BOOKS_API_BASE_PATH}/${encodeURIComponent(bookId)}`, {
    method: 'GET',
    credentials: 'include',
  });

  if (!response.ok) {
    throw await parseError(response, 'Failed to load book details.');
  }

  const payload = (await response.json()) as BookResponse;
  if (!payload || typeof payload !== 'object' || !payload.book) {
    throw new Error('Book response missing payload.');
  }
  return payload.book;
}

export async function renameBook(
  bookId: string,
  {
    title,
    description,
  }: {
    title: string;
    description?: string | null;
  },
  csrfToken: string | null,
): Promise<BookRecord> {
  const response = await fetch(`${BOOKS_API_BASE_PATH}/${encodeURIComponent(bookId)}/rename`, {
    method: 'PATCH',
    credentials: 'include',
    headers: {
      'content-type': 'application/json',
      ...(csrfToken ? { 'x-csrf-token': csrfToken } : {}),
    },
    body: JSON.stringify({ title, description }),
  });

  if (!response.ok) {
    throw await parseError(response, 'Failed to rename book.');
  }

  const payload = (await response.json()) as BookResponse;
  if (!payload || typeof payload !== 'object' || !payload.book) {
    throw new Error('Rename response missing updated book.');
  }
  return payload.book;
}

export async function deleteBook(bookId: string, csrfToken: string | null): Promise<void> {
  const response = await fetch(`${BOOKS_API_BASE_PATH}/${encodeURIComponent(bookId)}`, {
    method: 'DELETE',
    credentials: 'include',
    headers: csrfToken ? { 'x-csrf-token': csrfToken } : undefined,
  });

  if (!response.ok) {
    throw await parseError(response, 'Failed to delete book.');
  }
}
