import { act, render, screen } from '@testing-library/react';
import BookDetailsPage from '../app/(protected)/books/[bookId]/page';

describe('BookDetailsPage', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.clearAllMocks();
  });

  it('loads and displays book details', async () => {
    const fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        book: {
          id: 'book-1',
          title: 'First Book',
          status: 'indexed',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-02T00:00:00Z',
          metadata: { author: 'Author 1' },
          indexed_chunks: 42,
          file_hash: 'hash-123',
        },
      }),
    } as Response);

    global.fetch = fetchMock as unknown as typeof fetch;

    await act(async () => {
      render(<BookDetailsPage params={{ bookId: 'book-1' }} />);
    });

    expect(await screen.findByText('First Book')).toBeInTheDocument();
    expect(screen.getByText(/book id: book-1/i)).toBeInTheDocument();
    expect(screen.getByText('indexed')).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith('/api/books/book-1', expect.objectContaining({ method: 'GET' }));
  });

  it('shows an error when the API fails', async () => {
    const fetchMock = jest.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: 'Book not found' }),
      text: async () => 'Book not found',
    } as Response);

    global.fetch = fetchMock as unknown as typeof fetch;

    await act(async () => {
      render(<BookDetailsPage params={{ bookId: 'missing-book' }} />);
    });

    expect(await screen.findByRole('alert')).toHaveTextContent('Book not found');
  });
});
