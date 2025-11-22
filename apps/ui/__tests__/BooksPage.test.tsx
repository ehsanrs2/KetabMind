import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import BooksPage from '../app/(protected)/books/page';

jest.mock('../app/context/AuthContext', () => ({
  useAuth: () => ({ csrfToken: 'csrf-token' }),
}));

describe('BooksPage', () => {
  const originalFetch = global.fetch;
  const originalConsoleError = console.error;

  async function renderPage() {
    await act(async () => {
      render(<BooksPage />);
    });
    await act(async () => {});
  }

  beforeAll(() => {
    jest.spyOn(console, 'error').mockImplementation((message?: unknown, ...rest: unknown[]) => {
      if (typeof message === 'string' && message.includes('not wrapped in act')) {
        return;
      }
      originalConsoleError(message as string, ...rest);
    });
  });

  beforeEach(() => {
    global.fetch = originalFetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
    jest.clearAllMocks();
  });

  afterAll(() => {
    (console.error as jest.Mock).mockRestore();
  });

  it('renders books from the API', async () => {
    const fetchMock = jest
      .fn()
      .mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({
          total: 2,
          limit: 10,
          offset: 0,
          books: [
            {
              id: 'book-1',
              title: 'First Book',
              metadata: { author: 'Author 1' },
              created_at: '2024-01-01T00:00:00Z',
              status: 'indexed',
            },
            {
              id: 'book-2',
              title: 'Second Book',
              metadata: { author: 'Author 2' },
              created_at: '2024-02-01T00:00:00Z',
              status: 'pending',
            },
          ],
        }),
      } as Response);

    global.fetch = fetchMock as unknown as typeof fetch;

    await renderPage();

    expect(await screen.findByText('First Book')).toBeInTheDocument();
    expect(screen.getByText('Second Book')).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/books?limit=10&offset=0',
      expect.objectContaining({ method: 'GET', credentials: 'include' }),
    );
  });

  it('renames a book and updates the table', async () => {
    const fetchMock = jest
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          total: 1,
          limit: 10,
          offset: 0,
          books: [
            {
              id: 'book-1',
              title: 'First Book',
              metadata: { author: 'Author 1' },
              created_at: '2024-01-01T00:00:00Z',
              status: 'indexed',
            },
          ],
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          book: {
            id: 'book-1',
            title: 'Renamed Book',
            metadata: { author: 'Author 1' },
            created_at: '2024-01-01T00:00:00Z',
            status: 'indexed',
          },
        }),
      } as Response);

    global.fetch = fetchMock as unknown as typeof fetch;

    const user = userEvent.setup();
    await renderPage();

    expect(await screen.findByText('First Book')).toBeInTheDocument();

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /rename/i }));
    });
    const titleInput = screen.getByLabelText(/title/i);
    await act(async () => {
      await user.clear(titleInput);
    });
    await act(async () => {
      await user.type(titleInput, 'Renamed Book');
    });

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /save changes/i }));
    });

    await screen.findByText('Renamed Book');

    const renameCall = fetchMock.mock.calls[1];
    expect(renameCall[0]).toBe('/api/books/book-1/rename');
    expect(renameCall[1]).toMatchObject({
      method: 'PATCH',
      credentials: 'include',
      headers: expect.objectContaining({ 'x-csrf-token': 'csrf-token' }),
    });
  });

  it('deletes a book after confirmation', async () => {
    const fetchMock = jest
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          total: 1,
          limit: 10,
          offset: 0,
          books: [
            {
              id: 'book-1',
              title: 'First Book',
              metadata: { author: 'Author 1' },
              created_at: '2024-01-01T00:00:00Z',
              status: 'indexed',
            },
          ],
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        status: 204,
        json: async () => ({}),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ total: 0, limit: 10, offset: 0, books: [] }),
      } as Response);

    global.fetch = fetchMock as unknown as typeof fetch;

    const user = userEvent.setup();
    await renderPage();

    expect(await screen.findByText('First Book')).toBeInTheDocument();

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /delete/i }));
    });
    await act(async () => {
      await user.click(screen.getByRole('button', { name: /delete book/i }));
    });

    await waitFor(() => {
      expect(screen.queryByText('First Book')).not.toBeInTheDocument();
    });

    const deleteCall = fetchMock.mock.calls[1];
    expect(deleteCall[0]).toBe('/api/books/book-1');
    expect(deleteCall[1]).toMatchObject({
      method: 'DELETE',
      credentials: 'include',
      headers: { 'x-csrf-token': 'csrf-token' },
    });
  });
});
