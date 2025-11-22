import 'whatwg-fetch';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TextDecoder as NodeTextDecoder, TextEncoder as NodeTextEncoder } from 'util';
import ChatPage from '../app/(protected)/chat/page';

jest.mock('../app/context/AuthContext', () => ({
  useAuth: () => ({ csrfToken: 'csrf-token' }),
}));

if (typeof globalThis.TextDecoder === 'undefined') {
  globalThis.TextDecoder = NodeTextDecoder as unknown as typeof globalThis.TextDecoder;
}

type StreamController = {
  pushJson(chunk: Record<string, unknown>): void;
  pushDone(): void;
  close(): void;
  response: Response;
};

function createStreamController(): StreamController {
  const encoder = new NodeTextEncoder();
  const queue: Uint8Array[] = [];
  let closed = false;
  const pending: Array<(value: ReadableStreamReadResult<Uint8Array>) => void> = [];

  const flush = () => {
    while (pending.length > 0) {
      if (queue.length > 0) {
        const value = queue.shift()!;
        const resolve = pending.shift()!;
        resolve({ done: false, value });
      } else if (closed) {
        const resolve = pending.shift()!;
        resolve({ done: true, value: undefined });
      } else {
        break;
      }
    }
  };

  const body = {
    getReader() {
      return {
        read() {
          return new Promise<ReadableStreamReadResult<Uint8Array>>((resolve) => {
            pending.push(resolve);
            flush();
          });
        },
      };
    },
  };

  return {
    pushJson(chunk: Record<string, unknown>) {
      queue.push(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
      flush();
    },
    pushDone() {
      queue.push(encoder.encode('data: [DONE]\n\n'));
      flush();
    },
    close() {
      queue.push(encoder.encode('data: [DONE]\n\n'));
      closed = true;
      flush();
    },
    response: {
      ok: true,
      status: 200,
      body,
    } as unknown as Response,
  };
}

describe('ChatPage', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  function createJsonResponse(payload: unknown, init?: ResponseInit) {
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  }

  const MOCK_SESSION_ID = '11111111-1111-4111-8111-111111111111';
  const NEW_SESSION_ID = '22222222-2222-4222-8222-222222222222';
  const EXPECTED_STREAM_MODEL =
    ((candidates: Array<string | undefined>) => {
      for (const candidate of candidates) {
        if (typeof candidate === 'string') {
          const normalized = candidate.trim().toLowerCase();
          if (normalized === 'ollama' || normalized === 'openai') {
            return normalized;
          }
        }
      }
      return 'ollama';
    })([
      process.env.NEXT_PUBLIC_STREAM_MODEL,
      process.env.NEXT_PUBLIC_LLM_MODEL,
      process.env.NEXT_PUBLIC_LLM_BACKEND,
    ]);

  it('streams assistant tokens and renders citations as links', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const stream = createStreamController();
    const postBodies: Array<string | undefined> = [];
    const streamBodies: Array<unknown> = [];
    let assistantMessageIdCounter = 0;
    let bookmarkIdCounter = 0;

    const fetchMock = jest.spyOn(globalThis, 'fetch');
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const raw =
        typeof input === 'string'
          ? input
          : input instanceof URL
          ? input.toString()
          : input.url;
      let url = raw;
      if (/^https?:/i.test(raw)) {
        const parsed = new URL(raw);
        url = `${parsed.pathname}${parsed.search}`;
      }
      const method = (init?.method ?? 'GET').toUpperCase();

      if (url.startsWith('/api/books') && method === 'GET') {
        return Promise.resolve(
          createJsonResponse({
            books: [
              { book_id: 'book1', title: 'Physics 101' },
              { book_id: 'book2', title: 'Chemistry Basics' },
            ],
          }),
        );
      }

      if (url === '/bookmarks' && method === 'GET') {
        return Promise.resolve(createJsonResponse({ bookmarks: [] }));
      }

      const messagesBaseUrl = `/sessions/${MOCK_SESSION_ID}/messages`;

      if (url === messagesBaseUrl && method === 'GET') {
        return Promise.resolve(createJsonResponse({ messages: [] }));
      }

      if (url === messagesBaseUrl && method === 'POST') {
        postBodies.push(typeof init?.body === 'string' ? init?.body : undefined);
        const payload = typeof init?.body === 'string' ? JSON.parse(init.body) : {};
        const role = payload?.role;
        if (role === 'assistant') {
          assistantMessageIdCounter += 1;
          return Promise.resolve(
            createJsonResponse({
              message: {
                id: `assistant-${assistantMessageIdCounter}`,
                role: 'assistant',
                content: payload?.content,
                created_at: '2024-01-01T01:05:00Z',
              },
            }),
          );
        }
        return Promise.resolve(
          createJsonResponse({
            message: {
              id: `user-${postBodies.length}`,
              role: role ?? 'user',
              content: payload?.content,
              created_at: '2024-01-01T01:00:00Z',
            },
          }),
        );
      }

      if (url === '/bookmarks' && method === 'POST') {
        const body = typeof init?.body === 'string' ? JSON.parse(init.body) : {};
        bookmarkIdCounter += 1;
        return Promise.resolve(
          createJsonResponse({
            bookmark: {
              id: bookmarkIdCounter,
              session_id: MOCK_SESSION_ID,
              created_at: '2024-01-01T02:00:00Z',
              tag: body.tag ?? null,
              session: { id: MOCK_SESSION_ID, title: 'Mock Session' },
              message: {
                id: body.message_id,
                role: 'assistant',
                content: 'Partial answer with citation',
                citations: ['[book1:12-14]'],
                created_at: '2024-01-01T01:05:00Z',
              },
            },
          }),
        );
      }

      if (url.startsWith('/sessions') && method === 'GET') {
        return Promise.resolve(
          createJsonResponse([
            {
              id: MOCK_SESSION_ID,
              title: 'Mock Session',
              last_activity: '2024-01-01T00:00:00Z',
            },
          ]),
        );
      }

      const expectedStreamUrl = `/sessions/${MOCK_SESSION_ID}/messages/stream`;

      if (url.startsWith(`/sessions/${MOCK_SESSION_ID}/messages/stream`) && method === 'POST') {
        const body = typeof init?.body === 'string' ? JSON.parse(init.body) : {};
        streamBodies.push(body);
        return Promise.resolve(stream.response);
      }

      if (url.startsWith('/sessions') && method === 'POST') {
        return Promise.resolve(createJsonResponse({ id: NEW_SESSION_ID, title: 'New Session' }));
      }

      if (url.startsWith('/sessions/') && method === 'DELETE') {
        return Promise.resolve(new Response(null, { status: 204 }));
      }

      return Promise.reject(new Error(`Unexpected fetch call: ${url}`));
    });

    render(<ChatPage />);

    expect(await screen.findByRole('heading', { level: 2, name: 'Mock Session' })).toBeInTheDocument();

    const user = userEvent.setup();
    const textarea = screen.getByLabelText('Message') as HTMLTextAreaElement;
    await act(async () => {
      await user.type(textarea, 'Where is the library?');
    });

    const sendButton = screen.getByRole('button', { name: 'Send' });
    await act(async () => {
      await user.click(sendButton);
    });

    const streamCall = fetchMock.mock.calls.find(([requestedUrl]) =>
      typeof requestedUrl === 'string'
        ? requestedUrl.includes(`/sessions/${MOCK_SESSION_ID}/messages/stream`)
        : false,
    );

    expect(streamCall).toBeDefined();
    if (streamCall) {
      const [, init] = streamCall;
      expect(init).toMatchObject({
        method: 'POST',
        body: JSON.stringify({
          content: 'Where is the library?',
          context: true,
          model: EXPECTED_STREAM_MODEL,
        }),
      });
      expect(init?.headers).toMatchObject({ Accept: 'text/event-stream' });
    }

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining(`/sessions/${MOCK_SESSION_ID}/messages`),
      expect.objectContaining({ method: 'POST' }),
    );

    await waitFor(() => {
      expect(streamBodies.length).toBeGreaterThan(0);
    });

    const assistMessages = () => screen.getAllByTestId('chat-message-assistant');

    await act(async () => {
      stream.pushJson({ delta: 'Partial answer ' });
    });

    await waitFor(() => {
      const assistant = assistMessages().at(-1);
      expect(assistant).toHaveTextContent('Partial answer');
    });

    expect(streamBodies.at(0)).toMatchObject({
      content: 'Where is the library?',
      context: true,
      model: EXPECTED_STREAM_MODEL,
    });

    await act(async () => {
      stream.pushJson({
        answer: 'Partial answer with citation',
        contexts: [{ book_id: 'book1', page_start: 12, page_end: 14 }],
        meta: { coverage: 0.75, confidence: 0.64 },
      });
      stream.pushDone();
      stream.close();
    });

    await waitFor(() => {
      const assistant = assistMessages().at(-1);
      expect(assistant).toHaveTextContent('Partial answer with citation');
    });

    await waitFor(() => {
      expect(screen.getByText('Coverage: 75%')).toBeInTheDocument();
      expect(screen.getByText('Confidence: 64%')).toBeInTheDocument();
    });

    const assistant = assistMessages().at(-1) as HTMLElement;
    const assistantScrollSpy = jest.fn();
    assistant.scrollIntoView = assistantScrollSpy;

    const bookmarkButton = await screen.findByRole('button', { name: 'Bookmark' });
    const bookmarkTagInput = screen.getByPlaceholderText('Tag new bookmarks (optional)');

    await act(async () => {
      await user.type(bookmarkTagInput, 'math');
    });

    await act(async () => {
      await user.click(bookmarkButton);
    });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/bookmarks',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('assistant-'),
        }),
      );
      expect(fetchMock).toHaveBeenCalledWith(
        '/bookmarks',
        expect.objectContaining({
          body: expect.stringContaining('"tag":"math"'),
        }),
      );
      expect(screen.getByRole('button', { name: 'Bookmarked' })).toBeDisabled();
    });

    const bookmarkItem = await screen.findByTestId('bookmark-item-1');
    expect(bookmarkItem).toBeInTheDocument();

    await act(async () => {
      await user.click(bookmarkItem);
    });

    await waitFor(() => {
      expect(assistantScrollSpy).toHaveBeenCalled();
    });

    const citationLink = await screen.findByRole('link', { name: /Physics 101/i });
    expect(citationLink).toHaveAttribute('href', '/viewer?book=book1#page=12');

    await waitFor(() => {
      const assistantPersistCall = postBodies
        .map((body) => (body ? JSON.parse(body) : null))
        .find((payload) => payload?.role === 'assistant');
      expect(assistantPersistCall).toMatchObject({
        role: 'assistant',
        content: 'Partial answer with citation',
        citations: ['[book1:12-14]'],
      });
    });

    warnSpy.mockRestore();
  });

  it('allows searching within indexed books', async () => {
    const fetchMock = jest.spyOn(globalThis, 'fetch');
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const raw =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;
      let url = raw;
      if (/^https?:/i.test(raw)) {
        const parsed = new URL(raw);
        url = `${parsed.pathname}${parsed.search}`;
      }
      const method = (init?.method ?? 'GET').toUpperCase();

      if (url.startsWith('/api/books') && method === 'GET') {
        return Promise.resolve(
          createJsonResponse({
            books: [
              { book_id: 'book-1', title: 'Linear Algebra' },
              { book_id: 'book-2', title: 'Advanced Calculus' },
            ],
          }),
        );
      }

      if (url === '/bookmarks' && method === 'GET') {
        return Promise.resolve(createJsonResponse({ bookmarks: [] }));
      }

      if (url.startsWith(`/sessions/${MOCK_SESSION_ID}/messages`) && method === 'GET') {
        return Promise.resolve(createJsonResponse({ messages: [] }));
      }

      if (url.startsWith('/sessions') && method === 'GET') {
        return Promise.resolve(
          createJsonResponse([
            {
              id: MOCK_SESSION_ID,
              title: 'Mock Session',
            },
          ]),
        );
      }

      if (url.startsWith('/search') && method === 'GET') {
        return Promise.resolve(
          createJsonResponse({
            results: [
              {
                book_id: 'book-2',
                page_num: 42,
                text: 'The definite integral builds on the notion of limits and areas.',
                score: 0.87,
              },
            ],
          }),
        );
      }

      return Promise.reject(new Error(`Unexpected fetch call: ${url}`));
    });

    const user = userEvent.setup();
    render(<ChatPage />);

    const calculusChip = await screen.findByRole('button', { name: 'Advanced Calculus' });
    await act(async () => {
      await user.click(calculusChip);
    });

    const searchInput = screen.getByLabelText('Search across indexed books');
    await act(async () => {
      await user.type(searchInput, 'integral');
    });

    const searchButton = screen.getByRole('button', { name: 'Search' });
    await act(async () => {
      await user.click(searchButton);
    });

    await waitFor(() => {
      const searchCall = fetchMock.mock.calls.find(
        ([request]) => typeof request === 'string' && request.startsWith('/search?'),
      );
      expect(searchCall?.[0]).toContain('query=integral');
      expect(searchCall?.[0]).toContain('book_id=book-2');
    });

    expect(
      await screen.findByText(/The definite integral builds on the notion of limits and areas./i),
    ).toBeInTheDocument();
    expect(screen.getByText(/Score 0\.870/i)).toBeInTheDocument();

    const viewerLink = screen.getByRole('link', { name: 'Open in viewer' });
    expect(viewerLink).toHaveAttribute('href', '/viewer?book=book-2#page=42');
  });

  it('renames a session inline', async () => {
    const fetchMock = jest.spyOn(globalThis, 'fetch');
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const raw =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;
      let url = raw;
      if (/^https?:/i.test(raw)) {
        const parsed = new URL(raw);
        url = `${parsed.pathname}${parsed.search}`;
      }
      const method = (init?.method ?? 'GET').toUpperCase();

      if (url.startsWith('/api/books') && method === 'GET') {
        return Promise.resolve(createJsonResponse({ books: [] }));
      }

      if (url === '/bookmarks' && method === 'GET') {
        return Promise.resolve(createJsonResponse({ bookmarks: [] }));
      }

      if (url.startsWith(`/sessions/${MOCK_SESSION_ID}/messages`) && method === 'GET') {
        return Promise.resolve(createJsonResponse({ messages: [] }));
      }

      if (url.startsWith('/sessions') && method === 'GET') {
        return Promise.resolve(
          createJsonResponse([
            {
              id: MOCK_SESSION_ID,
              title: 'Untitled session',
              last_activity: '2024-01-01T00:00:00Z',
            },
          ]),
        );
      }

      if (url === `/sessions/${MOCK_SESSION_ID}` && method === 'PATCH') {
        return Promise.resolve(
          createJsonResponse({
            session: {
              id: MOCK_SESSION_ID,
              title: 'My Topic Notes',
              last_activity: '2024-01-01T00:05:00Z',
              updated_at: '2024-01-01T00:05:00Z',
            },
          }),
        );
      }

      return Promise.reject(new Error(`Unexpected fetch call: ${url}`));
    });

    const user = userEvent.setup();
    render(<ChatPage />);

    expect(await screen.findByRole('heading', { level: 2, name: 'Untitled session' })).toBeInTheDocument();

    await act(async () => {
      await user.click(screen.getByRole('button', { name: 'Rename' }));
    });

    const renameInput = screen.getByLabelText('Rename session');
    await act(async () => {
      await user.clear(renameInput);
      await user.type(renameInput, 'My Topic Notes');
    });

    await act(async () => {
      await user.click(screen.getByRole('button', { name: 'Save' }));
    });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        `/sessions/${MOCK_SESSION_ID}`,
        expect.objectContaining({
          method: 'PATCH',
          body: JSON.stringify({ title: 'My Topic Notes' }),
        }),
      );
    });

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 2, name: 'My Topic Notes' })).toBeInTheDocument();
    });
  });

  it('applies RTL styles for Persian assistant messages', async () => {
    jest
      .spyOn(globalThis, 'fetch')
      .mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
        const raw =
          typeof input === 'string'
            ? input
            : input instanceof URL
              ? input.toString()
              : input.url;
        let url = raw;
        if (/^https?:/i.test(raw)) {
          const parsed = new URL(raw);
          url = `${parsed.pathname}${parsed.search}`;
        }
        const method = (init?.method ?? 'GET').toUpperCase();

        if (url.startsWith('/api/books') && method === 'GET') {
          return Promise.resolve(createJsonResponse({ books: [] }));
        }

        if (url === '/bookmarks' && method === 'GET') {
          return Promise.resolve(createJsonResponse({ bookmarks: [] }));
        }

        if (url.startsWith(`/sessions/${MOCK_SESSION_ID}/messages`) && method === 'GET') {
          return Promise.resolve(
            createJsonResponse({
              messages: [
                {
                  id: 'assistant-rtl',
                  role: 'assistant',
                  content: 'پاسخ فارسی',
                  meta: { lang: 'fa' },
                },
              ],
            }),
          );
        }

        if (url.startsWith('/sessions') && method === 'GET') {
          return Promise.resolve(
            createJsonResponse([
              {
                id: MOCK_SESSION_ID,
                title: 'Mock Session',
              },
            ]),
          );
        }

        return Promise.reject(new Error(`Unexpected fetch call: ${url}`));
      });

    render(<ChatPage />);

    const message = await screen.findByTestId('chat-message-assistant');
    expect(message).toHaveClass('rtl');
    expect(message).toMatchSnapshot('persian-assistant-message');
  });

  it('keeps LTR layout for non-Persian assistant messages', async () => {
    jest
      .spyOn(globalThis, 'fetch')
      .mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
        const raw =
          typeof input === 'string'
            ? input
            : input instanceof URL
              ? input.toString()
              : input.url;
        let url = raw;
        if (/^https?:/i.test(raw)) {
          const parsed = new URL(raw);
          url = `${parsed.pathname}${parsed.search}`;
        }
        const method = (init?.method ?? 'GET').toUpperCase();

        if (url.startsWith('/api/books') && method === 'GET') {
          return Promise.resolve(createJsonResponse({ books: [] }));
        }

        if (url === '/bookmarks' && method === 'GET') {
          return Promise.resolve(createJsonResponse({ bookmarks: [] }));
        }

        if (url.startsWith(`/sessions/${MOCK_SESSION_ID}/messages`) && method === 'GET') {
          return Promise.resolve(
            createJsonResponse({
              messages: [
                {
                  id: 'assistant-default',
                  role: 'assistant',
                  content: 'English answer',
                  meta: { lang: 'en' },
                },
              ],
            }),
          );
        }

        if (url.startsWith('/sessions') && method === 'GET') {
          return Promise.resolve(
            createJsonResponse([
              {
                id: MOCK_SESSION_ID,
                title: 'Mock Session',
              },
            ]),
          );
        }

        return Promise.reject(new Error(`Unexpected fetch call: ${url}`));
      });

    render(<ChatPage />);

    const message = await screen.findByTestId('chat-message-assistant');
    expect(message).not.toHaveClass('rtl');
    expect(message).toMatchSnapshot('non-persian-assistant-message');
  });
});
