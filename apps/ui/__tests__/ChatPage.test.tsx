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
  push(chunk: Record<string, unknown>): void;
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
    push(chunk: Record<string, unknown>) {
      queue.push(encoder.encode(`${JSON.stringify(chunk)}\n`));
      flush();
    },
    close() {
      closed = true;
      flush();
    },
    response: {
      ok: true,
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

  it('streams assistant tokens and renders citations as links', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const stream = createStreamController();
    const postBodies: Array<string | undefined> = [];
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

      if (url === '/bookmarks' && method === 'GET') {
        return Promise.resolve(createJsonResponse({ bookmarks: [] }));
      }

      if (url.startsWith('/sessions/1/messages') && method === 'GET') {
        return Promise.resolve(createJsonResponse({ messages: [] }));
      }

      if (url.startsWith('/sessions/1/messages') && method === 'POST') {
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
              session_id: 1,
              created_at: '2024-01-01T02:00:00Z',
              tag: body.tag ?? null,
              session: { id: 1, title: 'Mock Session' },
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
          createJsonResponse({
            sessions: [
              {
                id: '1',
                title: 'Mock Session',
                last_activity: '2024-01-01T00:00:00Z',
              },
            ],
          }),
        );
      }

      if (url.startsWith('/sessions') && method === 'POST') {
        return Promise.resolve(createJsonResponse({ id: '2', title: 'New Session' }));
      }

      if (url.startsWith('/sessions/') && method === 'DELETE') {
        return Promise.resolve(new Response(null, { status: 204 }));
      }

      if (url.startsWith('/query')) {
        return Promise.resolve(stream.response);
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

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/sessions/1/messages'),
      expect.objectContaining({ method: 'POST' }),
    );

    const assistMessages = () => screen.getAllByTestId('chat-message-assistant');

    await act(async () => {
      stream.push({ delta: 'Partial answer ' });
    });

    await waitFor(() => {
      const assistant = assistMessages().at(-1);
      expect(assistant).toHaveTextContent('Partial answer');
    });

    await act(async () => {
      stream.push({
        answer: 'Partial answer with citation',
        contexts: [{ book_id: 'book1', page_start: 12, page_end: 14 }],
      });
      stream.close();
    });

    await waitFor(() => {
      const assistant = assistMessages().at(-1);
      expect(assistant).toHaveTextContent('Partial answer with citation');
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

    const citationLink = await screen.findByRole('link', { name: '[book1:12-14]' });
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

        if (url === '/bookmarks' && method === 'GET') {
          return Promise.resolve(createJsonResponse({ bookmarks: [] }));
        }

        if (url.startsWith('/sessions/1/messages') && method === 'GET') {
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
            createJsonResponse({
              sessions: [
                {
                  id: '1',
                  title: 'Mock Session',
                },
              ],
            }),
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

        if (url === '/bookmarks' && method === 'GET') {
          return Promise.resolve(createJsonResponse({ bookmarks: [] }));
        }

        if (url.startsWith('/sessions/1/messages') && method === 'GET') {
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
            createJsonResponse({
              sessions: [
                {
                  id: '1',
                  title: 'Mock Session',
                },
              ],
            }),
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
