import 'whatwg-fetch';
import { act, render, waitFor } from '@testing-library/react';
import ViewerPage from '../app/(protected)/viewer/page';

const jumpToPageMock = jest.fn();

jest.mock('@react-pdf-viewer/core', () => {
  const React = require('react');
  return {
    Worker: ({ children }: { children: React.ReactNode }) => <div data-testid="worker">{children}</div>,
    Viewer: ({ fileUrl, onDocumentLoad }: { fileUrl: string; onDocumentLoad?: () => void }) => {
      React.useEffect(() => {
        onDocumentLoad?.();
      }, [fileUrl, onDocumentLoad]);
      return <div data-testid="viewer" data-url={fileUrl} />;
    },
  };
});

jest.mock('@react-pdf-viewer/page-navigation', () => ({
  pageNavigationPlugin: () => ({
    jumpToPage: jumpToPageMock,
  }),
}));

jest.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams('book=book-123'),
}));

describe('ViewerPage', () => {
  beforeEach(() => {
    jumpToPageMock.mockClear();
  });

  afterEach(() => {
    (global.fetch as jest.Mock | undefined)?.mockRestore?.();
    jest.restoreAllMocks();
  });

  function createJsonResponse(payload: unknown, init?: ResponseInit) {
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  }

  it('fetches viewer URL and jumps when hash changes', async () => {
    window.location.hash = '#page=5';

    const fetchMock = jest.spyOn(global, 'fetch');
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (url.endsWith('/book/book-123/page/5/view')) {
        return Promise.resolve(createJsonResponse({ url: 'signed-url-5' }));
      }
      if (url.endsWith('/book/book-123/page/7/view')) {
        return Promise.resolve(createJsonResponse({ url: 'signed-url-7' }));
      }
      return Promise.reject(new Error(`Unexpected fetch call: ${url}`));
    });

    render(<ViewerPage />);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/book/book-123/page/5/view', expect.any(Object));
    });

    await waitFor(() => {
      expect(jumpToPageMock).toHaveBeenCalledWith(4);
    });

    act(() => {
      window.location.hash = '#page=7';
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/book/book-123/page/7/view', expect.any(Object));
    });

    await waitFor(() => {
      expect(jumpToPageMock).toHaveBeenCalledWith(6);
    });
  });
});
