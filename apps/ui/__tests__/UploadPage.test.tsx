import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UploadPage from '../app/(protected)/upload/page';

type ProgressHandler = ((event: ProgressEvent<EventTarget>) => void) | null;

type FormInput = {
  title: string;
  author: string;
  year: string;
  subject: string;
  file: File;
};

class MockXMLHttpRequest {
  static instances: MockXMLHttpRequest[] = [];

  public upload: { onprogress: ProgressHandler };

  public responseType = '';

  public status = 0;

  public response: unknown = null;

  public responseText = '';

  public onload: ((this: XMLHttpRequest, ev: ProgressEvent<EventTarget>) => unknown) | null = null;

  public onerror: ((this: XMLHttpRequest, ev: ProgressEvent<EventTarget>) => unknown) | null = null;

  public open = jest.fn();

  public send = jest.fn();

  public setRequestHeader = jest.fn();

  constructor() {
    MockXMLHttpRequest.instances.push(this);
    this.upload = { onprogress: null };
  }

  static reset() {
    MockXMLHttpRequest.instances = [];
  }

  simulateProgress(loaded: number, total: number) {
    const handler = this.upload.onprogress;
    if (handler) {
      handler({ lengthComputable: true, loaded, total } as ProgressEvent<EventTarget>);
    }
  }

  simulateResponse(status: number, body: unknown) {
    this.status = status;
    if (this.responseType === 'json') {
      this.response = body;
    }
    this.responseText = typeof body === 'string' ? body : JSON.stringify(body);
    if (this.onload) {
      this.onload({} as ProgressEvent<EventTarget>);
    }
  }

  simulateNetworkError() {
    if (this.onerror) {
      this.onerror({} as ProgressEvent<EventTarget>);
    }
  }
}

async function fillForm(user: ReturnType<typeof userEvent.setup>, data: FormInput) {
  await act(async () => {
    await user.type(screen.getByLabelText(/title/i), data.title);
    await user.type(screen.getByLabelText(/author/i), data.author);
    await user.type(screen.getByLabelText(/year/i), data.year);
    await user.type(screen.getByLabelText(/subject/i), data.subject);
    await user.upload(screen.getByLabelText(/file/i) as HTMLInputElement, data.file);
  });
}

describe('UploadPage', () => {
  const originalXhr = global.XMLHttpRequest;
  const originalFetch = global.fetch;

  beforeEach(() => {
    MockXMLHttpRequest.reset();
    global.XMLHttpRequest = jest.fn(() => new MockXMLHttpRequest()) as unknown as typeof XMLHttpRequest;
  });

  afterEach(() => {
    global.XMLHttpRequest = originalXhr;
    global.fetch = originalFetch;
    jest.clearAllMocks();
  });

  it('uploads a file and allows indexing', async () => {
    const user = userEvent.setup();
    render(<UploadPage />);

    await fillForm(user, {
      title: 'Test Book',
      author: 'Jane Doe',
      year: '2024',
      subject: 'History',
      file: new File(['content'], 'test.pdf', { type: 'application/pdf' }),
    });

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /upload/i }));
    });

    const xhr = MockXMLHttpRequest.instances[0];
    expect(xhr).toBeDefined();

    act(() => {
      xhr.simulateProgress(50, 100);
    });

    expect(screen.getByText(/upload progress: 50%/i)).toBeInTheDocument();

    act(() => {
      xhr.simulateResponse(200, {
        book_id: 'book-123',
        version: 1,
        file_hash: 'hash-abc',
        message: 'Upload completed successfully.',
      });
    });

    await screen.findByText(/upload completed successfully/i);

    await waitFor(() => {
      expect(screen.queryByText(/upload progress:/i)).not.toBeInTheDocument();
    });

    expect(screen.getByTestId('upload-result-book-id')).toHaveTextContent('book-123');
    expect(screen.getByTestId('upload-result-version')).toHaveTextContent('1');
    expect(screen.getByTestId('upload-result-file-hash')).toHaveTextContent('hash-abc');

    const fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify({ message: 'Indexing started.' })),
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /index now/i }));
    });

    expect(fetchMock).toHaveBeenCalledWith(
      '/index',
      expect.objectContaining({
        method: 'POST',
      }),
    );

    await screen.findByText(/indexing started/i);
  });

  it('shows deduplication message when book already indexed', async () => {
    const user = userEvent.setup();
    render(<UploadPage />);

    await fillForm(user, {
      title: 'Existing Book',
      author: 'John Doe',
      year: '2023',
      subject: 'Science',
      file: new File(['content'], 'existing.pdf', { type: 'application/pdf' }),
    });

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /upload/i }));
    });

    const xhr = MockXMLHttpRequest.instances[0];
    expect(xhr).toBeDefined();

    act(() => {
      xhr.simulateResponse(200, {
        book_id: 'book-duplicate',
        version: 2,
        file_hash: 'hash-duplicate',
        already_indexed: true,
        message: 'Already indexed',
      });
    });

    await screen.findByText(/this file was already indexed\. you can trigger indexing again if needed\./i);
  });

  it('shows an error when upload fails', async () => {
    const user = userEvent.setup();
    render(<UploadPage />);

    await fillForm(user, {
      title: 'Broken Book',
      author: 'Jane Smith',
      year: '2022',
      subject: 'Poetry',
      file: new File(['content'], 'broken.pdf', { type: 'application/pdf' }),
    });

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /upload/i }));
    });

    const xhr = MockXMLHttpRequest.instances[0];
    expect(xhr).toBeDefined();

    act(() => {
      xhr.simulateResponse(500, { detail: 'Internal server error' });
    });

    await screen.findByText(/internal server error/i);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /upload/i })).not.toBeDisabled();
    });

    expect(screen.queryByTestId('upload-result-book-id')).not.toBeInTheDocument();
  });
});
