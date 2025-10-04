'use client';

import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useRTL } from '../../hooks/useRTL';

type SessionSummary = {
  id: string;
  title: string;
  lastActivity?: string | null;
};

type CitationLink = {
  label: string;
  href: string;
  bookId: string;
  page: number;
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: CitationLink[];
  debug?: unknown;
  meta?: Record<string, unknown> | null;
  createdAt?: string | null;
};

type SessionsResponse = {
  sessions?: Array<{
    id?: string | number;
    title?: string | null;
    topic?: string | null;
    last_activity?: string | null;
    updated_at?: string | null;
  }>;
};

type MessagesResponse = {
  messages?: Array<{
    id?: string | number;
    role?: string;
    content?: string;
    citations?: string[];
    meta?: Record<string, unknown> | null;
    created_at?: string | null;
  }>;
};

type BookmarksResponse = {
  bookmarks?: Array<{
    id?: string | number;
    session_id?: string | number;
    created_at?: string | null;
    session?: {
      id?: string | number;
      title?: string | null;
      updated_at?: string | null;
    };
    message?: {
      id?: string | number;
      role?: string;
      content?: string;
      citations?: string[];
      meta?: Record<string, unknown> | null;
      created_at?: string | null;
    };
  }>;
};

type BookmarkRecord = {
  id: string;
  sessionId: string;
  sessionTitle: string;
  createdAt?: string | null;
  message: ChatMessage;
};

function createMessageId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `msg-${Math.random().toString(36).slice(2)}`;
}

function normaliseSession(item: SessionsResponse['sessions'][number]): SessionSummary | null {
  if (!item) {
    return null;
  }

  const id = item.id;
  if (id === null || id === undefined) {
    return null;
  }

  const title = item.title ?? item.topic ?? 'Untitled session';
  const lastActivity = item.last_activity ?? item.updated_at ?? null;

  return {
    id: String(id),
    title,
    lastActivity,
  };
}

function extractNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function mergeRanges(ranges: Array<[number, number]>): Array<[number, number]> {
  if (ranges.length === 0) {
    return [];
  }

  const sorted = [...ranges].sort(([aStart], [bStart]) => aStart - bStart);
  const merged: Array<[number, number]> = [];

  for (const [start, end] of sorted) {
    if (!merged.length) {
      merged.push([start, end]);
      continue;
    }

    const lastIndex = merged.length - 1;
    const [lastStart, lastEnd] = merged[lastIndex];
    if (start <= lastEnd + 1) {
      merged[lastIndex] = [lastStart, Math.max(lastEnd, end)];
    } else {
      merged.push([start, end]);
    }
  }

  return merged;
}

function buildCitationLinksFromContexts(
  contexts: Array<Record<string, unknown>>,
): CitationLink[] {
  const spansByBook = new Map<string, Array<[number, number]>>();

  for (const context of contexts) {
    const metadataRaw = (context as Record<string, unknown>).metadata;
    const metadata =
      metadataRaw && typeof metadataRaw === 'object'
        ? (metadataRaw as Record<string, unknown>)
        : undefined;
    const bookIdRaw =
      context.book_id ?? context.bookId ?? metadata?.book_id ?? metadata?.bookId;
    if (bookIdRaw === null || bookIdRaw === undefined) {
      continue;
    }
    const bookId = String(bookIdRaw);

    const start =
      extractNumber(context.page_start) ??
      extractNumber(context.page) ??
      extractNumber(context.page_num) ??
      extractNumber(metadata?.page_start) ??
      extractNumber(metadata?.page) ??
      extractNumber(metadata?.page_num);
    const end =
      extractNumber(context.page_end) ??
      extractNumber(context.page) ??
      extractNumber(context.page_num) ??
      extractNumber(metadata?.page_end) ??
      extractNumber(metadata?.page) ??
      extractNumber(metadata?.page_num);

    if (start === null && end === null) {
      continue;
    }

    const spanStart = start ?? end;
    const spanEnd = end ?? start;

    if (spanStart === null || spanEnd === null) {
      continue;
    }

    const normalisedStart = Math.min(spanStart, spanEnd);
    const normalisedEnd = Math.max(spanStart, spanEnd);

    const spans = spansByBook.get(bookId);
    if (spans) {
      spans.push([normalisedStart, normalisedEnd]);
    } else {
      spansByBook.set(bookId, [[normalisedStart, normalisedEnd]]);
    }
  }

  const links: CitationLink[] = [];

  spansByBook.forEach((spans, bookId) => {
    const merged = mergeRanges(spans);
    for (const [start, end] of merged) {
      const pageRange = start === end ? `${start}` : `${start}-${end}`;
      const label = `[${bookId}:${pageRange}]`;
      const href = `/viewer?book=${encodeURIComponent(bookId)}#page=${start}`;
      links.push({ label, href, page: start, bookId });
    }
  });

  const seen = new Set<string>();
  return links.filter((item) => {
    if (seen.has(item.label)) {
      return false;
    }
    seen.add(item.label);
    return true;
  });
}

function buildCitationLinksFromStrings(citations: string[] | undefined): CitationLink[] | undefined {
  if (!citations || citations.length === 0) {
    return undefined;
  }

  const normalised: CitationLink[] = [];
  for (const raw of citations) {
    if (typeof raw !== 'string' || raw.trim().length === 0) {
      continue;
    }

    const text = raw.trim();
    const match = text.match(/\[?([^:\]]+):(\d+)(?:-(\d+))?\]?/);
    if (!match) {
      continue;
    }

    const [, bookId, startRaw, endRaw] = match;
    const start = Number.parseInt(startRaw, 10);
    if (!Number.isFinite(start)) {
      continue;
    }
    const end = endRaw ? Number.parseInt(endRaw, 10) : start;
    const label = text.startsWith('[') ? text : `[${text}]`;
    const href = `/viewer?book=${encodeURIComponent(bookId)}#page=${start}`;
    normalised.push({ label, href, bookId, page: Number.isFinite(start) ? start : 1 });
    if (end && Number.isFinite(end) && end !== start) {
      // ensure the label captures the end of the range as provided
      normalised[normalised.length - 1].label = `[${bookId}:${start}-${end}]`;
    }
  }

  if (!normalised.length) {
    return undefined;
  }

  const seen = new Set<string>();
  return normalised.filter((item) => {
    if (seen.has(item.label)) {
      return false;
    }
    seen.add(item.label);
    return true;
  });
}

function renderMessageContent(content: string) {
  const citationPattern = /\[([^:\]]+):(\d+)(?:-(\d+))?\]/g;
  const segments: Array<string | { label: string; href: string }> = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = citationPattern.exec(content)) !== null) {
    const [full, bookId, startRaw, endRaw] = match;
    const start = Number.parseInt(startRaw, 10);
    if (match.index > lastIndex) {
      segments.push(content.slice(lastIndex, match.index));
    }
    if (Number.isFinite(start)) {
      const end = endRaw ? Number.parseInt(endRaw, 10) : start;
      const href = `/viewer?book=${encodeURIComponent(bookId)}#page=${start}`;
      const label = end && Number.isFinite(end) && end !== start ? `[${bookId}:${start}-${end}]` : full;
      segments.push({ label, href });
    } else {
      segments.push(full);
    }
    lastIndex = citationPattern.lastIndex;
  }

  if (lastIndex < content.length) {
    segments.push(content.slice(lastIndex));
  }

  return segments.map((segment, index) => {
    if (typeof segment === 'string') {
      return <span key={`text-${index}`}>{segment}</span>;
    }
    return (
      <a key={`citation-${index}`} href={segment.href} className="chat-citation" rel="noreferrer">
        {segment.label}
      </a>
    );
  });
}

function formatLastActivity(timestamp: string | null | undefined): string {
  if (!timestamp) {
    return 'No activity yet';
  }

  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return 'No activity yet';
  }

  return new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function truncateText(text: string, maxLength = 160): string {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxLength - 1))}…`;
}

function normaliseBookmark(item: BookmarksResponse['bookmarks'][number]): BookmarkRecord | null {
  if (!item) {
    return null;
  }

  const idRaw = item.id ?? item.session_id ?? item.message?.id;
  const sessionIdRaw = item.session_id ?? item.session?.id;
  const messageRaw = item.message;
  const messageIdRaw = messageRaw?.id;

  if (idRaw === null || idRaw === undefined) {
    return null;
  }
  if (sessionIdRaw === null || sessionIdRaw === undefined) {
    return null;
  }
  if (messageIdRaw === null || messageIdRaw === undefined) {
    return null;
  }

  const roleRaw = typeof messageRaw?.role === 'string' ? messageRaw.role : 'assistant';
  const role = roleRaw === 'user' || roleRaw === 'system' ? roleRaw : 'assistant';
  const content = messageRaw?.content ?? '';
  const citations = buildCitationLinksFromStrings(messageRaw?.citations ?? undefined);
  const meta = messageRaw?.meta ?? null;

  return {
    id: String(idRaw),
    sessionId: String(sessionIdRaw),
    sessionTitle: item.session?.title ?? 'Untitled session',
    createdAt: item.created_at ?? null,
    message: {
      id: String(messageIdRaw),
      role,
      content,
      citations,
      meta,
      createdAt: messageRaw?.created_at ?? null,
    },
  } satisfies BookmarkRecord;
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState('');
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<'fa' | 'en'>('fa');
  const [bookmarks, setBookmarks] = useState<BookmarkRecord[]>([]);
  const [bookmarkError, setBookmarkError] = useState<string | null>(null);
  const [pendingBookmarkId, setPendingBookmarkId] = useState<string | null>(null);
  const [pendingScrollMessageId, setPendingScrollMessageId] = useState<string | null>(null);
  const [activeBookmarkMessageId, setActiveBookmarkMessageId] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const messageRefs = useRef<Map<string, HTMLElement>>(new Map());

  useRTL(language);

  const bookmarkedMessageIds = useMemo(
    () => new Set(bookmarks.map((bookmark) => bookmark.message.id)),
    [bookmarks],
  );

  const registerMessageRef = useCallback(
    (messageId: string) =>
      (element: HTMLElement | null) => {
        if (!element) {
          messageRefs.current.delete(messageId);
        } else {
          messageRefs.current.set(messageId, element);
        }
      },
    [],
  );

  const loadBookmarks = useCallback(async () => {
    try {
      const response = await fetch('/bookmarks', {
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Failed to load bookmarks');
      }

      const payload = (await response.json()) as BookmarksResponse;
      const mapped = (payload.bookmarks ?? [])
        .map(normaliseBookmark)
        .filter((item): item is BookmarkRecord => item !== null);

      setBookmarks(mapped);
      setBookmarkError(null);
    } catch (err) {
      console.warn('Failed to load bookmarks', err);
      setBookmarkError('Unable to load bookmarks.');
      setBookmarks([]);
    }
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const response = await fetch('/sessions', {
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Failed to load sessions');
      }

      const payload = (await response.json()) as SessionsResponse;
      const mapped = (payload.sessions ?? [])
        .map(normaliseSession)
        .filter((item): item is SessionSummary => item !== null);

      setSessions(mapped);

      setSelectedSessionId((current) => {
        if (current && mapped.some((session) => session.id === current)) {
          return current;
        }
        if (mapped.length > 0) {
          return mapped[0].id;
        }
        return null;
      });
    } catch (err) {
      console.warn('Failed to load sessions', err);
      setError('Unable to load sessions.');
    }
  }, []);

  const loadMessages = useCallback(
    async (sessionId: string) => {
      setIsLoadingMessages(true);
      try {
        const response = await fetch(`/sessions/${sessionId}/messages`, {
          credentials: 'include',
        });

        if (!response.ok) {
          throw new Error('Failed to load messages');
        }

        const payload = (await response.json()) as MessagesResponse;
        const mapped = (payload.messages ?? []).map((message) => {
          const id = message.id ? String(message.id) : createMessageId();
          const role = (message.role as ChatMessage['role']) ?? 'assistant';
          const content = message.content ?? '';
          const citations = buildCitationLinksFromStrings(message.citations);
          return {
            id,
            role,
            content,
            citations,
            meta: message.meta ?? null,
            createdAt: message.created_at ?? null,
          } satisfies ChatMessage;
        });

        setMessages(mapped);

        const lastAssistant = [...mapped].reverse().find((msg) => msg.role === 'assistant');
        const lang = (lastAssistant?.meta?.lang as string | undefined) ?? null;
        if (lang === 'en') {
          setLanguage('en');
        } else if (lang === 'fa') {
          setLanguage('fa');
        }
      } catch (err) {
        console.warn('Failed to load messages', err);
        setError('Unable to load messages.');
        setMessages([]);
      } finally {
        setIsLoadingMessages(false);
      }
    },
    [],
  );

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    loadBookmarks();
  }, [loadBookmarks]);

  useEffect(() => {
    if (!selectedSessionId) {
      setMessages([]);
      return;
    }
    loadMessages(selectedSessionId);
  }, [loadMessages, selectedSessionId]);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    if (pendingScrollMessageId) {
      const target = messageRefs.current.get(pendingScrollMessageId);
      if (target && typeof target.scrollIntoView === 'function') {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        setActiveBookmarkMessageId(pendingScrollMessageId);
        setPendingScrollMessageId(null);
        return;
      }
    }
    const element = bottomRef.current;
    if (element && typeof element.scrollIntoView === 'function') {
      element.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages, pendingScrollMessageId]);

  const selectedSession = useMemo(
    () => sessions.find((session) => session.id === selectedSessionId) ?? null,
    [selectedSessionId, sessions],
  );

  const persistMessage = useCallback(async (sessionId: string, message: ChatMessage) => {
    try {
      const payload: Record<string, unknown> = {
        role: message.role,
        content: message.content,
      };
      if (message.citations?.length) {
        payload.citations = message.citations.map((item) => item.label);
      }
      if (message.meta) {
        payload.meta = message.meta;
      }
      const response = await fetch(`/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error('Failed to persist message');
      }
      const body = (await response.json().catch(() => null)) as
        | { message?: { id?: string | number; created_at?: string | null } }
        | null;
      const saved = (body?.message ?? body) as { id?: string | number; created_at?: string | null } | null;
      if (saved?.id !== undefined) {
        const savedId = String(saved.id);
        setMessages((previous) =>
          previous.map((item) =>
            item.id === message.id
              ? {
                  ...item,
                  id: savedId,
                  createdAt: saved.created_at ?? item.createdAt ?? null,
                }
              : item,
          ),
        );
      }
    } catch (err) {
      console.warn('Failed to persist message', err);
    }
  }, []);

  const handleCreateBookmark = useCallback(
    async (message: ChatMessage) => {
      setBookmarkError(null);
      setPendingBookmarkId(message.id);
      try {
        const response = await fetch('/bookmarks', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({ message_id: message.id }),
        });

        if (!response.ok) {
          throw new Error('Failed to create bookmark');
        }

        const payload = (await response.json()) as
          | { bookmark?: BookmarksResponse['bookmarks'][number] }
          | null;
        const normalised = normaliseBookmark(payload?.bookmark ?? (payload as any));

        if (normalised) {
          setBookmarks((current) => [normalised, ...current.filter((item) => item.id !== normalised.id)]);
        } else {
          await loadBookmarks();
        }
      } catch (err) {
        console.warn('Failed to create bookmark', err);
        setBookmarkError('Unable to create bookmark.');
      } finally {
        setPendingBookmarkId(null);
      }
    },
    [loadBookmarks],
  );

  const handleDeleteBookmark = useCallback(
    async (bookmarkId: string) => {
      try {
        const response = await fetch(`/bookmarks/${bookmarkId}`, {
          method: 'DELETE',
          credentials: 'include',
        });
        if (!response.ok && response.status !== 204) {
          throw new Error('Failed to delete bookmark');
        }
        setBookmarks((current) => {
          let removedMessageId: string | null = null;
          const filtered = current.filter((item) => {
            if (item.id === bookmarkId) {
              removedMessageId = item.message.id;
              return false;
            }
            return true;
          });
          if (removedMessageId && removedMessageId === activeBookmarkMessageId) {
            setActiveBookmarkMessageId(null);
          }
          return filtered;
        });
      } catch (err) {
        console.warn('Failed to delete bookmark', err);
        setBookmarkError('Unable to remove bookmark.');
      }
    },
    [activeBookmarkMessageId],
  );

  const handleBookmarkClick = useCallback(
    (bookmark: BookmarkRecord) => {
      setBookmarkError(null);
      setActiveBookmarkMessageId(null);
      if (bookmark.sessionId === selectedSessionId) {
        setPendingScrollMessageId(bookmark.message.id);
        return;
      }
      if (isStreaming) {
        abortControllerRef.current?.abort();
      }
      setSelectedSessionId(bookmark.sessionId);
      setMessages([]);
      setPendingScrollMessageId(bookmark.message.id);
    },
    [isStreaming, selectedSessionId],
  );

  const handleNewSession = useCallback(async () => {
    if (isStreaming) {
      abortControllerRef.current?.abort();
    }

    try {
      const response = await fetch('/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ title: 'New session' }),
      });

      if (!response.ok) {
        throw new Error('Failed to create session');
      }

      const payload = await response.json();
      const session = normaliseSession(payload.session ?? payload);
      if (session) {
        setSessions((current) => [session, ...current.filter((item) => item.id !== session.id)]);
        setSelectedSessionId(session.id);
        setMessages([]);
        setActiveBookmarkMessageId(null);
        setPendingScrollMessageId(null);
      } else {
        await loadSessions();
      }
    } catch (err) {
      console.warn('Failed to create session', err);
      setError('Unable to create a new session.');
    }
  }, [isStreaming, loadSessions]);

  const handleSelectSession = useCallback(
    (sessionId: string) => {
      if (sessionId === selectedSessionId) {
        return;
      }
      if (isStreaming) {
        abortControllerRef.current?.abort();
      }
      setSelectedSessionId(sessionId);
      setMessages([]);
      setPendingScrollMessageId(null);
      setActiveBookmarkMessageId(null);
    },
    [isStreaming, selectedSessionId],
  );

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!selectedSessionId) {
        setError('Please select or create a session first.');
        return;
      }
      const trimmedPrompt = prompt.trim();
      if (!trimmedPrompt || isStreaming) {
        return;
      }

      setError(null);
      setPrompt('');

      const userMessage: ChatMessage = {
        id: createMessageId(),
        role: 'user',
        content: trimmedPrompt,
      };

      const assistantMessage: ChatMessage = {
        id: createMessageId(),
        role: 'assistant',
        content: '',
        citations: [],
      };

      setMessages((previous) => [...previous, userMessage, assistantMessage]);
      await persistMessage(selectedSessionId, userMessage);

      const controller = new AbortController();
      abortControllerRef.current = controller;
      setIsStreaming(true);

      const params = new URLSearchParams({
        session_id: selectedSessionId,
        stream: 'true',
      });
      if (debugEnabled) {
        params.set('debug', 'true');
      }

      let finalAnswer: string | null = null;
      let finalCitations: CitationLink[] | undefined;
      let finalMeta: Record<string, unknown> | null = null;
      let debugPayload: unknown = null;

      try {
        const response = await fetch(`/query?${params.toString()}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({ q: trimmedPrompt }),
          signal: controller.signal,
        });

        const streamBody = response.body as {
          getReader?: () => ReadableStreamDefaultReader<Uint8Array>;
        } | null;
        const reader = streamBody?.getReader?.();
        if (!response.ok || !reader) {
          throw new Error('Query failed');
        }
        const decoder = new TextDecoder();
        let buffer = '';

        const applyAssistantUpdate = (updater: (previous: ChatMessage) => ChatMessage) => {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantMessage.id ? updater(message) : message,
            ),
          );
        };

        const processChunk = (chunk: Record<string, unknown>) => {
          if (typeof chunk.delta === 'string') {
            const delta = chunk.delta as string;
            applyAssistantUpdate((message) => ({
              ...message,
              content: message.content + delta,
            }));
          }

          if (typeof chunk.answer === 'string') {
            finalAnswer = chunk.answer as string;
            applyAssistantUpdate((message) => ({
              ...message,
              content: chunk.answer as string,
            }));
          }

          if (Array.isArray(chunk.contexts)) {
            finalCitations = buildCitationLinksFromContexts(
              chunk.contexts as Array<Record<string, unknown>>,
            );
          }

          if (chunk.meta && typeof chunk.meta === 'object') {
            finalMeta = chunk.meta as Record<string, unknown>;
            const lang = (chunk.meta as Record<string, unknown>).lang;
            if (lang === 'en') {
              setLanguage('en');
            } else if (lang === 'fa') {
              setLanguage('fa');
            }
          }

          if (chunk.debug !== undefined) {
            debugPayload = chunk.debug;
          }

          if (typeof chunk.error === 'string') {
            applyAssistantUpdate((message) => ({
              ...message,
              content: chunk.error as string,
            }));
            finalAnswer = chunk.error as string;
          }
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          buffer += decoder.decode(value, { stream: true });

          let newlineIndex = buffer.indexOf('\n');
          while (newlineIndex !== -1) {
            const line = buffer.slice(0, newlineIndex).trim();
            buffer = buffer.slice(newlineIndex + 1);
            if (line) {
              try {
                const parsed = JSON.parse(line) as Record<string, unknown>;
                processChunk(parsed);
              } catch (err) {
                console.warn('Failed to parse stream chunk', err);
              }
            }
            newlineIndex = buffer.indexOf('\n');
          }
        }

        if (buffer.trim().length > 0) {
          try {
            const parsed = JSON.parse(buffer) as Record<string, unknown>;
            processChunk(parsed);
          } catch (err) {
            console.warn('Failed to parse trailing stream chunk', err);
          }
        }

        applyAssistantUpdate((message) => ({
          ...message,
          content: finalAnswer ?? message.content,
          citations: finalCitations,
          meta: finalMeta,
          debug: debugPayload,
        }));

        const assistantToPersist: ChatMessage = {
          ...assistantMessage,
          content: finalAnswer ?? assistantMessage.content,
          citations: finalCitations,
          meta: finalMeta,
        };

        await persistMessage(selectedSessionId, assistantToPersist);
        await loadSessions();
      } catch (err) {
        console.warn('Chat request failed', err);
        setError('Chat failed. Please try again.');
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantMessage.id
              ? {
                  ...message,
                  content: 'Chat failed. Please try again.',
                }
              : message,
          ),
        );
      } finally {
        abortControllerRef.current = null;
        setIsStreaming(false);
      }
    },
    [debugEnabled, isStreaming, loadSessions, persistMessage, prompt, selectedSessionId],
  );

  return (
    <div className="chat-layout">
      <aside className="chat-sessions">
        <div className="chat-sessions__header">
          <h1>Sessions</h1>
          <button className="chat-button" type="button" onClick={handleNewSession}>
            New Session
          </button>
        </div>
        <ul className="chat-session-list">
          {sessions.map((session) => {
            const isActive = session.id === selectedSessionId;
            return (
              <li key={session.id}>
                <button
                  type="button"
                  className={`chat-session-item${isActive ? ' chat-session-item--active' : ''}`}
                  onClick={() => handleSelectSession(session.id)}
                >
                  <span className="chat-session-title">{session.title}</span>
                  <span className="chat-session-activity">
                    {formatLastActivity(session.lastActivity)}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </aside>
      <section className="chat-main">
        <header className="chat-main__header">
          <div>
            <h2>{selectedSession?.title ?? 'Select a session'}</h2>
            {selectedSession?.lastActivity ? (
              <p className="chat-main__meta">
                Last activity: {formatLastActivity(selectedSession.lastActivity)}
              </p>
            ) : null}
          </div>
          <label className="chat-debug-toggle">
            <input
              type="checkbox"
              checked={debugEnabled}
              onChange={(event) => setDebugEnabled(event.target.checked)}
            />
            Debug
          </label>
        </header>

        {error ? <div className="chat-error" role="alert">{error}</div> : null}

        <div className="chat-messages" aria-live="polite">
          {isLoadingMessages ? <p className="chat-status">Loading messages…</p> : null}
          {messages.map((message) => {
            const citations = message.citations ?? [];
            const isBookmarked = bookmarkedMessageIds.has(message.id);
            const isBookmarking = pendingBookmarkId === message.id;
            const isHighlighted = activeBookmarkMessageId === message.id;
            return (
              <article
                key={message.id}
                ref={registerMessageRef(message.id)}
                className={`chat-message chat-message--${message.role}${
                  isHighlighted ? ' chat-message--highlight' : ''
                }`}
                data-testid={`chat-message-${message.role}`}
              >
                <header className="chat-message__header">
                  <span className="chat-message__role">
                    {message.role === 'assistant' ? 'Assistant' : message.role === 'user' ? 'You' : 'System'}
                  </span>
                  {message.role === 'assistant' ? (
                    <div className="chat-message__actions">
                      <button
                        type="button"
                        className="chat-bookmark-button"
                        onClick={() => handleCreateBookmark(message)}
                        disabled={isBookmarked || isBookmarking}
                        aria-pressed={isBookmarked}
                      >
                        {isBookmarked ? 'Bookmarked' : isBookmarking ? 'Bookmarking…' : 'Bookmark'}
                      </button>
                    </div>
                  ) : null}
                </header>
                <div className="chat-message__content">{renderMessageContent(message.content)}</div>
                {citations.length > 0 ? (
                  <div className="chat-message__citations">
                    {citations.map((citation) => (
                      <a key={citation.label} href={citation.href} className="chat-citation" rel="noreferrer">
                        {citation.label}
                      </a>
                    ))}
                  </div>
                ) : null}
                {debugEnabled && message.debug ? (
                  <pre className="chat-debug-panel">{JSON.stringify(message.debug, null, 2)}</pre>
                ) : null}
              </article>
            );
          })}
          <div ref={bottomRef} />
        </div>

        <form className="chat-form" onSubmit={handleSubmit}>
          <label className="chat-form__label" htmlFor="chat-prompt">
            Message
          </label>
          <textarea
            id="chat-prompt"
            className="chat-form__textarea"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder={selectedSessionId ? 'Ask something…' : 'Select a session to start chatting'}
            disabled={!selectedSessionId || isStreaming}
          />
          <div className="chat-form__actions">
            <button className="chat-button" type="submit" disabled={!selectedSessionId || isStreaming}>
              {isStreaming ? 'Generating…' : 'Send'}
            </button>
          </div>
        </form>
      </section>
      <aside className="chat-bookmarks">
        <div className="chat-bookmarks__header">
          <h2>Bookmarks</h2>
        </div>
        {bookmarkError ? (
          <div className="chat-bookmarks__error" role="alert">
            {bookmarkError}
          </div>
        ) : null}
        {bookmarks.length === 0 ? (
          <p className="chat-status">No bookmarks yet.</p>
        ) : (
          <ul className="chat-bookmark-list">
            {bookmarks.map((bookmark) => {
              const isActive = activeBookmarkMessageId === bookmark.message.id;
              return (
                <li
                  key={bookmark.id}
                  className={`chat-bookmark${isActive ? ' chat-bookmark--active' : ''}`}
                >
                  <div className="chat-bookmark__item">
                    <button
                      type="button"
                      className="chat-bookmark__open"
                      onClick={() => handleBookmarkClick(bookmark)}
                      data-testid={`bookmark-item-${bookmark.id}`}
                    >
                      <span className="chat-bookmark__session">{bookmark.sessionTitle}</span>
                      <span className="chat-bookmark__preview">
                        {truncateText(bookmark.message.content)}
                      </span>
                      <span className="chat-bookmark__timestamp">
                        {formatLastActivity(bookmark.createdAt ?? bookmark.message.createdAt)}
                      </span>
                    </button>
                    <button
                      type="button"
                      className="chat-bookmark__remove"
                      onClick={(event) => {
                        event.stopPropagation();
                        handleDeleteBookmark(bookmark.id);
                      }}
                      aria-label={`Remove bookmark for ${bookmark.sessionTitle}`}
                    >
                      Remove
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </aside>
    </div>
  );
}

