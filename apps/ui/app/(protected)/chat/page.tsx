'use client';

import {
  ChangeEvent,
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import { useRTL } from '../../hooks/useRTL';

type SessionSummary = {
  id: string;
  title: string;
  lastActivity?: string | null;
};

type SessionPayload = NonNullable<SessionsResponse['sessions']>[number];

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

function extractSessions(payload: unknown): SessionPayload[] {
  if (Array.isArray(payload)) {
    return payload as SessionPayload[];
  }

  if (payload && typeof payload === 'object' && 'sessions' in payload) {
    const sessions = (payload as SessionsResponse).sessions;
    if (Array.isArray(sessions)) {
      return sessions;
    }
  }

  return [];
}

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
    tag?: string | null;
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
  tag: string | null;
  message: ChatMessage;
};

const ALL_BOOKMARKS_FILTER = 'all';
const UNTAGGED_BOOKMARKS_FILTER = '__untagged__';

function createMessageId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `msg-${Math.random().toString(36).slice(2)}`;
}

class SessionNotFoundError extends Error {
  constructor(message = 'Session not found') {
    super(message);
    this.name = 'SessionNotFoundError';
  }
}

class SSEParseError extends Error {
  constructor(message = 'Malformed SSE stream') {
    super(message);
    this.name = 'SSEParseError';
  }
}

const STREAM_INCOMPLETE_ERROR = 'STREAM_INCOMPLETE';

const STREAM_MODEL: 'ollama' | 'openai' = (() => {
  const candidates = [
    process.env.NEXT_PUBLIC_STREAM_MODEL,
    process.env.NEXT_PUBLIC_LLM_MODEL,
    process.env.NEXT_PUBLIC_LLM_BACKEND,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === 'string') {
      const normalized = candidate.trim().toLowerCase();
      if (normalized === 'ollama' || normalized === 'openai') {
        return normalized as 'ollama' | 'openai';
      }
    }
  }
  return 'ollama';
})();

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const NUMERIC_ID_PATTERN = /^[0-9]+$/;

function isSupportedSessionId(value: string | null | undefined): value is string {
  if (!value) {
    return false;
  }

  return UUID_PATTERN.test(value) || NUMERIC_ID_PATTERN.test(value);
}

function normaliseSession(item: SessionPayload | undefined): SessionSummary | null {
  if (!item) {
    return null;
  }

  let id: string | undefined;
  if (typeof item.id === 'number' && Number.isFinite(item.id)) {
    id = String(item.id);
  } else if (typeof item.id === 'string') {
    const trimmed = item.id.trim();
    if (trimmed.length > 0) {
      id = trimmed;
    }
  }

  if (!isSupportedSessionId(id)) {
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

function normaliseMetricValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const containsPercent = trimmed.includes('%');
    const parsed = Number.parseFloat(trimmed.replace(/%/g, ''));
    if (!Number.isFinite(parsed)) {
      return null;
    }
    return containsPercent ? parsed / 100 : parsed;
  }
  return null;
}

function formatMetricPercentage(value: number): string {
  const scaled = value <= 1 ? value * 100 : value;
  const clamped = Math.min(100, Math.max(0, scaled));
  return `${Math.round(clamped)}%`;
}

function readMetric(
  meta: Record<string, unknown> | null | undefined,
  key: 'coverage' | 'confidence',
): number | null {
  if (!meta || typeof meta !== 'object') {
    return null;
  }
  return normaliseMetricValue((meta as Record<string, unknown>)[key]);
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
    tag: item.tag ?? null,
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
  const { csrfToken } = useAuth();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionSearch, setSessionSearch] = useState('');
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState('');
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<'fa' | 'en'>('fa');
  const [allBookmarks, setAllBookmarks] = useState<BookmarkRecord[]>([]);
  const [bookmarkError, setBookmarkError] = useState<string | null>(null);
  const [bookmarkFilterTag, setBookmarkFilterTag] = useState<string>(ALL_BOOKMARKS_FILTER);
  const [bookmarkSearch, setBookmarkSearch] = useState('');
  const [bookmarkTagDraft, setBookmarkTagDraft] = useState('');
  const [pendingBookmarkId, setPendingBookmarkId] = useState<string | null>(null);
  const [pendingScrollMessageId, setPendingScrollMessageId] = useState<string | null>(null);
  const [activeBookmarkMessageId, setActiveBookmarkMessageId] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const messageRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useRTL(language);

  const handleSessionSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setSessionSearch(event.target.value);
  }, []);

  const handleBookmarkSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setBookmarkSearch(event.target.value);
  }, []);

  const handleBookmarkTagDraftChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setBookmarkTagDraft(event.target.value);
  }, []);

  const handleBookmarkTagSelect = useCallback((tag: string) => {
    setBookmarkFilterTag(tag);
    if (tag === ALL_BOOKMARKS_FILTER || tag === UNTAGGED_BOOKMARKS_FILTER) {
      setBookmarkTagDraft('');
    } else {
      setBookmarkTagDraft(tag);
    }
  }, []);

  const bookmarkedMessageIds = useMemo(
    () => new Set(allBookmarks.map((bookmark) => bookmark.message.id)),
    [allBookmarks],
  );

  const availableBookmarkTags = useMemo(() => {
    const tags = new Set<string>();
    let hasUntagged = false;
    for (const bookmark of allBookmarks) {
    if (bookmark.tag) {
      tags.add(bookmark.tag);
    } else {
      hasUntagged = true;
    }
  }
  const sorted = Array.from(tags).sort((a, b) => a.localeCompare(b));
    if (hasUntagged) {
      sorted.unshift(UNTAGGED_BOOKMARKS_FILTER);
    }
    return [ALL_BOOKMARKS_FILTER, ...sorted];
  }, [allBookmarks]);

  const filteredBookmarks = useMemo(() => {
    const searchValue = bookmarkSearch.trim().toLowerCase();
    return allBookmarks.filter((bookmark) => {
      if (bookmarkFilterTag === UNTAGGED_BOOKMARKS_FILTER) {
        if (bookmark.tag) {
          return false;
        }
      } else if (bookmarkFilterTag !== ALL_BOOKMARKS_FILTER) {
        if (bookmark.tag !== bookmarkFilterTag) {
          return false;
        }
      }

      if (!searchValue) {
        return true;
      }
      const haystack = `${bookmark.sessionTitle} ${bookmark.message.content}`.toLowerCase();
      return haystack.includes(searchValue);
    });
  }, [allBookmarks, bookmarkFilterTag, bookmarkSearch]);

  useEffect(() => {
    if (!availableBookmarkTags.includes(bookmarkFilterTag)) {
      setBookmarkFilterTag(ALL_BOOKMARKS_FILTER);
    }
  }, [availableBookmarkTags, bookmarkFilterTag]);

  const registerMessageRef = useCallback(
    (messageId: string) =>
      (element: HTMLDivElement | null) => {
        if (!element) {
          messageRefs.current.delete(messageId);
        } else {
          messageRefs.current.set(messageId, element);
        }
      },
    [],
  );

  const csrfHeaders = useMemo(() => (csrfToken ? { 'x-csrf-token': csrfToken } : {}), [csrfToken]);
  const hasCsrfToken = useMemo(() => Object.keys(csrfHeaders).length > 0, [csrfHeaders]);

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

      setAllBookmarks(mapped);
      setBookmarkError(null);
    } catch (err) {
      console.warn('Failed to load bookmarks', err);
      setBookmarkError('Unable to load bookmarks.');
      setAllBookmarks([]);
    }
  }, []);

  const loadSessions = useCallback(
    async (options?: { query?: string }) => {
      try {
        const params = new URLSearchParams({ sort: 'date_desc' });
        const search = options?.query?.trim();
        if (search) {
          params.set('query', search);
        }

        const response = await fetch(`/sessions?${params.toString()}`, {
          credentials: 'include',
        });

        if (!response.ok) {
          throw new Error('Failed to load sessions');
        }

        const payload = await response.json();
        const mapped = extractSessions(payload)
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
    },
    [],
  );

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
    void loadSessions({ query: sessionSearch });
  }, [loadSessions, sessionSearch]);

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
          ...csrfHeaders,
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
  }, [csrfHeaders]);

  const handleCreateBookmark = useCallback(
    async (message: ChatMessage) => {
      setBookmarkError(null);
      setPendingBookmarkId(message.id);
      try {
        const response = await fetch('/bookmarks', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...csrfHeaders,
          },
          credentials: 'include',
          body: JSON.stringify({
            message_id: message.id,
            tag:
              bookmarkTagDraft.trim() ||
              (bookmarkFilterTag !== ALL_BOOKMARKS_FILTER && bookmarkFilterTag !== UNTAGGED_BOOKMARKS_FILTER
                ? bookmarkFilterTag
                : undefined),
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to create bookmark');
        }

        const payload = (await response.json()) as
          | { bookmark?: BookmarksResponse['bookmarks'][number] }
          | null;
        const normalised = normaliseBookmark(payload?.bookmark ?? (payload as any));

        if (normalised) {
          setAllBookmarks((current) => [normalised, ...current.filter((item) => item.id !== normalised.id)]);
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
    [bookmarkFilterTag, bookmarkTagDraft, csrfHeaders, loadBookmarks],
  );

  const handleExportAnswer = useCallback(
    async (message: ChatMessage, requestedFormat: 'pdf' | 'word') => {
      setError(null);
      try {
        const response = await fetch('/export', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...csrfHeaders,
          },
          credentials: 'include',
          body: JSON.stringify({
            message_id: message.id,
            format: requestedFormat,
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to export answer');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = requestedFormat === 'pdf' ? 'answer.pdf' : 'answer.docx';
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        console.warn('Failed to export answer', err);
        setError('Unable to export answer.');
      }
    },
    [csrfHeaders],
  );

  const handleDeleteBookmark = useCallback(
    async (bookmarkId: string) => {
      try {
        const response = await fetch(`/bookmarks/${bookmarkId}`, {
          method: 'DELETE',
          headers: hasCsrfToken ? csrfHeaders : undefined,
          credentials: 'include',
        });
        if (!response.ok && response.status !== 204) {
          throw new Error('Failed to delete bookmark');
        }
        setAllBookmarks((current) => {
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
    [activeBookmarkMessageId, csrfHeaders, hasCsrfToken],
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
          ...csrfHeaders,
        },
        credentials: 'include',
        body: JSON.stringify({ title: 'New session' }),
      });

      if (!response.ok) {
        throw new Error('Failed to create session');
      }

      const payload = await response.json();
      const session = normaliseSession((payload?.session ?? payload) as SessionPayload | undefined);
      if (session) {
        setSessions((current) => [session, ...current.filter((item) => item.id !== session.id)]);
        setSelectedSessionId(session.id);
        setMessages([]);
        setActiveBookmarkMessageId(null);
        setPendingScrollMessageId(null);
      } else {
        await loadSessions({ query: sessionSearch });
      }
    } catch (err) {
      console.warn('Failed to create session', err);
      setError('Unable to create a new session.');
    }
  }, [csrfHeaders, isStreaming, loadSessions, sessionSearch]);

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

  const handleDeleteSession = useCallback(
    async (sessionId: string) => {
      if (isStreaming) {
        abortControllerRef.current?.abort();
      }

      const trimmedId = sessionId.trim();
      if (!isSupportedSessionId(trimmedId)) {
        setError('Invalid session identifier.');
        return;
      }

      try {
        const response = await fetch(`/sessions/${trimmedId}`, {
          method: 'DELETE',
          headers: csrfHeaders,
          credentials: 'include',
        });

        if (response.status === 404) {
          throw new Error('Session not found');
        }

        if (!response.ok) {
          throw new Error('Failed to delete session');
        }

        await loadSessions({ query: sessionSearch });
      } catch (err) {
        console.warn('Failed to delete session', err);
        setError('Unable to delete session.');
      }
    },
    [csrfHeaders, isStreaming, loadSessions, sessionSearch],
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

      const applyAssistantUpdate = (updater: (message: ChatMessage) => ChatMessage) => {
        setMessages((previous) =>
          previous.map((message) => (message.id === assistantMessage.id ? updater(message) : message)),
        );
      };

      const controller = new AbortController();
      abortControllerRef.current = controller;
      setIsStreaming(true);

      const baseStreamPath = `/sessions/${encodeURIComponent(selectedSessionId)}/messages/stream`;
      const streamParams = new URLSearchParams();
      if (debugEnabled) {
        streamParams.set('debug', 'true');
      }
      const streamUrl =
        streamParams.size > 0 ? `${baseStreamPath}?${streamParams.toString()}` : baseStreamPath;

      let aggregatedText = assistantMessage.content;
      let finalAnswer: string | null = null;
      let finalCitations: CitationLink[] | undefined;
      let finalMeta: Record<string, unknown> | null = null;
      let debugPayload: unknown = null;
      let receivedDone = false;
      let streamCompleted = false;

      try {
        const response = await fetch(streamUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            ...csrfHeaders,
          },
          credentials: 'include',
          body: JSON.stringify({
            content: trimmedPrompt,
            context: true,
            model: STREAM_MODEL,
          }),
          signal: controller.signal,
        });

        if (response.status === 404) {
          throw new SessionNotFoundError('Session not found');
        }

        if (!response.ok) {
          throw new Error('Query failed');
        }

        const streamBody = response.body as ReadableStream<Uint8Array> | null;

        type ReaderType =
          | ReadableStreamDefaultReader<string>
          | ReadableStreamDefaultReader<Uint8Array>
          | null;

        let reader: ReaderType = null;
        let usingTextDecoderStream = false;

        if (streamBody) {
          const canUseTextDecoderStream =
            typeof TextDecoderStream !== 'undefined' &&
            'pipeThrough' in streamBody &&
            typeof streamBody.pipeThrough === 'function';

          if (canUseTextDecoderStream) {
            try {
              const textStream = streamBody.pipeThrough(new TextDecoderStream());
              if (textStream && typeof textStream.getReader === 'function') {
                reader = textStream.getReader();
                usingTextDecoderStream = true;
              }
            } catch {
              // Fallback handled below when TextDecoderStream is unavailable.
            }
          }

          if (!reader && typeof streamBody.getReader === 'function') {
            reader = streamBody.getReader();
          }
        }

        if (!reader) {
          throw new Error('Query failed');
        }

        const decoder = usingTextDecoderStream ? null : new TextDecoder();
        let buffer = '';
        let eventDataLines: string[] = [];
        let shouldStop = false;

        const processChunk = (chunk: unknown) => {
          if (typeof chunk === 'string') {
            if (!chunk) {
              return;
            }
            aggregatedText += chunk;
            finalAnswer = aggregatedText;
            applyAssistantUpdate((message) => ({
              ...message,
              content: message.content + chunk,
            }));
            return;
          }

          if (!chunk || typeof chunk !== 'object') {
            return;
          }

          const payload = chunk as Record<string, unknown>;

          if (typeof payload.delta === 'string') {
            const delta = payload.delta as string;
            aggregatedText += delta;
            finalAnswer = aggregatedText;
            applyAssistantUpdate((message) => ({
              ...message,
              content: message.content + delta,
            }));
          }

          if (typeof payload.answer === 'string') {
            const answer = payload.answer as string;
            finalAnswer = answer;
            aggregatedText = answer;
            applyAssistantUpdate((message) => ({
              ...message,
              content: answer,
            }));
          }

          if (Array.isArray(payload.contexts)) {
            finalCitations = buildCitationLinksFromContexts(
              payload.contexts as Array<Record<string, unknown>>,
            );
          }

          if (payload.meta && typeof payload.meta === 'object') {
            finalMeta = payload.meta as Record<string, unknown>;
            const lang = (payload.meta as Record<string, unknown>).lang;
            if (lang === 'en') {
              setLanguage('en');
            } else if (lang === 'fa') {
              setLanguage('fa');
            }
          }

          if (payload.debug !== undefined) {
            debugPayload = payload.debug;
          }

          if (typeof payload.error === 'string') {
            const errorText = payload.error as string;
            finalAnswer = errorText;
            aggregatedText = errorText;
            applyAssistantUpdate((message) => ({
              ...message,
              content: errorText,
            }));
          }
        };

        const finalizeEvent = () => {
          if (!eventDataLines.length) {
            return;
          }
          const payload = eventDataLines.join('\n');
          eventDataLines = [];
          if (payload === '[DONE]') {
            receivedDone = true;
            shouldStop = true;
            streamCompleted = true;
            return;
          }
          let parsed: Record<string, unknown>;
          try {
            parsed = JSON.parse(payload) as Record<string, unknown>;
          } catch (error) {
            processChunk(payload);
            return;
          }
          processChunk(parsed);
        };

        const handleLine = (rawLine: string, allowFallbackData = false) => {
          const line = rawLine.replace(/\r$/, '');
          if (!line) {
            finalizeEvent();
            return;
          }
          if (line.trim().length === 0) {
            finalizeEvent();
            return;
          }
          if (line.startsWith(':')) {
            return;
          }
          if (line.startsWith('data:')) {
            const value = line.startsWith('data: ')
              ? line.slice('data: '.length)
              : line.slice('data:'.length);
            eventDataLines.push(value);
            return;
          }
          if (allowFallbackData) {
            eventDataLines.push(line);
            return;
          }
          throw new SSEParseError('Malformed SSE event');
        };

        const appendChunk = (text: string, flush = false) => {
          if (text) {
            buffer += text;
          }

          let newlineIndex = buffer.indexOf('\n');
          while (newlineIndex !== -1) {
            const line = buffer.slice(0, newlineIndex);
            buffer = buffer.slice(newlineIndex + 1);
            handleLine(line);
            if (shouldStop) {
              buffer = '';
              eventDataLines = [];
              return;
            }
          }

          if (flush) {
            if (buffer.length > 0) {
              handleLine(buffer, true);
              buffer = '';
            }
            finalizeEvent();
          }
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (!usingTextDecoderStream && decoder) {
              const remainder = decoder.decode();
              if (remainder) {
                appendChunk(remainder);
              }
            }
            appendChunk('', true);
            break;
          }

          const chunkText = usingTextDecoderStream
            ? (typeof value === 'string' ? value : String(value ?? ''))
            : decoder!.decode(value as Uint8Array, { stream: true });

          if (chunkText) {
            appendChunk(chunkText);
          }

          if (shouldStop) {
            break;
          }
        }

        if (!receivedDone) {
          throw new SSEParseError('Stream ended without completion signal');
        }

        if (!streamCompleted) {
          const error = new Error('Stream ended unexpectedly');
          error.name = STREAM_INCOMPLETE_ERROR;
          throw error;
        }

        applyAssistantUpdate((message) => {
          const resolvedContent = (finalAnswer ?? aggregatedText) || message.content;
          return {
            ...message,
            content: resolvedContent,
            citations: finalCitations,
            meta: finalMeta,
            debug: debugPayload,
          };
        });

        const assistantToPersist: ChatMessage = {
          ...assistantMessage,
          content: (finalAnswer ?? aggregatedText) || assistantMessage.content,
          citations: finalCitations,
          meta: finalMeta,
        };

        await persistMessage(selectedSessionId, assistantToPersist);
        await loadSessions({ query: sessionSearch });
      } catch (err) {
        console.warn('Chat request failed', err);
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }

        let errorMessage = 'Streaming response failed. Please try again.';
        if (err instanceof SessionNotFoundError) {
          errorMessage = 'Selected session could not be found. Please choose or create another session.';
          await loadSessions({ query: sessionSearch });
        } else if (err instanceof SSEParseError) {
          errorMessage = 'Received an invalid response from the server. Please try again.';
        } else if (err instanceof TypeError) {
          errorMessage = 'Network error. Please check your connection and try again.';
        }

        setError(errorMessage);
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantMessage.id
              ? {
                  ...message,
                  content: errorMessage,
                }
              : message,
          ),
        );
      } finally {
        abortControllerRef.current = null;
        setIsStreaming(false);
      }
    },
    [
      csrfHeaders,
      debugEnabled,
      isStreaming,
      loadSessions,
      persistMessage,
      prompt,
      selectedSessionId,
      sessionSearch,
    ],
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
        <div className="chat-sessions__controls">
          <input
            type="search"
            className="chat-session-search"
            placeholder="Search sessions"
            value={sessionSearch}
            onChange={handleSessionSearchChange}
            aria-label="Search sessions"
          />
        </div>
        <ul className="chat-session-list">
          {sessions.map((session) => {
            const isActive = session.id === selectedSessionId;
            return (
              <li key={session.id} className="chat-session-list__item">
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
                <button
                  type="button"
                  className="chat-session-delete"
                  onClick={(event) => {
                    event.stopPropagation();
                    void handleDeleteSession(session.id);
                  }}
                  aria-label={`Delete session ${session.title}`}
                >
                  Delete
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
          {isLoadingMessages ? (
            <div className="chat-message-skeleton__container" aria-hidden="true">
              {Array.from({ length: 3 }).map((_, index) => (
                <div className="chat-message-skeleton" key={`message-skeleton-${index}`}>
                  <div className="chat-message-skeleton__header">
                    <span className="skeleton skeleton-line skeleton-line--sm" />
                    <span className="skeleton skeleton-line skeleton-line--md" />
                  </div>
                  <div className="chat-message-skeleton__line-group">
                    <span className="skeleton skeleton-line skeleton-line--lg" />
                    <span className="skeleton skeleton-line skeleton-line--lg" />
                    <span className="skeleton skeleton-line skeleton-line--md" />
                  </div>
                </div>
              ))}
            </div>
          ) : null}
          <AnimatePresence initial={false}>
            {messages.map((message) => {
              const citations = message.citations ?? [];
              const isBookmarked = bookmarkedMessageIds.has(message.id);
              const isBookmarking = pendingBookmarkId === message.id;
              const isHighlighted = activeBookmarkMessageId === message.id;
              const messageLang = (message.meta?.lang as string | undefined) ?? null;
              const isRTLMessage = (messageLang ?? '').toLowerCase() === 'fa';
              const coverageValue = readMetric(message.meta ?? undefined, 'coverage');
              const confidenceValue = readMetric(message.meta ?? undefined, 'confidence');
              const hasMetrics = coverageValue !== null || confidenceValue !== null;
              return (
                <motion.article
                  key={message.id}
                  ref={registerMessageRef(message.id)}
                  className={`chat-message chat-message--${message.role}${
                    isHighlighted ? ' chat-message--highlight' : ''
                  }${isRTLMessage ? ' rtl' : ''}`}
                  data-testid={`chat-message-${message.role}`}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -16 }}
                  transition={{ type: 'spring', stiffness: 220, damping: 24, mass: 0.9 }}
                  layout
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
                        <div className="chat-export-buttons">
                          <button
                            type="button"
                            className="chat-export-button"
                            onClick={() => handleExportAnswer(message, 'pdf')}
                          >
                            Export PDF
                          </button>
                          <button
                            type="button"
                            className="chat-export-button"
                            onClick={() => handleExportAnswer(message, 'word')}
                          >
                            Export Word
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </header>
                  <div className="chat-message__content">{renderMessageContent(message.content)}</div>
                  {hasMetrics ? (
                    <div className="chat-message__metrics">
                      {coverageValue !== null ? (
                        <span data-testid={`chat-message-coverage-${message.id}`}>
                          Coverage: {formatMetricPercentage(coverageValue)}
                        </span>
                      ) : null}
                      {confidenceValue !== null ? (
                        <span data-testid={`chat-message-confidence-${message.id}`}>
                          Confidence: {formatMetricPercentage(confidenceValue)}
                        </span>
                      ) : null}
                    </div>
                  ) : null}
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
                </motion.article>
              );
            })}
          </AnimatePresence>
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
          <div className="chat-bookmarks__controls">
            <input
              type="search"
              className="chat-bookmarks__search"
              placeholder="Search bookmarks"
              value={bookmarkSearch}
              onChange={handleBookmarkSearchChange}
            />
            <input
              type="text"
              className="chat-bookmarks__tag-input"
              placeholder="Tag new bookmarks (optional)"
              value={bookmarkTagDraft}
              onChange={handleBookmarkTagDraftChange}
            />
          </div>
          <div className="chat-bookmarks__tabs" role="tablist" aria-label="Bookmark tags">
            {availableBookmarkTags.map((tag) => {
              const isActive = bookmarkFilterTag === tag;
              const label =
                tag === ALL_BOOKMARKS_FILTER
                  ? 'All'
                  : tag === UNTAGGED_BOOKMARKS_FILTER
                    ? 'No tag'
                    : tag;
              return (
                <button
                  type="button"
                  key={tag}
                  className={`chat-bookmarks__tab${isActive ? ' chat-bookmarks__tab--active' : ''}`}
                  onClick={() => handleBookmarkTagSelect(tag)}
                  role="tab"
                  aria-selected={isActive}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>
        {bookmarkError ? (
          <div className="chat-bookmarks__error" role="alert">
            {bookmarkError}
          </div>
        ) : null}
        {filteredBookmarks.length === 0 ? (
          <p className="chat-status">
            {allBookmarks.length === 0 ? 'No bookmarks yet.' : 'No bookmarks match your filters.'}
          </p>
        ) : (
          <ul className="chat-bookmark-list">
            {filteredBookmarks.map((bookmark) => {
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
                      {bookmark.tag ? (
                        <span className="chat-bookmark__tag">{bookmark.tag}</span>
                      ) : null}
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
