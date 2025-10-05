import { useMemo, useState } from 'react';
import { useI18n, isRTLLanguage } from '../i18n';

const API_BASE = process.env.REACT_APP_API_BASE ?? 'http://127.0.0.1:8000';

function extractCitations(text) {
  const citations = new Set();
  const pattern = /\[([^\]]+)\]/g;
  let match = pattern.exec(text);
  while (match) {
    citations.add(match[1]);
    match = pattern.exec(text);
  }
  return Array.from(citations);
}

export function Chat() {
  const { t, locale } = useI18n();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const direction = useMemo(() => (isRTLLanguage(locale) ? 'rtl' : 'ltr'), [locale]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const value = input.trim();
    if (!value) {
      return;
    }

    const userMessage = { role: 'user', content: value };
    setMessages((prev) => [...prev, userMessage, { role: 'assistant', content: '', citations: [] }]);
    setInput('');
    setPending(true);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: value }),
      });

      if (!response.body) {
        const payload = await response.json();
        const text = payload.response ?? '';
        const citations = extractCitations(text);
        setMessages((prev) => {
          const updated = [...prev];
          const assistantIndex = updated.length - 1;
          updated[assistantIndex] = {
            role: 'assistant',
            content: text,
            citations,
          };
          return updated;
        });
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = '';
      let buffer = '';

      while (true) {
        const { value: chunk, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(chunk, { stream: true });
        const parts = buffer.split('\n');
        buffer = parts.pop() ?? '';
        for (const part of parts) {
          if (!part.trim()) {
            continue;
          }
          try {
            const parsed = JSON.parse(part);
            if (parsed.text) {
              assistantText += parsed.text;
            }
            const citations = Array.isArray(parsed.citations)
              ? parsed.citations.map(String)
              : extractCitations(assistantText);
            setMessages((prev) => {
              const updated = [...prev];
              const assistantIndex = updated.length - 1;
              const previous = updated[assistantIndex] ?? { role: 'assistant', content: '', citations: [] };
              updated[assistantIndex] = {
                ...previous,
                role: 'assistant',
                content: assistantText,
                citations,
              };
              return updated;
            });
          } catch (error) {
            assistantText += part;
            const citations = extractCitations(assistantText);
            setMessages((prev) => {
              const updated = [...prev];
              const assistantIndex = updated.length - 1;
              updated[assistantIndex] = {
                role: 'assistant',
                content: assistantText,
                citations,
              };
              return updated;
            });
          }
        }
      }

      if (buffer) {
        assistantText += buffer;
      }
      const citations = extractCitations(assistantText);
      setMessages((prev) => {
        const updated = [...prev];
        const assistantIndex = updated.length - 1;
        updated[assistantIndex] = {
          role: 'assistant',
          content: assistantText,
          citations,
        };
        return updated;
      });
    } catch (error) {
      setMessages((prev) => {
        const updated = [...prev];
        const assistantIndex = updated.length - 1;
        updated[assistantIndex] = {
          role: 'assistant',
          content: error.message,
          citations: [],
        };
        return updated;
      });
    } finally {
      setPending(false);
    }
  };

  return (
    <section style={{ direction }}>
      <h1>{t('chatTitle')}</h1>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.5rem' }}>
        <input
          type="text"
          value={input}
          placeholder={t('chatPlaceholder')}
          onChange={(event) => setInput(event.target.value)}
          style={{ flex: 1 }}
        />
        <button type="submit" disabled={pending}>
          {t('send')}
        </button>
      </form>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {messages.map((message, index) => (
          <li key={`${message.role}-${index}`} style={{ marginTop: '1rem' }}>
            <strong>{message.role === 'user' ? t('userLabel') : t('assistantLabel')}:</strong>
            <p>{message.content || t('emptyResponse')}</p>
            {message.citations && message.citations.length > 0 ? (
              <details>
                <summary>{t('citationsHeading')}</summary>
                <ul>
                  {message.citations.map((citation) => (
                    <li key={citation}>{citation}</li>
                  ))}
                </ul>
              </details>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

export default Chat;
