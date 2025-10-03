'use client';

import { FormEvent, useState } from 'react';
import { useRTL } from '../../hooks/useRTL';

export default function ChatPage() {
  const [prompt, setPrompt] = useState('');
  const [language, setLanguage] = useState('fa');
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);

  useRTL(language);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!prompt.trim()) return;

    setMessages((prev) => [...prev, { role: 'user', content: prompt }]);
    setPrompt('');

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ prompt, language }),
      });

      if (!response.ok) {
        throw new Error('Chat failed');
      }

      const data = await response.json().catch(() => ({ reply: 'No response' }));
      setMessages((prev) => [...prev, { role: 'assistant', content: data.reply }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Chat failed.' }]);
    }
  };

  return (
    <div className="page-card">
      <h1>Chat</h1>
      <div className="language-toggle" role="group" aria-label="Language">
        <label>
          <input
            type="radio"
            name="language"
            value="fa"
            checked={language === 'fa'}
            onChange={(event) => setLanguage(event.target.value)}
          />
          فارسی
        </label>
        <label>
          <input
            type="radio"
            name="language"
            value="en"
            checked={language === 'en'}
            onChange={(event) => setLanguage(event.target.value)}
          />
          English
        </label>
      </div>
      <div className="messages" aria-live="polite">
        {messages.map((message, index) => (
          <p key={`${message.role}-${index}`}>
            <strong>{message.role}:</strong> {message.content}
          </p>
        ))}
      </div>
      <form className="form" onSubmit={handleSubmit}>
        <label>
          Prompt
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
        </label>
        <button type="submit">Send</button>
      </form>
    </div>
  );
}
