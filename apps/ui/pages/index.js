import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [contexts, setContexts] = useState([]);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${apiUrl}/upload`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        console.error("upload failed", res.statusText);
      }
    } catch (err) {
      console.error("upload error", err);
    }
  }

  async function onAsk() {
    try {
      const res = await fetch(`${apiUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: question }),
      });
      if (!res.ok) {
        console.error("query failed", res.statusText);
        return;
      }
      const data = await res.json();
      setAnswer(data.answer || "");
      setContexts(data.contexts || []);
    } catch (err) {
      console.error("query error", err);
    }
  }

  return (
    <main>
      <h1>KetabMind</h1>
      <input type="file" onChange={onUpload} />
      <div>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question"
        />
        <button onClick={onAsk}>Send</button>
      </div>
      {answer && <p>{answer}</p>}
      <ul>
        {contexts.map((c, i) => (
          <li key={i}>
            {c.book_id}: {c.page_start}-{c.page_end}
          </li>
        ))}
      </ul>
    </main>
  );
}
