'use client';

import { FormEvent, useState } from 'react';

export default function UploadPage() {
  const [title, setTitle] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      setStatus('Please choose a file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('title', title);
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      setStatus('Upload successful!');
      setTitle('');
      setFile(null);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Upload failed');
    }
  };

  return (
    <div className="page-card">
      <h1>Upload a book</h1>
      <form className="form" onSubmit={handleSubmit}>
        <label>
          Title
          <input value={title} onChange={(event) => setTitle(event.target.value)} required />
        </label>
        <label>
          File
          <input type="file" accept=".pdf,.epub" onChange={(event) => setFile(event.target.files?.[0] ?? null)} required />
        </label>
        <button type="submit">Upload</button>
        {status && <p role="status">{status}</p>}
      </form>
    </div>
  );
}
