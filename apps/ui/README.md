# KetabMind UI

KetabMind's web interface is built with Next.js (App Router) and custom CSS. The UI lives in the `apps/ui` workspace.

## Developing locally

```bash
cd apps/ui
npm install
npm run dev
```

The application expects the FastAPI backend to be available on the same origin; the configured Next.js rewrites proxy browser requests to the API service.

## Book Management UI

Authenticated users can manage indexed books from **Books** (`/books`). The section is linked from the global navigation once a session is established. It provides:

- A paginated table of indexed books with metadata, rename, and delete actions.
- Detail pages at `/books/[bookId]` that surface timestamps, indexing statistics, and stored metadata.
- Inline dialogs for rename/delete confirmations with optimistic UI updates.

The frontend talks to the following backend endpoints:

- `GET /books`
- `GET /books/{book_id}`
- `PATCH /books/{book_id}/rename`
- `DELETE /books/{book_id}`

## Testing

Run Jest unit tests from the UI workspace:

```bash
cd apps/ui
npm test -- BooksPage
npm test -- BookDetailsPage
```
