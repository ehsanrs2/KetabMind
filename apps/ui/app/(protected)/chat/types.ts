export type BookRecord = {
  bookId: string;
  title: string;
  metadata?: Record<string, unknown> | null;
  version?: string | null;
  fileHash?: string | null;
  indexedChunks?: number | null;
};
