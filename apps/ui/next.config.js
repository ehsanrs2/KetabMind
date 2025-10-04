const API_ORIGIN = process.env.KETABMIND_API_ORIGIN || 'http://127.0.0.1:8000';

function rewrite(pathPattern, destinationPath) {
  return {
    source: pathPattern,
    destination: `${API_ORIGIN}${destinationPath}`,
  };
}

/** @type {import('next').NextConfig} */
const config = {
  async rewrites() {
    return [
      rewrite('/api/login', '/auth/login'),
      rewrite('/api/logout', '/auth/logout'),
      rewrite('/api/me', '/me'),
      rewrite('/upload', '/upload'),
      rewrite('/index', '/index'),
      rewrite('/query', '/query'),
      rewrite('/sessions', '/sessions'),
      rewrite('/sessions/:sessionId/messages', '/sessions/:sessionId/messages'),
      rewrite('/bookmarks', '/bookmarks'),
      rewrite('/bookmarks/:bookmarkId', '/bookmarks/:bookmarkId'),
      rewrite('/book/:bookId/page/:page/view', '/book/:bookId/page/:page/view'),
      rewrite('/static/books/:token', '/static/books/:token'),
      rewrite('/metrics', '/metrics'),
      rewrite('/health', '/health'),
      rewrite('/ready', '/ready'),
    ];
  },
};

module.exports = config;
