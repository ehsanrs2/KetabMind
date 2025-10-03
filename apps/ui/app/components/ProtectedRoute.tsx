'use client';

import { type ReactNode, useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !user) {
      const params = new URLSearchParams();
      if (pathname && pathname !== '/login') {
        params.set('next', pathname);
      }
      router.replace(`/login${params.toString() ? `?${params}` : ''}`);
    }
  }, [loading, pathname, router, user]);

  if (loading) {
    return (
      <div role="status" className="loading">
        Loading...
      </div>
    );
  }

  return <>{children}</>;
}
