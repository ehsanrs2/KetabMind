import { render, screen, waitFor } from '@testing-library/react';
import ProtectedRoute from '../app/components/ProtectedRoute';
import { AuthContext } from '../app/context/AuthContext';
import { usePathname, useRouter } from 'next/navigation';

jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
  usePathname: jest.fn(),
}));

const mockedUseRouter = useRouter as jest.Mock;
const mockedUsePathname = usePathname as jest.Mock;

describe('ProtectedRoute', () => {
  beforeEach(() => {
    mockedUseRouter.mockReturnValue({ replace: jest.fn() });
    mockedUsePathname.mockReturnValue('/chat');
  });

  it('renders children when user exists', () => {
    render(
      <AuthContext.Provider
        value={{
          user: { id: '1', name: 'Sara' },
          loading: false,
          error: null,
          refresh: async () => {},
          logout: async () => {},
        }}
      >
        <ProtectedRoute>
          <p>Protected Content</p>
        </ProtectedRoute>
      </AuthContext.Provider>,
    );

    expect(screen.getByText(/protected content/i)).toBeInTheDocument();
  });

  it('redirects to login when user is missing', async () => {
    const replace = jest.fn();
    mockedUseRouter.mockReturnValue({ replace });

    render(
      <AuthContext.Provider
        value={{
          user: null,
          loading: false,
          error: null,
          refresh: async () => {},
          logout: async () => {},
        }}
      >
        <ProtectedRoute>
          <p>Hidden Content</p>
        </ProtectedRoute>
      </AuthContext.Provider>,
    );

    await waitFor(() => {
      expect(replace).toHaveBeenCalledWith('/login?next=%2Fchat');
    });
  });
});
