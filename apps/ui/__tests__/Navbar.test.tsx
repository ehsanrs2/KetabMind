import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import Navbar from '../app/components/Navbar';
import { AuthContext } from '../app/context/AuthContext';
import { useRouter } from 'next/navigation';

jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

const mockedUseRouter = useRouter as jest.Mock;

describe('Navbar', () => {
  beforeEach(() => {
    mockedUseRouter.mockReturnValue({ replace: jest.fn() });
  });

  it('shows login link when no user', () => {
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
        <Navbar />
      </AuthContext.Provider>,
    );

    expect(screen.getByText(/ketabmind/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /login/i })).toBeInTheDocument();
  });

  it('allows the user to open the menu and logout', async () => {
    const logout = jest.fn().mockResolvedValue(undefined);
    const replace = jest.fn();
    mockedUseRouter.mockReturnValue({ replace });

    render(
      <AuthContext.Provider
        value={{
          user: { id: '1', name: 'Sara' },
          loading: false,
          error: null,
          refresh: async () => {},
          logout,
        }}
      >
        <Navbar />
      </AuthContext.Provider>,
    );

    fireEvent.click(screen.getByRole('button', { name: /sara/i }));
    fireEvent.click(screen.getByRole('button', { name: /logout/i }));

    expect(logout).toHaveBeenCalledTimes(1);
    await waitFor(() => {
      expect(replace).toHaveBeenCalledWith('/login');
    });
  });
});
