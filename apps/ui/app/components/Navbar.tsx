'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';

export default function Navbar() {
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const router = useRouter();
  const { theme, toggleTheme } = useTheme();

  const handleLogout = async () => {
    await logout();
    router.replace('/login');
  };

  const themeLabel = theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';

  return (
    <header className="navbar">
      <div className="navbar__brand">
        <Link href="/">KetabMind</Link>
      </div>
      <nav className="navbar__links">
        <Link href="/chat">Chat</Link>
        <Link href="/upload">Upload</Link>
      </nav>
      <div className="navbar__spacer" />
      <button
        type="button"
        className="navbar__theme-toggle"
        onClick={toggleTheme}
        aria-label={themeLabel}
      >
        {theme === 'dark' ? 'Light' : 'Dark'}
      </button>
      {user ? (
        <div className="navbar__user">
          <button
            type="button"
            className="navbar__user-button"
            onClick={() => setMenuOpen((value) => !value)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            {user.name}
          </button>
          {menuOpen && (
            <div role="menu" className="navbar__dropdown">
              <button type="button" onClick={handleLogout} className="navbar__dropdown-item">
                Logout
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="navbar__auth-links">
          <Link href="/login">Login</Link>
        </div>
      )}
    </header>
  );
}
