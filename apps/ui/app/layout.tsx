import './globals.css';
import { Vazirmatn } from 'next/font/google';
import type { ReactNode } from 'react';
import Providers from './providers';
import Navbar from './components/Navbar';

const vazirmatn = Vazirmatn({
  subsets: ['arabic', 'latin'],
  variable: '--font-vazirmatn',
});

export const metadata = {
  title: 'KetabMind',
  description: 'Multilingual knowledge exploration',
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}) {
  const themeDefault = (() => {
    const raw = (process.env.NEXT_PUBLIC_THEME_DEFAULT ?? process.env.THEME_DEFAULT ?? 'light').toString();
    const normalised = raw.toLowerCase();
    return normalised === 'dark' || normalised === 'light' ? normalised : 'light';
  })();
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${vazirmatn.variable} font-sans`} data-theme={themeDefault}>
        <Providers>
          <Navbar />
          <main className="container">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
