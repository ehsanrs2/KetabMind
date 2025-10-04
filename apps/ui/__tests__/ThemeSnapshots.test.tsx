import { render, screen } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../app/context/ThemeContext';

function ThemeConsumer() {
  const { theme } = useTheme();
  return <div data-testid="current-theme">{theme}</div>;
}

describe('ThemeProvider snapshots', () => {
  beforeEach(() => {
    window.localStorage.clear();
    delete document.body.dataset.theme;
  });

  it('renders light theme snapshot', () => {
    const { asFragment } = render(
      <ThemeProvider initialTheme="light">
        <ThemeConsumer />
      </ThemeProvider>,
    );

    expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    expect(document.body.dataset.theme).toBe('light');
    expect(asFragment()).toMatchSnapshot('light-theme');
  });

  it('renders dark theme snapshot', () => {
    const { asFragment } = render(
      <ThemeProvider initialTheme="dark">
        <ThemeConsumer />
      </ThemeProvider>,
    );

    expect(screen.getByTestId('current-theme')).toHaveTextContent('dark');
    expect(document.body.dataset.theme).toBe('dark');
    expect(asFragment()).toMatchSnapshot('dark-theme');
  });
});

