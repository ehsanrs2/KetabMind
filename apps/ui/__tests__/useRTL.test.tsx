import { render } from '@testing-library/react';
import { useRTL } from '../app/hooks/useRTL';

describe('useRTL', () => {
  function RTLTester({ language }: { language?: string | null }) {
    useRTL(language);
    return null;
  }

  it('applies rtl direction for rtl languages', () => {
    render(<RTLTester language="fa" />);

    expect(document.body.getAttribute('dir')).toBe('rtl');
    expect(document.body.classList.contains('rtl')).toBe(true);
    expect(document.documentElement?.getAttribute('dir')).toBe('rtl');
  });

  it('applies ltr direction otherwise', () => {
    const { rerender } = render(<RTLTester language="fa" />);
    rerender(<RTLTester language="en" />);

    expect(document.body.getAttribute('dir')).toBe('ltr');
    expect(document.body.classList.contains('rtl')).toBe(false);
    expect(document.documentElement?.getAttribute('dir')).toBe('ltr');
  });
});
