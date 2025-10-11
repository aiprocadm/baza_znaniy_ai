import { fireEvent, render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import { ThemeToggle } from '../components/common/ThemeToggle';
import { useTheme } from '../context/ThemeContext';
import type { Mock } from 'vitest';

vi.mock('../context/ThemeContext', () => ({
  useTheme: vi.fn()
}));

const useThemeMock = useTheme as unknown as Mock;

describe('ThemeToggle', () => {
  beforeEach(() => {
    useThemeMock.mockReset();
  });

  it('renders the light theme icon by default', () => {
    const toggleTheme = vi.fn();
    useThemeMock.mockReturnValue({ theme: 'light', toggleTheme });

    render(<ThemeToggle />);

    expect(screen.getByRole('button', { name: /toggle theme/i })).toHaveTextContent('🌞');
  });

  it('renders the dark theme icon when theme is dark', () => {
    const toggleTheme = vi.fn();
    useThemeMock.mockReturnValue({ theme: 'dark', toggleTheme });

    render(<ThemeToggle />);

    expect(screen.getByRole('button', { name: /toggle theme/i })).toHaveTextContent('🌙');
  });

  it('invokes toggleTheme when clicked', () => {
    const toggleTheme = vi.fn();
    useThemeMock.mockReturnValue({ theme: 'light', toggleTheme });

    render(<ThemeToggle />);

    fireEvent.click(screen.getByRole('button', { name: /toggle theme/i }));

    expect(toggleTheme).toHaveBeenCalledTimes(1);
  });
});
