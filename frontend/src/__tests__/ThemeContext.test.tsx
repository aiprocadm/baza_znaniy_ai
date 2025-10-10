import { renderHook, act } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../context/ThemeContext';

const renderUseTheme = () =>
  renderHook(() => useTheme(), {
    wrapper: ({ children }) => <ThemeProvider>{children}</ThemeProvider>
  });

describe('ThemeContext', () => {
  it('toggles theme between light and dark', () => {
    const { result } = renderUseTheme();
    const initial = result.current.theme;
    act(() => {
      result.current.toggleTheme();
    });
    expect(result.current.theme).not.toBe(initial);
  });
});
