import { renderHook, act } from '@testing-library/react';
import { LocaleProvider, useLocale } from '../context/LocaleContext';

describe('LocaleContext', () => {
  it('switches between locales', () => {
    const { result } = renderHook(() => useLocale(), {
      wrapper: ({ children }) => <LocaleProvider>{children}</LocaleProvider>
    });
    const initial = result.current.locale;
    act(() => {
      result.current.setLocale(initial === 'en' ? 'ru' : 'en');
    });
    expect(result.current.locale).not.toBe(initial);
    expect(typeof result.current.t('search')).toBe('string');
  });
});
