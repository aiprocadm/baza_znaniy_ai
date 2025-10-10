import { useLocale } from '../../context/LocaleContext';

/**
 * LanguageSwitcher toggles RU/EN locales with accessible button.
 */
export const LanguageSwitcher = () => {
  const { locale, setLocale } = useLocale();

  const toggle = () => setLocale(locale === 'en' ? 'ru' : 'en');

  return (
    <button
      type="button"
      onClick={toggle}
      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 shadow-sm transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
    >
      {locale.toUpperCase()}
    </button>
  );
};
