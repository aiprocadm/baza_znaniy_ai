import { useTheme } from '../../context/ThemeContext';

/**
 * ThemeToggle switches between dark and light modes.
 */
export const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      type="button"
      aria-label="Toggle theme"
      onClick={toggleTheme}
      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 shadow-sm transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
    >
      {theme === 'light' ? '🌞' : '🌙'}
    </button>
  );
};
