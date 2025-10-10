import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { useLocale } from '../../context/LocaleContext';
import { ThemeToggle } from '../common/ThemeToggle';
import { LanguageSwitcher } from '../common/LanguageSwitcher';
import { useNotifications } from '../../context/NotificationContext';

/**
 * Topbar displays breadcrumbs, search input and session actions.
 */
export const Topbar = () => {
  const { session, logout } = useAuth();
  const { t } = useLocale();
  const navigate = useNavigate();
  const { push } = useNotifications();

  const handleLogout = () => {
    logout();
    navigate('/login');
    push({ title: t('logout'), description: 'Session terminated', type: 'info' });
  };

  return (
    <header className="flex h-16 items-center gap-4 border-b border-slate-200 bg-white/70 px-4 backdrop-blur dark:border-slate-800 dark:bg-slate-900/60">
      <div className="hidden flex-1 items-center gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-500 shadow-sm dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 lg:flex">
        <span>⌘K</span>
        <input
          type="search"
          placeholder={`${t('search')}...`}
          className="w-full bg-transparent outline-none"
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              navigate('/search');
            }
          }}
        />
      </div>
      <div className="ml-auto flex items-center gap-2">
        <LanguageSwitcher />
        <ThemeToggle />
        <button
          type="button"
          onClick={handleLogout}
          className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
        >
          {t('logout')}
        </button>
        <div className="hidden rounded-full border border-primary-200 bg-primary-50 px-4 py-1 text-sm font-semibold text-primary-700 dark:border-slate-700 dark:bg-slate-800 dark:text-white sm:flex">
          {session?.name}
        </div>
      </div>
    </header>
  );
};
