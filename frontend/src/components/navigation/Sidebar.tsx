import { NavLink } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { useLocale } from '../../context/LocaleContext';
import { cn } from '../../utils/cn';

/**
 * Sidebar renders main navigation with role-based sections.
 */
export const Sidebar = () => {
  const { t } = useLocale();
  const { hasRole } = useAuth();

  const links = [
    { to: '/', label: t('dashboard'), icon: '💡', roles: ['user', 'admin'] },
    { to: '/search', label: t('search'), icon: '🔍', roles: ['user', 'admin'] },
    { to: '/dashboard', label: t('documents'), icon: '📂', roles: ['user', 'admin'] },
    { to: '/dashboard#chat', label: t('chat'), icon: '💬', roles: ['user', 'admin'], external: true }
  ];

  const adminLinks = [
    { to: '/admin/users', label: t('users'), icon: '👥' },
    { to: '/admin/monitoring', label: t('monitoring'), icon: '📈' },
    { to: '/admin/settings', label: t('settings'), icon: '⚙️' }
  ];

  return (
    <aside className="hidden w-72 shrink-0 border-r border-slate-200 bg-white/80 p-6 dark:border-slate-800 dark:bg-slate-900/60 lg:flex lg:flex-col">
      <div className="flex items-center gap-3">
        <img src="/logo.svg" alt="KB.AI" className="h-10 w-10" />
        <div>
          <p className="text-lg font-semibold text-slate-900 dark:text-white">KB.AI</p>
          <p className="text-xs text-slate-500">Operations Console</p>
        </div>
      </div>
      <nav className="mt-8 space-y-1">
        {links
          .filter((item) => item.roles.some((role) => hasRole(role as never)))
          .map((link) =>
            link.external ? (
              <a
                key={link.to}
                href={link.to}
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
              >
                <span>{link.icon}</span>
                {link.label}
              </a>
            ) : (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition hover:bg-slate-100 hover:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-white',
                    isActive ? 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-white' : 'text-slate-500 dark:text-slate-300'
                  )
                }
                end
              >
                <span>{link.icon}</span>
                {link.label}
              </NavLink>
            )
          )}
      </nav>
      {hasRole('admin') && (
        <div className="mt-10">
          <p className="text-xs uppercase tracking-wide text-slate-400">{t('admin')}</p>
          <nav className="mt-3 space-y-1">
            {adminLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition hover:bg-primary-100 hover:text-primary-700 dark:hover:bg-slate-800',
                    isActive ? 'bg-primary-100 text-primary-700 dark:bg-slate-800 dark:text-white' : 'text-slate-500 dark:text-slate-300'
                  )
                }
              >
                <span>{link.icon}</span>
                {link.label}
              </NavLink>
            ))}
          </nav>
        </div>
      )}
      <div className="mt-auto rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 p-4 text-sm text-white shadow-lg">
        <p className="font-medium">{t('recentActivity')}</p>
        <p className="mt-1 text-xs text-white/70">
          Stay on top of ingestion jobs, chat conversations and search performance in real-time.
        </p>
      </div>
    </aside>
  );
};
