import { useNotifications } from '../../context/NotificationContext';
import { cn } from '../../utils/cn';

/**
 * NotificationStack renders toasts on the right side of the viewport.
 */
export const NotificationStack = () => {
  const { notifications, remove } = useNotifications();

  return (
    <div className="pointer-events-none fixed inset-y-0 right-0 flex w-80 flex-col gap-3 p-6">
      {notifications.map((notification) => (
        <div
          key={notification.id}
          className={cn(
            'pointer-events-auto rounded-xl border bg-white/90 p-4 shadow-lg backdrop-blur transition dark:bg-slate-900/90',
            notification.type === 'success' && 'border-green-200 text-green-600 dark:border-green-800 dark:text-green-300',
            notification.type === 'error' && 'border-rose-200 text-rose-600 dark:border-rose-800 dark:text-rose-300',
            notification.type === 'info' && 'border-slate-200 text-slate-700 dark:border-slate-700 dark:text-slate-200'
          )}
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-semibold">{notification.title}</p>
              {notification.description && <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{notification.description}</p>}
            </div>
            <button
              type="button"
              onClick={() => remove(notification.id)}
              className="text-xs uppercase tracking-wide text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-slate-300"
            >
              Close
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};
