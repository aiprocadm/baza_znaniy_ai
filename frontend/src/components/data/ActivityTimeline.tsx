import type { ActivityItem } from '../../api';
import { formatDateTime } from '../../utils/format';

/**
 * ActivityTimeline shows chronological events across modules.
 */
export const ActivityTimeline = ({ items }: { items: ActivityItem[] }) => (
  <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
    <h3 className="text-base font-semibold text-slate-800 dark:text-slate-100">Timeline</h3>
    <ul className="mt-4 space-y-4 text-sm text-slate-600 dark:text-slate-300">
      {items.map((item) => (
        <li key={item.id} className="flex items-start gap-4">
          <span className="mt-1 text-lg">{item.type === 'upload' ? '📤' : item.type === 'chat' ? '💬' : '🔎'}</span>
          <div>
            <p className="font-semibold text-slate-700 dark:text-slate-100">{item.title}</p>
            <p className="text-xs text-slate-400 dark:text-slate-500">{formatDateTime(item.created_at)}</p>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-300">{item.description}</p>
          </div>
        </li>
      ))}
      {items.length === 0 && <p className="text-xs text-slate-400">No activity yet.</p>}
    </ul>
  </div>
);
