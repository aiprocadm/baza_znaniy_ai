import { useMemo } from 'react';
import { fetchFiles, type FileMeta, fetchActivities, type ActivityItem } from '../api';
import { DataTable, type Column } from '../components/data/DataTable';
import { useApi } from '../hooks/useApi';
import { formatBytes, formatDateTime, mapStatusColor } from '../utils/format';
import { ChatPanel } from '../components/chat/ChatPanel';
import { ActivityTimeline } from '../components/data/ActivityTimeline';
import { useAuth } from '../context/AuthContext';

/**
 * DashboardPage is the personalized workspace for operators.
 */
export const DashboardPage = () => {
  const { data: files } = useApi<FileMeta[]>(() => fetchFiles().then((res) => res.data), []);
  const { data: activities } = useApi<ActivityItem[]>(() => fetchActivities().then((res) => res.data), []);
  const { user } = useAuth();

  const columns: Column<FileMeta>[] = useMemo(
    () => [
      { key: 'name', header: 'Name' },
      {
        key: 'size',
        header: 'Size',
        render: (row) => formatBytes(row.size)
      },
      { key: 'mime_type', header: 'Type' },
      {
        key: 'status',
        header: 'Status',
        render: (row) => <span className={`font-semibold ${mapStatusColor(row.status)}`}>{row.status}</span>
      },
      {
        key: 'created_at',
        header: 'Uploaded',
        render: (row) => formatDateTime(row.created_at)
      }
    ],
    []
  );

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-gradient-to-r from-primary-600 to-primary-800 p-6 text-white shadow-lg dark:border-slate-800">
        <h1 className="text-2xl font-semibold">Welcome back, {user?.name ?? 'operator'} 👋</h1>
        <p className="mt-2 max-w-3xl text-sm text-white/80">
          Track your ingestion jobs, share collections with the team, and collaborate with the assistant using the chat panel on the right.
        </p>
      </div>
      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Your files</h2>
          <DataTable data={files ?? []} columns={columns} emptyState="Upload your first document to see it here." />
        </div>
        <ActivityTimeline items={activities?.filter((item) => item.type !== 'chat') ?? []} />
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Saved searches</h2>
          <ul className="mt-3 space-y-3 text-sm text-slate-600 dark:text-slate-300">
            <li className="rounded-xl border border-slate-200 px-4 py-3 shadow-sm transition hover:border-primary-300 dark:border-slate-700 dark:hover:border-primary-500">
              🔖 Production incidents (top 20)
            </li>
            <li className="rounded-xl border border-slate-200 px-4 py-3 shadow-sm transition hover:border-primary-300 dark:border-slate-700 dark:hover:border-primary-500">
              🔖 Employee onboarding checklist
            </li>
            <li className="rounded-xl border border-slate-200 px-4 py-3 shadow-sm transition hover:border-primary-300 dark:border-slate-700 dark:hover:border-primary-500">
              🔖 Database replication recipes
            </li>
          </ul>
        </div>
        <ChatPanel />
      </div>
    </div>
  );
};
