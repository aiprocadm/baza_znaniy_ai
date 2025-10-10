import { useCallback } from 'react';
import { fetchActivities, fetchFiles, fetchSystemStatus, type SystemStatus, type ActivityItem, type FileMeta } from '../api';
import { StatsCards, ServiceStatusList } from '../components/data/StatsCards';
import { ActivityTimeline } from '../components/data/ActivityTimeline';
import { FileUploader } from '../components/files/FileUploader';
import { DataTable, type Column } from '../components/data/DataTable';
import { useApi } from '../hooks/useApi';
import { useNotifications } from '../context/NotificationContext';
import { formatDateTime, formatBytes, mapStatusColor } from '../utils/format';

/**
 * HomePage aggregates system overview for operators.
 */
export const HomePage = () => {
  const { data: statusResponse, loading: statusLoading } = useApi<SystemStatus>(() => fetchSystemStatus().then((res) => res.data), []);
  const { data: activityResponse } = useApi<ActivityItem[]>(() => fetchActivities().then((res) => res.data), []);
  const { data: filesResponse, loading: filesLoading } = useApi<FileMeta[]>(() => fetchFiles().then((res) => res.data), []);
  const { push } = useNotifications();

  const refresh = useCallback(async () => {
    await Promise.all([fetchSystemStatus(), fetchActivities(), fetchFiles()]);
    push({ title: 'Data refreshed', type: 'success' });
  }, [push]);

  const fileColumns: Column<FileMeta>[] = [
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
      header: 'Created at',
      render: (row) => formatDateTime(row.created_at)
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">KB.AI Operations Console</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">Monitor ingestion, search, and chat in real time.</p>
        </div>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={refresh}
            className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-600 shadow-sm transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
          >
            Refresh
          </button>
          <a
            href="/dashboard"
            className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary-500"
          >
            Go to dashboard
          </a>
        </div>
      </div>
      <StatsCards status={statusLoading ? null : statusResponse ?? null} />
      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <ActivityTimeline items={activityResponse ?? []} />
        <ServiceStatusList status={statusLoading ? null : statusResponse ?? null} />
      </div>
      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Recent files</h2>
          <DataTable data={filesLoading ? [] : filesResponse ?? []} columns={fileColumns} emptyState={filesLoading ? 'Loading…' : 'No files yet'} />
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <h3 className="text-base font-semibold text-slate-800 dark:text-slate-100">Upload center</h3>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Drag-and-drop files or connect via API. The queue updates in real time with ingestion progress.
          </p>
          <div className="mt-4">
            <FileUploader />
          </div>
        </div>
      </div>
    </div>
  );
};
