import { useMemo } from 'react';
import { fetchActivities, fetchSystemStatus, type ActivityItem, type SystemStatus } from '../api';
import { useApi } from '../hooks/useApi';
import { formatDateTime, mapStatusColor } from '../utils/format';
import { DataTable, type Column } from '../components/data/DataTable';

/**
 * AdminMonitoringPage visualizes service health and operational logs.
 */
export const AdminMonitoringPage = () => {
  const { data: status } = useApi<SystemStatus>(() => fetchSystemStatus().then((res) => res.data), []);
  const { data: activities } = useApi<ActivityItem[]>(() => fetchActivities().then((res) => res.data), []);

  const columns: Column<ActivityItem>[] = useMemo(
    () => [
      { key: 'type', header: 'Type' },
      { key: 'title', header: 'Title' },
      {
        key: 'description',
        header: 'Details',
        render: (row) => <span className="text-xs text-slate-500">{row.description}</span>
      },
      {
        key: 'created_at',
        header: 'Timestamp',
        render: (row) => formatDateTime(row.created_at)
      }
    ],
    []
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">Infrastructure monitoring</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Inspect service health, queue latencies, and ingestion logs to keep operations resilient.
        </p>
      </div>
      <div className="grid gap-6 md:grid-cols-3">
        {status?.services.map((service) => (
          <div key={service.name} className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            <p className="text-sm font-medium text-slate-500">{service.name}</p>
            <p className={`mt-2 text-3xl font-semibold ${mapStatusColor(service.status)} text-balance capitalize`}>
              {service.status}
            </p>
            <p className="mt-1 text-xs text-slate-400">Latency {service.latency_ms} ms</p>
            {service.last_error && <p className="mt-2 text-xs text-rose-500">{service.last_error}</p>}
          </div>
        ))}
        {!status && (
          <div className="md:col-span-3 rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-500 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            Loading metrics…
          </div>
        )}
      </div>
      <DataTable data={activities ?? []} columns={columns} emptyState="No activity captured yet." />
    </div>
  );
};
