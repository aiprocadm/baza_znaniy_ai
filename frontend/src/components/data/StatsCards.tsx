import type { SystemStatus } from '../../api';
import { mapStatusColor } from '../../utils/format';

/**
 * StatsCards visualizes aggregated metrics from the backend.
 */
export const StatsCards = ({ status }: { status: SystemStatus | null }) => {
  if (!status) {
    return (
      <div className="grid gap-4 md:grid-cols-3">
        {[1, 2, 3].map((key) => (
          <div key={key} className="h-28 animate-pulse rounded-2xl bg-slate-200/60 dark:bg-slate-800/60" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <p className="text-sm font-medium text-slate-500">Documents</p>
        <p className="mt-2 text-3xl font-semibold text-slate-900 dark:text-white">{status.stats.documents}</p>
        <p className="mt-1 text-xs text-slate-400">Indexed assets</p>
      </div>
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <p className="text-sm font-medium text-slate-500">Active ingestions</p>
        <p className="mt-2 text-3xl font-semibold text-slate-900 dark:text-white">{status.stats.ingestions}</p>
        <p className="mt-1 text-xs text-slate-400">Queue throughput</p>
      </div>
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <p className="text-sm font-medium text-slate-500">Errors (24h)</p>
        <p className="mt-2 text-3xl font-semibold text-slate-900 dark:text-white">{status.stats.errors}</p>
        <p className="mt-1 text-xs text-slate-400">Alerts in the last 24 hours</p>
      </div>
    </div>
  );
};

/**
 * ServiceStatusList renders health states for dependencies.
 */
export const ServiceStatusList = ({ status }: { status: SystemStatus | null }) => (
  <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
    <h3 className="text-base font-semibold text-slate-800 dark:text-slate-100">Service health</h3>
    <ul className="mt-4 space-y-3 text-sm text-slate-600 dark:text-slate-300">
      {status?.services.map((service) => (
        <li key={service.name} className="flex items-center justify-between">
          <div>
            <p className="font-medium">{service.name}</p>
            <p className="text-xs text-slate-400">{service.latency_ms} ms</p>
          </div>
          <span className={`text-sm font-semibold ${mapStatusColor(service.status)}`}>{service.status}</span>
        </li>
      ))}
      {!status && <li className="text-xs text-slate-400">Loading services…</li>}
    </ul>
  </div>
);
