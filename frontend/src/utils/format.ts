/**
 * Utility helpers for consistent formatting.
 */
export const formatDateTime = (value: string | number | Date) =>
  new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short'
  }).format(typeof value === 'string' || typeof value === 'number' ? new Date(value) : value);

export const formatBytes = (size: number) => {
  if (size === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const exponent = Math.min(Math.floor(Math.log(size) / Math.log(1024)), units.length - 1);
  return `${(size / 1024 ** exponent).toFixed(1)} ${units[exponent]}`;
};

export const mapStatusColor = (status: string) => {
  switch (status) {
    case 'healthy':
    case 'active':
    case 'indexed':
      return 'text-green-500';
    case 'degraded':
    case 'processing':
      return 'text-amber-500';
    case 'offline':
    case 'error':
    case 'blocked':
      return 'text-rose-500';
    default:
      return 'text-slate-500';
  }
};
