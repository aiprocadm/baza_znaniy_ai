import type { ReactNode } from 'react';

/**
 * DataTable renders responsive table with sticky header.
 */
export type Column<T> = {
  key: keyof T | string;
  header: string;
  render?: (row: T) => ReactNode;
};

export type DataTableProps<T> = {
  data: T[];
  columns: Column<T>[];
  emptyState?: ReactNode;
};

export const DataTable = <T extends Record<string, unknown>>({ data, columns, emptyState }: DataTableProps<T>) => (
  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
    <div className="max-h-[420px] overflow-auto">
      <table className="min-w-full divide-y divide-slate-200 text-sm dark:divide-slate-800">
        <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500 dark:bg-slate-800 dark:text-slate-400">
          <tr>
            {columns.map((column) => (
              <th key={String(column.key)} className="px-4 py-3">
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 text-slate-700 dark:divide-slate-800 dark:text-slate-200">
          {data.map((row) => (
            <tr key={String(row.id ?? crypto.randomUUID())} className="hover:bg-slate-50 dark:hover:bg-slate-800/80">
              {columns.map((column) => (
                <td key={String(column.key)} className="px-4 py-3">
                  {column.render ? column.render(row) : String(row[column.key as keyof T] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.length === 0 && (
        <div className="p-6 text-center text-sm text-slate-500 dark:text-slate-400">{emptyState ?? 'No data available.'}</div>
      )}
    </div>
  </div>
);
