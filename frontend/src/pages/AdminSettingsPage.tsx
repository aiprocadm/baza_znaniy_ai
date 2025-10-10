import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { DataTable, type Column } from '../components/data/DataTable';
import { fetchApiKeys, fetchSettings, rotateApiKey, updateSettings, type ApiKey, type SystemSettings } from '../api';
import { useApi } from '../hooks/useApi';
import { useNotifications } from '../context/NotificationContext';

/**
 * AdminSettingsPage handles platform configuration and API credentials.
 */
const schema = z.object({
  qdrant_url: z.string().url(),
  llm_model: z.string().min(3),
  ingestion_parallelism: z.coerce.number().min(1).max(32),
  allow_guest_access: z.boolean()
});

type FormValues = z.infer<typeof schema>;

export const AdminSettingsPage = () => {
  const { data: settings, loading: settingsLoading } = useApi<SystemSettings>(() => fetchSettings().then((res) => res.data), []);
  const { data: apiKeys, loading: keysLoading } = useApi<ApiKey[]>(() => fetchApiKeys().then((res) => res.data), []);
  const { register, handleSubmit, reset, formState } = useForm<FormValues>({ resolver: zodResolver(schema) });
  const { push } = useNotifications();

  useEffect(() => {
    if (settings) {
      reset(settings);
    }
  }, [settings, reset]);

  const onSubmit = handleSubmit(async (values) => {
    try {
      await updateSettings(values);
      push({ title: 'Settings saved', type: 'success' });
    } catch (error) {
      push({ title: 'Failed to save', description: (error as Error).message, type: 'error' });
    }
  });

  const rotate = async (key: ApiKey) => {
    try {
      const response = await rotateApiKey(key.id);
      push({ title: 'API key rotated', description: response.data.secret, type: 'success', ttl: 8000 });
    } catch (error) {
      push({ title: 'Rotation failed', description: (error as Error).message, type: 'error' });
    }
  };

  const columns: Column<ApiKey>[] = [
    { key: 'name', header: 'Name' },
    { key: 'prefix', header: 'Prefix' },
    {
      key: 'created_at',
      header: 'Created',
      render: (row) => new Date(row.created_at).toLocaleString()
    },
    {
      key: 'last_used_at',
      header: 'Last used',
      render: (row) => (row.last_used_at ? new Date(row.last_used_at).toLocaleString() : '—')
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (row) => (
        <button
          type="button"
          onClick={() => rotate(row)}
          className="rounded-lg border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
        >
          Rotate
        </button>
      )
    }
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">Platform settings</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Configure storage endpoints, LLM providers, and access policies.
        </p>
      </div>
      <form
        onSubmit={onSubmit}
        className="grid gap-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900 md:grid-cols-2"
      >
        <label className="block text-sm md:col-span-2">
          <span className="font-medium text-slate-600 dark:text-slate-300">Qdrant URL</span>
          <input
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
            {...register('qdrant_url')}
            placeholder="http://localhost:6333"
          />
          {formState.errors.qdrant_url && <span className="mt-1 block text-xs text-rose-500">{formState.errors.qdrant_url.message}</span>}
        </label>
        <label className="block text-sm">
          <span className="font-medium text-slate-600 dark:text-slate-300">LLM model ID</span>
          <input
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
            {...register('llm_model')}
            placeholder="meta-llama/Meta-Llama-3-8B-Instruct"
          />
          {formState.errors.llm_model && <span className="mt-1 block text-xs text-rose-500">{formState.errors.llm_model.message}</span>}
        </label>
        <label className="block text-sm">
          <span className="font-medium text-slate-600 dark:text-slate-300">Ingestion parallelism</span>
          <input
            type="number"
            min={1}
            max={32}
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
            {...register('ingestion_parallelism', { valueAsNumber: true })}
          />
          {formState.errors.ingestion_parallelism && (
            <span className="mt-1 block text-xs text-rose-500">{formState.errors.ingestion_parallelism.message}</span>
          )}
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-slate-300 text-primary-600 focus:ring-primary-500"
            {...register('allow_guest_access')}
          />
          Allow guest read-only access
        </label>
        <div className="md:col-span-2">
          <button
            type="submit"
            disabled={settingsLoading}
            className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary-500 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            Save settings
          </button>
        </div>
      </form>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">API keys</h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">Rotate secrets frequently to reduce risk.</p>
        </div>
        <DataTable data={keysLoading ? [] : apiKeys ?? []} columns={columns} emptyState={keysLoading ? 'Loading…' : 'No keys yet.'} />
      </div>
    </div>
  );
};
