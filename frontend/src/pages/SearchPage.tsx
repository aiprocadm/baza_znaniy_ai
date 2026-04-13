import { useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { searchDocuments, type SearchResult } from '../api';
import { DataTable, type Column } from '../components/data/DataTable';
import { useNotifications } from '../context/NotificationContext';
import { useDebounce } from '../hooks/useDebounce';

/**
 * SearchPage provides advanced filtering and ranking preview.
 */
const schema = z.object({
  query: z.string().min(2, 'Enter at least 2 characters'),
  top_k: z.coerce.number().min(1).max(50).default(10)
});

type FormValues = z.infer<typeof schema>;

export const SearchPage = () => {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors }
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { query: '', top_k: 10 }
  });
  const { push } = useNotifications();
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  const query = watch('query');
  const topK = watch('top_k');

  const debouncedQuery = useDebounce(query, 400);

  const columns: Column<SearchResult>[] = useMemo(
    () => [
      { key: 'file', header: 'File' },
      {
        key: 'text',
        header: 'Snippet',
        render: (row) => <span className="text-xs text-slate-500">{row.text}</span>
      },
      {
        key: 'score',
        header: 'Score',
        render: (row) => row.score.toFixed(2)
      },
      {
        key: 'page',
        header: 'Page',
        render: (row) => row.page ?? '—'
      }
    ],
    []
  );

  useEffect(() => {
    if (!debouncedQuery || debouncedQuery.length < 2) {
      setResults([]);
      return;
    }
    setLoading(true);
    searchDocuments({
      query: debouncedQuery,
      top_k: topK
    })
      .then((response) => setResults(response.data.hits))
      .catch((error) => push({ title: 'Search failed', description: error.message, type: 'error' }))
      .finally(() => setLoading(false));
  }, [debouncedQuery, topK, push]);

  const onSubmit = handleSubmit(async (values) => {
    try {
      setLoading(true);
      const response = await searchDocuments({
        query: values.query,
        top_k: values.top_k
      });
      setResults(response.data.hits);
    } catch (error) {
      push({ title: 'Search failed', description: (error as Error).message, type: 'error' });
    } finally {
      setLoading(false);
    }
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">Search knowledge base</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">Combine filters to refine RAG retrieval.</p>
      </div>
      <form
        onSubmit={onSubmit}
        className="grid gap-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900 md:grid-cols-2 lg:grid-cols-4"
      >
        <label className="block text-sm">
          <span className="font-medium text-slate-600 dark:text-slate-300">Query</span>
          <input
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
            placeholder="How to configure replication?"
            {...register('query')}
          />
          {errors.query && <span className="mt-1 block text-xs text-rose-500">{errors.query.message}</span>}
        </label>
        <label className="block text-sm">
          <span className="font-medium text-slate-600 dark:text-slate-300">Top K</span>
          <input
            type="number"
            min={1}
            max={50}
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
            {...register('top_k', { valueAsNumber: true })}
          />
          {errors.top_k && <span className="mt-1 block text-xs text-rose-500">{errors.top_k.message}</span>}
        </label>
        <div className="lg:col-span-4">
          <button
            type="submit"
            className="inline-flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary-500"
          >
            {loading ? 'Searching…' : 'Run search'}
          </button>
        </div>
      </form>
      <DataTable
        data={results}
        columns={columns}
        emptyState={loading ? 'Searching…' : 'No results yet. Try adjusting the filters.'}
      />
    </div>
  );
};
