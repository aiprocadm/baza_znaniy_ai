import { useEffect, useState } from 'react';

/**
 * useApi executes provided async function and returns request lifecycle state.
 */
export const useApi = <TData, TError = unknown>(request: () => Promise<TData>, deps: unknown[] = []) => {
  const [data, setData] = useState<TData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<TError | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    request()
      .then((response) => {
        if (!mounted) return;
        setData(response);
        setError(null);
      })
      .catch((err: TError) => {
        if (!mounted) return;
        setError(err);
      })
      .finally(() => mounted && setLoading(false));
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, error } as const;
};
