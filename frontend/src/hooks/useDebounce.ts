import { useEffect, useState } from 'react';

/**
 * useDebounce returns debounced value for expensive operations (search filters).
 */
export const useDebounce = <T>(value: T, delay = 300) => {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);

  return debounced;
};
