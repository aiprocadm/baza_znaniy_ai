import { useEffect, useState } from 'react';

/**
 * useFilePreview converts file into a data URL for preview widgets.
 */
export const useFilePreview = (file: File | null) => {
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }
    const reader = new FileReader();
    reader.onload = () => setPreview(typeof reader.result === 'string' ? reader.result : null);
    reader.readAsDataURL(file);
    return () => reader.abort();
  }, [file]);

  return preview;
};
