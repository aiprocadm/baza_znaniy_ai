import { useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import { uploadFile } from '../../api';
import { useNotifications } from '../../context/NotificationContext';
import { useFilePreview } from '../../hooks/useFilePreview';
import { formatBytes } from '../../utils/format';

/**
 * FileUploader supports drag-and-drop, preview and optimistic upload.
 */
export const FileUploader = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const preview = useFilePreview(file);
  const { push } = useNotifications();

  const handleSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    setFile(selected ?? null);
  };

  const handleUpload = async () => {
    if (!file) {
      push({ title: 'No file selected', type: 'error' });
      return;
    }
    setLoading(true);
    try {
      await uploadFile(file, { source: 'console' });
      push({ title: 'Upload started', description: file.name, type: 'success' });
      setFile(null);
      if (inputRef.current) {
        inputRef.current.value = '';
      }
    } catch (error) {
      push({ title: 'Upload failed', description: (error as Error).message, type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="rounded-2xl border border-dashed border-slate-300 bg-white/70 p-6 shadow-sm transition hover:border-primary-300 dark:border-slate-700 dark:bg-slate-900/70">
      <div
        className="flex flex-col items-center justify-center gap-3"
        onDragOver={(event) => {
          event.preventDefault();
        }}
        onDrop={(event) => {
          event.preventDefault();
          const dropped = event.dataTransfer.files?.[0];
          setFile(dropped ?? null);
        }}
      >
        <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">Drag & drop or click to upload</p>
        <p className="text-xs text-slate-500 dark:text-slate-400">PDF, DOCX, TXT, XLSX, PPTX up to 50 MB</p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,.pptx,.xlsx,.md"
          onChange={handleSelect}
          className="hidden"
          data-testid="file-input"
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary-500"
        >
          Browse files
        </button>
        {file && (
          <div className="mt-4 w-full rounded-xl border border-slate-200 bg-white p-4 text-left text-sm shadow dark:border-slate-700 dark:bg-slate-800">
            <p className="font-medium text-slate-700 dark:text-slate-200">{file.name}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">{formatBytes(file.size)}</p>
            {preview && preview.startsWith('data:image') && (
              <img src={preview} alt={file.name} className="mt-3 h-32 w-full rounded-lg object-cover" />
            )}
            <button
              type="button"
              onClick={handleUpload}
              disabled={loading}
              className="mt-4 inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-500"
            >
              {loading ? 'Uploading…' : 'Upload'}
            </button>
          </div>
        )}
      </div>
    </section>
  );
};
