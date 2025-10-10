import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import type * as ApiModule from '../api';
import { FileUploader } from '../components/files/FileUploader';
import { NotificationProvider } from '../context/NotificationContext';

vi.mock('../api', async () => {
  const actual = (await vi.importActual('../api')) as typeof ApiModule;
  return {
    ...actual,
    uploadFile: vi.fn().mockResolvedValue({ data: { id: '1' } })
  };
});

const { uploadFile } = await import('../api');

afterEach(() => {
  vi.clearAllMocks();
});

describe('FileUploader', () => {
  it('uploads selected file', async () => {
    render(
      <NotificationProvider>
        <FileUploader />
      </NotificationProvider>
    );

    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const file = new File(['content'], 'manual.pdf', { type: 'application/pdf' });
    fireEvent.change(input, {
      target: { files: [file] }
    });
    await screen.findByText('manual.pdf');
    fireEvent.click(screen.getByRole('button', { name: /upload/i }));
    await waitFor(() => {
      expect(uploadFile).toHaveBeenCalledWith(file, { source: 'console' });
    });
  });
});
