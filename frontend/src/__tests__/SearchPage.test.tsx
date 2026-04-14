import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { act } from 'react';
import type * as ApiModule from '../api';
import { SearchPage } from '../pages/SearchPage';
import { NotificationProvider } from '../context/NotificationContext';
import { ThemeProvider } from '../context/ThemeContext';
import { LocaleProvider } from '../context/LocaleContext';

vi.mock('../api', async () => {
  const actual = (await vi.importActual('../api')) as typeof ApiModule;
  return {
    ...actual,
    searchDocuments: vi.fn().mockResolvedValue({
      data: {
        hits: [
          {
            file: 'docs/replication.md',
            page: 1,
            text: 'Step-by-step instructions',
            score: 0.98,
          }
        ],
        query: 'replication'
      }
    })
  };
});

vi.mock('../hooks/useDebounce', () => ({
  useDebounce: (value: unknown) => value
}));

const { searchDocuments } = await import('../api');

afterEach(() => {
  vi.clearAllMocks();
});

const renderPage = () =>
  render(
    <LocaleProvider>
      <ThemeProvider>
        <NotificationProvider>
          <SearchPage />
        </NotificationProvider>
      </ThemeProvider>
    </LocaleProvider>
  );

describe('SearchPage', () => {
  it('performs search on submit', async () => {
    renderPage();
    const input = screen.getByPlaceholderText('How to configure replication?');
    await act(async () => {
      await userEvent.type(input, 'replication');
      await userEvent.click(screen.getByRole('button', { name: /run search/i }));
    });
    await waitFor(() => {
      expect(searchDocuments).toHaveBeenCalledWith({
        query: 'replication',
        top_k: 10,
        owner: undefined,
        tags: undefined
      });
    });
  });

  it('passes owner and tags filters', async () => {
    renderPage();
    await act(async () => {
      await userEvent.type(screen.getByPlaceholderText('How to configure replication?'), 'replication');
      await userEvent.type(screen.getByPlaceholderText('alice@kb.ai'), 'alice@kb.ai');
      await userEvent.type(screen.getByPlaceholderText('production,onboarding'), 'production, onboarding');
      await userEvent.click(screen.getByRole('button', { name: /run search/i }));
    });

    await waitFor(() => {
      expect(searchDocuments).toHaveBeenCalledWith({
        query: 'replication',
        top_k: 10,
        owner: 'alice@kb.ai',
        tags: ['production', 'onboarding']
      });
    });
  });
});
