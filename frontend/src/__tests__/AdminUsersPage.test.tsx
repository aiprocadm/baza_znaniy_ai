import { describe, it, expect, vi, afterEach, beforeEach, type Mock } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { act } from 'react';
import userEvent from '@testing-library/user-event';
import type * as ApiModule from '../api';
import { AdminUsersPage } from '../pages/AdminUsersPage';
import { NotificationProvider } from '../context/NotificationContext';
import { ThemeProvider } from '../context/ThemeContext';
import { LocaleProvider } from '../context/LocaleContext';

const user = userEvent.setup();

vi.mock('../api', async () => {
  const actual = (await vi.importActual('../api')) as typeof ApiModule;
  return {
    ...actual,
    fetchUsers: vi.fn().mockResolvedValue({
      data: [
        {
          id: 1,
          full_name: 'Alice',
          email: 'alice@example.com',
          role: 'member',
          is_active: true,
          tenant_slug: 'default',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          last_login_at: null
        }
      ]
    }),
    createUser: vi.fn().mockResolvedValue({ data: { id: '2' } })
  };
});

const { fetchUsers, createUser } = await import('../api');
const fetchUsersMock = fetchUsers as unknown as Mock;
const createUserMock = createUser as unknown as Mock;

const renderPage = () =>
  render(
    <LocaleProvider>
      <ThemeProvider>
        <NotificationProvider>
          <AdminUsersPage />
        </NotificationProvider>
      </ThemeProvider>
    </LocaleProvider>
  );

afterEach(() => {
  vi.clearAllMocks();
});

beforeEach(() => {
  fetchUsersMock.mockClear().mockResolvedValue({
    data: [
      {
        id: 1,
        full_name: 'Alice',
        email: 'alice@example.com',
        role: 'member',
        is_active: true,
        tenant_slug: 'default',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        last_login_at: null
      }
    ]
  });
});

describe('AdminUsersPage', () => {
  it('loads users and submits create form', async () => {
    renderPage();
    await waitFor(() => expect(fetchUsersMock).toHaveBeenCalled());
    await act(async () => {
      await user.click(screen.getByRole('button', { name: /new user/i }));
    });
    const nameInput = await screen.findByLabelText('Name');
    await act(async () => {
      await user.clear(nameInput);
      await user.type(nameInput, 'Bob');
    });
    const emailInput = await screen.findByLabelText('Email');
    await act(async () => {
      await user.clear(emailInput);
      await user.type(emailInput, 'bob@example.com');
    });
    const passwordInput = await screen.findByLabelText('Password');
    await act(async () => {
      await user.type(passwordInput, 'password123');
    });
    await act(async () => {
      await user.click(screen.getByRole('button', { name: /create/i }));
    });
    await waitFor(() => {
      expect(createUserMock).toHaveBeenCalledWith({
        full_name: 'Bob',
        email: 'bob@example.com',
        password: 'password123',
        role: 'member',
        tenant_slug: 'default',
        is_active: true
      });
    });
  });
});
