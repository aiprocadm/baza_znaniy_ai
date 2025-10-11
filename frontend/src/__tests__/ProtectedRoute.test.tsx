import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { vi } from 'vitest';
import { ProtectedRoute } from '../components/navigation/ProtectedRoute';
import { useAuth } from '../context/AuthContext';
import type { Mock } from 'vitest';

vi.mock('../context/AuthContext', () => ({
  useAuth: vi.fn()
}));

const useAuthMock = useAuth as unknown as Mock;

describe('ProtectedRoute', () => {
  beforeEach(() => {
    useAuthMock.mockReset();
  });

  it('redirects unauthenticated users to the login page', async () => {
    useAuthMock.mockReturnValue({ isAuthenticated: false, hasRole: vi.fn() });

    render(
      <MemoryRouter initialEntries={['/admin']}>
        <Routes>
          <Route element={<ProtectedRoute roles={['admin']} />}> 
            <Route path="/admin" element={<div>Admin console</div>} />
          </Route>
          <Route path="/login" element={<div>Login page</div>} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Login page')).toBeInTheDocument();
    });
  });

  it('redirects users without the required role', async () => {
    useAuthMock.mockReturnValue({ isAuthenticated: true, hasRole: vi.fn().mockReturnValue(false) });

    render(
      <MemoryRouter initialEntries={['/admin']}>
        <Routes>
          <Route element={<ProtectedRoute roles={['admin']} />}>
            <Route path="/admin" element={<div>Admin console</div>} />
          </Route>
          <Route path="/" element={<div>Home page</div>} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Home page')).toBeInTheDocument();
    });
  });

  it('renders the outlet for authorized users', () => {
    useAuthMock.mockReturnValue({ isAuthenticated: true, hasRole: vi.fn().mockReturnValue(true) });

    render(
      <MemoryRouter initialEntries={['/admin']}>
        <Routes>
          <Route element={<ProtectedRoute roles={['admin']} />}>
            <Route path="/admin" element={<div>Admin console</div>} />
          </Route>
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByText('Admin console')).toBeInTheDocument();
  });
});
