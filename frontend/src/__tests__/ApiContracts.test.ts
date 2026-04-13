import { describe, it, expect, vi, afterEach } from 'vitest';
import { apiClient } from '../api/client';
import { createUser, fetchUsers, login, refreshToken, searchDocuments } from '../api';

describe('API SDK smoke contracts', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('uses auth endpoints with backend-compatible payloads', async () => {
    const postSpy = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });

    await login('alice@example.com', 'password123');
    await refreshToken({ refresh_token: 'refresh-token' });

    expect(postSpy).toHaveBeenNthCalledWith(1, '/auth/login', {
      email: 'alice@example.com',
      password: 'password123'
    });
    expect(postSpy).toHaveBeenNthCalledWith(2, '/auth/refresh', {
      refresh_token: 'refresh-token'
    });
  });

  it('calls search as GET /search with query params', async () => {
    const getSpy = vi.spyOn(apiClient, 'get').mockResolvedValue({ data: {} });

    await searchDocuments({ query: 'replication', top_k: 5 });

    expect(getSpy).toHaveBeenCalledWith('/search', {
      params: { query: 'replication', top_k: 5 }
    });
  });

  it('uses /users endpoints for list and create', async () => {
    const getSpy = vi.spyOn(apiClient, 'get').mockResolvedValue({ data: {} });
    const postSpy = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });

    await fetchUsers();
    await createUser({
      full_name: 'Bob',
      email: 'bob@example.com',
      password: 'password123',
      role: 'member',
      is_active: true,
      tenant_slug: 'default'
    });

    expect(getSpy).toHaveBeenCalledWith('/users');
    expect(postSpy).toHaveBeenCalledWith('/users', {
      full_name: 'Bob',
      email: 'bob@example.com',
      password: 'password123',
      role: 'member',
      is_active: true,
      tenant_slug: 'default'
    });
  });
});
