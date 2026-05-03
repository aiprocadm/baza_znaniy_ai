import axios, { AxiosHeaders } from 'axios';
import type { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { __resetClientInterceptorsStateForTests, apiClient } from '../api/client';

const make401Error = (url: string): AxiosError =>
  ({
    config: {
      url,
      headers: AxiosHeaders.from({})
    } as InternalAxiosRequestConfig,
    response: {
      status: 401,
      statusText: 'Unauthorized',
      headers: {},
      config: {} as InternalAxiosRequestConfig,
      data: { detail: 'Unauthorized' }
    },
    isAxiosError: true,
    name: 'AxiosError',
    message: 'Unauthorized',
    toJSON: () => ({})
  }) as AxiosError;

const getRejectedResponseHandler = () => {
  const responseHandlers = (apiClient.interceptors.response as { handlers: Array<{ rejected: (error: AxiosError) => Promise<unknown> }> }).handlers;
  return responseHandlers[0].rejected;
};

describe('apiClient interceptors', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    __resetClientInterceptorsStateForTests();
    window.localStorage.clear();
    window.localStorage.setItem(
      'operations-console.session',
      JSON.stringify({ access_token: 'stale-token', refresh_token: 'refresh-token' })
    );
  });

  it('performs token refresh once for parallel 401 responses and retries original requests', async () => {
    const refreshSpy = vi.spyOn(axios, 'post').mockResolvedValue({
      data: { access_token: 'fresh-token', refresh_token: 'fresh-refresh' }
    });
    const requestSpy = vi.spyOn(apiClient, 'request').mockResolvedValue({ data: { ok: true } });

    const rejected = getRejectedResponseHandler();

    await Promise.all([rejected(make401Error('/files')), rejected(make401Error('/files'))]);

    expect(refreshSpy).toHaveBeenCalledTimes(1);
    expect(requestSpy).toHaveBeenCalledTimes(2);
    expect(
      (requestSpy.mock.calls[0][0] as InternalAxiosRequestConfig).headers.get('Authorization')
    ).toBe('Bearer fresh-token');

    const storedSession = JSON.parse(window.localStorage.getItem('operations-console.session') ?? '{}');
    expect(storedSession.access_token).toBe('fresh-token');
    expect(storedSession.refresh_token).toBe('fresh-refresh');
  });

  it('forces logout and redirects to /login when refresh fails', async () => {
    vi.spyOn(axios, 'post').mockRejectedValue(make401Error('/auth/refresh'));
    const requestSpy = vi.spyOn(apiClient, 'request').mockResolvedValue({ data: { ok: true } });
    const rejected = getRejectedResponseHandler();

    await expect(rejected(make401Error('/search'))).rejects.toMatchObject({ status: 401 });
    expect(requestSpy).not.toHaveBeenCalled();
    expect(window.localStorage.getItem('operations-console.session')).toBeNull();
    expect(window.location.pathname).toBe('/login');
  });

  it('retries failed request with updated access token after successful refresh', async () => {
    vi.spyOn(axios, 'post').mockResolvedValue({
      data: { access_token: 'updated-token', refresh_token: 'updated-refresh' }
    });

    const requestSpy = vi.spyOn(apiClient, 'request').mockResolvedValue({ data: { retried: true } });

    const rejected = getRejectedResponseHandler();
    await rejected(make401Error('/admin/users'));

    expect(requestSpy).toHaveBeenCalledTimes(1);
    const retriedConfig = requestSpy.mock.calls[0][0] as InternalAxiosRequestConfig;
    expect(retriedConfig._retry).toBe(true);
    expect(retriedConfig.headers.get('Authorization')).toBe('Bearer updated-token');
  });
});
