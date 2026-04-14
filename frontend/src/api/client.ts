import axios from 'axios';
import type { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { forceLogout, getStoredTokens, updateStoredTokens } from '../context/authStorage';

type ApiError = {
  status: number;
  message: string;
  details: unknown;
};

type RetryableRequestConfig = InternalAxiosRequestConfig & {
  _retry?: boolean;
};

/**
 * Axios instance with sane defaults and error normalization.
 */
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '/api/v1',
  timeout: 10_000,
  withCredentials: true
});

let refreshPromise: Promise<string> | null = null;

const normalizeError = (error: AxiosError): ApiError => {
  if (error.response) {
    return {
      status: error.response.status,
      message: (error.response.data as { detail?: string } | undefined)?.detail ?? 'Request failed',
      details: error.response.data
    };
  }

  if (error.request) {
    return { status: 0, message: 'Network error', details: null };
  }

  return { status: 0, message: error.message, details: null };
};

const refreshAccessToken = async (): Promise<string> => {
  const { refreshToken } = getStoredTokens();
  if (!refreshToken) {
    throw new Error('Missing refresh token');
  }

  const response = await axios.post<{ access_token: string; refresh_token: string }>(
    '/auth/refresh',
    { refresh_token: refreshToken },
    {
      baseURL: apiClient.defaults.baseURL,
      timeout: apiClient.defaults.timeout,
      withCredentials: apiClient.defaults.withCredentials
    }
  );

  updateStoredTokens(response.data);
  return response.data.access_token;
};

apiClient.interceptors.request.use((config) => {
  const { accessToken } = getStoredTokens();

  if (accessToken) {
    config.headers.set('Authorization', `Bearer ${accessToken}`);
  }

  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as RetryableRequestConfig | undefined;
    const isUnauthorized = error.response?.status === 401;
    const isRefreshRequest = originalRequest?.url?.includes('/auth/refresh');

    if (originalRequest && isUnauthorized && !originalRequest._retry && !isRefreshRequest) {
      originalRequest._retry = true;

      try {
        if (!refreshPromise) {
          refreshPromise = refreshAccessToken().finally(() => {
            refreshPromise = null;
          });
        }

        const freshAccessToken = await refreshPromise;
        originalRequest.headers.set('Authorization', `Bearer ${freshAccessToken}`);

        return apiClient.request(originalRequest);
      } catch (refreshError) {
        forceLogout();
        return Promise.reject(
          normalizeError((refreshError as AxiosError) ?? error)
        );
      }
    }

    return Promise.reject(normalizeError(error));
  }
);

export const __resetClientInterceptorsStateForTests = () => {
  refreshPromise = null;
};
