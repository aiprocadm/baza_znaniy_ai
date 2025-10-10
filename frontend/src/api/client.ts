import axios from 'axios';

/**
 * Axios instance with sane defaults and error normalization.
 */
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '/api/v1',
  timeout: 10_000,
  withCredentials: true
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      return Promise.reject({
        status: error.response.status,
        message: error.response.data?.detail ?? 'Request failed',
        details: error.response.data
      });
    }
    if (error.request) {
      return Promise.reject({ status: 0, message: 'Network error', details: null });
    }
    return Promise.reject({ status: 0, message: error.message, details: null });
  }
);
