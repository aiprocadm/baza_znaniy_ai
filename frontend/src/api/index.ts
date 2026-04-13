import { apiClient } from './client';

/**
 * Central API SDK describing backend contracts.
 */
export type SystemStatus = {
  status: 'ok' | 'degraded' | 'error';
  version: string;
  services: Array<{
    name: string;
    status: 'healthy' | 'degraded' | 'offline';
    latency_ms: number;
    last_error?: string | null;
  }>;
  stats: {
    documents: number;
    ingestions: number;
    errors: number;
  };
};

export type SearchFilter = {
  query: string;
  top_k?: number;
};

export type SearchResult = {
  file?: string | null;
  page?: number | null;
  score: number;
  text: string;
};

export type ActivityItem = {
  id: string;
  type: 'upload' | 'ingest' | 'chat' | 'search';
  title: string;
  description: string;
  created_at: string;
};

export type FileMeta = {
  id: string;
  name: string;
  size: number;
  mime_type: string;
  status: 'processing' | 'indexed' | 'error';
  created_at: string;
};

export type User = {
  id: number;
  full_name?: string | null;
  email: string;
  role: BackendRole;
  is_active: boolean;
  tenant_slug?: string | null;
  created_at: string;
  updated_at: string;
  last_login_at?: string | null;
};

export type CreateUserPayload = {
  full_name: string;
  email: string;
  password: string;
  role: 'admin' | 'manager' | 'member';
  is_active: boolean;
  tenant_slug: string;
};

export type TokenResponse = {
  access_token: string;
  refresh_token: string;
  token_type: 'bearer';
  expires_in: number;
};

export type RefreshRequest = {
  refresh_token: string;
};

export type LogoutRequest = {
  refresh_token: string;
};

export type ApiKey = {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at?: string | null;
};

export type SystemSettings = {
  qdrant_url: string;
  llm_model: string;
  ingestion_parallelism: number;
  allow_guest_access: boolean;
};

export const fetchSystemStatus = () =>
  apiClient.get<{ status: string; version: string }>('/ops/health').then((response) => ({
    ...response,
    data: {
      status: response.data.status === 'ok' ? 'ok' : 'degraded',
      version: response.data.version,
      services: [
        {
          name: 'api',
          status: response.data.status === 'ok' ? 'healthy' : 'degraded',
          latency_ms: 0,
          last_error: null
        }
      ],
      stats: {
        documents: 0,
        ingestions: 0,
        errors: 0
      }
    } satisfies SystemStatus
  }));

export const searchDocuments = (payload: SearchFilter) =>
  apiClient.get<{ query: string; hits: SearchResult[] }>('/search', { params: payload });

export const fetchActivities = () => apiClient.get<ActivityItem[]>('/activities');

export const fetchFiles = () => apiClient.get<FileMeta[]>('/files');

export const uploadFile = (file: File, metadata: Record<string, unknown>) => {
  const formData = new FormData();
  formData.append('file', file);
  Object.entries(metadata).forEach(([key, value]) => {
    formData.append(key, String(value));
  });
  return apiClient.post<FileMeta>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
};

export const fetchUsers = () => apiClient.get<User[]>('/users');

export const createUser = (payload: CreateUserPayload) => apiClient.post<User>('/users', payload);

export const updateUser = () =>
  Promise.reject(new Error('User update endpoint is not available on backend yet.'));

export const deleteUser = () =>
  Promise.reject(new Error('User delete endpoint is not available on backend yet.'));

export const fetchApiKeys = () => apiClient.get<ApiKey[]>('/admin/api-keys');

export const rotateApiKey = (id: string) => apiClient.post<{ secret: string }>(`/admin/api-keys/${id}/rotate`, {});

export const fetchSettings = () => apiClient.get<SystemSettings>('/admin/settings');

export const updateSettings = (payload: SystemSettings) => apiClient.put<SystemSettings>('/admin/settings', payload);

export const login = (email: string, password: string) =>
  apiClient.post<TokenResponse>('/auth/login', { email, password });
export const refreshToken = (payload: RefreshRequest) =>
  apiClient.post<TokenResponse>('/auth/refresh', payload);
export const logout = (payload: LogoutRequest) => apiClient.post<{ ok: true }>('/auth/logout', payload);
export type BackendRole = 'admin' | 'manager' | 'member';
