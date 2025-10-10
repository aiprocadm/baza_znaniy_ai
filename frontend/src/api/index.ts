import { apiClient } from './client';
import type { Role, Session } from '../context/AuthContext';

/**
 * Central API SDK describing backend contracts.
 */
export type SystemStatus = {
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
  tags?: string[];
  owner?: string;
};

export type SearchResult = {
  id: string;
  title: string;
  snippet: string;
  score: number;
  source: string;
  updated_at: string;
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
  id: string;
  name: string;
  email: string;
  roles: Role[];
  status: 'active' | 'invited' | 'blocked';
};

export type CreateUserPayload = {
  name: string;
  email: string;
  roles: Role[];
};

export type UpdateUserPayload = Partial<CreateUserPayload> & { status?: User['status'] };

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

export const fetchSystemStatus = () => apiClient.get<SystemStatus>('/status');

export const searchDocuments = (payload: SearchFilter) =>
  apiClient.post<{ results: SearchResult[]; total: number }>('/search', payload);

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

export const fetchUsers = () => apiClient.get<User[]>('/admin/users');

export const createUser = (payload: CreateUserPayload) => apiClient.post<User>('/admin/users', payload);

export const updateUser = (id: string, payload: UpdateUserPayload) =>
  apiClient.patch<User>(`/admin/users/${id}`, payload);

export const deleteUser = (id: string) => apiClient.delete<void>(`/admin/users/${id}`);

export const fetchApiKeys = () => apiClient.get<ApiKey[]>('/admin/api-keys');

export const rotateApiKey = (id: string) => apiClient.post<{ secret: string }>(`/admin/api-keys/${id}/rotate`, {});

export const fetchSettings = () => apiClient.get<SystemSettings>('/admin/settings');

export const updateSettings = (payload: SystemSettings) => apiClient.put<SystemSettings>('/admin/settings', payload);

export const fetchSession = () => apiClient.get<Session>('/auth/session');

export const refreshToken = () => apiClient.post<{ token: string }>('/auth/refresh', {});
