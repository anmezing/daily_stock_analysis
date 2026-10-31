import apiClient from './index';
import { toCamelCase } from './utils';
import type {
  ScreenerAlerts,
  ScreenerHistoryItem,
  ScreenerLatest,
  ScreenerPool,
  ScreenerReport,
} from '../types/screener';

export const screenerApi = {
  getLatest: async (): Promise<ScreenerLatest> => {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/screener/latest');
    return toCamelCase<ScreenerLatest>(response.data);
  },

  getHistory: async (): Promise<ScreenerHistoryItem[]> => {
    const response = await apiClient.get<{ items?: Record<string, unknown>[] }>(
      '/api/v1/screener/history',
    );
    return (response.data.items || []).map((item) =>
      toCamelCase<ScreenerHistoryItem>(item),
    );
  },

  getReport: async (date: string): Promise<ScreenerReport> => {
    const response = await apiClient.get<Record<string, unknown>>(
      `/api/v1/screener/report/${encodeURIComponent(date)}`,
    );
    return toCamelCase<ScreenerReport>(response.data);
  },

  getAlerts: async (): Promise<ScreenerAlerts> => {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/screener/alerts');
    return toCamelCase<ScreenerAlerts>(response.data);
  },

  getPool: async (): Promise<ScreenerPool> => {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/screener/pool');
    return toCamelCase<ScreenerPool>(response.data);
  },
};
