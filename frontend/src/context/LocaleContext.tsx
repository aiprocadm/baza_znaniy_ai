import { createContext, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

/**
 * LocaleContext provides lightweight i18n with RU/EN dictionaries.
 */
export type Locale = 'en' | 'ru';

export type TranslationKey = keyof typeof en;

const STORAGE_KEY = 'operations-console.locale';

const en = {
  welcome: 'Welcome back',
  quickActions: 'Quick actions',
  uploadDocument: 'Upload document',
  newChat: 'Start new chat',
  refresh: 'Refresh status',
  recentActivity: 'Recent activity',
  search: 'Search',
  filters: 'Filters',
  logout: 'Log out',
  theme: 'Theme',
  language: 'Language',
  dashboard: 'Dashboard',
  documents: 'Documents',
  files: 'Files',
  tasks: 'Tasks',
  chat: 'Chat',
  admin: 'Admin',
  monitoring: 'Monitoring',
  settings: 'Settings',
  users: 'Users',
  create: 'Create',
  save: 'Save',
  cancel: 'Cancel',
  status: 'Status',
  role: 'Role',
  email: 'Email',
  name: 'Name',
  apiKeys: 'API keys',
  general: 'General',
  queue: 'Queue',
  uptime: 'Uptime',
  errorRate: 'Error rate',
  downloads: 'Downloads'
};

const ru: typeof en = {
  welcome: 'С возвращением',
  quickActions: 'Быстрые действия',
  uploadDocument: 'Загрузить документ',
  newChat: 'Новый чат',
  refresh: 'Обновить статус',
  recentActivity: 'Лента событий',
  search: 'Поиск',
  filters: 'Фильтры',
  logout: 'Выйти',
  theme: 'Тема',
  language: 'Язык',
  dashboard: 'Личный кабинет',
  documents: 'Документы',
  files: 'Файлы',
  tasks: 'Задачи',
  chat: 'Чат',
  admin: 'Админ-панель',
  monitoring: 'Мониторинг',
  settings: 'Настройки',
  users: 'Пользователи',
  create: 'Создать',
  save: 'Сохранить',
  cancel: 'Отмена',
  status: 'Статус',
  role: 'Роль',
  email: 'Email',
  name: 'Имя',
  apiKeys: 'API-ключи',
  general: 'Общее',
  queue: 'Очередь',
  uptime: 'Аптайм',
  errorRate: 'Ошибки',
  downloads: 'Загрузки'
};

const dictionaries = { en, ru } satisfies Record<Locale, typeof en>;

type LocaleContextValue = {
  locale: Locale;
  t: (key: TranslationKey) => string;
  setLocale: (value: Locale) => void;
};

const LocaleContext = createContext<LocaleContextValue | undefined>(undefined);

const detectLocale = (): Locale => {
  if (typeof window === 'undefined') {
    return 'en';
  }
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === 'ru' || stored === 'en') {
    return stored;
  }
  return navigator.language.startsWith('ru') ? 'ru' : 'en';
};

export const LocaleProvider = ({ children }: { children: ReactNode }) => {
  const [locale, setLocaleState] = useState<Locale>(detectLocale);

  const setLocale = (value: Locale) => {
    window.localStorage.setItem(STORAGE_KEY, value);
    setLocaleState(value);
  };

  const value = useMemo(
    () => ({
      locale,
      t: (key: TranslationKey) => dictionaries[locale][key],
      setLocale
    }),
    [locale]
  );

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
};

export const useLocale = () => {
  const context = useContext(LocaleContext);
  if (!context) {
    throw new Error('useLocale must be used within LocaleProvider');
  }
  return context;
};
