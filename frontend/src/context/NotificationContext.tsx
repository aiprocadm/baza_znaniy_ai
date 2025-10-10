import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

/**
 * NotificationContext keeps toast notifications and exposes helpers.
 */
type Notification = {
  id: string;
  title: string;
  description?: string;
  type?: 'success' | 'error' | 'info';
  ttl?: number;
};

type NotificationContextValue = {
  notifications: Notification[];
  push: (notification: Omit<Notification, 'id'>) => void;
  remove: (id: string) => void;
  clear: () => void;
};

const NotificationContext = createContext<NotificationContextValue | undefined>(undefined);

const randomId = () => crypto.randomUUID();

export const NotificationProvider = ({ children }: { children: ReactNode }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const remove = useCallback((id: string) => {
    setNotifications((items) => items.filter((notification) => notification.id !== id));
  }, []);

  const push = useCallback(
    ({ ttl = 4000, ...notification }: Omit<Notification, 'id'>) => {
      const id = randomId();
      setNotifications((items) => [...items, { ...notification, id, ttl }]);
      if (ttl > 0) {
        window.setTimeout(() => remove(id), ttl);
      }
    },
    [remove]
  );

  const clear = useCallback(() => setNotifications([]), []);

  const value = useMemo(() => ({ notifications, push, remove, clear }), [notifications, push, remove, clear]);

  return <NotificationContext.Provider value={value}>{children}</NotificationContext.Provider>;
};

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
};
