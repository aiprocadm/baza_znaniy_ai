import { renderHook, act } from '@testing-library/react';
import { NotificationProvider, useNotifications } from '../context/NotificationContext';

describe('NotificationContext', () => {
  it('adds and removes notifications', () => {
    const { result } = renderHook(() => useNotifications(), {
      wrapper: ({ children }) => <NotificationProvider>{children}</NotificationProvider>
    });

    act(() => {
      result.current.push({ title: 'Saved', type: 'success', ttl: 0 });
    });
    expect(result.current.notifications).toHaveLength(1);

    act(() => {
      result.current.clear();
    });
    expect(result.current.notifications).toHaveLength(0);
  });
});
