import { fireEvent, render, screen, within } from '@testing-library/react';
import { vi } from 'vitest';
import { NotificationStack } from '../components/feedback/NotificationStack';
import { useNotifications } from '../context/NotificationContext';
import type { Mock } from 'vitest';

vi.mock('../context/NotificationContext', () => ({
  useNotifications: vi.fn()
}));

const useNotificationsMock = useNotifications as unknown as Mock;

describe('NotificationStack', () => {
  beforeEach(() => {
    useNotificationsMock.mockReset();
  });

  it('renders notifications with contextual styling', () => {
    const remove = vi.fn();
    useNotificationsMock.mockReturnValue({
      remove,
      notifications: [
        {
          id: '1',
          title: 'Upload complete',
          description: 'The dataset is ready.',
          type: 'success'
        },
        {
          id: '2',
          title: 'Rate limit nearing',
          description: 'Consider throttling requests.',
          type: 'info'
        }
      ]
    });

    render(<NotificationStack />);

    const successCard = screen.getByTestId('notification-1');
    expect(successCard).toHaveClass('border-green-200');

    expect(screen.getByText('Rate limit nearing')).toBeInTheDocument();
    expect(screen.getByText('Consider throttling requests.')).toBeInTheDocument();

    const closeButtons = screen.getAllByRole('button', { name: /close/i });
    fireEvent.click(closeButtons[0]);

    expect(remove).toHaveBeenCalledWith('1');
  });

  it('shows nothing when there are no notifications', () => {
    useNotificationsMock.mockReturnValue({ notifications: [], remove: vi.fn() });

    render(<NotificationStack />);

    const container = screen.getByRole('region', { name: 'Notifications' });
    expect(within(container).queryAllByRole('button')).toHaveLength(0);
  });
});
