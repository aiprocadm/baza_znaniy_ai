import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ChatPanel } from '../components/chat/ChatPanel';

const pushMock = vi.fn();

vi.mock('../context/NotificationContext', () => ({
  useNotifications: () => ({ push: pushMock })
}));

vi.mock('../context/LocaleContext', () => ({
  useLocale: () => ({ t: (key: string) => key })
}));

type Listener = (event?: MessageEvent) => void;

class MockWebSocket {
  static OPEN = 1;

  readyState = MockWebSocket.OPEN;
  sent: string[] = [];
  listeners = new Map<string, Listener[]>();

  constructor(public readonly url: string) {}

  addEventListener(type: string, cb: Listener) {
    const next = this.listeners.get(type) ?? [];
    next.push(cb);
    this.listeners.set(type, next);
  }

  removeEventListener(type: string, cb: Listener) {
    const next = (this.listeners.get(type) ?? []).filter((listener) => listener !== cb);
    this.listeners.set(type, next);
  }

  emit(type: string, payload?: unknown) {
    const handlers = this.listeners.get(type) ?? [];
    if (type === 'message') {
      handlers.forEach((handler) => handler({ data: JSON.stringify(payload) } as MessageEvent));
      return;
    }
    handlers.forEach((handler) => handler());
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.emit('close');
  }
}

describe('ChatPanel websocket flow', () => {
  let sockets: MockWebSocket[] = [];

  beforeEach(() => {
    Object.defineProperty(Element.prototype, 'scrollIntoView', {
      configurable: true,
      writable: true,
      value: vi.fn()
    });
    sockets = [];
    pushMock.mockReset();
    const webSocketCtor = vi.fn().mockImplementation((url: string) => {
      const socket = new MockWebSocket(url);
      sockets.push(socket);
      return socket;
    });
    Object.assign(webSocketCtor, { OPEN: 1 });
    vi.stubGlobal('WebSocket', webSocketCtor);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('supports chat round-trip over websocket protocol', async () => {
    render(<ChatPanel />);

    act(() => {
      sockets[0].emit('open');
    });

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText('Ask about indexed knowledge…'), {
        target: { value: 'Привет' }
      });
      fireEvent.submit(screen.getByRole('button', { name: 'Send' }).closest('form') as HTMLFormElement);
    });

    const outbound = JSON.parse(sockets[0].sent[0]);
    expect(outbound.type).toBe('request');
    expect(outbound.payload.message).toBe('Привет');

    act(() => {
      sockets[0].emit('message', {
        type: 'partial',
        request_id: outbound.request_id,
        delta: 'Здравствуйте',
        token_index: 1
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Здравствуйте')).toBeInTheDocument();
    });

    act(() => {
      sockets[0].emit('message', {
        type: 'response',
        request_id: outbound.request_id,
        payload: { answer: 'Здравствуйте! Чем помочь?' }
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Здравствуйте! Чем помочь?')).toBeInTheDocument();
    });
  });

  it('handles connection level errors and server error envelopes', async () => {
    render(<ChatPanel />);

    act(() => {
      sockets[0].emit('error');
    });

    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'WebSocket error', type: 'error' })
      );
    });

    act(() => {
      sockets[0].emit('message', {
        type: 'error',
        code: 'BAD_MESSAGE_TYPE',
        message: "Expected message type 'request' or 'pong'"
      });
    });

    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Chat error', type: 'error' })
      );
    });
  });
});
