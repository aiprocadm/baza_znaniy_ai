import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { useNotifications } from '../../context/NotificationContext';
import { useLocale } from '../../context/LocaleContext';

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
};

type WsEnvelope =
  | { type: 'ack'; request_id: string }
  | { type: 'partial'; request_id: string; delta: string; token_index: number }
  | {
      type: 'response';
      request_id: string;
      payload: {
        answer: string;
      };
    }
  | {
      type: 'error';
      request_id?: string;
      code: string;
      message: string;
      status?: number;
    }
  | { type: 'ping' }
  | { type: 'pong' };

const WS_URL = import.meta.env.VITE_WS_BASE_URL ?? 'ws://localhost:8000/api/v1/ws/chat';
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 500;

const createSocket = () => new WebSocket(WS_URL);

export const ChatPanel = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [connected, setConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const isUnmountingRef = useRef(false);
  const activeAssistantByRequestRef = useRef<Map<string, string>>(new Map());
  const { push } = useNotifications();
  const { t } = useLocale();
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    isUnmountingRef.current = false;

    const connect = () => {
      const socket = createSocket();
      socketRef.current = socket;

      socket.addEventListener('open', () => {
        setConnected(true);
        setIsReconnecting(false);
        reconnectAttemptsRef.current = 0;
      });

      socket.addEventListener('close', () => {
        setConnected(false);

        if (isUnmountingRef.current) {
          return;
        }

        if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
          setIsReconnecting(false);
          push({
            title: 'Connection lost',
            description: 'Unable to reconnect to chat server.',
            type: 'error'
          });
          return;
        }

        reconnectAttemptsRef.current += 1;
        setIsReconnecting(true);
        const timeoutMs = BASE_RECONNECT_DELAY_MS * 2 ** (reconnectAttemptsRef.current - 1);
        reconnectTimeoutRef.current = window.setTimeout(connect, timeoutMs);
      });

      socket.addEventListener('error', () => {
        push({ title: 'WebSocket error', description: 'Chat connection failed.', type: 'error' });
      });

      socket.addEventListener('message', (event) => {
        try {
          const payload = JSON.parse(event.data) as WsEnvelope;
          if (payload.type === 'ping') {
            socket.send(JSON.stringify({ type: 'pong' }));
            return;
          }

          if (payload.type === 'ack') {
            return;
          }

          if (payload.type === 'partial') {
            const assistantId = activeAssistantByRequestRef.current.get(payload.request_id);
            if (!assistantId) {
              return;
            }
            setMessages((items) =>
              items.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: `${message.content}${payload.delta}`
                    }
                  : message
              )
            );
            return;
          }

          if (payload.type === 'response') {
            const assistantId = activeAssistantByRequestRef.current.get(payload.request_id);
            if (assistantId) {
              setMessages((items) =>
                items.map((message) =>
                  message.id === assistantId
                    ? {
                        ...message,
                        content: payload.payload.answer
                      }
                    : message
                )
              );
              activeAssistantByRequestRef.current.delete(payload.request_id);
            }
            return;
          }

          if (payload.type === 'error') {
            const description = payload.status
              ? `${payload.code}: ${payload.message} (status ${payload.status})`
              : `${payload.code}: ${payload.message}`;
            push({ title: 'Chat error', description, type: 'error' });
            return;
          }
        } catch (error) {
          push({ title: 'Invalid message', description: String(error), type: 'error' });
        }
      });
    };

    connect();

    return () => {
      isUnmountingRef.current = true;
      if (reconnectTimeoutRef.current !== null) {
        window.clearTimeout(reconnectTimeoutRef.current);
      }
      socketRef.current?.close();
    };
  }, [push]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!input.trim()) return;
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      push({ title: 'Disconnected', description: 'WebSocket is not connected.', type: 'error' });
      return;
    }

    const nowIso = new Date().toISOString();
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
      created_at: nowIso
    };

    const requestId = crypto.randomUUID();
    const assistantMessageId = crypto.randomUUID();

    activeAssistantByRequestRef.current.set(requestId, assistantMessageId);

    setMessages((items) => [
      ...items,
      userMessage,
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        created_at: nowIso
      }
    ]);

    socketRef.current.send(
      JSON.stringify({
        type: 'request',
        request_id: requestId,
        stream: true,
        payload: {
          user_id: 'web-user',
          message: input,
          conversation_id: null
        }
      })
    );

    setInput('');
  };

  return (
    <section id="chat" className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white/70 shadow-sm dark:border-slate-800 dark:bg-slate-900/70">
      <header className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
        <div>
          <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100">{t('chat')}</h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {connected ? 'Connected' : isReconnecting ? 'Reconnecting…' : 'Disconnected'}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setMessages([])}
          className="rounded-lg border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
        >
          Clear
        </button>
      </header>
      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs rounded-2xl px-4 py-2 text-sm shadow-sm ${
                message.role === 'user'
                  ? 'bg-primary-600 text-white'
                  : 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-100'
              }`}
            >
              <p>{message.content || '…'}</p>
              <span className="mt-1 block text-[10px] text-white/70 dark:text-slate-400">
                {new Date(message.created_at).toLocaleTimeString()}
              </span>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <form onSubmit={handleSubmit} className="border-t border-slate-200 px-4 py-3 dark:border-slate-800">
        <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-800">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about indexed knowledge…"
            className="flex-1 bg-transparent text-sm text-slate-700 outline-none dark:text-slate-100"
          />
          <button
            type="submit"
            disabled={!connected}
            className="rounded-lg bg-primary-600 px-3 py-1 text-xs font-semibold text-white shadow disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            Send
          </button>
        </div>
      </form>
    </section>
  );
};
