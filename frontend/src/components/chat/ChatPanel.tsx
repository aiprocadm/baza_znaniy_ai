import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { useNotifications } from '../../context/NotificationContext';
import { useLocale } from '../../context/LocaleContext';

/**
 * ChatPanel orchestrates WebSocket chat with streaming updates.
 */
type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
};

const createSocket = () => new WebSocket(import.meta.env.VITE_WS_BASE_URL ?? 'ws://localhost:8000/ws/chat');

export const ChatPanel = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const { push } = useNotifications();
  const { t } = useLocale();
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const socket = createSocket();
    socketRef.current = socket;

    socket.addEventListener('open', () => setConnected(true));
    socket.addEventListener('close', () => setConnected(false));
    socket.addEventListener('message', (event) => {
      try {
        const payload = JSON.parse(event.data) as ChatMessage;
        setMessages((items) => [...items, payload]);
      } catch (error) {
        push({ title: 'Invalid message', description: String(error), type: 'error' });
      }
    });

    return () => {
      socket.close();
    };
  }, [push]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!input.trim()) return;
    const message: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
      created_at: new Date().toISOString()
    };
    setMessages((items) => [...items, message]);
    socketRef.current?.send(JSON.stringify({ message: input }));
    setInput('');
  };

  return (
    <section id="chat" className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white/70 shadow-sm dark:border-slate-800 dark:bg-slate-900/70">
      <header className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
        <div>
          <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100">{t('chat')}</h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">{connected ? 'Connected' : 'Disconnected'}</p>
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
              <p>{message.content}</p>
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
