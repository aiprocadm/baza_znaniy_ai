import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useNotifications } from '../context/NotificationContext';

/**
 * LoginPage authenticates operators with email/password.
 */
export const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const { push } = useNotifications();
  const navigate = useNavigate();
  const location = useLocation();

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    try {
      await login(email, password);
      push({ title: 'Welcome back', type: 'success' });
      const redirect = (location.state as { from?: Location })?.from?.pathname ?? '/';
      navigate(redirect, { replace: true });
    } catch (error) {
      push({ title: 'Login failed', description: (error as Error).message, type: 'error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4 dark:bg-slate-950">
      <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-8 shadow-xl dark:border-slate-800 dark:bg-slate-900">
        <div className="flex items-center gap-3">
          <img src="/logo.svg" alt="KB.AI" className="h-12 w-12" />
          <div>
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">KB.AI Console</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">Securely manage knowledge operations.</p>
          </div>
        </div>
        <form onSubmit={handleSubmit} className="mt-8 space-y-4 text-sm">
          <label className="block">
            <span className="font-medium text-slate-600 dark:text-slate-300">Email</span>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              required
            />
          </label>
          <label className="block">
            <span className="font-medium text-slate-600 dark:text-slate-300">Password</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              required
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary-500 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  );
};
