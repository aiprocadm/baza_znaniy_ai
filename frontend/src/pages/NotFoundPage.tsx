import { Link } from 'react-router-dom';

/**
 * NotFoundPage surfaces when route is missing.
 */
export const NotFoundPage = () => (
  <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-slate-50 px-6 text-center dark:bg-slate-950">
    <h1 className="text-6xl font-black text-slate-900 dark:text-white">404</h1>
    <p className="max-w-md text-sm text-slate-500 dark:text-slate-400">
      We could not find the page you were looking for. Use the navigation or jump back to the dashboard.
    </p>
    <Link
      to="/"
      className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary-500"
    >
      Return to console
    </Link>
  </div>
);
