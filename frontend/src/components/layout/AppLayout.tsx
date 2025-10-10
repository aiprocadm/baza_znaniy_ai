import { Outlet } from 'react-router-dom';
import { Sidebar } from '../navigation/Sidebar';
import { Topbar } from '../navigation/Topbar';
import { NotificationStack } from '../feedback/NotificationStack';

/**
 * AppLayout composes the shell with sidebar, topbar and routed content.
 */
export const AppLayout = () => (
  <div className="flex h-full bg-slate-50 dark:bg-slate-950">
    <Sidebar />
    <div className="flex flex-1 flex-col overflow-hidden">
      <Topbar />
      <main className="flex-1 overflow-y-auto px-6 py-6 lg:px-10">
        <Outlet />
      </main>
    </div>
    <NotificationStack />
  </div>
);
