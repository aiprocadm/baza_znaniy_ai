import { useCallback, useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  createUser,
  fetchUsers,
  type CreateUserPayload,
  type User
} from '../api';
import { DataTable, type Column } from '../components/data/DataTable';
import { Modal } from '../components/modals/Modal';
import { useNotifications } from '../context/NotificationContext';
import { FormInput } from '../components/forms/FormInput';

/**
 * AdminUsersPage covers CRUD operations for accounts and RBAC.
 */
const schema = z.object({
  full_name: z.string().min(2),
  email: z.string().email(),
  password: z.string().min(8),
  role: z.enum(['admin', 'manager', 'member']),
  tenant_slug: z.string().min(1),
  is_active: z.boolean().default(true)
});

type FormValues = z.infer<typeof schema>;

export const AdminUsersPage = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const { push } = useNotifications();

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors }
  } = useForm<FormValues>({ resolver: zodResolver(schema), defaultValues: { role: 'member', is_active: true, tenant_slug: 'default' } });

  const loadUsers = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetchUsers();
      setUsers(response.data);
    } catch (error) {
      push({ title: 'Failed to load users', description: (error as Error).message, type: 'error' });
    } finally {
      setLoading(false);
    }
  }, [push]);

  useEffect(() => {
    void loadUsers();
  }, [loadUsers]);

  const onSubmit = handleSubmit(async (values) => {
    try {
      setLoading(true);
      const payload: CreateUserPayload = values;
      await createUser(payload);
      push({ title: 'User created', type: 'success' });
      setOpen(false);
      reset({ full_name: '', email: '', password: '', role: 'member', is_active: true, tenant_slug: 'default' });
      await loadUsers();
    } catch (error) {
      push({ title: 'Failed to save user', description: (error as Error).message, type: 'error' });
    } finally {
      setLoading(false);
    }
  });

  const columns: Column<User>[] = useMemo(
    () => [
      { key: 'full_name', header: 'Name' },
      { key: 'email', header: 'Email' },
      {
        key: 'role',
        header: 'Roles',
        render: (row) => row.role
      },
      {
        key: 'is_active',
        header: 'Status',
        render: (row) => (row.is_active ? 'active' : 'inactive')
      }
    ],
    []
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">User management</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">Invite teammates and assign granular permissions.</p>
        </div>
        <button
          type="button"
          onClick={() => {
            setOpen(true);
            reset({ full_name: '', email: '', password: '', role: 'member', is_active: true, tenant_slug: 'default' });
          }}
          className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary-500"
        >
          New user
        </button>
      </div>
      <DataTable data={users} columns={columns} emptyState={loading ? 'Loading users…' : 'No users found.'} />
      <Modal
        open={open}
        onClose={() => {
          setOpen(false);
        }}
        title="Create user"
        description="Define access and permissions for operators."
        footer={
          <>
            <button
              type="button"
              onClick={() => {
                setOpen(false);
              }}
              className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => onSubmit()}
              className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary-500"
            >
              Create
            </button>
          </>
        }
      >
        <form className="space-y-4" onSubmit={onSubmit}>
          <FormInput id="full_name" label="Name" register={register('full_name')} error={errors.full_name} />
          <FormInput id="email" label="Email" register={register('email')} error={errors.email} />
          <FormInput id="password" label="Password" register={register('password')} error={errors.password} type="password" />
          <label className="block text-sm">
            <span className="font-medium text-slate-600 dark:text-slate-300">Role</span>
            <select
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              {...register('role')}
            >
              <option value="member">member</option>
              <option value="manager">manager</option>
              <option value="admin">admin</option>
            </select>
          </label>
          <FormInput id="tenant_slug" label="Tenant" register={register('tenant_slug')} error={errors.tenant_slug} />
          <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
            <input type="checkbox" {...register('is_active')} />
            Active user
          </label>
        </form>
      </Modal>
    </div>
  );
};
