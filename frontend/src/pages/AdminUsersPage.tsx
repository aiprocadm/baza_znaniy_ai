import { useCallback, useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  createUser,
  deleteUser,
  fetchUsers,
  updateUser,
  type CreateUserPayload,
  type UpdateUserPayload,
  type User
} from '../api';
import { DataTable, type Column } from '../components/data/DataTable';
import { Modal } from '../components/modals/Modal';
import { useNotifications } from '../context/NotificationContext';
import { FormSelect } from '../components/forms/FormSelect';
import { FormInput } from '../components/forms/FormInput';
import { mapStatusColor } from '../utils/format';

/**
 * AdminUsersPage covers CRUD operations for accounts and RBAC.
 */
const schema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
  roles: z.array(z.enum(['user', 'admin'])).min(1),
  status: z.enum(['active', 'invited', 'blocked']).optional()
});

type FormValues = z.infer<typeof schema>;

export const AdminUsersPage = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const { push } = useNotifications();

  const {
    register,
    handleSubmit,
    reset,
    setValue,
    watch,
    formState: { errors }
  } = useForm<FormValues>({ resolver: zodResolver(schema), defaultValues: { roles: ['user'], status: 'active' } });

  const rolesValue = watch('roles');

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
      if (editingUser) {
        const payload: UpdateUserPayload = {
          name: values.name,
          email: values.email,
          roles: values.roles,
          status: values.status
        };
        await updateUser(editingUser.id, payload);
        push({ title: 'User updated', type: 'success' });
      } else {
        const payload: CreateUserPayload = {
          name: values.name,
          email: values.email,
          roles: values.roles
        };
        await createUser(payload);
        push({ title: 'User created', type: 'success' });
      }
      setOpen(false);
      setEditingUser(null);
      reset({ name: '', email: '', roles: ['user'], status: 'active' });
      await loadUsers();
    } catch (error) {
      push({ title: 'Failed to save user', description: (error as Error).message, type: 'error' });
    } finally {
      setLoading(false);
    }
  });

  const handleEdit = useCallback((user: User) => {
    setEditingUser(user);
    setOpen(true);
    reset({
      name: user.name,
      email: user.email,
      roles: user.roles,
      status: user.status
    });
  }, [reset]);

  const handleDelete = useCallback(async (user: User) => {
    if (!window.confirm(`Delete ${user.email}?`)) return;
    try {
      await deleteUser(user.id);
      push({ title: 'User deleted', type: 'success' });
      await loadUsers();
    } catch (error) {
      push({ title: 'Delete failed', description: (error as Error).message, type: 'error' });
    }
  }, [loadUsers, push]);

  const toggleRole = (role: 'user' | 'admin') => {
    const exists = rolesValue?.includes(role);
    if (exists) {
      setValue('roles', rolesValue.filter((item) => item !== role), { shouldValidate: true });
    } else {
      setValue('roles', [...(rolesValue ?? []), role], { shouldValidate: true });
    }
  };

  const columns: Column<User>[] = useMemo(
    () => [
      { key: 'name', header: 'Name' },
      { key: 'email', header: 'Email' },
      {
        key: 'roles',
        header: 'Roles',
        render: (row) => row.roles.join(', ')
      },
      {
        key: 'status',
        header: 'Status',
        render: (row) => <span className={`font-semibold ${mapStatusColor(row.status)}`}>{row.status}</span>
      },
      {
        key: 'actions',
        header: 'Actions',
        render: (row) => (
          <div className="flex gap-2 text-xs">
            <button
              type="button"
              onClick={() => handleEdit(row)}
              className="rounded-lg border border-slate-200 px-3 py-1 font-semibold text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              Edit
            </button>
            <button
              type="button"
              onClick={() => handleDelete(row)}
              className="rounded-lg border border-rose-200 px-3 py-1 font-semibold text-rose-600 hover:bg-rose-50 dark:border-rose-800 dark:text-rose-300 dark:hover:bg-rose-950/30"
            >
              Delete
            </button>
          </div>
        )
      }
    ],
    [handleDelete, handleEdit]
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
            setEditingUser(null);
            reset({ name: '', email: '', roles: ['user'], status: 'active' });
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
          setEditingUser(null);
        }}
        title={editingUser ? 'Edit user' : 'Invite user'}
        description="Define access and permissions for operators."
        footer={
          <>
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                setEditingUser(null);
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
              {editingUser ? 'Update' : 'Create'}
            </button>
          </>
        }
      >
        <form className="space-y-4" onSubmit={onSubmit}>
          <FormInput id="name" label="Name" register={register('name')} error={errors.name} />
          <FormInput id="email" label="Email" register={register('email')} error={errors.email} />
          <div className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
            <span className="font-medium">Roles</span>
            <div className="flex gap-3">
              {(['user', 'admin'] as const).map((role) => (
                <label key={role} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={rolesValue?.includes(role) ?? false}
                    onChange={() => toggleRole(role)}
                    className="h-4 w-4 rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="capitalize">{role}</span>
                </label>
              ))}
            </div>
            {errors.roles && <span className="block text-xs text-rose-500">Select at least one role.</span>}
          </div>
          <FormSelect
            id="status"
            label="Status"
            register={register('status')}
            options={[
              { value: 'active', label: 'Active' },
              { value: 'invited', label: 'Invited' },
              { value: 'blocked', label: 'Blocked' }
            ]}
            placeholder="Choose status"
            error={errors.status}
          />
        </form>
      </Modal>
    </div>
  );
};
