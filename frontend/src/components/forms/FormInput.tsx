import type { FieldError, UseFormRegisterReturn } from 'react-hook-form';
import { cn } from '../../utils/cn';

/**
 * FormInput standardizes text inputs with validation state.
 */
export type FormInputProps = {
  id: string;
  label: string;
  error?: FieldError;
  type?: string;
  placeholder?: string;
  register: UseFormRegisterReturn;
};

export const FormInput = ({ id, label, error, type = 'text', placeholder, register }: FormInputProps) => (
  <label className="block text-sm">
    <span className="font-medium text-slate-600 dark:text-slate-300">{label}</span>
    <input
      id={id}
      type={type}
      placeholder={placeholder}
      className={cn(
        'mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100',
        error && 'border-rose-400 focus:border-rose-400 focus:ring-rose-100'
      )}
      {...register}
    />
    {error && <span className="mt-1 block text-xs text-rose-500">{error.message}</span>}
  </label>
);
