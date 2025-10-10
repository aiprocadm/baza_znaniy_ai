import type { FieldError, UseFormRegisterReturn } from 'react-hook-form';
import { cn } from '../../utils/cn';

/**
 * FormSelect normalizes select appearance and validation.
 */
export type FormSelectProps = {
  id: string;
  label: string;
  options: Array<{ value: string; label: string }>;
  placeholder?: string;
  error?: FieldError;
  register: UseFormRegisterReturn;
  multiple?: boolean;
};

export const FormSelect = ({ id, label, options, placeholder, error, register, multiple }: FormSelectProps) => (
  <label className="block text-sm">
    <span className="font-medium text-slate-600 dark:text-slate-300">{label}</span>
    <select
      id={id}
      multiple={multiple}
      className={cn(
        'mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-primary-400 focus:ring-2 focus:ring-primary-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100',
        error && 'border-rose-400 focus:border-rose-400 focus:ring-rose-100'
      )}
      defaultValue={multiple ? [] : ''}
      {...register}
    >
      {!multiple && (
        <option value="" disabled>
          {placeholder ?? 'Select'}
        </option>
      )}
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
    {error && <span className="mt-1 block text-xs text-rose-500">{error.message}</span>}
  </label>
);
