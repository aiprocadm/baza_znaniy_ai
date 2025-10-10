import type { ReactNode } from 'react';
import { Dialog } from '@headlessui/react';
import { cn } from '../../utils/cn';

/**
 * Modal is a shared accessible dialog component with transitions.
 */
export type ModalProps = {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  children: ReactNode;
  footer?: ReactNode;
};

export const Modal = ({ open, onClose, title, description, children, footer }: ModalProps) => (
  <Dialog
    open={open}
    onClose={onClose}
    className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur"
  >
    <Dialog.Panel className={cn('w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl dark:bg-slate-900')}>
      <Dialog.Title className="text-lg font-semibold text-slate-900 dark:text-white">{title}</Dialog.Title>
      {description && (
        <Dialog.Description className="mt-1 text-sm text-slate-500 dark:text-slate-400">{description}</Dialog.Description>
      )}
      <div className="mt-4 space-y-3 text-sm text-slate-600 dark:text-slate-200">{children}</div>
      {footer && <div className="mt-6 flex justify-end gap-3">{footer}</div>}
    </Dialog.Panel>
  </Dialog>
);
