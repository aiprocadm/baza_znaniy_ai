import clsx from 'clsx';
import type { ClassValue } from 'clsx';

/**
 * cn merges Tailwind classes and conditionals.
 */
export const cn = (...inputs: ClassValue[]) => clsx(inputs);
