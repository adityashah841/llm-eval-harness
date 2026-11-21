import type { RunStatus } from '../api/types';

const CLASS: Record<RunStatus, string> = {
  completed: 'pill-good',
  partial: 'pill-warning',
  failed: 'pill-critical',
  pending: 'pill-neutral',
  running: 'pill-neutral',
};

export default function StatusPill({ status }: { status: RunStatus }) {
  return <span className={`pill ${CLASS[status] ?? 'pill-neutral'}`}>{status}</span>;
}
