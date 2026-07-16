import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';

export interface RecentRun {
  id: string;
  dataset: string;
  models: string[];
  created_at: string;
}

const STORAGE_KEY = 'llm-eval-recent-runs';
const MAX_RECENT = 20;

interface RunsContextValue {
  recentRuns: RecentRun[];
  addRun: (run: RecentRun) => void;
}

const RunsContext = createContext<RunsContextValue | null>(null);

function loadInitial(): RecentRun[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as RecentRun[]) : [];
  } catch {
    return [];
  }
}

export function RunsProvider({ children }: { children: ReactNode }) {
  const [recentRuns, setRecentRuns] = useState<RecentRun[]>(loadInitial);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(recentRuns));
  }, [recentRuns]);

  const addRun = (run: RecentRun) => {
    setRecentRuns((prev) => [run, ...prev.filter((r) => r.id !== run.id)].slice(0, MAX_RECENT));
  };

  return <RunsContext.Provider value={{ recentRuns, addRun }}>{children}</RunsContext.Provider>;
}

export function useRuns(): RunsContextValue {
  const ctx = useContext(RunsContext);
  if (!ctx) throw new Error('useRuns must be used within a RunsProvider');
  return ctx;
}
