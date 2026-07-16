// Mirrors llm_eval/dashboard/app.py — the API has no dataset-listing endpoint,
// so both frontends hardcode the same preset dataset paths.
export const DATASETS: { label: string; path: string }[] = [
  { label: 'Legal QA', path: 'legal_qa' },
  { label: 'Code generation', path: 'code_gen' },
  { label: 'Summarization', path: 'summarization' },
];
