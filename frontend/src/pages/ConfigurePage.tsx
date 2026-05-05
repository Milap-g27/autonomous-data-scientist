import { useNavigate } from 'react-router-dom';
import { Play, ChevronDown, CheckCircle2, Loader2, Circle, AlertTriangle } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import {
  configureSession,
  analyzeSession,
  getAnalysisStatus,
  getAnalysisResult,
  type DatasetInfo,
  type AnalysisStatusResponse,
  type AnalyzeResponse,
} from '../services/api';

interface Props {
  sessionId: string;
  datasetInfo: DatasetInfo;
  onAnalysisComplete: (result: AnalyzeResponse, target: string | null) => void;
}

type TimelineStep = NonNullable<AnalysisStatusResponse['progress_timeline']>[number];

const DEFAULT_TIMELINE: TimelineStep[] = [
  { key: 'cleaning', label: 'Cleaning', status: 'pending', detail: '' },
  { key: 'eda', label: 'EDA', status: 'pending', detail: '' },
  { key: 'baseline', label: 'Baseline model', status: 'pending', detail: '' },
  { key: 'advanced', label: 'Advanced plots', status: 'pending', detail: '' },
];

const MIN_RUNNING_UI_MS = 1800;
const STATUS_POLL_INTERVAL_MS = 3000;

function normalizeTimeline(
  incoming: AnalysisStatusResponse['progress_timeline'] | undefined,
  status: AnalysisStatusResponse['status']
): TimelineStep[] {
  if (incoming && incoming.length > 0) return incoming;

  if (status === 'completed') {
    return DEFAULT_TIMELINE.map((step) => ({ ...step, status: 'completed' as const }));
  }

  if (status === 'running') {
    return DEFAULT_TIMELINE.map((step, idx) => ({
      ...step,
      status: idx === 0 ? ('running' as const) : ('pending' as const),
    }));
  }

  if (status === 'failed') {
    return DEFAULT_TIMELINE.map((step, idx) => ({
      ...step,
      status: idx === 0 ? ('failed' as const) : ('pending' as const),
    }));
  }

  return DEFAULT_TIMELINE;
}

export default function ConfigurePage({ sessionId, datasetInfo, onAnalysisComplete }: Props) {
  const navigate = useNavigate();
  const [target, setTarget] = useState<string | null>(
    datasetInfo.column_names[datasetInfo.column_names.length - 1] || null
  );
  const [problemHint, setProblemHint] = useState('Auto-detect');
  const [testSize, setTestSize] = useState(0.2);
  const [scaling, setScaling] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [timeline, setTimeline] = useState<TimelineStep[]>(DEFAULT_TIMELINE);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const handleRun = async () => {
    setError('');
    setRunning(true);
    setElapsedSeconds(0);
    setTimeline(DEFAULT_TIMELINE);
    setCurrentStep('cleaning');
    try {
      await configureSession({
        session_id: sessionId,
        target,
        problem_hint: problemHint,
        random_seed: 42,
        test_size: testSize,
        scaling,
        feature_selection: false,
      });

      await analyzeSession(sessionId);

      const pollStart = Date.now();
      const timeoutMs = 12 * 60 * 1000;
      while (true) {
        const status = await getAnalysisStatus(sessionId);
        const elapsed = Math.floor((Date.now() - pollStart) / 1000);
        if (isMountedRef.current) {
          setElapsedSeconds(elapsed);
          setTimeline(normalizeTimeline(status.progress_timeline, status.status));
          setCurrentStep(status.current_step || null);
        }

        if (status.status === 'completed') {
          const elapsedMs = Date.now() - pollStart;
          if (elapsedMs < MIN_RUNNING_UI_MS) {
            await new Promise((resolve) => setTimeout(resolve, MIN_RUNNING_UI_MS - elapsedMs));
          }
          const res = await getAnalysisResult(sessionId);
          onAnalysisComplete(res, target);
          navigate('/results');
          return;
        }

        if (status.status === 'failed') {
          throw new Error(status.error || 'Analysis failed during processing.');
        }

        if (Date.now() - pollStart > timeoutMs) {
          throw new Error('Analysis is taking longer than expected. Please try again shortly.');
        }

        await new Promise((resolve) => setTimeout(resolve, STATUS_POLL_INTERVAL_MS));
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Analysis failed');
        setRunning(false);
      }
    }
  };

  const activeStepLabel = timeline.find((step) => step.key === currentStep)?.label || 'Preparing';

  if (running) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-6">
        {/* Animated loading */}
        <div className="relative">
          <div className="w-20 h-20 rounded-full border-2 border-neutral-700 border-t-white animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-8 h-8 rounded-full bg-neutral-800" />
          </div>
        </div>
        <p className="text-neutral-400 text-sm font-medium">Agent is working — analyzing, modeling, explaining…</p>
        <p className="text-neutral-600 text-xs">This may take a few minutes depending on dataset size.</p>
        <p className="text-neutral-500 text-xs">Elapsed: {elapsedSeconds}s</p>

        <div className="w-full max-w-xl mt-2 rounded-2xl border border-white/10 bg-neutral-900/40 p-4">
          <p className="text-[11px] uppercase tracking-widest text-neutral-500 mb-3">
            Live Timeline · Current: <span className="text-neutral-300 normal-case tracking-normal">{activeStepLabel}</span>
          </p>

          <div className="space-y-3">
            {timeline.map((step) => {
              const isRunning = step.status === 'running';
              const isDone = step.status === 'completed';
              const isFailed = step.status === 'failed';

              return (
                <div key={step.key} className="flex items-center gap-3">
                  <div className="shrink-0">
                    {isDone && <CheckCircle2 className="h-4 w-4 text-emerald-400" />}
                    {isRunning && <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />}
                    {isFailed && <AlertTriangle className="h-4 w-4 text-red-400" />}
                    {!isDone && !isRunning && !isFailed && <Circle className="h-4 w-4 text-neutral-600" />}
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className={`text-sm ${isDone ? 'text-neutral-200' : isRunning ? 'text-blue-300' : isFailed ? 'text-red-300' : 'text-neutral-500'}`}>
                      {step.label}
                    </p>
                    {step.detail && (
                      <p className="text-[11px] text-neutral-500 truncate">{step.detail}</p>
                    )}
                  </div>

                  <span className="text-[10px] uppercase tracking-wide text-neutral-500">
                    {step.status}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 mb-2">
          Configure Analysis
        </h1>
        <p className="text-sm text-neutral-500">Set up your target, problem type, and parameters before running the AI agent.</p>
      </div>

      {error && (
        <div className="p-4 rounded-xl border border-red-500/20 bg-red-900/10 text-red-400 text-sm">{error}</div>
      )}

      {/* Dataset Info Badges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Rows', value: datasetInfo.rows.toLocaleString() },
          { label: 'Columns', value: datasetInfo.columns },
          { label: 'Numeric', value: datasetInfo.numeric_columns.length },
          { label: 'Categorical', value: datasetInfo.categorical_columns.length },
        ].map(b => (
          <div key={b.label} className="bg-neutral-900/50 border border-white/10 rounded-xl p-5">
            <p className="text-[10px] text-neutral-500 uppercase tracking-widest mb-1">{b.label}</p>
            <p className="text-2xl font-bold text-neutral-200">{b.value}</p>
          </div>
        ))}
      </div>

      {/* Config Controls */}
      <div className="bg-black/[0.96] border border-white/10 rounded-2xl p-8 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-xs text-neutral-500 mb-2 uppercase tracking-wider">🎯 Target Column</label>
            <select
              value={target || '__none__'}
              onChange={e => setTarget(e.target.value === '__none__' ? null : e.target.value)}
              className="w-full px-4 py-3 bg-neutral-900 border border-white/10 rounded-lg text-sm text-neutral-200 focus:outline-none focus:border-neutral-500"
            >
              <option value="__none__">None (Clustering)</option>
              {datasetInfo.column_names.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-2 uppercase tracking-wider">📑 Problem Type</label>
            <select
              value={problemHint}
              onChange={e => setProblemHint(e.target.value)}
              className="w-full px-4 py-3 bg-neutral-900 border border-white/10 rounded-lg text-sm text-neutral-200 focus:outline-none focus:border-neutral-500"
            >
              {['Auto-detect', 'Classification', 'Regression', 'Clustering'].map(o => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Advanced */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
        >
          <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
          Advanced Settings
        </button>
        {showAdvanced && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pl-4 border-l border-white/5">
            <div>
              <label className="block text-xs text-neutral-500 mb-2 uppercase tracking-wider">Test Size</label>
              <input
                type="range" min="0.1" max="0.4" step="0.05"
                value={testSize}
                onChange={e => setTestSize(Number(e.target.value))}
                className="w-full accent-white"
              />
              <p className="text-xs text-neutral-400 mt-1">{(testSize * 100).toFixed(0)}%</p>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="checkbox" checked={scaling}
                onChange={e => setScaling(e.target.checked)}
                className="w-4 h-4 rounded accent-white"
              />
              <label className="text-sm text-neutral-400">Apply Feature Scaling</label>
            </div>
          </div>
        )}

        {/* Run Button */}
        <button
          onClick={handleRun}
          className="w-full py-4 bg-white text-black font-bold rounded-xl hover:bg-neutral-200 transition-all flex items-center justify-center gap-2 text-sm"
        >
          <Play className="w-4 h-4" />
          Run AI Data Scientist
        </button>
      </div>

      {/* Dataset Preview */}
      <div className="bg-black/[0.96] border border-white/10 rounded-2xl p-6 overflow-x-auto">
        <h3 className="text-sm font-bold text-neutral-300 uppercase tracking-widest mb-4">Dataset Preview</h3>
        <table className="w-full text-xs text-left text-neutral-400">
          <thead className="bg-neutral-900/50 text-neutral-500 uppercase tracking-wider">
            <tr>
              {datasetInfo.column_names.map(c => <th key={c} className="px-3 py-2 font-medium whitespace-nowrap">{c}</th>)}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {datasetInfo.preview.slice(0, 8).map((row, i) => (
              <tr key={i} className="hover:bg-white/5">
                {datasetInfo.column_names.map(c => <td key={c} className="px-3 py-2 font-mono whitespace-nowrap">{String(row[c] ?? '—')}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
