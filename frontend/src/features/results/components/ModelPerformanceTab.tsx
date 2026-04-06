import { useEffect, useMemo, useState } from 'react';
import type { AnalyzeResponse } from '../../../services/api';

interface Props { result: AnalyzeResponse; }

function formatParamValue(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export default function ModelPerformanceTab({ result }: Props) {
  const metrics = result.metrics || {};
  const bestParams = result.best_params || {};
  const bestModel = result.best_model || 'N/A';
  const problemType = result.problem_type || '';

  const modelNames = Object.keys(metrics);
  const scoreCol = problemType === 'Clustering' ? 'Silhouette Score'
    : problemType === 'Regression' ? 'R2'
    : (metrics[modelNames[0]]?.['F1 Score'] !== undefined ? 'F1 Score' : 'Accuracy');

  // Sort models by score descending
  const sorted = [...modelNames].sort((a, b) => {
    const sa = Number(metrics[a]?.[scoreCol]) || 0;
    const sb = Number(metrics[b]?.[scoreCol]) || 0;
    return sb - sa;
  });

  const defaultModelForParams = bestModel && modelNames.includes(bestModel)
    ? bestModel
    : (sorted[0] || '');

  const [selectedModel, setSelectedModel] = useState<string>(defaultModelForParams);

  useEffect(() => {
    setSelectedModel(defaultModelForParams);
  }, [defaultModelForParams]);

  const selectedParamEntries = useMemo(() => {
    if (!selectedModel) return [] as [string, unknown][];
    const params = bestParams[selectedModel] || {};
    return Object.entries(params).sort(([a], [b]) => a.localeCompare(b));
  }, [bestParams, selectedModel]);

  const maxScore = Math.max(...sorted.map(m => Number(metrics[m]?.[scoreCol]) || 0), 1);
  const metricKeys = modelNames.length > 0 ? Object.keys(metrics[modelNames[0]]) : [];

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
        Model Evaluation
      </h2>

      {/* Horizontal Bar Chart */}
      <div>
        <h3 className="text-sm font-bold text-neutral-300 uppercase tracking-widest mb-6">
          Model Comparison — {scoreCol}
        </h3>
        <div className="space-y-2">
          {sorted.map((name) => {
            const score = Number(metrics[name]?.[scoreCol]) || 0;
            const pct = (score / maxScore) * 100;
            const isBest = name === bestModel;
            return (
              <div key={name} className="flex items-center gap-4 group">
                <div className="w-44 text-right text-xs text-neutral-400 font-medium truncate shrink-0">{name}</div>
                <div className="flex-1 h-8 bg-neutral-900 rounded-sm overflow-hidden relative">
                  <div
                    className={`h-full rounded-sm transition-all duration-500 ${isBest ? 'bg-white' : 'bg-neutral-600'}`}
                    style={{ width: `${Math.max(pct, 2)}%` }}
                  />
                </div>
                <div className={`w-16 text-right text-xs font-mono ${isBest ? 'text-white font-bold' : 'text-neutral-500'}`}>
                  {score.toFixed(4)}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Full Metrics Table */}
      {metricKeys.length > 0 && (
        <div>
          <h3 className="text-sm font-bold text-neutral-300 uppercase tracking-widest mb-4">📊 Full Metrics Table</h3>
          <div className="overflow-x-auto rounded-xl border border-white/10">
            <table className="w-full text-xs text-left text-neutral-400">
              <thead className="bg-neutral-900/50 text-neutral-500 uppercase tracking-wider">
                <tr>
                  <th className="px-4 py-3 font-medium">Model</th>
                  {metricKeys.map(k => <th key={k} className="px-4 py-3 font-medium">{k}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {sorted.map(name => (
                  <tr key={name} className={`hover:bg-white/5 transition-colors ${name === bestModel ? 'bg-white/[0.03]' : ''}`}>
                    <td className="px-4 py-3 font-medium text-neutral-200">{name}</td>
                    {metricKeys.map(k => (
                      <td key={k} className="px-4 py-3 font-mono">
                        {typeof metrics[name][k] === 'number' ? metrics[name][k].toFixed(4) : String(metrics[name][k])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Best Hyperparameters */}
      <div>
        <h3 className="text-sm font-bold text-neutral-300 uppercase tracking-widest mb-4">Best Hyperparameters</h3>

        {sorted.length > 0 && (
          <div className="mb-4 max-w-sm">
            <label className="block text-[11px] uppercase tracking-widest text-neutral-500 mb-2">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full rounded-lg bg-neutral-900 border border-white/10 text-neutral-200 px-3 py-2 text-sm focus:outline-none focus:border-white/30"
            >
              {sorted.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>
        )}

        {selectedParamEntries.length > 0 ? (
          <div className="overflow-x-auto rounded-xl border border-white/10">
            <table className="w-full text-xs text-left text-neutral-400">
              <thead className="bg-neutral-900/50 text-neutral-500 uppercase tracking-wider">
                <tr>
                  <th className="px-4 py-3 font-medium">Parameter</th>
                  <th className="px-4 py-3 font-medium">Best Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {selectedParamEntries.map(([key, value]) => (
                  <tr key={key} className="hover:bg-white/5 transition-colors">
                    <td className="px-4 py-3 font-mono text-neutral-300">{key}</td>
                    <td className="px-4 py-3 font-mono text-neutral-200 break-all">{formatParamValue(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-neutral-500">No parameters available for the selected model.</p>
        )}
      </div>


    </div>
  );
}
