import { useState } from 'react';
import { Sparkles } from 'lucide-react';
import type { AnalyzeResponse } from '../../../services/api';
import { predictWithModel } from '../../../services/api';

interface Props {
  result: AnalyzeResponse;
  sessionId: string;
  preview: Record<string, unknown>[];
  target: string | null;
}

export default function PredictionsTab({ result, sessionId, preview, target }: Props) {
  const [prediction, setPrediction] = useState<{ value: unknown; model: string } | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, unknown>>(() => {
    const initial: Record<string, unknown> = {};
    if (preview.length > 0) {
      const firstRow = preview[0];
      Object.keys(firstRow).forEach(col => {
        if (col !== target) {
          initial[col] = firstRow[col];
        }
      });
    }
    return initial;
  });

  const problemType = result.problem_type || '';

  if (problemType === 'Clustering') {
    return (
      <div className="space-y-8">
        <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">Predictions</h2>
        <p className="text-neutral-500 text-sm">Prediction is not applicable for clustering tasks.</p>
      </div>
    );
  }

  const columns = preview.length > 0 ? Object.keys(preview[0]).filter(c => c !== target) : [];

  const getFieldType = (col: string): 'number' | 'select' => {
    const val = preview[0]?.[col];
    return typeof val === 'number' ? 'number' : 'select';
  };

  const getUniqueValues = (col: string): string[] => {
    const seen = new Set<string>();
    preview.forEach(row => { if (row[col] != null) seen.add(String(row[col])); });
    return Array.from(seen).sort();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);
    try {
      const inputData: Record<string, unknown> = {};
      columns.forEach(col => {
        const v = formValues[col];
        inputData[col] = v !== undefined ? v : (getFieldType(col) === 'number' ? 0 : getUniqueValues(col)[0] || '');
      });
      const res = await predictWithModel(sessionId, inputData);
      setPrediction({ value: res.predicted_value, model: res.model_used });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
        Predict with Best Model
      </h2>

      <div className="rounded-xl border border-white/10 bg-neutral-950/50 p-4 text-sm text-neutral-400">
        Fill in the feature values below and click <strong className="text-neutral-200">Predict</strong> to get a prediction
        from <strong className="text-white">{result.best_model}</strong>.
      </div>

      <form onSubmit={handleSubmit} className="max-w-2xl mx-auto space-y-4">
        {columns.map(col => {
          const type = getFieldType(col);
          return (
            <div key={col}>
              <label className="block text-xs text-neutral-500 mb-1.5 uppercase tracking-wider">{col}</label>
              {type === 'number' ? (
                <input
                  type="number"
                  step="any"
                  className="w-full px-4 py-3 bg-neutral-900 border border-white/10 rounded-lg text-sm text-neutral-200 focus:outline-none focus:border-neutral-500 transition-colors"
                  value={String(formValues[col] ?? '')}
                  onChange={e => setFormValues({ ...formValues, [col]: Number(e.target.value) })}
                />
              ) : (
                <select
                  className="w-full px-4 py-3 bg-neutral-900 border border-white/10 rounded-lg text-sm text-neutral-200 focus:outline-none focus:border-neutral-500 transition-colors"
                  value={String(formValues[col] ?? '')}
                  onChange={e => setFormValues({ ...formValues, [col]: e.target.value })}
                >
                  {getUniqueValues(col).map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              )}
            </div>
          );
        })}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-4 bg-white text-black font-bold rounded-xl hover:bg-neutral-200 transition-all disabled:opacity-50 flex items-center justify-center gap-2 text-sm"
        >
          <Sparkles className="w-4 h-4" />
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {error && (
        <div className="max-w-2xl mx-auto p-4 rounded-xl border border-red-500/30 bg-red-900/10 text-red-400 text-sm">{error}</div>
      )}

      {prediction && (
        <div className="max-w-2xl mx-auto p-8 rounded-2xl bg-neutral-900/50 border border-white/10 text-center">
          <p className="text-neutral-500 text-sm mb-2">Predicted Value</p>
          <h3 className="text-white font-bold text-lg mb-1">{target}</h3>
          <h2 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 my-4">
            {problemType === 'Regression' && typeof prediction.value === 'number'
              ? prediction.value.toFixed(2)
              : String(prediction.value)}
          </h2>
          <p className="text-neutral-600 text-xs">Model: {prediction.model}</p>
        </div>
      )}
    </div>
  );
}
