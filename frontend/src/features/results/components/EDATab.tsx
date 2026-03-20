import type { AnalyzeResponse } from '../../../services/api';

interface Props { result: AnalyzeResponse; preview: Record<string, unknown>[]; }

export default function EDATab({ result, preview }: Props) {
  const eda = result.eda_results as Record<string, unknown>;
  const description = eda?.description as Record<string, Record<string, number>> | undefined;

  // Build column info from preview
  const columns = preview.length > 0 ? Object.keys(preview[0]) : [];
  
  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
        Exploratory Data Analysis
      </h2>

      {/* Summary Statistics */}
      {description && (
        <div>
          <h3 className="text-sm font-bold text-neutral-300 uppercase tracking-widest mb-4">📋 Summary Statistics</h3>
          <div className="overflow-x-auto rounded-xl border border-white/10">
            <table className="w-full text-xs text-left text-neutral-400">
              <thead className="bg-neutral-900/50 text-neutral-500 uppercase tracking-wider">
                <tr>
                  <th className="px-4 py-3 font-medium">Column</th>
                  {Object.keys(Object.values(description)[0] || {}).map(stat => (
                    <th key={stat} className="px-4 py-3 font-medium">{stat}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {Object.entries(description).map(([col, stats]) => (
                  <tr key={col} className="hover:bg-white/5 transition-colors">
                    <td className="px-4 py-3 font-medium text-neutral-200">{col}</td>
                    {Object.values(stats).map((val, i) => (
                      <td key={i} className="px-4 py-3 font-mono">
                        {typeof val === 'number' ? val.toFixed(4) : String(val)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Missing Values & Unique Counts */}
      <div>
        <h3 className="text-sm font-bold text-neutral-300 uppercase tracking-widest mb-4">🔍 Missing Values & Unique Counts</h3>
        <div className="overflow-x-auto rounded-xl border border-white/10">
          <table className="w-full text-xs text-left text-neutral-400">
            <thead className="bg-neutral-900/50 text-neutral-500 uppercase tracking-wider">
              <tr>
                <th className="px-4 py-3 font-medium">Column</th>
                <th className="px-4 py-3 font-medium">Type</th>
                <th className="px-4 py-3 font-medium">Sample Value</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {columns.map(col => (
                <tr key={col} className="hover:bg-white/5 transition-colors">
                  <td className="px-4 py-3 font-medium text-neutral-200">{col}</td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 bg-neutral-800 rounded text-[10px] tracking-wider">
                      {typeof preview[0][col] === 'number' ? 'numeric' : 'categorical'}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono">{String(preview[0][col] ?? '—')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Preview */}
      <div>
        <h3 className="text-sm font-bold text-neutral-300 uppercase tracking-widest mb-4">📐 Dataset Preview (first 10 rows)</h3>
        <div className="overflow-x-auto rounded-xl border border-white/10">
          <table className="w-full text-xs text-left text-neutral-400">
            <thead className="bg-neutral-900/50 text-neutral-500 uppercase tracking-wider">
              <tr>
                {columns.map(col => (
                  <th key={col} className="px-4 py-3 font-medium whitespace-nowrap">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {preview.slice(0, 10).map((row, i) => (
                <tr key={i} className="hover:bg-white/5 transition-colors">
                  {columns.map(col => (
                    <td key={col} className="px-4 py-3 font-mono whitespace-nowrap">{String(row[col] ?? '—')}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
