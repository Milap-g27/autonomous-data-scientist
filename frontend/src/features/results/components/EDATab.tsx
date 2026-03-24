import type { AnalyzeResponse } from '../../../services/api';
import { FileText, Search, Table2 } from 'lucide-react';

interface Props { result: AnalyzeResponse; preview: Record<string, unknown>[]; }

export default function EDATab({ result, preview }: Props) {
  const eda = result.eda_results as Record<string, unknown>;
  const description = eda?.description as Record<string, Record<string, number>> | undefined;

  // Build column info from preview
  const columns = preview.length > 0 ? Object.keys(preview[0]) : [];
  
  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex flex-col space-y-2">
        <h2 className="text-3xl font-bold tracking-tight text-white">
          Exploratory Data Analysis
        </h2>
        <p className="text-sm text-neutral-400">
          Discover patterns, spot anomalies, and check assumptions with summary statistics.
        </p>
      </div>

      {/* Summary Statistics */}
      {description && (
        <div className="rounded-2xl border border-white/10 bg-[#0a0a0a] overflow-hidden shadow-2xl">
          <div className="flex items-center gap-3 px-6 py-4 border-b border-white/10 bg-white/[0.02]">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">
              <Table2 className="h-4 w-4" />
            </div>
            <h3 className="text-sm font-semibold text-neutral-200 tracking-wide">Summary Statistics</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-[#111] text-xs uppercase tracking-wider text-neutral-400 border-b border-white/5">
                <tr>
                  <th className="px-6 py-4 font-semibold">Feature</th>
                  {Object.keys(Object.values(description)[0] || {}).map(stat => (
                    <th key={stat} className="px-6 py-4 font-semibold text-right">{stat}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {Object.entries(description).map(([col, stats], idx) => (
                  <tr key={col} className={`${idx % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.01]'} hover:bg-white/[0.04] transition-colors`}>
                    <td className="px-6 py-4 font-medium text-neutral-200">{col}</td>
                    {Object.values(stats).map((val, i) => (
                      <td key={i} className="px-6 py-4 font-mono text-right text-neutral-400">
                        {typeof val === 'number' ? val.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(val)}
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
      <div className="rounded-2xl border border-white/10 bg-[#0a0a0a] overflow-hidden shadow-2xl">
        <div className="flex items-center gap-3 px-6 py-4 border-b border-white/10 bg-white/[0.02]">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-500/10 text-purple-400">
            <Search className="h-4 w-4" />
          </div>
          <h3 className="text-sm font-semibold text-neutral-200 tracking-wide">Data Types & Samples</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-[#111] text-xs uppercase tracking-wider text-neutral-400 border-b border-white/5">
              <tr>
                <th className="px-6 py-4 font-semibold">Column</th>
                <th className="px-6 py-4 font-semibold">Assigned Type</th>
                <th className="px-6 py-4 font-semibold">Random Sample</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {columns.map((col, idx) => {
                const isNumeric = typeof preview[0][col] === 'number';
                return (
                  <tr key={col} className={`${idx % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.01]'} hover:bg-white/[0.04] transition-colors`}>
                    <td className="px-6 py-4 font-medium text-neutral-200">{col}</td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${
                        isNumeric ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                      }`}>
                        {isNumeric ? 'Numeric' : 'Categorical'}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono text-neutral-400 truncate max-w-xs">{String(preview[0][col] ?? '—')}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Preview */}
      <div className="rounded-2xl border border-white/10 bg-[#0a0a0a] overflow-hidden shadow-2xl">
        <div className="flex items-center gap-3 px-6 py-4 border-b border-white/10 bg-white/[0.02]">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-500/10 text-orange-400">
            <FileText className="h-4 w-4" />
          </div>
          <h3 className="text-sm font-semibold text-neutral-200 tracking-wide">Preview (First 10 Rows)</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-[#111] text-xs uppercase tracking-wider text-neutral-400 border-b border-white/5">
              <tr>
                {columns.map(col => (
                  <th key={col} className="px-6 py-4 font-semibold whitespace-nowrap">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {preview.slice(0, 10).map((row, i) => (
                <tr key={i} className={`${i % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.01]'} hover:bg-white/[0.04] transition-colors`}>
                  {columns.map(col => (
                    <td key={col} className="px-6 py-4 font-mono text-neutral-400 whitespace-nowrap">{String(row[col] ?? '—')}</td>
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
