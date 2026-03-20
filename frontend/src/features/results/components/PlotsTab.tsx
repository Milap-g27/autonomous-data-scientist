import type { AnalyzeResponse } from '../../../services/api';

interface Props { result: AnalyzeResponse; }

export default function PlotsTab({ result }: Props) {
  const figures = result.eda_figures || [];
  
  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
        Plots & Visualizations
      </h2>

      {figures.length === 0 && (
        <div className="text-center py-12 text-neutral-500 text-sm">No visualizations generated yet.</div>
      )}

      {figures.map((fig, i) => (
        <div key={i} className="rounded-xl border border-white/10 bg-neutral-950/50 overflow-hidden">
          {fig.heading && (
            <div className="px-6 pt-6">
              <h3 className="text-base font-bold text-neutral-200 mb-1">{fig.heading}</h3>
              {fig.description && <p className="text-xs text-neutral-500">{fig.description}</p>}
            </div>
          )}
          {fig.image_base64 && (
            <div className="p-4 flex justify-center">
              <img
                src={`data:image/png;base64,${fig.image_base64}`}
                alt={fig.heading || `Plot ${i + 1}`}
                className="max-w-full rounded-lg"
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
