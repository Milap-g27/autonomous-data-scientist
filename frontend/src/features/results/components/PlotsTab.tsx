import type { AnalyzeResponse } from '../../../services/api';
import { Image as ImageIcon } from 'lucide-react';

interface Props { result: AnalyzeResponse; }

export default function PlotsTab({ result }: Props) {
  const figures = result.eda_figures || [];
  
  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex flex-col space-y-2">
        <h2 className="text-3xl font-bold tracking-tight text-white">
          Visualizations
        </h2>
        <p className="text-sm text-neutral-400">
          Graphical representations of the data distributions, correlations, and target dependencies.
        </p>
      </div>

      {figures.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 px-4 mt-8 rounded-2xl border border-dashed border-white/10 bg-white/[0.01]">
          <div className="h-12 w-12 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <ImageIcon className="h-6 w-6 text-neutral-500" />
          </div>
          <h3 className="text-lg font-medium text-neutral-300 mb-1">No Visualizations Found</h3>
          <p className="text-sm text-neutral-500 text-center max-w-sm">
            The dataset might have been too small or lacked numeric features to generate visual insights.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {figures.map((fig, i) => (
          <div 
            key={i} 
            className="group rounded-2xl border border-white/10 bg-[#0a0a0a] overflow-hidden shadow-2xl transition-all hover:border-white/20"
          >
            {fig.heading && (
              <div className="px-6 py-5 border-b border-white/10 bg-white/[0.02]">
                <h3 className="text-base font-semibold text-neutral-200 tracking-wide">{fig.heading}</h3>
                {fig.description && (
                  <p className="text-sm text-neutral-400 mt-1.5 leading-relaxed">{fig.description}</p>
                )}
              </div>
            )}
            {fig.image_base64 && (
              <div className="p-6 flex items-center justify-center bg-[#111] min-h-[300px]">
                <img
                  src={`data:image/png;base64,${fig.image_base64}`}
                  alt={fig.heading || `Plot ${i + 1}`}
                  className="w-full h-auto object-contain rounded-lg border border-white/5 shadow-inner transition-transform duration-500 group-hover:scale-[1.02]"
                  loading="lazy"
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
