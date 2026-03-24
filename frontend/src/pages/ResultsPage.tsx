import { useState } from 'react';
import { BarChart2, LineChart, Activity, Sparkles, Brain, Settings2 } from 'lucide-react';
import EDATab from '../features/results/components/EDATab';
import PlotsTab from '../features/results/components/PlotsTab';
import ModelPerformanceTab from '../features/results/components/ModelPerformanceTab';
import PredictionsTab from '../features/results/components/PredictionsTab';
import AIExplanationTab from '../features/results/components/AIExplanationTab';
import ProcessingLogTab from '../features/results/components/ProcessingLogTab';
import type { AnalyzeResponse, DatasetInfo } from '../services/api';

const TABS = [
  { key: 'eda', label: 'EDA & Insights', icon: BarChart2 },
  { key: 'plots', label: 'Plots & Visualizations', icon: LineChart },
  { key: 'model', label: 'Model Performance', icon: Activity },
  { key: 'predictions', label: 'Predictions', icon: Sparkles },
  { key: 'explanation', label: 'AI Explanation', icon: Brain },
  { key: 'log', label: 'Processing Log', icon: Settings2 },
] as const;

type TabKey = typeof TABS[number]['key'];

interface Props {
  result: AnalyzeResponse;
  datasetInfo: DatasetInfo;
  sessionId: string;
  target: string | null;
}

export default function ResultsPage({ result, datasetInfo, sessionId, target }: Props) {
  const [activeTab, setActiveTab] = useState<TabKey>('eda');

  const bestMetrics = result.metrics?.[result.best_model] || {};
  const pt = result.problem_type;
  const metricCards = pt === 'Clustering'
    ? [
      { label: 'Best Algorithm', value: result.best_model },
      { label: 'Silhouette', value: bestMetrics['Silhouette Score'] },
      { label: 'Calinski-Harabasz', value: bestMetrics['Calinski-Harabasz'] },
      { label: 'Training Time', value: `${result.training_time}s` },
    ]
    : pt === 'Regression'
    ? [
      { label: 'Best Model', value: result.best_model },
      { label: 'R² Score', value: bestMetrics['R2'] },
      { label: 'MAE', value: bestMetrics['MAE'] },
      { label: 'Training Time', value: `${result.training_time}s` },
    ]
    : [
      { label: 'Best Model', value: result.best_model },
      { label: 'Accuracy', value: bestMetrics['Accuracy'] },
      { label: 'F1 Score', value: bestMetrics['F1 Score'] },
      { label: 'Training Time', value: `${result.training_time}s` },
    ];

  return (
    <div className="space-y-8">
      {/* Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metricCards.map(c => (
          <div key={c.label} className="bg-neutral-900/50 border border-white/10 rounded-xl p-5">
            <p className="text-[10px] text-neutral-500 uppercase tracking-widest mb-1">{c.label}</p>
            <p className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 truncate">
              {typeof c.value === 'number' ? c.value.toFixed(4) : String(c.value ?? 'N/A')}
            </p>
          </div>
        ))}
      </div>

      {/* Tab Navigation */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4 pb-2">
        {TABS.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center justify-center gap-2 px-2 py-3 rounded-xl text-xs xl:text-sm font-medium whitespace-nowrap transition-all w-full ${
                activeTab === tab.key
                  ? 'bg-neutral-800 text-neutral-200 shadow-[0_0_20px_rgba(255,255,255,0.05)]'
                  : 'text-neutral-500 hover:text-neutral-300 bg-neutral-900/50 hover:bg-neutral-900'
              }`}
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span className="truncate">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <div className="bg-black/[0.96] border border-white/10 rounded-2xl p-6 md:p-8 min-h-[400px]">
        {activeTab === 'eda' && <EDATab result={result} preview={datasetInfo.preview} />}
        {activeTab === 'plots' && <PlotsTab result={result} />}
        {activeTab === 'model' && <ModelPerformanceTab result={result} />}
        {activeTab === 'predictions' && (
          <PredictionsTab result={result} sessionId={sessionId} preview={datasetInfo.preview} target={target} />
        )}
        {activeTab === 'explanation' && <AIExplanationTab result={result} />}
        {activeTab === 'log' && <ProcessingLogTab result={result} />}
      </div>
    </div>
  );
}
