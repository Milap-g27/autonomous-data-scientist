import type { AnalyzeResponse } from '../../../services/api';

interface Props { result: AnalyzeResponse; }

export default function ProcessingLogTab({ result }: Props) {
  const cleaningReport = result.cleaning_report || 'No report';
  const featureReport = result.feature_report || 'No report';

  const toBullets = (text: string) => {
    return text
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0);
  };

  const renderBullets = (items: string[]) => (
    <ul className="space-y-2">
      {items.map((item, i) => {
        // Color-code different types of log entries
        const isAction = item.toLowerCase().includes('filled') || item.toLowerCase().includes('encoded') || item.toLowerCase().includes('created');
        const isRemove = item.toLowerCase().includes('dropped') || item.toLowerCase().includes('removed');
        const isSeparated = item.toLowerCase().includes('separated');

        let dotColor = 'bg-neutral-600';
        if (isAction) dotColor = 'bg-neutral-400';
        if (isRemove) dotColor = 'bg-neutral-500';
        if (isSeparated) dotColor = 'bg-neutral-300';

        // Highlight code-like tokens
        const rendered = item.replace(
          /`([^`]+)`|'([^']+)'|(\b[A-Z_]{2,}\b)/g,
          (match) => `<code class="px-1.5 py-0.5 bg-neutral-800 rounded text-neutral-300 text-[11px] font-mono">${match.replace(/[`']/g, '')}</code>`
        );

        return (
          <li key={i} className="flex items-start gap-3 text-sm text-neutral-400">
            <span className={`w-1.5 h-1.5 rounded-full mt-2 shrink-0 ${dotColor}`}></span>
            <span dangerouslySetInnerHTML={{ __html: rendered }} />
          </li>
        );
      })}
    </ul>
  );

  return (
    <div className="space-y-10 max-w-4xl">
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
        Processing Logs
      </h2>

      <div>
        <h3 className="text-base font-bold text-neutral-200 mb-4">Data Cleaning Report</h3>
        {renderBullets(toBullets(cleaningReport))}
      </div>

      <div>
        <h3 className="text-base font-bold text-neutral-200 mb-4">Feature Engineering Report</h3>
        {renderBullets(toBullets(featureReport))}
      </div>
    </div>
  );
}
