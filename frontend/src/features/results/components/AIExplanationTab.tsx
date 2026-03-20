import type { AnalyzeResponse } from '../../../services/api';

interface Props { result: AnalyzeResponse; }

export default function AIExplanationTab({ result }: Props) {
  const explanation = result.explanation || 'No explanation generated.';

  // Simple markdown-to-HTML: bold, headings, bullets, code
  const renderMarkdown = (text: string) => {
    const lines = text.split('\n');
    return lines.map((line, i) => {
      // Headings
      if (line.startsWith('### ')) return <h4 key={i} className="text-base font-bold text-neutral-200 mt-6 mb-2">{line.slice(4)}</h4>;
      if (line.startsWith('## ')) return <h3 key={i} className="text-lg font-bold text-neutral-200 mt-8 mb-3">{line.slice(3)}</h3>;
      if (line.startsWith('# ')) return <h2 key={i} className="text-xl font-bold text-neutral-200 mt-8 mb-3">{line.slice(2)}</h2>;
      
      // Bullet points
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        const content = line.trim().slice(2);
        return (
          <li key={i} className="text-sm text-neutral-400 leading-relaxed ml-4 mb-1 list-disc">
            <span dangerouslySetInnerHTML={{ __html: formatInline(content) }} />
          </li>
        );
      }

      // Empty line
      if (line.trim() === '') return <div key={i} className="h-3" />;
      
      // Regular paragraph
      return <p key={i} className="text-sm text-neutral-400 leading-relaxed mb-2" dangerouslySetInnerHTML={{ __html: formatInline(line) }} />;
    });
  };

  return (
    <div className="space-y-4 max-w-4xl">
      <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
        AI Explanation
      </h2>
      <div className="prose prose-invert max-w-none">
        {renderMarkdown(explanation)}
      </div>
    </div>
  );
}

function formatInline(text: string): string {
  // Bold: **text**
  let out = text.replace(/\*\*(.*?)\*\*/g, '<strong class="text-neutral-200">$1</strong>');
  // Inline code: `text`
  out = out.replace(/`(.*?)`/g, '<code class="px-1.5 py-0.5 bg-neutral-800 rounded text-neutral-300 text-xs">$1</code>');
  return out;
}
