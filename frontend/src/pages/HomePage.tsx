import { useNavigate } from 'react-router-dom';
import { Upload, Zap, Brain, BarChart2 } from 'lucide-react';
import { useState, useCallback } from 'react';
import { SplineSceneBasic } from '../components/ui/demo';
import { EvervaultCard, Icon } from '../components/ui/evervault-card';
import { uploadDataset, type DatasetInfo } from '../services/api';

interface Props {
  onSessionCreated: (sessionId: string, info: DatasetInfo) => void;
}

export default function HomePage({ onSessionCreated }: Props) {
  const navigate = useNavigate();
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  const handleFile = useCallback(async (file: File) => {
    setError('');
    setUploading(true);
    try {
      const res = await uploadDataset(file);
      onSessionCreated(res.session_id, res.dataset_info);
      navigate('/configure');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [navigate, onSessionCreated]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div className="space-y-12">
      {/* 3D Hero — always visible on home */}
      <section>
        <SplineSceneBasic />
      </section>

      {/* Upload Section */}
      <section className="bg-black/[0.96] border border-white/10 rounded-2xl p-8 md:p-12">
        <h2 className="text-2xl md:text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 mb-2">
          Data Injection
        </h2>
        <p className="text-sm text-neutral-400 mb-8">Upload your CSV dataset to start the autonomous analysis pipeline.</p>

        {error && (
          <div className="mb-6 p-4 rounded-xl border border-red-500/20 bg-red-900/10 text-red-400 text-sm">{error}</div>
        )}

        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
          className={`w-full border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer transition-all ${dragOver ? 'border-white/50 bg-neutral-900' : 'border-neutral-700/50 bg-neutral-950/50 hover:bg-neutral-900/50 hover:border-neutral-600'
            }`}
        >
          <div className="w-16 h-16 rounded-full bg-neutral-800/50 flex items-center justify-center mb-4">
            <Upload className={`w-8 h-8 text-neutral-300 ${uploading ? 'animate-pulse' : ''}`} />
          </div>
          <p className="text-neutral-200 font-medium text-lg mb-2">
            {uploading ? 'Processing...' : 'Drag & Drop CSV Here'}
          </p>
          <p className="text-neutral-500 text-xs tracking-widest uppercase">or click to browse</p>
        </div>
        <input
          id="file-input"
          type="file"
          accept=".csv"
          className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />
      </section>

      {/* Features Grid — EvervaultCard hover effect */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { icon: Zap, title: 'Auto-Clean & Engineer', desc: 'Handles missing data, encoding, and feature creation autonomously.' },
          { icon: BarChart2, title: 'Multi-Model Training', desc: 'Trains 15+ models and benchmarks them automatically.' },
          { icon: Brain, title: 'AI Explanation', desc: 'Generates human-readable insights about your data and model performance.' },
        ].map(f => (
          <div key={f.title} className="border border-white/[0.2] flex flex-col items-center p-4 relative h-[22rem] rounded-2xl bg-neutral-950/50">
            <Icon className="absolute h-6 w-6 -top-3 -left-3 text-white/30" />
            <Icon className="absolute h-6 w-6 -bottom-3 -left-3 text-white/30" />
            <Icon className="absolute h-6 w-6 -top-3 -right-3 text-white/30" />
            <Icon className="absolute h-6 w-6 -bottom-3 -right-3 text-white/30" />

            <EvervaultCard text={f.title} description={f.desc} />
          </div>
        ))}
      </section>
    </div>
  );
}
