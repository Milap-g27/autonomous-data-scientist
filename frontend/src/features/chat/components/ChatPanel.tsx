import { useState, useRef, useEffect } from 'react';
import { X, Send, Bot, Copy, Check, Maximize2, Minimize2 } from 'lucide-react';
import { chatWithAssistant } from '../../../services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  image_base64?: string;
}

interface Props {
  sessionId: string;
  isOpen: boolean;
  onClose: () => void;
}

export default function ChatPanel({ sessionId, isOpen, onClose }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);
    try {
      const res = await chatWithAssistant(sessionId, userMsg);
      setMessages(prev => [...prev, { role: 'assistant', content: res.reply, image_base64: res.image_base64 }]);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error: Could not reach the assistant.' }]);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className={`fixed z-[100] bg-neutral-950 shadow-2xl flex flex-col overflow-hidden transition-all duration-300 origin-bottom-right ${
      isExpanded 
        ? 'inset-4 md:inset-8 border border-white/20 rounded-3xl' 
        : 'bottom-24 md:bottom-28 right-6 md:right-10 w-[380px] max-w-[calc(100vw-2rem)] h-[500px] border border-white/10 rounded-2xl'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/10 bg-neutral-900/50">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-neutral-300" />
          <span className="font-bold text-neutral-200 text-sm">AI Assistant</span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setIsExpanded(!isExpanded)} className="text-neutral-500 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-white/5">
            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
          <button onClick={onClose} className="text-neutral-500 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-white/5 text-red-500 hover:bg-red-500/10 hover:text-red-400">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 text-sm">
        {messages.length === 0 && (
          <p className="text-neutral-600 text-center pt-12 text-xs">Ask about your data or models…</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`relative group max-w-[85%] px-4 py-3 rounded-2xl ${
              msg.role === 'user' 
                ? 'bg-neutral-800 text-neutral-200 rounded-br-sm' 
                : 'bg-neutral-900/50 border border-white/5 text-neutral-300 rounded-bl-sm'
            }`}>
              <button
                onClick={() => handleCopy(msg.content, i)}
                className="absolute top-2 right-2 p-1.5 rounded-lg bg-neutral-800/80 backdrop-blur-sm border border-white/10 text-neutral-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity z-10"
                title="Copy text"
              >
                {copiedIndex === i ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
              <p className="text-[13px] leading-relaxed whitespace-pre-wrap">
                {msg.content.split(/`([^`]+)`/g).map((part, idx) => 
                  idx % 2 === 1 ? (
                    <code key={idx} className="bg-neutral-800 text-neutral-300 font-mono text-[12px] px-1.5 py-0.5 rounded mx-0.5">
                      {part}
                    </code>
                  ) : (
                    <span key={idx}>{part}</span>
                  )
                )}
              </p>
              {msg.image_base64 && (
                <div className="mt-3">
                  <img src={`data:image/png;base64,${msg.image_base64}`} alt="Generated Plot" className="w-full rounded-lg bg-white/5 border border-white/10" />
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-neutral-900/50 border border-white/5 px-4 py-3 rounded-2xl rounded-bl-sm">
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 bg-neutral-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                <span className="w-1.5 h-1.5 bg-neutral-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                <span className="w-1.5 h-1.5 bg-neutral-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t border-white/10 bg-neutral-900/30">
        <div className="flex items-center gap-2">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask about your data or models…"
            rows={isExpanded ? 3 : 2}
            className="flex-1 bg-neutral-900 border border-white/10 rounded-xl px-4 py-3 text-sm text-neutral-200 placeholder:text-neutral-600 focus:outline-none focus:border-neutral-500 transition-colors resize-none overflow-y-auto"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="p-3 bg-white text-black rounded-xl hover:bg-neutral-200 transition-all disabled:opacity-30"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
