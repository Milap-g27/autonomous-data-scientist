import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { Home, Settings2, BarChart2, Bot } from 'lucide-react';
import { useState } from 'react';
import ChatPanel from '../features/chat/components/ChatPanel';

interface Props {
  sessionId: string;
  hasResults: boolean;
}

export default function Layout({ sessionId, hasResults }: Props) {
  const location = useLocation();
  const [chatOpen, setChatOpen] = useState(false);

  const navItems = [
    { to: '/', label: 'Home', icon: Home },
    ...(sessionId ? [{ to: '/configure', label: 'Configure', icon: Settings2 }] : []),
    ...(hasResults ? [{ to: '/results', label: 'Results', icon: BarChart2 }] : []),
  ];

  return (
    <div className="min-h-screen bg-black text-neutral-300 font-sans overflow-x-hidden selection:bg-neutral-800">
      {/* Top Nav */}
      <header className="fixed top-0 w-full z-[60] flex items-center px-6 h-16 bg-black/80 backdrop-blur-md border-b border-white/10">
        <div className="flex items-center gap-3 mr-8">
          <img src="/logo.svg" alt="Logo" className="w-6 h-6" />
          <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
            Autonomous Data Scientist
          </h1>
        </div>

        <nav className="flex gap-1">
          {navItems.map(item => {
            const Icon = item.icon;
            const isActive = location.pathname === item.to;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${isActive
                  ? 'bg-neutral-800 text-neutral-200'
                  : 'text-neutral-500 hover:text-neutral-300 hover:bg-neutral-900'
                  }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>

        <div className="flex-1" />

        {/* Status Chip */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-neutral-900/50 border border-white/5 rounded-full">
          <span className={`w-2 h-2 rounded-full ${!sessionId ? 'bg-neutral-500' :
            hasResults ? 'bg-green-400' : 'bg-amber-400 animate-pulse'
            }`}></span>
          <span className="text-[10px] font-mono text-neutral-400 uppercase tracking-wider">
            {!sessionId ? 'IDLE' : hasResults ? 'COMPLETE' : 'ACTIVE'}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-24 pb-24 px-4 md:px-8 lg:px-16 max-w-[1400px] mx-auto min-h-screen">
        <Outlet />
      </main>

      {/* Floating Chatbot */}
      {sessionId && (
        <>
          <button
            onClick={() => setChatOpen(!chatOpen)}
            className="fixed bottom-6 right-6 z-50 w-14 h-14 bg-white text-black rounded-full shadow-[0_0_40px_rgba(255,255,255,0.15)] flex items-center justify-center transition-all hover:scale-105 hover:bg-neutral-200"
          >
            <Bot className="w-6 h-6" />
          </button>
          <ChatPanel sessionId={sessionId} isOpen={chatOpen} onClose={() => setChatOpen(false)} />
        </>
      )}
    </div>
  );
}
