import { useState, useRef, useEffect } from 'react';
import { useNarration } from '../hooks/useNarration';

interface NarrationChatProps {
  engineId: number;
  datasetId: string;
}

export default function NarrationChat({ engineId, datasetId }: NarrationChatProps) {
  const { messages, loading, error, narrationAvailable, sendMessage } = useNarration(engineId, datasetId);
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || loading) return;
    sendMessage(inputText);
    setInputText('');
  };

  if (!narrationAvailable) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted text-sm font-mono border border-dashed border-border rounded-lg bg-panel2 bg-opacity-50 min-h-[300px] p-6 text-center">
        <p>Narration unavailable.</p>
        <p className="mt-2 text-xs">Ensure GEMINI_API_KEY is configured in the environment.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-[300px] max-h-[500px]">
      {error && (
        <div className="bg-[#E0533A]/10 border border-[#E0533A]/30 text-[#E0533A] text-xs p-2 rounded mb-3">
          Error: {error}
        </div>
      )}
      
      <div className="flex-1 overflow-y-auto pr-2 mb-4 flex flex-col space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div 
              className={`max-w-[90%] rounded-lg px-3 py-2 text-sm ${
                msg.role === 'user' 
                  ? 'bg-[#3ECF8E]/20 border border-[#3ECF8E]/30 text-text' 
                  : 'bg-panel2 border border-border text-muted whitespace-pre-wrap'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-panel2 border border-border text-muted rounded-lg px-3 py-2 text-sm italic">
              Gemini is thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="mt-auto relative flex items-center">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask about this engine snapshot..."
          disabled={loading}
          className="w-full bg-panel2 border border-border rounded-lg pl-3 pr-10 py-2 text-sm text-text focus:outline-none focus:border-[#3ECF8E] disabled:opacity-50"
        />
        <button 
          type="submit" 
          disabled={!inputText.trim() || loading}
          className="absolute right-2 p-1 text-muted hover:text-[#3ECF8E] disabled:opacity-50 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </form>
    </div>
  );
}
