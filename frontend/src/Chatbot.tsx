import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css';

const Chatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isVisible] = useState(true);
  const [messages, setMessages] = useState<{ text: string; isBot: boolean }[]>([]);
  const [displayMessage, setDisplayMessage] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [isCentered, setIsCentered] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, displayMessage]);

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      const fullMessage = "Hello fellow Novoxian! ✨ Welcome to the Novox community. I'm Novox AI Prime, your personal guide to excellence. How can I make your journey amazing today? 🚀";
      
      let charIndex = 0;
      setDisplayMessage("");
      
      const typingInterval = setInterval(() => {
        if (charIndex < fullMessage.length) {
          setDisplayMessage(fullMessage.substring(0, charIndex + 1));
          charIndex++;
        } else {
          clearInterval(typingInterval);
          setMessages([{ text: fullMessage, isBot: true }]);
        }
      }, 40);

      return () => clearInterval(typingInterval);
    }
  }, [isOpen, messages.length]);

  const toggleChat = () => setIsOpen(!isOpen);

  const handleSend = async () => {
    if (inputValue.trim() === "") return;
    
    setIsCentered(true); // Expand to center Command Center
    const currentInput = inputValue;
    const newUserMessage = { text: currentInput, isBot: false };
    setMessages((prev) => [...prev, newUserMessage]);
    setInputValue("");

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: currentInput })
      });
      
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      let botResponseText = data.answer;
      // Optional: Log sources to console instead of displaying them to the user
      if (data.sources && data.sources.length > 0) {
        console.log("Sources:", data.sources);
      }
      
      setMessages((prev) => [...prev, { text: botResponseText, isBot: true }]);
    } catch (error: any) {
      console.error("Failed to fetch response:", error);
      const errorMessage = error.message || "Could not connect to Neural Link. Please check backend status.";
      setMessages((prev) => [...prev, { text: `Error: ${errorMessage}`, isBot: true }]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  const NovoxAvatar = () => (
    <svg viewBox="0 0 100 100" className="novox-avatar-svg">
      <circle cx="50" cy="50" r="48" fill="#ffffff" stroke="#e0e6ed" strokeWidth="1" />
      <path 
        d="M50 50 L75 35 L90 50 L75 65 Z" 
        fill="#0047AB" 
        stroke="#0047AB" 
        strokeWidth="1"
        opacity="0.8"
      />
      <path 
        d="M10 50 L25 35 L50 50 L25 65 Z" 
        fill="#0047AB" 
        stroke="#0047AB" 
        strokeWidth="1.2"
      />
    </svg>
  );

  return (
    <div className={`chatbot-container ${isOpen ? 'open' : ''} ${isVisible ? 'visible' : 'hidden'} ${isCentered ? 'centered-mode' : ''}`}>
      {!isOpen && (
        <button className="chatbot-toggle light-pulse" onClick={toggleChat} aria-label="Connect to Novox AI">
          <div className="avatar-preview">
             <NovoxAvatar />
          </div>
        </button>
      )}

      {isOpen && (
        <div className="chatbot-window light-frame">
          <div className="chatbot-header light-header">
            <div className="chatbot-avatar-container">
              <div className="chatbot-avatar light-glow">
                <NovoxAvatar />
              </div>
            </div>
            <div className="header-text">
              <div className="name">Novox AI Prime</div>
              <div className="status">Neural Link Active 💎</div>
            </div>
            <button className="close-btn light-close" onClick={toggleChat} aria-label="Disconnect">×</button>
          </div>
          
          <div className="chatbot-messages light-content">
            {messages.length === 0 && displayMessage && (
              <div className="message-wrapper bot">
                <div className="message light-bot slide-in-left">
                  {displayMessage}
                  <span className="cursor">_</span>
                </div>
              </div>
            )}
            
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-wrapper ${msg.isBot ? 'bot' : 'user'}`}>
                <div className={`message ${msg.isBot ? 'light-bot slide-in-left' : 'light-user slide-in-right'}`}>
                  {msg.text}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input light-footer">
            <div className="input-circuitry">
              <input 
                type="text" 
                placeholder="Enter command..." 
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
              />
            </div>
            <button className="send-btn cobalt-btn" onClick={handleSend} aria-label="Send message">
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path fill="currentColor" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
