import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaPaperPlane, FaRobot, FaUser, FaSmile, FaPaperclip, FaImage } from 'react-icons/fa';
import Avatar from 'react-avatar';

export default function Chat() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your insomnia risk assistant. How can I help you today?",
      sender: 'bot',
      timestamp: new Date().toISOString()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    setTimeout(() => {
      const botResponses = [
        "Based on your questions, I recommend checking your sleep hygiene. Try to maintain a consistent sleep schedule.",
        "The risk assessment takes into account age, BMI, smoking status, and various medical conditions.",
        "You can view your complete history in the History tab. Each assessment is saved automatically.",
        "If you're concerned about your sleep patterns, consider keeping a sleep diary for 2 weeks.",
        "The dashboard shows trends over time. More data will give you better insights.",
        "Remember that this tool is for informational purposes. Always consult with healthcare professionals."
      ];
      
      const botMessage = {
        id: Date.now() + 1,
        text: botResponses[Math.floor(Math.random() * botResponses.length)],
        sender: 'bot',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, botMessage]);
      setIsTyping(false);
    }, 1500);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-7xl mx-auto h-[calc(100vh-200px)] flex flex-col"
    >
      {/* Header */}
      <div className="bg-white rounded-t-2xl shadow-lg p-6 border-b">
        <div className="flex items-center gap-4">
          <div className="relative">
            <Avatar name="AI Assistant" size="50" round={true} />
            <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
          </div>
          <div>
            <h2 className="text-xl font-bold">AI Health Assistant</h2>
            <p className="text-sm text-gray-500">Online • Usually responds instantly</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
        <div className="space-y-4">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={'flex ' + (message.sender === 'user' ? 'justify-end' : 'justify-start')}
              >
                <div className={'flex gap-3 max-w-[70%] ' + (message.sender === 'user' ? 'flex-row-reverse' : '')}>
                  <div className="flex-shrink-0">
                    {message.sender === 'bot' ? (
                      <Avatar name="AI" size="40" round={true} />
                    ) : (
                      <Avatar name="User" size="40" round={true} />
                    )}
                  </div>
                  <div>
                    <div className={'p-4 rounded-2xl ' + (message.sender === 'user' ? 'bg-primary-600 text-white rounded-tr-none' : 'bg-white text-gray-800 rounded-tl-none shadow-sm')}>
                      <p className="whitespace-pre-wrap">{message.text}</p>
                    </div>
                    <p className={'text-xs mt-1 ' + (message.sender === 'user' ? 'text-right' : 'text-left') + ' text-gray-400'}>
                      {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {isTyping && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-start"
            >
              <div className="flex gap-3">
                <Avatar name="AI" size="40" round={true} />
                <div className="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-white rounded-b-2xl shadow-lg p-4 border-t">
        <div className="flex items-center gap-3">
          <button className="p-2 text-gray-500 hover:text-primary-600 transition-colors">
            <FaPaperclip size={20} />
          </button>
          <button className="p-2 text-gray-500 hover:text-primary-600 transition-colors">
            <FaImage size={20} />
          </button>
          <button className="p-2 text-gray-500 hover:text-primary-600 transition-colors">
            <FaSmile size={20} />
          </button>
          
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1 p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none"
            rows="1"
            style={{ minHeight: '50px', maxHeight: '120px' }}
          />
          
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim()}
            className="p-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <FaPaperPlane size={20} />
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          This AI assistant can answer questions about insomnia risk and help you understand your assessments.
        </p>
      </div>
    </motion.div>
  );
}
