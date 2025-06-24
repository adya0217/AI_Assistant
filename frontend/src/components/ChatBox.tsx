import React, { useEffect, useRef } from 'react';

interface Message {
    sender: string;
    text: string;
    timestamp: string;
    type: 'text' | 'voice' | 'image' | 'audio';
}

interface ChatBoxProps {
    messages: Message[];
}

const ChatBox: React.FC<ChatBoxProps> = ({ messages }) => {
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const getMessageIcon = (type: Message['type']) => {
        switch (type) {
            case 'text':
                return '💬';
            case 'voice':
            case 'audio':
                return '🎤';
            case 'image':
                return '🖼️';
            default:
                return '💬';
        }
    };

    const scrollToBottom = () => {
        console.log('📜 Scrolling to bottom of chat');
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        console.log('🔄 Messages updated:', messages);
        scrollToBottom();
    }, [messages]);

    return (
        <div className="chat-box">
            {messages.map((message, index) => {
                console.log(`📨 Rendering message ${index}:`, message);
                return (
                    <div
                        key={index}
                        className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
                    >
                        <div className="message-header">
                            <span className="message-icon">
                                {getMessageIcon(message.type)}
                            </span>
                            <span className="message-sender">
                                {message.sender === 'user' ? 'You' : 'AI Assistant'}
                            </span>
                            <span className="message-timestamp">{message.timestamp}</span>
                        </div>
                        <div className="message-content">{message.text}</div>
                    </div>
                );
            })}
            <div ref={messagesEndRef} />
        </div>
    );
};

export default ChatBox;
