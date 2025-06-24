import React, { useState } from "react";
import { BookOpen, Brain, Lightbulb, Star, Sparkles, Mic, Camera, FileText } from "lucide-react";
import InputArea from "./components/InputArea";
import './App.css';
import axios from "axios";
import { Card } from "./components/ui/card";

interface ChatMessage {
  id: number;
  text: string;
  isBot: boolean;
  timestamp: Date;
  inputType?: string;
  analysisSummary?: any;
}

export default function EducationalChatbot() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      text: "Hello! I'm your AI learning companion. I can help you with text, voice, and image analysis. What would you like to explore today?",
      isBot: true,
      timestamp: new Date(),
    },
  ]);
  const [uploading, setUploading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [currentSession, setCurrentSession] = useState<string | null>(null);

  // Enhanced unified send handler for multimodal input
  const handleUnifiedSend = async (msg: string, image?: File, audio?: File) => {
    console.log('🟦 handleUnifiedSend called:', { msg, image, audio });
    if (!msg.trim() && !image && !audio) return;

    const newMessage: ChatMessage = {
      id: messages.length + 1,
      text: msg || (audio ? '[Audio message]' : image ? '[Image uploaded]' : ''),
      isBot: false,
      timestamp: new Date(),
      inputType: audio ? 'audio' : image ? 'image' : 'text'
    };
    setMessages((prev) => [...prev, newMessage]);
    setGenerating(true);

    try {
      // Use the new unified multimodal endpoint
      const formData = new FormData();

      if (msg.trim()) {
        formData.append('text_query', msg);
      }

      if (audio) {
        formData.append('voice_file', audio);
      }

      if (image) {
        formData.append('image_file', image);
      }

      console.log('🟧 Sending multimodal request to unified endpoint');
      const res = await axios.post('http://localhost:8000/api/unified/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log('🟩 Received unified response:', res.data);

      // Store session ID for context
      if (res.data.session_id) {
        setCurrentSession(res.data.session_id);
      }

      // Create bot response with enhanced information
      const botMessage: ChatMessage = {
        id: messages.length + 2,
        text: res.data.response,
        isBot: true,
        timestamp: new Date(),
        inputType: res.data.input_types?.[0] || 'text',
        analysisSummary: res.data.analysis_summary
      };

      setMessages((prev) => [...prev, botMessage]);

    } catch (err: any) {
      console.error('Error in unified send:', err);
      const errorMessage: ChatMessage = {
        id: messages.length + 2,
        text: err.response?.data?.detail || 'Sorry, I encountered an error. Please try again.',
        isBot: true,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setGenerating(false);
    }
  };

  // Function to get input type icon
  const getInputTypeIcon = (inputType?: string) => {
    switch (inputType) {
      case 'voice':
        return <Mic size={16} className="inline mr-2" />;
      case 'image':
        return <Camera size={16} className="inline mr-2" />;
      case 'text':
        return <FileText size={16} className="inline mr-2" />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-400 via-pink-400 to-orange-400 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-10 left-10 w-20 h-20 bg-yellow-300 rounded-full opacity-20 animate-bounce"></div>
        <div className="absolute top-32 right-20 w-16 h-16 bg-green-300 rounded-full opacity-20 animate-pulse"></div>
        <div className="absolute bottom-20 left-20 w-24 h-24 bg-blue-300 rounded-full opacity-20 animate-bounce delay-1000"></div>
        <div className="absolute bottom-40 right-10 w-12 h-12 bg-pink-300 rounded-full opacity-20 animate-pulse delay-500"></div>
        {/* Floating Educational Icons */}
        <div className="absolute top-20 left-1/4 text-white opacity-10 animate-float">
          <BookOpen size={40} />
        </div>
        <div className="absolute top-40 right-1/4 text-white opacity-10 animate-float delay-1000">
          <Brain size={35} />
        </div>
        <div className="absolute bottom-32 left-1/3 text-white opacity-10 animate-float delay-2000">
          <Lightbulb size={45} />
        </div>
        <div className="absolute top-60 left-1/2 text-white opacity-10 animate-float delay-500">
          <Star size={30} />
        </div>
      </div>
      {/* Main Container */}
      <div className="relative z-10 container mx-auto px-4 py-6 h-screen flex flex-col">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="flex items-center justify-center gap-3 mb-2">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg">
                <BookOpen className="text-white" size={24} />
              </div>
              <Sparkles className="absolute -top-1 -right-1 text-yellow-400" size={16} />
            </div>
            <h1 className="text-4xl font-bold text-white drop-shadow-lg">EduBot AI</h1>
          </div>
          <p className="text-white/90 text-lg font-medium drop-shadow">Your Multimodal Learning Companion</p>
          <div className="flex justify-center gap-4 mt-2 text-white/80 text-sm">
            <span className="flex items-center gap-1">
              <Mic size={14} /> Voice
            </span>
            <span className="flex items-center gap-1">
              <Camera size={14} /> Image
            </span>
            <span className="flex items-center gap-1">
              <FileText size={14} /> Text
            </span>
          </div>
        </div>
        {/* Chat Area */}
        <Card className="flex-1 bg-white/95 backdrop-blur-sm shadow-2xl rounded-3xl p-6 mb-6 overflow-hidden">
          <div className="h-full flex flex-col">
            <div className="flex-1 overflow-y-auto space-y-4 mb-4">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.isBot ? "justify-start" : "justify-end"}`}>
                  <div
                    className={`max-w-[80%] p-4 rounded-2xl shadow-md ${msg.isBot
                      ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white"
                      : "bg-gradient-to-r from-green-400 to-blue-500 text-white"
                      }`}
                  >
                    <div className="flex items-center mb-2">
                      {getInputTypeIcon(msg.inputType)}
                      <p className="text-xs opacity-75">
                        {msg.inputType ? msg.inputType.charAt(0).toUpperCase() + msg.inputType.slice(1) : 'Text'}
                      </p>
                    </div>
                    <p className="text-sm leading-relaxed">{msg.text}</p>
                    <p className="text-xs opacity-75 mt-2">{msg.timestamp.toLocaleTimeString()}</p>

                    {/* Show analysis summary for bot messages */}
                    {msg.isBot && msg.analysisSummary && (
                      <div className="mt-3 pt-3 border-t border-white/20">
                        <p className="text-xs opacity-75 mb-1">Analysis Summary:</p>
                        <div className="text-xs opacity-75 space-y-1">
                          {msg.analysisSummary.input_types && (
                            <p>Input types: {msg.analysisSummary.input_types.join(', ')}</p>
                          )}
                          {msg.analysisSummary.voice_transcription && (
                            <p>Voice: "{msg.analysisSummary.voice_transcription}"</p>
                          )}
                          {msg.analysisSummary.image_objects_detected > 0 && (
                            <p>Objects detected: {msg.analysisSummary.image_objects_detected}</p>
                          )}
                          {msg.analysisSummary.image_subject !== 'none' && (
                            <p>Subject: {msg.analysisSummary.image_subject}</p>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {uploading && (
                <div className="flex justify-end">
                  <div className="max-w-[80%] p-4 rounded-2xl shadow-md bg-gradient-to-r from-green-400 to-blue-500 text-white">
                    <p className="text-sm leading-relaxed">Processing...</p>
                  </div>
                </div>
              )}
              {generating && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] p-4 rounded-2xl shadow-md bg-gradient-to-r from-blue-500 to-purple-600 text-white flex items-center gap-2">
                    <span className="loader" />
                    <span>Generating response...</span>
                  </div>
                </div>
              )}
            </div>
            {/* Input Area */}
            <div className="border-t pt-4">
              <InputArea onSend={handleUnifiedSend} disabled={uploading || generating} />
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
