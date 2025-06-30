import React, { useState } from "react";
import { BookOpen, Brain, Lightbulb, Star, Sparkles, Mic, Camera, FileText } from "lucide-react";
import InputArea from "./components/InputArea";
import './App.css';
import axios from "axios";
import { Card } from "./components/ui/card";
import EducationalChatbot from "./components/EducationalChatbot";

interface ChatMessage {
  id: number;
  text: string;
  isBot: boolean;
  timestamp: Date;
  inputType?: string;
  analysisSummary?: any;
}

function App() {
  return <EducationalChatbot />;
}

export default App;
