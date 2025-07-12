"use client"

import React, { useState, useRef } from "react"
import { Send, Mic, BookOpen, Brain, Lightbulb, Star, Sparkles, Image as ImageIcon, X, Edit3, Play, Pause } from "lucide-react"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Card } from "./ui/card"
import { sendMessage, uploadFile } from "../services/api"

// Message type supports text, image, and audio
interface Message {
    id: number;
    text?: string;
    image?: string; // data URL
    audio?: string; // data URL
    isBot: boolean;
    timestamp: Date;
}

// Preview state for pending uploads
interface PreviewState {
    type: 'image' | 'audio';
    data: string;
    text?: string; // For image captions
}

export default function EducationalChatbot() {
    const [message, setMessage] = useState("")
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 1,
            text: "Hello! I'm your AI learning companion. What would you like to explore today?",
            isBot: true,
            timestamp: new Date(),
        },
    ])
    const [isRecording, setIsRecording] = useState(false)
    const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
    const [audioChunks, setAudioChunks] = useState<Blob[]>([])
    const fileInputRef = useRef<HTMLInputElement>(null)
    const streamRef = useRef<MediaStream | null>(null)
    const [preview, setPreview] = useState<PreviewState | null>(null)
    const [botLoading, setBotLoading] = useState(false)
    const audioRef = useRef<HTMLAudioElement>(null)

    const handleSend = async () => {
        if (botLoading || preview) return;
        if (message.trim()) {
            const newMessage: Message = {
                id: messages.length + 1,
                text: message,
                isBot: false,
                timestamp: new Date(),
            }
            setMessages([...messages, newMessage])
            setMessage("")
            await handleBotResponse({ text: message })
        }
    }

    // Helper to convert dataURL to File
    function dataURLtoFile(dataurl: string, filename: string) {
        const arr = dataurl.split(",")
        const match = arr[0].match(/:(.*?);/)
        const mime = match ? match[1] : ''
        const bstr = atob(arr[1])
        let n = bstr.length
        const u8arr = new Uint8Array(n)
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n)
        }
        return new File([u8arr], filename, { type: mime })
    }

    // Handles bot response for text, image, or audio
    const handleBotResponse = async (userMsg: Partial<Message>) => {
        setBotLoading(true)
        try {
            let botText = ""
            if (userMsg.text) {
                const res = await sendMessage(userMsg.text)
                botText = res.message
            } else if (userMsg.image) {
                // Convert dataURL to File
                const file = dataURLtoFile(userMsg.image, "image.png")
                const res = await uploadFile(file, "image")
                botText = res.message
            } else if (userMsg.audio) {
                const file = dataURLtoFile(userMsg.audio, "audio.webm")
                const res = await uploadFile(file, "audio")
                botText = res.message
            }
            const botResponse: Message = {
                id: messages.length + 2,
                text: botText,
                isBot: true,
                timestamp: new Date(),
            }
            setMessages((prev) => [...prev, botResponse])
        } catch (err) {
            const botResponse: Message = {
                id: messages.length + 2,
                text: "Sorry, I couldn't process your request. Please try again.",
                isBot: true,
                timestamp: new Date(),
            }
            setMessages((prev) => [...prev, botResponse])
        } finally {
            setBotLoading(false)
        }
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === "Enter") {
            handleSend()
        }
    }

    const handleFileUpload = () => {
        if (botLoading || preview) return;
        fileInputRef.current?.click()
        console.log('[UI] Image upload button clicked')
    }

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return
        const reader = new FileReader()
        reader.onload = () => {
            setPreview({
                type: 'image',
                data: reader.result as string,
                text: '' // Initialize empty text for image caption
            })
        }
        reader.readAsDataURL(file)
        e.target.value = "" // reset input
    }

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            streamRef.current = stream

            // Try different audio formats for better compatibility
            let mimeType = 'audio/webm;codecs=opus'
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/webm'
            }
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/mp4'
            }
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/wav'
            }
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                // Fallback to default format
                mimeType = ''
            }

            const recorder = new window.MediaRecorder(stream, {
                mimeType: mimeType
            })
            setMediaRecorder(recorder)
            setAudioChunks([])
            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) setAudioChunks((prev) => [...prev, e.data])
            }
            recorder.onstop = () => {
                const blob = new Blob(audioChunks, { type: mimeType })
                if (blob.size === 0) {
                    alert('Recording failed: no audio data captured. Please try again.');
                    return;
                }
                const reader = new FileReader()
                reader.onload = () => {
                    setPreview({
                        type: 'audio',
                        data: reader.result as string
                    })
                }
                reader.readAsDataURL(blob)
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop())
                    streamRef.current = null
                }
            }
            recorder.start()
            setIsRecording(true)
            console.log('[UI] Audio recording started with format:', mimeType)
        } catch (err) {
            alert('Could not start audio recording. Please check your microphone permissions.')
            setIsRecording(false)
            console.error('[UI] Audio recording error', err)
        }
    }

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop()
            console.log('[UI] Audio recording stopped')
        }
        setIsRecording(false)
    }

    const handleMicClick = () => {
        if (botLoading || preview) return;
        if (isRecording) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    const handleSendPreview = async () => {
        if (!preview) return;

        // Convert blob URL to data URL for sending
        let dataToSend = preview.data;
        if (preview.type === 'audio' && preview.data.startsWith('blob:')) {
            try {
                const response = await fetch(preview.data);
                const blob = await response.blob();
                const reader = new FileReader();
                dataToSend = await new Promise((resolve) => {
                    reader.onload = () => resolve(reader.result as string);
                    reader.readAsDataURL(blob);
                });
            } catch (error) {
                console.error('Error converting blob URL to data URL:', error);
            }
        }

        const newMessage: Message = {
            id: messages.length + 1,
            [preview.type]: dataToSend,
            text: preview.text, // Include caption text for images
            isBot: false,
            timestamp: new Date(),
        }
        setMessages([...messages, newMessage])
        setPreview(null)
        await handleBotResponse({ [preview.type]: dataToSend })
    }

    const handleCancelPreview = () => {
        setPreview(null)
        if (audioRef.current) {
            audioRef.current.pause()
            audioRef.current.currentTime = 0
        }
    }





    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-800 relative overflow-hidden">
            {/* Animated Background Elements */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute top-10 left-10 w-20 h-20 bg-cyan-400 rounded-full opacity-15 animate-bounce blur-sm"></div>
                <div className="absolute top-32 right-20 w-16 h-16 bg-emerald-400 rounded-full opacity-15 animate-pulse blur-sm"></div>
                <div className="absolute bottom-20 left-20 w-24 h-24 bg-violet-400 rounded-full opacity-15 animate-bounce delay-1000 blur-sm"></div>
                <div className="absolute bottom-40 right-10 w-12 h-12 bg-teal-400 rounded-full opacity-15 animate-pulse delay-500 blur-sm"></div>
                <div className="absolute top-1/2 left-1/2 w-32 h-32 bg-indigo-500 rounded-full opacity-10 animate-pulse delay-700 blur-lg"></div>

                {/* Floating Educational Icons */}
                <div className="absolute top-20 left-1/4 text-cyan-300 opacity-20 animate-float">
                    <BookOpen size={40} />
                </div>
                <div className="absolute top-40 right-1/4 text-emerald-300 opacity-20 animate-float delay-1000">
                    <Brain size={35} />
                </div>
                <div className="absolute bottom-32 left-1/3 text-violet-300 opacity-20 animate-float delay-2000">
                    <Lightbulb size={45} />
                </div>
                <div className="absolute top-60 left-1/2 text-teal-300 opacity-20 animate-float delay-500">
                    <Star size={30} />
                </div>
            </div>

            {/* Main Container */}
            <div className="relative z-10 flex justify-center items-center min-h-screen px-4 py-6">
                <div className="w-full max-w-4xl h-[95vh] flex flex-col">
                    {/* Header */}
                    <div className="text-center mb-6">
                        <div className="flex items-center justify-center gap-3 mb-2">
                            <div className="relative">
                                <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 via-violet-500 to-emerald-500 rounded-full flex items-center justify-center shadow-lg shadow-cyan-500/25">
                                    <BookOpen className="text-white" size={24} />
                                </div>
                                <Sparkles className="absolute -top-1 -right-1 text-emerald-400" size={16} />
                            </div>
                            <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-violet-400 to-emerald-400 bg-clip-text text-transparent drop-shadow-lg">
                                EduBot AI
                            </h1>
                        </div>
                        <p className="text-slate-300 text-lg font-medium drop-shadow">Your Intelligent Learning Companion</p>
                    </div>

                    {/* Chat Area */}
                    <Card className="flex-1 bg-gradient-to-br from-slate-900/95 via-gray-900/95 to-slate-800/95 backdrop-blur-xl shadow-2xl rounded-3xl p-6 mb-6 overflow-hidden border border-gradient-to-r from-cyan-500/20 via-violet-500/20 to-emerald-500/20 overflow-x-hidden">
                        <div className="h-full flex flex-col">
                            <div className="flex-1 overflow-y-auto space-y-4 mb-4 overflow-x-hidden">
                                {messages.map((msg) => (
                                    <div key={msg.id} className={`flex ${msg.isBot ? "justify-start" : "justify-end"}`}>
                                        <div
                                            className={`break-words whitespace-pre-line w-fit max-w-2xl p-4 rounded-2xl shadow-lg ${msg.isBot
                                                ? "bg-gradient-to-r from-violet-600/90 to-indigo-600/90 text-white border border-violet-400/30 shadow-violet-500/20"
                                                : "bg-gradient-to-r from-emerald-600/90 to-teal-600/90 text-white border border-emerald-400/30 shadow-emerald-500/20"
                                                }`}
                                        >
                                            {/* Show image if present */}
                                            {msg.image && (
                                                <img src={msg.image} alt="uploaded" className="mb-2 max-w-xs max-h-60 rounded-lg border border-slate-700" />
                                            )}
                                            {/* Show audio if present */}
                                            {msg.audio && (
                                                <audio controls className="mb-2 w-full">
                                                    <source src={msg.audio} />
                                                    Your browser does not support the audio element.
                                                </audio>
                                            )}
                                            {/* Show text if present */}
                                            {msg.text && (
                                                <p className="text-sm leading-relaxed break-words whitespace-pre-line">{msg.text}</p>
                                            )}
                                            <p className="text-xs opacity-75 mt-2">{msg.timestamp.toLocaleTimeString()}</p>
                                        </div>
                                    </div>
                                ))}

                                {/* Preview Area */}
                                {preview && (
                                    <div className="flex justify-end">
                                        <div className="break-words whitespace-pre-line w-fit max-w-2xl p-4 rounded-2xl shadow-lg bg-gradient-to-r from-emerald-600/90 to-teal-600/90 text-white border border-emerald-400/30 shadow-emerald-500/20 relative">
                                            {/* Preview Header */}
                                            <div className="flex items-center justify-between mb-3 pb-2 border-b border-emerald-400/30">
                                                <span className="text-sm font-medium text-emerald-200">
                                                    {preview.type === 'image' ? '📷 Image Preview' : '🎤 Audio Preview'}
                                                </span>
                                                <Button
                                                    onClick={handleCancelPreview}
                                                    className="p-1 h-6 w-6 rounded-full bg-red-500/20 hover:bg-red-500/30 text-red-300 hover:text-red-200"
                                                    size="sm"
                                                >
                                                    <X size={12} />
                                                </Button>
                                            </div>

                                            {/* Preview Content */}
                                            {preview.type === 'image' && (
                                                <div className="space-y-3">
                                                    <img
                                                        src={preview.data}
                                                        alt="preview"
                                                        className="max-w-xs max-h-60 rounded-lg border border-slate-700"
                                                    />
                                                    <div className="space-y-2">
                                                        <div className="flex items-center gap-2">
                                                            <Edit3 size={14} className="text-emerald-300" />
                                                            <span className="text-xs text-emerald-200">Add caption (optional):</span>
                                                        </div>
                                                        <Input
                                                            value={preview.text || ''}
                                                            onChange={(e) => setPreview({ ...preview, text: e.target.value })}
                                                            placeholder="Describe what you see in this image..."
                                                            className="text-xs bg-slate-800/60 border-emerald-400/30 text-white placeholder:text-slate-400"
                                                        />
                                                    </div>
                                                </div>
                                            )}

                                            {preview.type === 'audio' && (
                                                <div className="space-y-3">
                                                    <div className="text-sm text-emerald-200 mb-2">Audio Recording Preview</div>
                                                    <audio
                                                        ref={audioRef}
                                                        controls
                                                        className="w-full"
                                                        onLoadedMetadata={() => console.log('Audio metadata loaded')}
                                                        onCanPlay={() => console.log('Audio can play')}
                                                        onError={(e) => console.error('Audio error:', e)}
                                                    >
                                                        <source src={preview.data} type="audio/webm" />
                                                        <source src={preview.data} type="audio/webm;codecs=opus" />
                                                        Your browser does not support the audio element.
                                                    </audio>
                                                </div>
                                            )}

                                            {/* Preview Actions */}
                                            <div className="flex gap-2 mt-3 pt-3 border-t border-emerald-400/30">
                                                <Button
                                                    onClick={handleCancelPreview}
                                                    className="flex-1 bg-slate-700/60 hover:bg-slate-600/60 text-slate-300 hover:text-slate-200"
                                                    size="sm"
                                                >
                                                    Cancel
                                                </Button>
                                                <Button
                                                    onClick={handleSendPreview}
                                                    className="flex-1 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white"
                                                    size="sm"
                                                    disabled={botLoading}
                                                >
                                                    <Send size={14} className="mr-1" />
                                                    Send
                                                </Button>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Bot loading bubble */}
                                {botLoading && (
                                    <div className="flex justify-start">
                                        <div className="break-words whitespace-pre-line w-fit max-w-2xl p-4 rounded-2xl shadow-lg bg-gradient-to-r from-violet-600/90 to-indigo-600/90 text-white border border-violet-400/30 shadow-violet-500/20 opacity-80 relative animate-pulse">
                                            <span className="text-xs text-slate-200">Thinking...</span>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Input Area */}
                            <div className="border-t border-gradient-to-r from-cyan-500/20 via-violet-500/20 to-emerald-500/20 pt-4">
                                <div className="flex items-center gap-3">
                                    <div className="flex-1 relative">
                                        <Input
                                            value={message}
                                            onChange={(e) => setMessage(e.target.value)}
                                            onKeyPress={handleKeyPress}
                                            placeholder="Ask me anything about your studies..."
                                            className="pr-12 py-3 text-base rounded-full border-2 border-slate-600/50 focus:border-cyan-400/60 bg-slate-800/80 text-white placeholder:text-slate-400 shadow-inner w-full"
                                            disabled={botLoading || !!preview}
                                        />
                                    </div>

                                    {/* Action Buttons */}
                                    <div className="flex gap-2">
                                        <Button
                                            onClick={handleFileUpload}
                                            className="rounded-full bg-slate-700/80 border border-violet-500/30 hover:bg-violet-600/20 hover:border-violet-400 text-violet-300 hover:text-violet-200 shadow-lg hover:shadow-violet-500/25 p-2"
                                            type="button"
                                            variant="outline"
                                            size="icon"
                                            title="Upload Image"
                                            disabled={botLoading || !!preview}
                                        >
                                            <ImageIcon size={18} />
                                        </Button>
                                        <Button
                                            onClick={handleMicClick}
                                            className={`rounded-full shadow-lg ${isRecording
                                                ? "bg-red-600 text-white hover:bg-red-700 border-red-500 shadow-red-500/25 animate-pulse"
                                                : "bg-slate-700/80 border-emerald-500/30 hover:bg-emerald-600/20 text-emerald-300 hover:text-emerald-200 hover:border-emerald-400 hover:shadow-emerald-500/25"
                                                } p-2 border`}
                                            type="button"
                                            variant="outline"
                                            size="icon"
                                            title={isRecording ? "Stop Recording" : "Record Audio"}
                                            disabled={botLoading || !!preview}
                                        >
                                            <Mic size={18} />
                                        </Button>
                                        <Button
                                            onClick={handleSend}
                                            className="rounded-full bg-gradient-to-r from-cyan-600 via-violet-600 to-emerald-600 hover:from-cyan-700 hover:via-violet-700 hover:to-emerald-700 text-white shadow-lg px-6 border border-cyan-400/30 shadow-cyan-500/25 p-2"
                                            type="button"
                                            disabled={botLoading || !!preview}
                                        >
                                            <Send size={18} />
                                        </Button>
                                    </div>
                                </div>

                                {/* Quick Action Buttons */}
                                <div className="flex flex-wrap gap-2 mt-4">
                                    <Button
                                        className="rounded-full bg-slate-700/60 border border-cyan-500/30 hover:bg-cyan-600/20 text-cyan-300 hover:text-cyan-200 hover:border-cyan-400 shadow-sm hover:shadow-cyan-500/25 px-3 py-1 text-sm"
                                        onClick={() => { if (!(botLoading || !!preview)) setMessage("Explain this concept to me") }}
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        disabled={botLoading || !!preview}
                                    >
                                        <Lightbulb size={14} className="mr-1 inline" />
                                        Explain Concept
                                    </Button>
                                    <Button
                                        className="rounded-full bg-slate-700/60 border border-violet-500/30 hover:bg-violet-600/20 text-violet-300 hover:text-violet-200 hover:border-violet-400 shadow-sm hover:shadow-violet-500/25 px-3 py-1 text-sm"
                                        onClick={() => { if (!(botLoading || !!preview)) setMessage("Give me practice questions") }}
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        disabled={botLoading || !!preview}
                                    >
                                        <Brain size={14} className="mr-1 inline" />
                                        Practice Quiz
                                    </Button>
                                    <Button
                                        className="rounded-full bg-slate-700/60 border border-emerald-500/30 hover:bg-emerald-600/20 text-emerald-300 hover:text-emerald-200 hover:border-emerald-400 shadow-sm hover:shadow-emerald-500/25 px-3 py-1 text-sm"
                                        onClick={() => { if (!(botLoading || !!preview)) setMessage("Create a study plan") }}
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        disabled={botLoading || !!preview}
                                    >
                                        <BookOpen size={14} className="mr-1 inline" />
                                        Study Plan
                                    </Button>
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>
            </div>

            {/* Hidden file inputs */}
            <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleFileChange}
            />
        </div>
    )
} 