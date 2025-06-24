import React, { useState, useRef } from 'react';

interface MessageInputProps {
    onSend: (content: string | File, type: 'text' | 'audio' | 'image') => void;
    disabled?: boolean;
}

const MessageInput: React.FC<MessageInputProps> = ({ onSend, disabled = false }) => {
    const [message, setMessage] = useState('');
    const [attachment, setAttachment] = useState<{ file: File; type: 'audio' | 'image' } | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleStartRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            mediaRecorderRef.current.ondataavailable = (event) => {
                audioChunksRef.current.push(event.data);
            };
            mediaRecorderRef.current.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                const audioFile = new File([audioBlob], `recording-${Date.now()}.webm`, { type: 'audio/webm' });
                onSend(audioFile, 'audio');
                audioChunksRef.current = [];
                // Stop all tracks on the stream to turn off the mic indicator
                stream.getTracks().forEach(track => track.stop());
            };
            audioChunksRef.current = [];
            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Error starting recording:", err);
            alert("Could not start recording. Please ensure you have given microphone permissions.");
        }
    };

    const handleStopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const handleMicClick = () => {
        if (isRecording) {
            handleStopRecording();
        } else {
            handleStartRecording();
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        console.log('📝 Form submitted:', { message, attachment });

        if (message.trim() || attachment) {
            if (attachment) {
                console.log('📤 Sending file:', {
                    name: attachment.file.name,
                    type: attachment.type,
                    size: attachment.file.size
                });
                onSend(attachment.file, attachment.type);
            } else {
                console.log('📝 Sending text message:', message);
                onSend(message, 'text');
            }
            setMessage('');
            setAttachment(null);
        } else {
            console.log('⚠️ Form submitted but no content to send');
        }
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const type = file.type.startsWith('audio/') ? 'audio' : 'image';
            console.log('📎 File selected:', {
                name: file.name,
                type: file.type,
                size: file.size,
                detectedType: type
            });
            setAttachment({ file, type });
        } else {
            console.log('⚠️ No file selected');
        }
    };

    const removeAttachment = () => {
        console.log('🗑️ Removing attachment:', attachment?.file.name);
        setAttachment(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <form onSubmit={handleSubmit} className="message-input">
            <div className="input-container">
                <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Type your message..."
                    disabled={disabled}
                />
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    accept="audio/*,image/*"
                    style={{ display: 'none' }}
                />
                <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={disabled || isRecording}
                    className="attach-button"
                >
                    📎
                </button>
                <button
                    type="button"
                    onClick={handleMicClick}
                    disabled={disabled}
                    className={`mic-button ${isRecording ? 'recording' : ''}`}
                >
                    {isRecording ? '■' : '🎤'}
                </button>
                <button
                    type="submit"
                    disabled={disabled || (!message.trim() && !attachment)}
                >
                    Send
                </button>
            </div>
            {attachment && (
                <div className="attachment-preview">
                    <span>{attachment.file.name}</span>
                    <button
                        type="button"
                        onClick={removeAttachment}
                        className="remove-attachment"
                    >
                        ×
                    </button>
                </div>
            )}
        </form>
    );
};

export default MessageInput; 