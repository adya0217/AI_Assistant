import React, { useRef, useState } from 'react';

interface InputAreaProps {
    onSend: (message: string, image?: File, audio?: File) => void;
    disabled?: boolean;
}

const InputArea: React.FC<InputAreaProps> = ({ onSend, disabled = false }) => {
    const [message, setMessage] = useState('');
    const [image, setImage] = useState<File | null>(null);
    const [audio, setAudio] = useState<File | null>(null);
    const [recording, setRecording] = useState(false);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunks = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);

    const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
    const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if ((message.trim() || image || audio) && !disabled) {
            onSend(message, image || undefined, audio || undefined);
            setMessage('');
            setImage(null);
            setAudio(null);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            if (file.type.startsWith('image/')) {
                setImage(file);
                setAudio(null);
                if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
                setImagePreviewUrl(URL.createObjectURL(file));
                if (audioPreviewUrl) {
                    URL.revokeObjectURL(audioPreviewUrl);
                    setAudioPreviewUrl(null);
                }
            } else if (file.type.startsWith('audio/')) {
                setAudio(file);
                setImage(null);
                if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
                setAudioPreviewUrl(URL.createObjectURL(file));
                if (imagePreviewUrl) {
                    URL.revokeObjectURL(imagePreviewUrl);
                    setImagePreviewUrl(null);
                }
            } else {
                alert('Unsupported file type. Please select an image or audio file.');
            }
        }
    };

    const startRecording = async () => {
        try {
            setImage(null);
            setAudio(null);
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunks.current = [];
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.current.push(e.data);
            };
            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunks.current, { type: 'audio/webm' });
                const audioFile = new File([blob], 'audio.webm', { type: 'audio/webm' });
                setAudio(audioFile);
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop());
                    streamRef.current = null;
                }
            };
            mediaRecorder.start();
            setRecording(true);
        } catch (err) {
            alert('Could not start audio recording.');
            setRecording(false);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
            mediaRecorderRef.current.stop();
        }
        setRecording(false);
    };

    React.useEffect(() => {
        return () => {
            if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
            if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
        };
    }, []);

    return (
        <form onSubmit={handleSubmit} className="input-area" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder={image ? "Add a message about this image..." : "Type your message..."}
                className="message-input"
                disabled={disabled || recording}
                style={{ flex: 1 }}
            />

            <input
                type="file"
                accept="image/*,audio/*"
                style={{ display: 'none' }}
                ref={fileInputRef}
                onChange={handleFileChange}
                disabled={disabled || recording}
            />
            <button
                type="button"
                className="attach-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled || recording}
                title="Attach image or audio"
                style={{ fontSize: 20 }}
            >
                📎
            </button>

            <button
                type="button"
                onClick={recording ? stopRecording : startRecording}
                disabled={disabled || !!image || !!audio}
                style={{ fontSize: 20 }}
                title={recording ? "Stop recording" : "Record audio"}
            >
                {recording ? '⏹️' : '🎤'}
            </button>

            {image && imagePreviewUrl && (
                <div style={{ position: 'relative', display: 'inline-block', marginLeft: 8 }}>
                    <img
                        src={imagePreviewUrl}
                        alt="preview"
                        style={{ width: 120, height: 120, objectFit: 'cover', borderRadius: 8, border: '1px solid #ccc', marginBottom: 4 }}
                    />
                    <button
                        type="button"
                        onClick={() => { setImage(null); if (imagePreviewUrl) { URL.revokeObjectURL(imagePreviewUrl); setImagePreviewUrl(null); } }}
                        style={{ position: 'absolute', top: 2, right: 2, background: '#fff', border: '1px solid #ccc', borderRadius: '50%', width: 24, height: 24, fontSize: 16, cursor: 'pointer' }}
                        title="Remove image"
                    >
                        ×
                    </button>
                    <textarea
                        value={message}
                        onChange={e => setMessage(e.target.value)}
                        placeholder="Add a message about this image..."
                        style={{ width: 120, marginTop: 4, borderRadius: 4, border: '1px solid #ccc', resize: 'none', fontSize: 14 }}
                        rows={2}
                        disabled={disabled}
                    />
                </div>
            )}
            {audio && audioPreviewUrl && (
                <div style={{ display: 'inline-block', marginLeft: 8 }}>
                    <audio controls src={audioPreviewUrl} style={{ height: 40 }} />
                    <button
                        type="button"
                        onClick={() => { setAudio(null); if (audioPreviewUrl) { URL.revokeObjectURL(audioPreviewUrl); setAudioPreviewUrl(null); } }}
                        style={{ marginLeft: 4, fontSize: 14, cursor: 'pointer' }}
                        title="Remove audio"
                    >
                        ×
                    </button>
                </div>
            )}
            <button type="submit" className="send-button" disabled={disabled || recording} style={{ fontWeight: 600 }}>
                {disabled ? 'Sending...' : 'Send'}
            </button>
        </form>
    );
};

export default InputArea; 