import whisper
import numpy as np
import pyaudio
import os

model = whisper.load_model("base")

def record_audio(duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=16000, input=True, frames_per_buffer=1024)
    frames = [stream.read(1024) for _ in range(0, int(16000 / 1024 * duration))]
    stream.stop_stream(); stream.close(); p.terminate()
    audio_data = b''.join(frames)
    return np.frombuffer(audio_data, dtype=np.int16)

def transcribe_audio(audio_path):
    if isinstance(audio_path, str) and os.path.exists(audio_path):
        return model.transcribe(audio_path)["text"]
    elif isinstance(audio_path, np.ndarray):
        return model.transcribe(audio_path)["text"]
    else:
        raise ValueError(f"Invalid audio input: {type(audio_path)}")
