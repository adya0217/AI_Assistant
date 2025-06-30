import whisper
import numpy as np
import pyaudio
import os
from app.config.openvino_config import OpenVINOConfig
import time
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from app.config.system_config import SystemConfig as config


model = None
processor = None
model_pt = None 


def _load_pytorch_whisper():
    """Load PyTorch Whisper model (fallback)"""
    try:
        print("Loading Whisper from OpenAI library...")
        model = whisper.load_model("base")
        return model, None
    except Exception as e:
        print(f"Error loading whisper library model: {e}")
        try:
            print("Loading Whisper from Hugging Face...")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            return model, processor
        except Exception as e2:
            print(f"Error loading Hugging Face Whisper: {e2}")
            return None, None

def load_whisper_model():
    """Load Whisper model with OpenVINO optimization if available"""
    global model, processor, model_pt
    try:
        
        print("ℹ️ Loading PyTorch Whisper model for fallback...")
        model_pt, _ = _load_pytorch_whisper()
        if model_pt is None:
            print("⚠️  Warning: PyTorch fallback model could not be loaded.")

        if OpenVINOConfig.should_use_openvino():
            model_id = OpenVINOConfig.WHISPER_MODEL_NAME
            openvino_cache_path = OpenVINOConfig.get_model_cache_path("whisper-base")
            if not os.path.exists(openvino_cache_path):
                print(f"Exporting Whisper to OpenVINO format at {openvino_cache_path}...")
                ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
                ov_model.save_pretrained(openvino_cache_path)
            model = OVModelForSpeechSeq2Seq.from_pretrained(openvino_cache_path)
            processor = WhisperProcessor.from_pretrained(model_id)
            # Compile the model for best performance
            model.to("AUTO")
            model.compile()
            print("✅ OpenVINO Whisper model loaded and compiled successfully")
        else:
            print("ℹ️ OpenVINO disabled, using PyTorch Whisper model as primary.")
            
            model = None
            if model_pt and processor is None: 
                processor = WhisperProcessor.from_pretrained(OpenVINOConfig.WHISPER_MODEL_NAME)

    except Exception as e:
       
        if model_pt is None:
            model_pt, processor = _load_pytorch_whisper()
        model = None
    return model, processor


model, processor = load_whisper_model()


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
        if model is not None:
            print("\U0001F501 Using OpenVINO Whisper for transcription...")
            return _transcribe_with_openvino(audio_path)
        else:
            print("\U0001F501 Using PyTorch Whisper for transcription...")
            return model_pt.transcribe(audio_path)["text"]
    elif isinstance(audio_path, np.ndarray):
        if model is not None:
            print("\U0001F501 Using OpenVINO Whisper for array transcription...")
            return _transcribe_array_with_openvino(audio_path)
        else:
            print("\U0001F501 Using PyTorch Whisper for array transcription...")
            return model_pt.transcribe(audio_path)["text"]
    else:
        raise ValueError(f"Invalid audio input: {type(audio_path)}")

def _transcribe_with_openvino(audio_path):
    """Transcribe audio using OpenVINO optimized Whisper"""
    try:
        import librosa
        print(f"\U0001F4C1 Loading audio from: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)
        print("\U0001F501 Processing audio...")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        # Set forced_decoder_ids for language if needed
        forced_decoder_ids = getattr(model.generation_config, 'forced_decoder_ids', None)
        if forced_decoder_ids is None:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            model.generation_config.forced_decoder_ids = forced_decoder_ids
        print("\U0001F3A4 Generating transcription...")
        start_time = time.time()
        generated_ids = model.generate(
            inputs["input_features"],
            max_length=config.MAX_RESPONSE_TOKENS,
            num_beams=5
        )
        inference_time = time.time() - start_time
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"\u2705 Transcription completed in {inference_time:.3f}s")
        return transcription.strip()
    except Exception as e:
        print(f"\u274C Error in OpenVINO transcription: {e}")
        print("\U0001F501 Falling back to PyTorch model...")
        if model_pt is None:
            err_msg = "OpenVINO transcription failed, and the PyTorch fallback model is not available."
            print(f"\u274C [FATAL] {err_msg}")
            raise RuntimeError(err_msg) from e
        return model_pt.transcribe(audio_path)["text"]

def _transcribe_array_with_openvino(audio_array):
    """Transcribe audio array using OpenVINO optimized Whisper"""
    try:
        audio = audio_array.astype(np.float32) / 32768.0
        print("\U0001F501 Processing audio array...")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        forced_decoder_ids = getattr(model.generation_config, 'forced_decoder_ids', None)
        if forced_decoder_ids is None:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            model.generation_config.forced_decoder_ids = forced_decoder_ids
        print("\U0001F3A4 Generating transcription...")
        start_time = time.time()
        generated_ids = model.generate(
            inputs["input_features"],
            max_length=config.MAX_RESPONSE_TOKENS,
            num_beams=5
        )
        inference_time = time.time() - start_time
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"\u2705 Transcription completed in {inference_time:.3f}s")
        return transcription.strip()
    except Exception as e:
        print(f"\u274C Error in OpenVINO array transcription: {e}")
        print("\U0001F501 Falling back to PyTorch model...")
        if model_pt is None:
            err_msg = "OpenVINO array transcription failed, and the PyTorch fallback model is not available."
            print(f"\u274C [FATAL] {err_msg}")
            raise RuntimeError(err_msg) from e
        return model_pt.transcribe(audio_array)["text"]
