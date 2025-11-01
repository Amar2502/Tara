import numpy as np
import wave
from config import SILENCE_THRESHOLD, CHUNK, RATE
from models import whisper_model
import pyaudio

def audio_energy(data):
    """Compute normalized energy of audio chunk."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(samples**2)) / 32768.0

def record_until_silence(p, stream, file_path, silence_limit=20):
    """Record audio until silence is detected for a number of chunks."""
    frames, silence_counter, recording = [], 0, False
    print("🎙️ Listening...")

    while True:
        data = stream.read(CHUNK)
        energy = audio_energy(data)

        if energy > SILENCE_THRESHOLD:
            frames.append(data)
            recording, silence_counter = True, 0
        elif recording:
            silence_counter += 1
            if silence_counter > silence_limit:
                break

    if not frames:
        return None

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return file_path

def transcribe_chunk(file_path):
    """Convert audio file to text using Whisper model."""
    segments, _ = whisper_model.transcribe(
        file_path,
        beam_size=5,
        language="en",
        initial_prompt="This is an English conversation with an AI assistant called Tara."
    )
    return " ".join([segment.text for segment in segments]).strip()
