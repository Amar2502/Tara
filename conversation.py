import os
import pyaudio
from config import NEON_GREEN, RESET_COLOR, RATE, CHUNK
from audio_utils import record_until_silence, transcribe_chunk
from tts import speak_text_streamed

async def handle_conversation():
    """Handle a full conversation after wake word is detected."""
    p = pyaudio.PyAudio()
    stream_in = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    try:
        while True:
            file_path = record_until_silence(p, stream_in, "temp_chunk.wav")
            if not file_path:
                continue

            transcription = transcribe_chunk(file_path)
            os.remove(file_path)

            if not transcription:
                continue

            print(f"\nYou: {NEON_GREEN}{transcription}{RESET_COLOR}")

            trigger_phrases = ["bye", "tata", "ok, bye", "ok, tata", "goodbye", "see you later", "see you", "see you soon"]
            if transcription and any(phrase in transcription.lower() for phrase in trigger_phrases):
                await speak_text_streamed(transcription)
                return

            await speak_text_streamed(transcription)    

    finally:
        stream_in.stop_stream()
        stream_in.close()
        p.terminate()


async def wake_word_detection():
    """Listen continuously for wake word 'Ok Nova' before starting conversation."""
    p = pyaudio.PyAudio()
    stream_in = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n=== NOVA Voice Assistant Started ===\n")
    print("Listening for wake word 'Ok Nova'...")

    try:
        while True:
            file_path = record_until_silence(p, stream_in, "temp_chunk.wav")
            if not file_path:
                continue

            try:
                transcription = transcribe_chunk(file_path)
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

            if transcription:
                print(f"You: {NEON_GREEN}{transcription}{RESET_COLOR}")

            trigger_phrases = ["ok tara", "okay tara", "ok, tara", "okay, tara", "tara."]
            if any(phrase in transcription.lower() for phrase in trigger_phrases):
                await speak_text_streamed(transcription)
                break

    finally:
        stream_in.stop_stream()
        stream_in.close()
        p.terminate()
