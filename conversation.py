import os
import pyaudio
from config import NEON_GREEN, RESET_COLOR, RATE, CHUNK
from audio_utils import record_until_silence, transcribe_chunk
from tts import speak_text_streamed
# from embeddings import embed_data
import asyncio

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
                # await asyncio.to_thread(embed_data)
                return

            await speak_text_streamed(transcription)    

    finally:
        stream_in.stop_stream()
        stream_in.close()
        p.terminate()