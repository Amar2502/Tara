# background_listener.py
import time
import wave
import pyaudio
import os
from audio_utils import audio_energy, transcribe_chunk
from config import CHUNK, RATE, SILENCE_THRESHOLD, SILENCE_LIMIT


def record_stop_phrases(stop_event, stream):
    """
    Continuously monitor microphone input *only* while TTS is playing.
    Stop listening once playback ends or a stop phrase is detected.
    """
    p = pyaudio.PyAudio()
    mic_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    try:
        while not stop_event.is_set():
            # Wait until TTS starts speaking
            if not stream.is_playing():
                time.sleep(0.05)
                continue

            frames = []
            silence_counter = 0
            recording = False

            while stream.is_playing() and not stop_event.is_set():
                data = mic_stream.read(CHUNK, exception_on_overflow=False)
                energy = audio_energy(data)

                if energy > SILENCE_THRESHOLD:
                    frames.append(data)
                    recording = True
                    silence_counter = 0
                elif recording:
                    silence_counter += 1
                    if silence_counter > SILENCE_LIMIT:
                        break

            # Process if voice captured
            if frames:
                filename = f"temp_{int(time.time() * 1000)}.wav"
                wf = wave.open(filename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                transcription = transcribe_chunk(filename)
                os.remove(filename)

                print("Stop phrase detected:", transcription)

                trigger_phrases = ["ok nova", "stop", "ok, stop", "stop nova", "quit", "break"]
                if transcription and any(phrase in transcription.lower() for phrase in trigger_phrases):
                    stop_event.set()
                    stream.stop()
                    break

            if not stream.is_playing():
                break

    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        p.terminate()


def stop_tts(stream, stop_event):
    """Stop TTS streaming and background listener."""
    stream.stop()
    stop_event.set()
