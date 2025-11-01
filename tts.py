import time
import threading
from RealtimeTTS import TextToAudioStream, EdgeEngine
from config import TTSArgs, NEON_BLUE, RESET_COLOR
from models import llm_model
from background_listener import record_stop_phrases, stop_tts

# === Initialize TTS Engine ===
args = TTSArgs()
engine = EdgeEngine(rate=args.rate, pitch=args.pitch, volume=args.volume)
stream = TextToAudioStream(engine)
engine.set_voice(args.voice)

# Shared stop event
stop_event = threading.Event()


async def speak_text_streamed(prompt: str):
    """
    Stream AI responses to TTS while continuously listening
    for stop phrases in the background.
    """
    stop_event.clear()
    print(f"Nova: {NEON_BLUE}{RESET_COLOR}", end="", flush=True)

    # Background listener
    listener_thread = threading.Thread(target=record_stop_phrases, args=(stop_event, stream))
    listener_thread.start()

    def ai_response_generator():

        full_response = ""
        for chunk in llm_model.stream(prompt):
            if stop_event.is_set():
                break
            if chunk.content:
                yield chunk.content
                full_response += chunk.content
                print(f"{NEON_BLUE}{chunk.content}{RESET_COLOR}", end="", flush=True)
        print("\n")

    stream.feed(ai_response_generator()).play_async(log_synthesized_text=False)

    # Wait until finished or stopped
    while stream.is_playing():
        time.sleep(0.02)

    stop_event.set()
    listener_thread.join()
