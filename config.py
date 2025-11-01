# === COLORS ===
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"
NEON_BLUE = "\033[94m"

# === AUDIO SETTINGS ===
SILENCE_THRESHOLD = 0.007
CHUNK = 1024
RATE = 16000
SILENCE_LIMIT = 10

# === TTS VOICE SETTINGS ===
class TTSArgs:
    voice = "en-US-AvaNeural"
    pitch = 0
    rate = 0
    volume = 0