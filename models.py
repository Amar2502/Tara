from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama

print("Loading models...")
llm_model = ChatOllama(model="Tara")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
