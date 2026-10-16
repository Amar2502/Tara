"""Unified Voice Engine providing ultra-fast local STT (Faster-Whisper) and native GPU streaming TTS (Qwen3-TTS via RealtimeTTS)."""

import io
import logging
import queue
import re
import threading
import time
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

# Silence noisy third-party loggers
logging.getLogger("RealtimeTTS").setLevel(logging.ERROR)
logging.getLogger("RealtimeTTS.stream_player").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_REF_AUDIO = ROOT_DIR / "reference.wav"
DEFAULT_REF_TEXT = (
    "When was Elizabeth Olsen born? February 16, 1989. When did Elizabeth Olsen start acting? "
    "That's a bigger question because I think my first professional job was when I was really young, "
    "but I didn't really start acting then. I don't think I would say my sophomore year of college "
    "when I started understudying."
)

_QWEN_LIB = None

try:
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
    from RealtimeTTS import QwenEngine, QwenVoice, TextToAudioStream

    VOICE_DEPS_INSTALLED = True
    VOICE_IMPORT_ERROR = ""

    # Monkey-patch RealtimeTTS StreamPlayer and AudioStream to prevent
    # concurrent PortAudio Pa_CloseStream() double-free heap corruption (Windows error 0xc0000374)
    try:
        from RealtimeTTS.stream_player import StreamPlayer, AudioStream

        _audio_close_lock = threading.RLock()
        _orig_close_stream = AudioStream.close_stream

        def _safe_close_stream(self):
            with _audio_close_lock:
                s = getattr(self, "stream", None)
                if s is not None:
                    self.stream = None
                    try:
                        if s.is_active():
                            s.stop_stream()
                    except Exception:
                        pass
                    try:
                        s.close()
                    except Exception:
                        pass
                elif getattr(self, "mpv_process", None) is not None:
                    proc = self.mpv_process
                    self.mpv_process = None
                    try:
                        if proc.stdin:
                            proc.stdin.close()
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
                    try:
                        proc.terminate()
                    except Exception:
                        pass

        AudioStream.close_stream = _safe_close_stream

        _player_stop_lock = threading.RLock()
        _orig_player_stop = StreamPlayer.stop

        def _safe_player_stop(self, immediate: bool = False):
            with _player_stop_lock:
                try:
                    return _orig_player_stop(self, immediate=immediate)
                except Exception:
                    pass

        StreamPlayer.stop = _safe_player_stop
    except Exception:
        pass

    # Silence native qwentts.cpp verbose C++ logs ([Pipeline], [Perf], [Sample], etc.)
    try:
        from qwentts_cpp import QwenLibrary
        _QWEN_LIB = QwenLibrary()
        _QWEN_LIB.set_log_callback(lambda level, msg: None)
    except Exception:
        pass

except ImportError as err:
    VOICE_DEPS_INSTALLED = False
    VOICE_IMPORT_ERROR = str(err)


def is_voice_available() -> Tuple[bool, str]:
    """Check if all required voice dependencies are installed and available."""
    if not VOICE_DEPS_INSTALLED:
        return (
            False,
            f"Voice dependencies are missing ({VOICE_IMPORT_ERROR}). "
            "Please install: pip install --only-binary=realtimetts-qwen-native \"realtimetts[qwen]\" faster-whisper sounddevice soundfile numpy",
        )
    return True, "Voice dependencies are available."


class QwenTTS:
    """Ultra-fast, native GPU Text-to-Speech using Qwen3-TTS via RealtimeTTS."""

    def __init__(
        self,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: Optional[str] = None,
        voice_name: str = "narrator",
        language: str = "english",
    ):
        self.ref_audio = Path(ref_audio) if ref_audio else DEFAULT_REF_AUDIO
        self.ref_text = ref_text if ref_text else DEFAULT_REF_TEXT
        self.voice_name = voice_name
        self.language = language

        self.engine: Optional[QwenEngine] = None
        self.voice: Optional[QwenVoice] = None
        self.stream: Optional[TextToAudioStream] = None
        self._init_lock = threading.Lock()
        self._is_active: bool = False
        self._last_feed_time: float = 0.0

    def _ensure_initialized(self) -> None:
        """Lazy-load the native Qwen model and stream handler on first call."""
        if self.engine is None:
            with self._init_lock:
                if self.engine is None:
                    if self.ref_audio.exists():
                        self.voice = QwenVoice(
                            name=self.voice_name,
                            ref_audio=str(self.ref_audio),
                            ref_text=self.ref_text,
                            language=self.language,
                        )
                        self.engine = QwenEngine(voice=self.voice)
                    else:
                        # Fallback to default voice if reference audio is missing
                        self.engine = QwenEngine()

                    self.stream = TextToAudioStream(self.engine)

    def speak_sentence(self, sentence: str) -> None:
        """Feed text and play asynchronously with native streaming."""
        clean_text = sentence.strip()
        # Skip markdown symbols, emojis, URLs, and code blocks from TTS
        clean_text = re.sub(r"```.*?```", "", clean_text, flags=re.DOTALL)
        clean_text = re.sub(r"[*_~`#>\[\]\(\)]", "", clean_text)
        clean_text = re.sub(r"https?://\S+", "", clean_text)
        clean_text = " ".join(clean_text.split())

        if not clean_text or len(clean_text) < 2:
            return

        self._ensure_initialized()
        with self._init_lock:
            if self.stream:
                self._is_active = True
                self._last_feed_time = time.time()
                self.stream.feed(clean_text + " ")
                if not self.stream.is_playing():
                    self.stream.play_async()

    def is_speaking(self) -> bool:
        """Check if audio is currently being synthesized or played."""
        if not self.stream:
            return False
        if self.stream.is_playing():
            return True
        # Grace window right after feed before playback thread marks stream_running
        if self._is_active and (time.time() - self._last_feed_time < 1.2):
            return True
        self._is_active = False
        return False

    def wait_until_done(self, timeout: float = 60.0) -> None:
        """Block until the current audio playback has finished."""
        start_time = time.time()
        while self.is_speaking():
            if time.time() - start_time > timeout:
                break
            time.sleep(0.08)
        # Brief pause to allow room acoustic echo to settle
        time.sleep(0.15)

    def stop(self) -> None:
        """Immediately stop currently playing audio (barge-in) and clean pipeline state."""
        self._is_active = False
        with self._init_lock:
            if self.stream:
                try:
                    if self.stream.is_playing():
                        self.stream.stop()
                except Exception:
                    pass
                try:
                    if self.engine:
                        self.engine.stop_synthesis_event.clear()
                    self.stream = TextToAudioStream(self.engine)
                except Exception:
                    pass

    def shutdown(self) -> None:
        """Shut down the native Qwen engine and release GPU memory."""
        self._is_active = False
        if self.stream:
            try:
                self.stream.stop()
            except Exception:
                pass
        if self.engine:
            try:
                self.engine.shutdown()
            except Exception:
                pass
            self.engine = None


class FasterWhisperSTT:
    """Local, low-latency Speech-to-Text using faster-whisper and energy VAD."""

    def __init__(self, model_size: str = "distil-whisper/distil-small.en"):
        self.model_size = model_size
        self.sample_rate = 16000
        self.model: Optional[WhisperModel] = None
        self._init_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Lazy-load the Whisper model on first transcription."""
        if self.model is None:
            with self._init_lock:
                if self.model is None:
                    try:
                        self.model = WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=4,
                        )
                    except Exception:
                        self.model = WhisperModel(
                            "base.en",
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=4,
                        )

    def transcribe_audio_data(self, audio_data: np.ndarray) -> str:
        """Transcribe a 1D float32 numpy audio array in-memory with Silero VAD."""
        self._ensure_initialized()
        if audio_data is None or len(audio_data) < int(self.sample_rate * 0.25):
            return ""

        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_io.seek(0)

        try:
            segments, _ = self.model.transcribe(
                wav_io,
                beam_size=3,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=350),
            )
            return " ".join([seg.text for seg in segments]).strip()
        except Exception:
            return ""

    def listen_and_transcribe(
        self,
        silence_duration: float = 1.1,
        max_duration: float = 20.0,
        energy_threshold: float = 0.015,
        max_initial_silence: float = 5.0,
    ) -> str:
        """Record from default microphone until speech ends, then transcribe with VAD."""
        self._ensure_initialized()

        audio_buffer = []
        chunk_duration = 0.1  # 100ms
        chunk_samples = int(self.sample_rate * chunk_duration)
        silent_chunks = 0
        max_silent_chunks = int(silence_duration / chunk_duration)
        max_unstarted_chunks = int(max_initial_silence / chunk_duration)
        unstarted_chunks = 0
        has_spoken = False

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            ) as stream:
                total_samples = 0
                while total_samples < max_duration * self.sample_rate:
                    chunk, _ = stream.read(chunk_samples)
                    audio_buffer.append(chunk)
                    total_samples += chunk_samples

                    rms = np.sqrt(np.mean(chunk**2))
                    if rms > energy_threshold:
                        has_spoken = True
                        silent_chunks = 0
                    elif has_spoken:
                        silent_chunks += 1
                        if silent_chunks >= max_silent_chunks:
                            # Finished speaking
                            break
                    else:
                        unstarted_chunks += 1
                        if unstarted_chunks >= max_unstarted_chunks:
                            # No speech detected within initial silence window
                            break
        except Exception:
            return ""

        if not has_spoken or not audio_buffer:
            return ""

        audio_data = np.concatenate(audio_buffer, axis=0)
        return self.transcribe_audio_data(audio_data)

    def listen_with_interruption(
        self,
        is_speaking_fn,
        stop_speaking_fn,
        silence_duration: float = 1.1,
        max_duration: float = 30.0,
        barge_in_energy_threshold: float = 0.020,
    ) -> Tuple[Optional[str], bool]:
        """
        Actively monitor microphone while assistant is speaking (barge-in).
        If user speaks or key is pressed:
          - Immediately calls stop_speaking_fn() to cut off assistant audio.
          - Continues recording the user's speech until they finish.
          - Transcribes and returns (transcribed_question, True).
        If assistant finishes speaking without interruption:
          - Returns (None, False).
        """
        from collections import deque

        self._ensure_initialized()

        chunk_duration = 0.08  # 80ms chunks
        chunk_samples = int(self.sample_rate * chunk_duration)
        max_silent_chunks = int(silence_duration / chunk_duration)
        silent_chunks = 0
        has_interrupted = False
        consecutive_speech = 0

        pre_roll = deque(maxlen=5)
        recorded_audio = []
        noise_floor = 0.008

        try:
            import msvcrt
            has_msvcrt = True
        except ImportError:
            has_msvcrt = False

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            ) as stream:
                total_samples = 0
                while total_samples < max_duration * self.sample_rate:
                    # 1. Non-blocking keypress check for instant manual tap barge-in
                    if has_msvcrt and msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b"\x03":
                            raise KeyboardInterrupt
                        has_interrupted = True
                        stop_speaking_fn()
                        break

                    chunk, _ = stream.read(chunk_samples)
                    total_samples += chunk_samples
                    rms = float(np.sqrt(np.mean(chunk**2)))

                    # While assistant is speaking, monitor user speech with adaptive threshold
                    if is_speaking_fn() and not has_interrupted:
                        pre_roll.append(chunk)

                        if rms < noise_floor:
                            noise_floor = noise_floor * 0.85 + rms * 0.15
                        else:
                            noise_floor = noise_floor * 0.96 + rms * 0.04

                        speech_threshold = max(barge_in_energy_threshold, noise_floor * 1.55)

                        if rms > speech_threshold:
                            consecutive_speech += 1
                        else:
                            consecutive_speech = max(0, consecutive_speech - 1)

                        if consecutive_speech >= 2:
                            # User voice detected! Cut off speaker immediately!
                            has_interrupted = True
                            stop_speaking_fn()
                            recorded_audio.extend(list(pre_roll))
                            recorded_audio.append(chunk)
                            silent_chunks = 0
                        continue

                    # If assistant finished speaking normally without any interruption
                    if not is_speaking_fn() and not has_interrupted:
                        return (None, False)

                    # If interrupted, record user's question until silence
                    if has_interrupted:
                        recorded_audio.append(chunk)
                        if rms > max(0.014, noise_floor * 1.1):
                            silent_chunks = 0
                        else:
                            silent_chunks += 1
                            if silent_chunks >= max_silent_chunks:
                                break
        except KeyboardInterrupt:
            raise
        except Exception:
            return (None, has_interrupted)

        if not has_interrupted or not recorded_audio:
            return (None, has_interrupted)

        audio_data = np.concatenate(recorded_audio, axis=0)
        text = self.transcribe_audio_data(audio_data)
        return (text if text else None, True)


class BargeInMonitor:
    """Active background microphone listener (Thread 2) that monitors for barge-in user speech
    concurrently while the assistant (Thread 1) is streaming text or vocalizing audio output."""

    def __init__(
        self,
        stt: FasterWhisperSTT,
        is_speaking_fn,
        stop_speaking_fn,
        silence_duration: float = 0.9,
        max_duration: float = 25.0,
    ):
        self.stt = stt
        self.is_speaking_fn = is_speaking_fn
        self.stop_speaking_fn = stop_speaking_fn
        self.silence_duration = silence_duration
        self.max_duration = max_duration

        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.triggered_event = threading.Event()
        self.done_event = threading.Event()

        self.was_interrupted: bool = False
        self.transcribed_text: Optional[str] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start background microphone monitoring."""
        with self._lock:
            self.stop()
            self.stop_event.clear()
            self.triggered_event.clear()
            self.done_event.clear()
            self.was_interrupted = False
            self.transcribed_text = None

            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

    def is_triggered(self) -> bool:
        """Return True immediately when user interruption has been triggered."""
        return self.triggered_event.is_set()

    def stop(self) -> None:
        """Signal monitor to stop and wait briefly for thread to terminate."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.6)
            self.thread = None

    def wait_for_text(self, timeout: float = 30.0) -> Optional[str]:
        """Wait for speech capture and transcription to complete, and return transcribed text."""
        self.done_event.wait(timeout=timeout)
        return self.transcribed_text

    def wait_for_completion(self, timeout: float = 35.0) -> Tuple[Optional[str], bool]:
        """Wait for playback to finish naturally or for speech capture/transcription to complete."""
        self.done_event.wait(timeout=timeout)
        return (self.transcribed_text, self.was_interrupted)

    def _monitor_loop(self) -> None:
        """Background thread reading microphone chunks with dynamic VAD and pre-roll."""
        from collections import deque

        self.stt._ensure_initialized()

        chunk_duration = 0.05  # 50ms chunks for sub-100ms rapid response
        chunk_samples = int(self.stt.sample_rate * chunk_duration)
        max_silent_chunks = max(1, int(self.silence_duration / chunk_duration))
        silent_chunks = 0

        pre_roll = deque(maxlen=7)  # 350ms pre-roll circular buffer to save word beginnings
        recorded_chunks = []
        noise_floor = 0.008
        consecutive_speech = 0
        min_consecutive_chunks = 2  # ~100ms speech activity to trigger

        try:
            import msvcrt
            has_msvcrt = True
        except ImportError:
            has_msvcrt = False

        try:
            with sd.InputStream(
                samplerate=self.stt.sample_rate,
                channels=1,
                dtype="float32",
            ) as stream:
                while not self.stop_event.is_set():
                    # Check for non-blocking keyboard tap barge-in
                    if has_msvcrt and msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b"\x03":
                            break
                        self.was_interrupted = True
                        self.triggered_event.set()
                        self.stop_speaking_fn()
                        break

                    chunk, _ = stream.read(chunk_samples)
                    rms = float(np.sqrt(np.mean(chunk**2)))

                    if not self.was_interrupted:
                        pre_roll.append(chunk)

                        # Dynamically update ambient/speaker bleed noise floor
                        if rms < noise_floor:
                            noise_floor = noise_floor * 0.82 + rms * 0.18
                        else:
                            noise_floor = noise_floor * 0.97 + rms * 0.03

                        # Sensitive threshold adapting to headphones vs laptop speakers
                        speech_threshold = max(0.016, noise_floor * 1.5)

                        if rms > speech_threshold:
                            consecutive_speech += 1
                        else:
                            consecutive_speech = max(0, consecutive_speech - 1)

                        if consecutive_speech >= min_consecutive_chunks:
                            # User voice detected! Trigger barge-in immediately!
                            self.was_interrupted = True
                            self.triggered_event.set()
                            self.stop_speaking_fn()

                            # Prepend pre-roll buffer to preserve start of first word
                            recorded_chunks.extend(list(pre_roll))
                            recorded_chunks.append(chunk)
                            silent_chunks = 0
                            continue

                    else:
                        # User has interrupted - now recording user's speech until silence
                        recorded_chunks.append(chunk)
                        user_voice_threshold = max(0.013, noise_floor * 1.1)
                        if rms > user_voice_threshold:
                            silent_chunks = 0
                        else:
                            silent_chunks += 1
                            if silent_chunks >= max_silent_chunks:
                                # User finished speaking!
                                break

                        if len(recorded_chunks) * chunk_duration > self.max_duration:
                            break

        except Exception:
            pass

        # Transcribe user voice if speech was captured
        if self.was_interrupted and recorded_chunks:
            audio_data = np.concatenate(recorded_chunks, axis=0)
            text = self.stt.transcribe_audio_data(audio_data)
            self.transcribed_text = text if text else None

        self.done_event.set()


class VoiceEngine:
    """Unified controller managing both STT listener and streaming Qwen TTS playback."""

    def __init__(
        self,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: Optional[str] = None,
    ):
        self.tts = QwenTTS(ref_audio=ref_audio, ref_text=ref_text)
        self.stt = FasterWhisperSTT()
        self._barge_in_monitor: Optional[BargeInMonitor] = None
        self._is_ready = False

    def warmup(self) -> None:
        """Pre-load model weights into memory."""
        self.tts._ensure_initialized()
        self.stt._ensure_initialized()
        self._is_ready = True

    def is_speaking(self) -> bool:
        """Check if assistant is actively speaking."""
        return self.tts.is_speaking()

    def wait_until_done(self, timeout: float = 60.0) -> None:
        """Wait until speech output is complete before listening."""
        self.tts.wait_until_done(timeout=timeout)

    def start_barge_in_monitor(self, silence_duration: float = 0.9) -> None:
        """Start listening for user voice interruption in Thread 2 while AI is speaking."""
        if self._barge_in_monitor:
            self._barge_in_monitor.stop()
        self._barge_in_monitor = BargeInMonitor(
            stt=self.stt,
            is_speaking_fn=self.is_speaking,
            stop_speaking_fn=self.stop_speaking,
            silence_duration=silence_duration,
        )
        self._barge_in_monitor.start()

    def is_barge_in_triggered(self) -> bool:
        """Check if user speech or keypress has triggered an interruption."""
        return bool(self._barge_in_monitor and self._barge_in_monitor.is_triggered())

    def stop_barge_in_monitor(self) -> None:
        """Stop background barge-in monitor without waiting."""
        if self._barge_in_monitor:
            self._barge_in_monitor.stop()
            self._barge_in_monitor = None

    def wait_for_barge_in_text(self, timeout: float = 30.0) -> Optional[str]:
        """Wait for Thread 2 to finish capturing and transcribing user speech, and return the text."""
        if not self._barge_in_monitor:
            return None
        text = self._barge_in_monitor.wait_for_text(timeout=timeout)
        self._barge_in_monitor = None
        return text

    def wait_for_speech_or_interruption(self, timeout: float = 35.0) -> Tuple[Optional[str], bool]:
        """
        Wait until assistant finishes speaking naturally or until user finishes their barge-in question.
        Returns (transcribed_user_speech, was_interrupted).
        """
        if not self._barge_in_monitor:
            return (None, False)
        result = self._barge_in_monitor.wait_for_completion(timeout=timeout)
        self._barge_in_monitor = None
        return result

    def wait_or_listen_for_interruption(self, silence_duration: float = 1.1) -> Tuple[Optional[str], bool]:
        """
        Wait while assistant is speaking, actively monitoring for user interruption.
        Returns (transcribed_user_question, was_interrupted).
        """
        if self._barge_in_monitor:
            return self.wait_for_speech_or_interruption()
        if not self.is_speaking():
            return (None, False)
        self.start_barge_in_monitor(silence_duration=silence_duration)
        return self.wait_for_speech_or_interruption()

    def listen(self, silence_duration: float = 1.1) -> str:
        """Listen to the microphone and transcribe spoken words."""
        # Ensure audio output is completely finished before recording
        if self.tts.is_speaking():
            self.tts.wait_until_done()
        return self.stt.listen_and_transcribe(silence_duration=silence_duration)

    def speak_sentence(self, sentence: str) -> None:
        """Queue a sentence for real-time speech synthesis and playback."""
        self.tts.speak_sentence(sentence)

    def stop_speaking(self) -> None:
        """Immediately stop TTS speech output (barge-in interruption)."""
        self.tts.stop()

    def shutdown(self) -> None:
        """Shutdown speech synthesizer, stop background listeners, and release GPU memory."""
        self.stop_barge_in_monitor()
        self.tts.shutdown()


