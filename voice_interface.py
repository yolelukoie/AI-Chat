import io
import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from gtts import gTTS
import pygame
from faster_whisper import WhisperModel

# Avoid HuggingFace symlink errors on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize Whisper model
whisper_model = WhisperModel("base", compute_type="auto")

# Audio settings
FS = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(FS * FRAME_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 1
HOTWORD_CHUNK_SECONDS = 2

# Initialize VAD and audio playback
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
pygame.mixer.init(frequency=FS)


def transcribe_local(wav_io: io.BytesIO) -> str:
    wav_io.seek(0)
    with sf.SoundFile(wav_io) as audio_file:
        audio = audio_file.read(dtype="int16")
        audio = audio[:, 0] if audio.ndim > 1 else audio
        audio = audio.astype(np.float32) / 32768.0
        segments, _ = whisper_model.transcribe(audio, language="en")
        return " ".join([seg.text.strip() for seg in segments])


def listen_for_hotword(hotword: str = "lama"):
    print("[hotword] Listening for hotword...", flush=True)
    while True:
        audio_chunk = sd.rec(
            int(HOTWORD_CHUNK_SECONDS * FS), samplerate=FS, channels=1, dtype='int16'
        )
        sd.wait()

        wav_io = io.BytesIO()
        sf.write(wav_io, audio_chunk, FS, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        wav_io.name = "hotword.wav"

        try:
            text = transcribe_local(wav_io).lower()
            print(f"[hotword] Transcribed: {text}")
            if hotword in text:
                print(f"[hotword] Detected '{hotword}'", flush=True)
                return
        except Exception as e:
            print(f"[hotword] Local ASR error: {e}", flush=True)

        time.sleep(0.1)


def record_until_silence() -> io.BytesIO:
    buffer = []
    speech_started = False
    silence_frames = 0
    speech_frames = 0

    with sd.RawInputStream(
        samplerate=FS, blocksize=FRAME_SIZE, dtype='int16', channels=1
    ) as stream:
        print("[VAD] Waiting for speech...")
        while True:
            frame, _ = stream.read(FRAME_SIZE)
            frame_np = np.frombuffer(frame, dtype='int16')
            frame_bytes = frame_np.tobytes()
            is_speech = vad.is_speech(frame_bytes, FS)

            if not speech_started:
                if is_speech:
                    speech_frames += 1
                else:
                    speech_frames = 0
                if speech_frames * FRAME_DURATION_MS >= 200:
                    speech_started = True
                    print("[VAD] Speech started")
            else:
                buffer.append(frame_np)
                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
                if silence_frames * FRAME_DURATION_MS > 500:
                    print("[VAD] Speech ended")
                    break

    audio_np = np.concatenate(buffer)
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_np, FS, format='WAV', subtype='PCM_16')
    wav_io.seek(0)
    wav_io.name = "speech.wav"
    return wav_io


def detect_interrupting_speech() -> bool:
    duration_samples = int(0.4 * FS)
    audio = sd.rec(duration_samples, samplerate=FS, channels=1, dtype='int16')
    sd.wait()
    audio_np = audio[:, 0] if audio.ndim > 1 else audio

    max_amp = np.max(np.abs(audio_np))
    if max_amp < 2000:
        print(f"[VAD] Ignored quiet background (amp={max_amp})")
        return False

    freqs = np.fft.rfft(audio_np * np.hamming(len(audio_np)))
    freqs_power = np.abs(freqs)
    speech_band_power = np.sum(freqs_power[100:300])
    low_freq_noise = np.sum(freqs_power[10:80])

    print(f"[VAD] Amp={max_amp}, SpeechBand={speech_band_power:.0f}, Hum={low_freq_noise:.0f}")

    if speech_band_power < 20000 or speech_band_power < 3 * low_freq_noise:
        print("[VAD] Frequency pattern not matching speech — ignored")
        return False

    return True


def speak(text: str = "(no reply)"):
    if not text.strip():
        print("[TTS] Warning: empty reply, skipping speech.")
        return
    print(f"[TTS] Speaking: {text}")
    tts = gTTS(text=text, lang="en")
    mp3_io = io.BytesIO()
    tts.write_to_fp(mp3_io)
    mp3_io.seek(0)

    try:
        pygame.mixer.music.load(mp3_io, 'mp3')
        pygame.mixer.music.play()
        start_time = time.time()
        while pygame.mixer.music.get_busy():
            if time.time() - start_time < 1.0:
                time.sleep(0.1)
                continue
            if detect_interrupting_speech():
                pygame.mixer.music.stop()
                print("[TTS] Interrupted by user speech")
                break
    except pygame.error as e:
        print(f"[TTS] Playback error: {e}")


def ask_ollama(prompt_text: str) -> str:
    return ""


def handle_exit_flow() -> bool:
    speak("Are you sure you want to exit? Say 'yes' or 'no'.")
    wav_io = record_until_silence()
    answer = transcribe_local(wav_io).strip().lower()
    if "yes" in answer:
        speak("Goodbye!")
        return True
    else:
        speak("Okay, continuing.")
        return False


def main():
    while True:
        listen_for_hotword()
        speak("Hello! I'm listening.")

        while True:
            wav_io = record_until_silence()
            text = transcribe_local(wav_io).strip()
            if not text:
                continue

            print(f"[User] {text}")
            if "exit" in text.lower():
                if handle_exit_flow():
                    return
                else:
                    break

            reply = ask_ollama(text)
            print(f"[LLM] {reply}")
            speak(reply)


if __name__ == "__main__":
    main()
