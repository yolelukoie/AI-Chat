import io
import os
import re
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from gtts import gTTS
import pygame
import json
from pathlib import Path
from faster_whisper import WhisperModel
from instructions_store import add_instruction, get_instructions, remove_instruction
from profile_updater import load_static_profile, save_static_profile
from promotion_tracker import should_run_promotion, update_promotion_time
from memory_promoter import promote_summaries_to_facts as run_memory_promotion
from compression_tracker import should_run_compression, update_compression_time
from chat_history import save_chat_history as save_history
from session_summary import summarize_session 
from memory_promoter import compress_old_memory
from ollama_api import ask_ollama
from helpers import chat, chat_about_users, retrieve_memory, retrieve_memory_by_type, client

PROFILE_FILE = str(Path(__file__).resolve().parent / "memory.json")

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

def extract_user_switch(text: str) -> str | None:
    # Match: "I'm Yana", "I am Kate", "Hey Lama I'm Stav", etc.
    match = re.search(r"(?:hey\s+lama[, ]*)?\b(?:i[’'`]?m|i am)\s+([a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().capitalize()
    return None

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

def detect_mentioned_users(user_id: str, text: str, profiles: dict) -> list[str]:
    input_lower = text.lower()
    return [
        name for name in profiles
        if name.lower() != user_id.lower() and name.lower() in input_lower
    ]

def load_all_memory(filename=PROFILE_FILE):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_user_memory(user_id, filename="PROFILE_FILE"):
    all_memory = load_all_memory(filename)
    return all_memory.get(user_id, {})

def main():

    global user_id
    user_id = "Yana"
    memory = load_user_memory(user_id)
    all_profiles = load_static_profile()
    instructions = get_instructions(user_id)
    chat_history = []

    INSTRUCTION_TRIGGERS = [
        "instruction", "from now on", "please", "always", "i want you to",
        "don't", "never", "stop", "no longer"
    ]

    while True:
        listen_for_hotword()
        speak("Hello! I'm listening.")

        while True:
            wav_io = record_until_silence()
            text = transcribe_local(wav_io).strip()
            if not text:
                continue

            new_user = extract_user_switch(text)
            if new_user:
                user_id = new_user

                if user_id in all_profiles:
                    speak(f"Hi {user_id}!")
                else:
                    speak(f"Hi {user_id}! It looks like we haven't chatted before. Please, tell me something about yourself.")
                    save_static_profile(user_id, {"user_name": user_id})  # minimally bootstrap
                continue 


            print(f"[User] {text}")
            lower_text = text.lower()

            # === Exit handler ===
            if "exit" in lower_text:
                if handle_exit_flow():
                    try:
                        summarize_session(user_id, chat_history)
                    except Exception as e:
                        print(f"⚠️ Failed to summarize session: {type(e).__name__}: {e}")
                    if should_run_compression(user_id):
                        compress_old_memory(user_id)
                        update_compression_time(user_id)
                    if should_run_promotion(user_id):
                        run_memory_promotion(user_id)
                        update_promotion_time(user_id)
                    break
                else:
                    break

            # === Instruction handler ===
            if any(k in lower_text for k in INSTRUCTION_TRIGGERS):
                prompt = f"""
                You're an instruction parser for an AI assistant.

                1. Extract any persistent instruction the user is giving to the assistant.
                2. Check if the new instruction conflicts with existing ones: {get_instructions(user_id)}
                3. If so, return both the new instruction and the one to remove.

                Return a **JSON list**:
                - [new_instruction] → if only adding
                - [new_instruction, instruction_to_remove] → if replacing

                If there's no instruction, return [].

                User said: "{text}"
                """
                raw_response = ask_ollama(prompt).strip()
                print(f"[Instruction Raw Response] {raw_response}")

                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError:
                    print("❌ Failed to parse instruction response.")
                    speak("I didn’t understand that instruction. Could you say it again?")
                    continue

                if not parsed:
                    print("🫥 No instruction extracted.")
                    continue

                new_instruction = parsed[0]
                instruction_to_remove = parsed[1] if len(parsed) > 1 else None

                # Remove conflicting instruction first
                if instruction_to_remove:
                    removed = remove_instruction(user_id, instruction_to_remove)
                    print(f"🗑 Removed: {instruction_to_remove}") if removed else print("⚠️ Nothing removed.")

                # Add new instruction
                add_instruction(user_id, new_instruction)
                speak(f"Got it! From now on I will {new_instruction}.")
                continue
             
            # === Memory and chat handling ===
            mentioned = detect_mentioned_users(user_id, text, all_profiles)
            if mentioned:
                reply = chat_about_users(user_id, text, mentioned, all_profiles)
            else:
                reply = chat(user_id, text, memory, all_profiles, instructions)

            # === Normal assistant reply ===
            chat_history.append({"role": "user", "content": text})
            chat_history.append({"role": "assistant", "content": reply})
            save_history(user_id, chat_history)
            print(f"[LLM] {reply}")
            speak(reply)

if __name__ == "__main__":
    main()
