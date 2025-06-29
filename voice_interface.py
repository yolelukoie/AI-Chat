import io
import os
import re
import time
from urllib import response
import pygame
import json
import uuid
import numpy as np
import sys
import librosa
from pathlib import Path
import webrtcvad
import sounddevice as sd
import soundfile as sf
from queue import Queue
from TTS.api import TTS
from threading import Thread
from collections import deque
from faster_whisper import WhisperModel
from instructions_store import add_instruction, get_instructions, remove_instruction
from promotion_tracker import should_run_promotion, update_promotion_time
from memory_promoter import promote_summaries_to_facts as run_memory_promotion
from compression_tracker import should_run_compression, update_compression_time
from memory_engine import is_memory_removal_request, find_and_remove_matching_memory, query_profile_memory, update_profile_vector, load_all_vector_profiles, get_collections
from session_summary import summarize_session 
from memory_promoter import compress_old_memory
from ollama_api import ask_ollama
from helpers import chat, chat_about_users
from vosk import Model, KaldiRecognizer

# Avoid HuggingFace symlink errors on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize Whisper model
whisper_model = WhisperModel("base", compute_type="auto")

# Initialize TTS model
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
"""
tts_models/en/ljspeech/tacotron2-DDC	Female (LJ Speech)	Clear, high-quality single-speaker
tts_models/en/vctk/vits	Multi-speaker (VCTK)	~100 voices – British, American, global
tts_models/en/multi-dataset/tortoise-v2	Expressive	Slow but highly natural, creative tone
tts_models/en/jenny/jenny	AI assistant style	Fast and very natural, female
"""

# Audio settings
FS = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(FS * FRAME_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 1
HOTWORD_CHUNK_SECONDS = 2

# Initialize Vosk model
vosk_model = Model("models/vosk-model-small-en-us-0.15")
vosk_rec = KaldiRecognizer(vosk_model, FS)
vosk_rec.SetWords(False)

# Initialize VAD and audio playback
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
pygame.mixer.init(frequency=FS)

interrupt_audio_queue = Queue()

# region Rolling stats for audio analysis
class RollingStats:
    def __init__(self, maxlen=30):
        self.values = deque(maxlen=maxlen)

    def add(self, value):
        self.values.append(value)

    def mean(self):
        return np.mean(self.values) if self.values else 0

    def std(self):
        return np.std(self.values) if self.values else 0

    def last(self):
        return self.values[-1] if self.values else 0
        
rolling_amp = RollingStats()
rolling_band = RollingStats()
rolling_noise = RollingStats()

# endregion

# region Audio transcription and hotword detection
def transcribe_local(wav_io: io.BytesIO) -> str:
    wav_io.seek(0)
    with sf.SoundFile(wav_io) as audio_file:
        audio = audio_file.read(dtype="int16")
        audio = audio[:, 0] if audio.ndim > 1 else audio
        audio = audio.astype(np.float32) / 32768.0
        segments, _ = whisper_model.transcribe(audio, language="en")
        return " ".join([seg.text.strip() for seg in segments])


def listen_for_hotword(hotword: str = "lama") -> io.BytesIO | None:
    print("[hotword] Listening for hotword...")

    with sd.RawInputStream(samplerate=FS, blocksize=FRAME_SIZE, dtype='int16', channels=1) as stream:
        rec = KaldiRecognizer(vosk_model, FS)
        rec.SetWords(True)

        frames = []
        hotword_detected = False
        silence_counter = 0

        while True:
            data, _ = stream.read(FRAME_SIZE)
            frame_bytes = bytes(data)
            frames.append(frame_bytes)

            if rec.AcceptWaveform(frame_bytes):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower()
                print(f"[hotword] Final: {text}")
                if hotword in text:
                    print(f"[hotword] Detected '{hotword}' in full result")
                    hotword_detected = True
                    break
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if hotword in partial.lower():
                    print(f"[hotword] Partial match: {partial}")
                    hotword_detected = True
                    break

        if not hotword_detected:
            return None

        # Record extra 3 seconds of audio for intent
        post_hotword = sd.rec(int(FS * 3), samplerate=FS, channels=1, dtype='int16')
        sd.wait()
        frames.append(post_hotword.tobytes())

    # Combine and convert to BytesIO
    full_audio = b''.join(frames)
    wav_io = io.BytesIO()
    sf.write(wav_io, np.frombuffer(full_audio, dtype='int16'), FS, format='WAV', subtype='PCM_16')
    wav_io.seek(0)
    wav_io.name = "intent.wav"
    return wav_io

def stream_until_silence(chunk_duration=6.0, silence_threshold=1.5, max_total_duration=60.0) -> str:
    print("[Whisper] Listening in chunks...")

    all_audio = []
    start_time = time.time()
    last_voice_time = time.time()
    silence_detected = False

    with sd.RawInputStream(samplerate=FS, blocksize=FRAME_SIZE, dtype='int16', channels=1) as stream:
        while True:
            chunk = []
            chunk_start = time.time()
            while time.time() - chunk_start < chunk_duration:
                frame, _ = stream.read(FRAME_SIZE)
                chunk.append(frame)
                frame_bytes = bytes(frame)

                if vad.is_speech(frame_bytes, FS):
                    last_voice_time = time.time()

                if time.time() - last_voice_time > silence_threshold:
                    print("[Whisper] Detected end of speech mid-chunk.")
                    silence_detected = True
                    break

            all_audio.extend(chunk)

            # Check if we have silence at the end of the chunk
            if silence_detected:
                print("[Whisper] Final end of speech.")
                break
            silence_duration = time.time() - last_voice_time

            # Check if silence threshold exceeded
            if silence_duration > silence_threshold:
                print("[Whisper] Silence threshold exceeded.")
                speak("hey, are you still there?")
                break

            if time.time() - start_time > max_total_duration:
                print("[Whisper] Max duration reached.")
                break

            if silence_duration > max_total_duration:
                print("[Whisper] Too much silence, stopping.")
                
                break


    # Combine all frames and convert to BytesIO
    full_audio = b''.join(all_audio)
    wav_io = io.BytesIO()
    sf.write(wav_io, np.frombuffer(full_audio, dtype='int16'), FS, format='WAV', subtype='PCM_16')
    wav_io.seek(0)

    segments, _ = whisper_model.transcribe(wav_io, language="en")
    transcript = " ".join([seg.text.strip() for seg in segments])
    print(f"[Whisper] Final transcript: {transcript}")
    return transcript


"""def stream_until_silence(timeout=6.0) -> str:
    print("[Whisper] Listening for full utterance...")

    buffer = []
    transcript_parts = []
    start_time = time.time()
    last_voice_time = time.time()

    with sd.RawInputStream(samplerate=FS, blocksize=FRAME_SIZE, dtype='int16', channels=1) as stream:
        while True:
            frame, _ = stream.read(FRAME_SIZE)
            buffer.append(frame)
            frame_bytes = bytes(frame)

            # VAD check
            if vad.is_speech(frame_bytes, FS):
                last_voice_time = time.time()

            # Check for silence
            if time.time() - last_voice_time > 1.5:
                print("[Whisper] Detected end of speech.")
                break

            # Optional timeout
            if time.time() - start_time > timeout:
                print("[Whisper] Timeout.")
                break

    # Save buffered audio as WAV
    full_audio = b''.join(buffer)
    wav_io = io.BytesIO()
    sf.write(wav_io, np.frombuffer(full_audio, dtype='int16'), FS, format='WAV', subtype='PCM_16')
    wav_io.seek(0)

    # Transcribe once with Whisper
    segments, _ = whisper_model.transcribe(wav_io, language="en")
    transcript = " ".join([seg.text.strip() for seg in segments])
    print(f"[Whisper] Final transcript: {transcript}")
    return transcript"""
# endregion

# region User interrupt detection


def detect_user_interrupt(similarity_threshold=0.80) -> bool:
    if interrupt_audio_queue.empty():
        return False

    user_audio = interrupt_audio_queue.get()
    user_audio = user_audio.flatten().astype(np.float32) / 32768.0

    try:
        # Load last TTS audio (if you saved it)
        tts_audio, _ = sf.read("last_tts.wav")
        user_mfcc = librosa.feature.mfcc(y=user_audio, sr=FS, n_mfcc=13)
        tts_mfcc = librosa.feature.mfcc(y=tts_audio, sr=FS, n_mfcc=13)

        # Average across time, then compare
        sim = cosine_similarity(user_mfcc.mean(axis=1), tts_mfcc.mean(axis=1))
        print(f"[DIAR] Cosine similarity = {sim:.2f}")

        return sim < similarity_threshold
    except Exception as e:
        print(f"[DIAR] Error comparing voices: {e}")
        return False


def cosine_similarity(vec1, vec2):
    a = np.dot(vec1, vec2)
    b = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return a / b if b != 0 else 0

def detect_interrupting_speech(hotwords={"stop", "lama", "wait", "excuse"}):
    print("[INTERRUPT] Listening for interrupt command...")

    rec = KaldiRecognizer(vosk_model, FS)
    rec.SetWords(False)

    # Record ~1 sec of audio (non-blocking)
    duration_samples = int(1.0 * FS)
    audio = sd.rec(duration_samples, samplerate=FS, channels=1, dtype='int16')
    sd.wait()

    # Feed audio to recognizer
    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.Result())
    text = result.get("text", "").lower()

    print(f"[INTERRUPT] Heard: {text}")
    for hw in hotwords:
        if hw in text:
            print(f"🛑 Interrupt word detected: {hw}")
            return True

    return False

"""def detect_interrupting_speech() -> bool:
    # uses FFT/VAD
    duration_samples = int(0.4 * FS)
    audio = sd.rec(duration_samples, samplerate=FS, channels=1, dtype='int16')
    sd.wait()
    audio_np = audio[:, 0] if audio.ndim > 1 else audio
    audio_np = audio_np * np.hamming(len(audio_np))

    max_amp = np.max(np.abs(audio_np))
    freqs = np.fft.rfft(audio_np)
    freqs_power = np.abs(freqs)
    speech_band = np.sum(freqs_power[100:300])
    low_noise = np.sum(freqs_power[10:80])

    # Update rolling stats
    rolling_amp.add(max_amp)
    rolling_band.add(speech_band)
    rolling_noise.add(low_noise)

    print(f"[VAD] Amp={max_amp}, SpeechBand={speech_band:.0f}, Hum={low_noise:.0f}")
    print(f"[ROLLING] Amp µ={rolling_amp.mean():.0f}, Speech µ={rolling_band.mean():.0f}, Noise µ={rolling_noise.mean():.0f}")

    # Dynamic thresholds (tuned as needed)
    amp_threshold = rolling_amp.mean() + 2 * rolling_amp.std()
    band_threshold = rolling_band.mean() + 1.5 * rolling_band.std()

    # Adaptive logic
    if max_amp < amp_threshold:
        # print("[VAD] ❌ Rejected: Amplitude too low")
        return False
    if speech_band < band_threshold:
        # print("[VAD] ❌ Rejected: Not enough speech band energy")
        return False
    if speech_band < 3 * low_noise:
        # print("[VAD] ❌ Rejected: Too much background hum")
        return False

    print("🛑 Real speech detected — interrupting.")
    return True
"""
# endregion

# region Confidence scoring for input

def should_process_text(text: str) -> bool:
    if not text:
        print("[Confidence] ❌ Empty.")
        return False
    
    special_intents = {"stop", "wait", "exit", "repeat", "continue"}
    if text in special_intents:
        print(f"[Confidence] ⚠️ Command word detected: {text}")
        return True

    if len(text.split()) < 2:
        print("[Confidence] ⚠️ Only one word — might be junk.")
        # Pass to LLM check
        return check_with_llm(text)
    
    return True

def check_with_llm(text: str) -> bool:
    prompt = f"""A user said: "{text}"

Is this a real question or answer or command that an AI assistant should respond to?

Reply only with YES or NO."""
    try:
        reply = ask_ollama(prompt).strip().upper()
        print(f"[LLM-Check] LLM replied:Is it a meaningful input? - {reply}")
        return "YES" in reply
    except Exception as e:
        print(f"[LLM-Check Error] {e}")
        return False

# endregion

# region Speaking
def clean_text_for_tts(text: str) -> str:
    # Remove emojis and characters not in TTS vocabulary
    return re.sub(r"[^\x00-\x7F]+", "", text)

def buffer_mic_during_speak(duration=0.2): # set back tp 0.4 if it crashes
    while pygame.mixer.music.get_busy():
        audio = sd.rec(int(FS * duration), samplerate=FS, channels=1, dtype='int16')
        sd.wait()
        interrupt_audio_queue.put(audio)

def speak(text: str):
    text = clean_text_for_tts(text.strip())
    if not text.strip():
        print("[TTS] Warning: empty reply, skipping speech.")
        return
    if len(text.split()) < 2:
        print(f"[TTS] Skipping too-short TTS input: '{text}'")
        return

    print(f"[TTS] Speaking: {text}")
    audio = tts_model.tts(text)

    # Normalize and save to WAV for diarization
    audio_np = np.array(audio).astype(np.float32)
    sf.write("last_tts.wav", audio_np, FS)

    # Save to temp playable file for pygame
    temp_filename = f"tts_output_{uuid.uuid4().hex[:8]}.wav"
    sf.write(temp_filename, audio_np, FS, format="WAV")
    
    # Initialize Pygame if not already
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=FS)

    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()

    # Start mic buffering in parallel
    Thread(target=buffer_mic_during_speak, daemon=True).start()

    start_time = time.time()

    while pygame.mixer.music.get_busy():
        elapsed = time.time() - start_time
        if elapsed < 1.0:
            time.sleep(0.05) # set back tp 0.1 if it crashes
            continue

        # Check for user interrupt
        if detect_interrupting_speech():
            #if detect_user_interrupt():
            pygame.mixer.music.stop()
            print("[TTS] 🔇 Interrupted by user speech")
            return

        time.sleep(0.05)
# endregion

# region Command hotwords detection

def deal_with_instruction(user_id: str, user_text: str):
    print(f"[DEBUG] deal_with_instruction() called with text: {user_text}")

    prompt = f"""
    You're an instruction parser for an AI assistant.

    1. Make a decision how the users wants you to talk to him from now on.
    2. Check if the new instruction conflicts with existing ones: {get_instructions(user_id)}
    3. If so, return both the new instruction and the one to remove.

    Return a **JSON list**:
    - [new_instruction] → if only adding
    - [new_instruction, instruction_to_remove] → if replacing

    If it's not an instruction, return [].

    User said: "{user_text}"
    """
    raw_response = ask_ollama(prompt).strip()

    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_response:
        raw_response = raw_response.split("```")[1].split("```")[0].strip()
   
    print(f"[Instruction Raw Response] {raw_response}")
    try:
        parsed = json.loads(raw_response)
        if parsed:
            new_instruction = parsed[0]
            instruction_to_remove = parsed[1] if len(parsed) > 1 else None
            if instruction_to_remove:
                removed = remove_instruction(user_id, instruction_to_remove)
                print(f"🗑 Removed: {instruction_to_remove}") if removed else print("⚠️ Nothing removed.")
            add_instruction(user_id, new_instruction)
            print(f"Got it! From now on I will {new_instruction}.")
    except json.JSONDecodeError:
        print("❌ Failed to parse instruction response.")
        speak("I didn’t understand that instruction. Could you say it again?")
    user_text = ""

def handle_exit_flow(chat_history, exit = False) -> bool:
    speak("Are you sure you want to exit? Say 'yes' or 'no'.")
    wav_io = stream_until_silence()
    answer = transcribe_local(wav_io).strip().lower()
    if exit or "yes" in answer:
        speak("I didn't hear you for a while, so I stopped listening.")
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
        return True
    else:
        speak("Okay, continuing.")
        return False

def user_exists(user_id: str) -> bool:
    memory_collection, profile_collection = get_collections(user_id)

    # Check general memory
    results = memory_collection.get(where={"user": user_id}, limit=1)
    if results.get("documents"):
        return True

    # Check profile memory
    results = profile_collection.get(where={"user": user_id}, limit=1)
    return bool(results.get("documents"))

# endregion

def main():
    global user_id, memory, calibration_done, calibration
    user_id = "Yana"
    memory = query_profile_memory(user_id, "__FULL__")
    # all_profiles = load_all_vector_profiles ()
    instructions = get_instructions(user_id)
    chat_history = []

    INSTRUCTION_TRIGGERS = [
        "instruction", "from now on", "please", "always", "i want you to",
        "don't", "never", "stop", "no longer"
    ]

    while True:
        hotword_audio = listen_for_hotword(hotword="lama")
        if not hotword_audio:
            continue

        hotword_text = transcribe_local(hotword_audio).strip()

        # === Remove excessive repetition
        sentences = re.split(r'[.?!]', hotword_text)
        unique = []
        for s in sentences:
            s = s.strip()
            if s and s not in unique:
                unique.append(s)
        if len(unique) < len(sentences):
            print("[User Text] Repetition detected — cleaned.")
            hotword_text = ". ".join(unique)

        if not hotword_text or hotword_text.lower() == "lama":
            speak("Hey! I'm listening.")
            time.sleep(1.0)  # Let the user prepare to speak
            user_text = stream_until_silence().strip()
        else:
            user_text = hotword_text

        # === Session Loop ===
        while True:
            if not user_text:
                user_text = stream_until_silence().strip()
                if not should_process_text(user_text):
                    print("[Main] Skipping junk or background noise.")
                    break

            print(f"[User] {user_text}")
            lower_text = user_text.lower()

             # === Exit
            if "exit" in lower_text:
                if not handle_exit_flow(chat_history):
                    user_text = ""  # Reset after exit confirmation

            # Pattern match for common name-introduction formats
            match = re.search(r"\bmy name is\s+([A-Z][a-z]+)\b", user_text, re.IGNORECASE)
            print("Name detected: ", match)

            possible_name = match.group(1).strip().capitalize() if match else ""

            # Extract potential names from the user text
            tokens = re.findall(r'\b(?:[A-Z][a-z]+)\b', user_text)
            mentioned_users = list(set(tokens))
            if possible_name in mentioned_users:
                mentioned_users.remove(possible_name)

            prompt = f"""
            You are Lama, the user's assistant. The message below is what the *user* said to you:
            "{user_text}"
            Your task is to analyze the user's message and decide what action should be taken:

            1. is the user instructing you on how to talk to him from now on, setting a conversation style? If so, return 1.
            2. is the user trying to remove or forget some memory or information stored about them? If so, return 2.
            3. Does the user clearly indicate that user's name is {possible_name} ("Lama" is your name, not user's, so ignore name "Lama")? If so, return 3. 
            4. Does the user clearly ask you a question about someone named {', '.join(mentioned_users)} ("Lama" is your name, not user's, so ignore name "Lama")? If so, return 4. 
            5. If none of the above and the user is just talking, return 0.
            """
            response = ask_ollama(prompt)
            matches = re.findall(r"\b([0-4])\b", response)
            response_clean = matches[-1] if matches else None
            print(f"[LLM Action] Parsed action code: {response_clean}")

            if not response_clean:
                return None
            if response_clean == "1":
                deal_with_instruction(user_id, user_text)
                user_text = ""  # Reset after handling instruction
            elif response_clean == "2":
                find_and_remove_matching_memory(user_id, user_text)
                print("Okay, I've removed that from memory.")
                user_text = ""  # Reset after handling memory removal
            elif response_clean == "3" and match:
                if user_exists(possible_name):
                    speak(f"Hi {possible_name}!")
                    user_id = possible_name
                    memory = query_profile_memory(user_id, "__FULL__")
                else:
                    speak(f"Hi {user_id}! It looks like we haven't chatted before. Would you like to tell me something about yourself?")
                user_text = ""  # Reset after handling name introduction
            elif response_clean == "4":
                for name in mentioned_users:
                    if user_exists(name):
                        user_memory = query_profile_memory(name, "__FULL__")
                        reply = chat_about_users(user_id, user_text, name, user_memory)
                        chat_history.append({"role": "user", "content": user_text})
                        chat_history.append({"role": "assistant", "content": reply})
                        speak(reply)
                        user_text = ""
            elif response_clean == "0":
                reply = chat(user_id, user_text, memory, instructions)
                chat_history.append({"role": "user", "content": user_text})
                chat_history.append({"role": "assistant", "content": reply})
                speak(reply)
                user_text = ""

if __name__ == "__main__":
    main()
