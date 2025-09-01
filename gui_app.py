import tkinter as tk
from tkinter import font, scrolledtext
import threading
import requests
import json
import os
import subprocess
import pyttsx3
import re
import time
from pywhispercpp.model import Model

# --- Constants ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"
WHISPER_MODEL_NAME = "large-v3"
TEMP_REC_FILE = "temp_rec.wav"
FFMPEG_EXE = "./ffmpeg/ffmpeg.exe"
MICROPHONE_NAME = "Mikrofon (Realtek(R) Audio)"
MAX_RECORD_DURATION = 15

# --- Global Conversation History ---
conversation_history = [
    {"role": "system", "content": "Du bist ein freundlicher und hilfreicher deutscher KI-Assistent. Antworte immer auf Deutsch und benutze keine Emojis."}
]

# --- AI Logic Functions ---
def chat_with_ollama(history):
    full_prompt = "".join(
        f"You: {msg['content']}\n" if msg['role'] == 'user' else
        f"AI: {msg['content']}\n" if msg['role'] == 'assistant' else
        f"{msg['content']}\n"
        for msg in history
    ) + "AI:"
    payload = {"model": MODEL_NAME, "prompt": full_prompt, "stream": False}
    response = requests.post(OLLAMA_ENDPOINT, json=payload)
    response.raise_for_status()
    response_data = response.json()
    return response_data.get('response', '').strip()

# --- Main GUI Application Class ---
class AssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("German AI Partner")
        self.root.geometry("700x600")
        self.root.configure(bg="#2E3440")
        
        self.title_font = font.Font(family="Arial", size=24, weight="bold")
        self.chat_font = font.Font(family="Arial", size=12)
        self.button_font = font.Font(family="Arial", size=16, weight="bold")

        self.whisper_model = None
        self.tts_engine = None
        self.recorder_process = None
        self.last_ai_response = ""
        self.is_speaking = threading.Event()
        
        # --- UI Elements ---
        self.title_label = tk.Label(root, text="German AI Partner", font=self.title_font, fg="#ECEFF4", bg="#2E3440")
        self.title_label.pack(pady=15)

        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=self.chat_font, bg="#3B4252", fg="#D8DEE9",
            bd=0, highlightthickness=0, relief="flat", padx=10, pady=10
        )
        self.chat_area.pack(pady=10, padx=20, expand=True, fill="both")
        self.chat_area.configure(state='disabled')
        self.chat_area.tag_config('user', foreground="#88C0D0")
        self.chat_area.tag_config('ai', foreground="#A3BE8C")

        self.button_frame = tk.Frame(root, bg="#2E3440")
        self.button_frame.pack(pady=20)

        self.talk_button = tk.Button(
            self.button_frame, text=" Lade... ", font=self.button_font, bg="#434C5E", fg="#ECEFF4",
            width=20, height=2, relief="flat", state="disabled"
        )
        self.talk_button.pack(side="left", padx=5)
        
        self.talk_button.bind("<ButtonPress-1>", self.on_button_press)
        self.talk_button.bind("<ButtonRelease-1>", self.on_button_release)
        
        self.repeat_button = tk.Button(
            self.button_frame, text="Wiederholen", font=self.button_font, bg="#4C566A", fg="#ECEFF4",
            width=12, height=2, relief="flat", state="disabled", command=self.on_repeat_press
        )
        self.repeat_button.pack(side="left", padx=5)

        self.skip_button = tk.Button(
            self.button_frame, text="Überspringen", font=self.button_font, bg="#BF616A", fg="#ECEFF4",
            width=12, height=2, relief="flat", state="disabled", command=self.on_skip_press
        )
        self.skip_button.pack(side="left", padx=5)

        threading.Thread(target=self.initialize_ai_models, daemon=True).start()

    def update_chat_display(self):
        self.root.after(0, self._update_chat_display_thread_safe)
        
    def _update_chat_display_thread_safe(self):
        self.chat_area.configure(state='normal')
        self.chat_area.delete(1.0, tk.END)
        # Start from the system prompt but don't display it
        for msg in conversation_history[1:]:
            if msg['role'] == 'user':
                self.chat_area.insert(tk.END, f"You: {msg['content']}\n\n", 'user')
            elif msg['role'] == 'assistant':
                self.chat_area.insert(tk.END, f"AI: {msg['content']}\n\n", 'ai')
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def initialize_ai_models(self):
        try:
            self.update_ui_state("Loading STT...", talk_text=" Lade... ", talk_enabled=False)
            self.whisper_model = Model(WHISPER_MODEL_NAME, n_threads=4, language='de')
            
            self.update_ui_state("Loading TTS...", talk_text=" Lade... ", talk_enabled=False)
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'german' in voice.name.lower() or 'de-de' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            self.update_ui_state("Ready!", talk_text=" Halten zum Sprechen ", talk_enabled=True)
        except Exception as e:
            self.update_ui_state(f"Error: {e}", talk_text=" Restart ", talk_enabled=True, is_error=True)

    def on_button_press(self, event):
        if self.recorder_process: return
        self.update_ui_state("Recording...", talk_text=" Aufnahme... ", talk_enabled=False)
        record_command = [
            FFMPEG_EXE, '-f', 'dshow', '-i', f'audio={MICROPHONE_NAME}',
            '-t', str(MAX_RECORD_DURATION), '-ar', '16000', '-ac', '1', '-y', TEMP_REC_FILE
        ]
        self.recorder_process = subprocess.Popen(
            record_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def on_button_release(self, event):
        if self.recorder_process:
            try:
                self.recorder_process.communicate(b'q', timeout=5)
            except subprocess.TimeoutExpired:
                self.recorder_process.kill()
            self.recorder_process = None
        threading.Thread(target=self.process_and_respond, daemon=True).start()

    def on_repeat_press(self):
        if self.last_ai_response:
            self.update_ui_state("Repeating...", repeat_enabled=False, skip_enabled=True)
            self.speak(self.last_ai_response)

    def on_skip_press(self):
        if self.is_speaking.is_set():
            self.tts_engine.stop()
            # The callback in the speak thread will handle the rest
            
    def speak(self, text):
        def run_tts():
            self.is_speaking.set()
            try:
                clean_text = re.sub(r'[\*_`]', '', text)
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            finally:
                self.is_speaking.clear()
                self.root.after(0, self.reset_ui_after_speaking)
        threading.Thread(target=run_tts, daemon=True).start()

    def process_and_respond(self):
        try:
            self.update_ui_state("Processing...", talk_text=" Verarbeite... ", talk_enabled=False)
            time.sleep(0.5)

            if not os.path.exists(TEMP_REC_FILE) or os.path.getsize(TEMP_REC_FILE) == 0:
                self.update_ui_state("Recording too short", talk_text=" Halten zum Sprechen ", talk_enabled=True)
                return

            result_list = self.whisper_model.transcribe(TEMP_REC_FILE)
            user_input = "".join(segment.text for segment in result_list).strip()
            if os.path.exists(TEMP_REC_FILE): os.remove(TEMP_REC_FILE)
            
            if not user_input:
                self.update_ui_state("Could not hear you", talk_text=" Halten zum Sprechen ", talk_enabled=True)
                return

            conversation_history.append({"role": "user", "content": user_input})
            self.update_chat_display()
            self.update_ui_state("Thinking...")
            
            self.last_ai_response = chat_with_ollama(conversation_history)
            conversation_history.append({"role": "assistant", "content": self.last_ai_response})
            self.update_chat_display()
            
            self.update_ui_state("Speaking...", skip_enabled=True)
            self.speak(self.last_ai_response)

        except Exception as e:
            self.update_ui_state(f"Error: {e}", talk_text=" Restart ", talk_enabled=True, is_error=True)

    def reset_ui_after_speaking(self):
        self.update_ui_state("Ready!", talk_text=" Halten zum Sprechen ", talk_enabled=True, repeat_enabled=True, skip_enabled=False)

    def update_ui_state(self, status_text, talk_text=None, talk_enabled=None, repeat_enabled=None, skip_enabled=None, is_error=False):
        def do_update():
            # This is a dummy status update for now.
            if talk_text: self.talk_button.config(text=talk_text)
            if talk_enabled is not None: self.talk_button.config(state="normal" if talk_enabled else "disabled")
            if repeat_enabled is not None: self.repeat_button.config(state="normal" if repeat_enabled else "disabled")
            if skip_enabled is not None: self.skip_button.config(state="normal" if skip_enabled else "disabled")
        self.root.after(0, do_update)

if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantApp(root)
    root.mainloop()

