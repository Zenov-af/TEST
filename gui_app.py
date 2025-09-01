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
from enum import Enum

# --- App State Enum ---
class AppState(Enum):
    IDLE = 1
    RECORDING = 2
    PROCESSING = 3
    SPEAKING = 4
    ERROR = 5

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

        # --- Font and Color Definitions ---
        self.title_font = font.Font(family="Arial", size=24, weight="bold")
        self.chat_font = font.Font(family="Arial", size=12)
        self.button_font = font.Font(family="Arial", size=16, weight="bold")
        self.status_font = font.Font(family="Arial", size=10)
        self.colors = {
            "bg": "#2E3440", "fg": "#ECEFF4", "user": "#88C0D0", "ai": "#A3BE8C",
            "chat_bg": "#3B4252", "button_bg": "#434C5E", "button_fg": "#ECEFF4",
            "repeat_bg": "#4C566A", "skip_bg": "#BF616A", "status_fg": "#D8DEE9",
            "error_fg": "#BF616A"
        }

        # --- State Variables ---
        self.current_state = None
        self.whisper_model = None
        self.tts_engine = None
        self.recorder_process = None
        self.last_ai_response = ""
        self.is_speaking = threading.Event()
        self.stop_tts_event = threading.Event()

        # --- UI Layout using .grid() ---
        self.root.grid_rowconfigure(0, weight=0)  # Title
        self.root.grid_rowconfigure(1, weight=1)  # Chat Area
        self.root.grid_rowconfigure(2, weight=0)  # Button Frame
        self.root.grid_rowconfigure(3, weight=0)  # Status Bar
        self.root.grid_columnconfigure(0, weight=1)

        # --- UI Elements ---
        self.title_label = tk.Label(
            root, text="German AI Partner", font=self.title_font,
            fg=self.colors['fg'], bg=self.colors['bg']
        )
        self.title_label.grid(row=0, column=0, pady=10, sticky="ew")

        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=self.chat_font, bg=self.colors['chat_bg'], fg=self.colors['fg'],
            bd=0, highlightthickness=0, relief="flat", padx=10, pady=10
        )
        self.chat_area.grid(row=1, column=0, pady=5, padx=20, sticky="nsew")
        self.chat_area.configure(state='disabled')
        self.chat_area.tag_config('user', foreground=self.colors['user'])
        self.chat_area.tag_config('ai', foreground=self.colors['ai'])

        self.button_frame = tk.Frame(root, bg=self.colors['bg'])
        self.button_frame.grid(row=2, column=0, pady=10)

        self.talk_button = tk.Button(
            self.button_frame, text=" Lade... ", font=self.button_font,
            bg=self.colors['button_bg'], fg=self.colors['button_fg'],
            width=20, height=2, relief="flat", state="disabled"
        )
        self.talk_button.pack(side="left", padx=10)

        self.talk_button.bind("<ButtonPress-1>", self.on_button_press)
        self.talk_button.bind("<ButtonRelease-1>", self.on_button_release)

        self.repeat_button = tk.Button(
            self.button_frame, text="Wiederholen", font=self.button_font,
            bg=self.colors['repeat_bg'], fg=self.colors['button_fg'],
            width=12, height=2, relief="flat", state="disabled", command=self.on_repeat_press
        )
        self.repeat_button.pack(side="left", padx=10)

        self.skip_button = tk.Button(
            self.button_frame, text="Überspringen", font=self.button_font,
            bg=self.colors['skip_bg'], fg=self.colors['button_fg'],
            width=12, height=2, relief="flat", state="disabled", command=self.on_skip_press
        )
        self.skip_button.pack(side="left", padx=10)

        self.status_bar = tk.Label(
            root, text="Initializing...", font=self.status_font, fg=self.colors['status_fg'],
            bg=self.colors['bg'], anchor='w', padx=10
        )
        self.status_bar.grid(row=3, column=0, pady=(0, 5), sticky="ew")

        threading.Thread(target=self.initialize_ai_models, daemon=True).start()

    def set_state(self, state, status_message=None):
        def do_update():
            self.current_state = state
            if state == AppState.IDLE:
                self.status_bar.config(text=status_message or "Ready", fg=self.colors['status_fg'])
                self.talk_button.config(text=" Halten zum Sprechen ", state="normal")
                self.repeat_button.config(state="normal" if self.last_ai_response else "disabled")
                self.skip_button.config(state="disabled")
            elif state == AppState.RECORDING:
                self.status_bar.config(text="Recording...", fg=self.colors['status_fg'])
                self.talk_button.config(text=" Aufnahme... ", state="disabled")
                self.repeat_button.config(state="disabled")
                self.skip_button.config(state="disabled")
            elif state == AppState.PROCESSING:
                self.status_bar.config(text=status_message or "Processing...", fg=self.colors['status_fg'])
                self.talk_button.config(text=" Verarbeite... ", state="disabled")
                self.repeat_button.config(state="disabled")
                self.skip_button.config(state="disabled")
            elif state == AppState.SPEAKING:
                self.status_bar.config(text="Speaking...", fg=self.colors['status_fg'])
                self.talk_button.config(text=" Spricht... ", state="disabled")
                self.repeat_button.config(state="disabled")
                self.skip_button.config(state="normal")
            elif state == AppState.ERROR:
                self.status_bar.config(text=f"Error: {status_message}", fg=self.colors['error_fg'])
                self.talk_button.config(text=" Restart ", state="normal")
                self.repeat_button.config(state="disabled")
                self.skip_button.config(state="disabled")
        self.root.after(0, do_update)

    def update_chat_display(self):
        self.root.after(0, self._update_chat_display_thread_safe)

    def _update_chat_display_thread_safe(self):
        self.chat_area.configure(state='normal')
        self.chat_area.delete(1.0, tk.END)
        for msg in conversation_history[1:]:
            if msg['role'] == 'user':
                self.chat_area.insert(tk.END, f"You: {msg['content']}\n\n", 'user')
            elif msg['role'] == 'assistant':
                self.chat_area.insert(tk.END, f"AI: {msg['content']}\n\n", 'ai')
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def _initialize_tts(self):
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'german' in voice.name.lower() or 'de-de' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            # If initialization fails, ensure engine is None so we can retry
            self.tts_engine = None
            # We are in a thread, so we can't directly update UI easily without risking race conditions.
            # The error will be caught and handled in the speak() method.
            raise e


    def initialize_ai_models(self):
        try:
            self.set_state(AppState.PROCESSING, status_message="Loading STT model...")
            self.whisper_model = Model(WHISPER_MODEL_NAME, n_threads=4, language='de')

            self.set_state(AppState.PROCESSING, status_message="Loading TTS engine...")
            self._initialize_tts()

            self.set_state(AppState.IDLE)
        except Exception as e:
            self.set_state(AppState.ERROR, status_message=str(e))

    def on_button_press(self, event):
        if self.current_state != AppState.IDLE: return
        self.set_state(AppState.RECORDING)
        record_command = [
            FFMPEG_EXE, '-f', 'dshow', '-i', f'audio={MICROPHONE_NAME}',
            '-t', str(MAX_RECORD_DURATION), '-ar', '16000', '-ac', '1', '-y', TEMP_REC_FILE
        ]
        self.recorder_process = subprocess.Popen(
            record_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def on_button_release(self, event):
        if self.current_state != AppState.RECORDING: return

        if self.recorder_process:
            try:
                self.recorder_process.communicate(b'q', timeout=5)
            except subprocess.TimeoutExpired:
                self.recorder_process.kill()
            self.recorder_process = None

        threading.Thread(target=self.process_and_respond, daemon=True).start()

    def on_repeat_press(self):
        if self.last_ai_response and self.current_state == AppState.IDLE:
            self.speak(self.last_ai_response)

    def on_skip_press(self):
        if self.current_state == AppState.SPEAKING:
            self.stop_tts_event.set()
            # We also set the state here to make the UI feel instantly responsive
            self.set_state(AppState.IDLE, status_message="Skipped")

    def speak(self, text):
        if self.current_state not in [AppState.IDLE, AppState.PROCESSING]: return

        def run_tts():
            self.is_speaking.set()
            self.stop_tts_event.clear()
            try:
                if self.tts_engine is None:
                    self._initialize_tts()

                self.set_state(AppState.SPEAKING)
                clean_text = re.sub(r'[\*_`]', '', text)
                self.tts_engine.say(clean_text)

                # Custom non-blocking loop
                self.tts_engine.startLoop(False)
                while self.tts_engine.isBusy() and not self.stop_tts_event.is_set():
                    self.tts_engine.iterate()
                    time.sleep(0.1)
                self.tts_engine.endLoop()
                # If the loop was stopped manually, the engine might be in a bad state
                if self.stop_tts_event.is_set():
                    self.tts_engine = None

            except Exception as e:
                self.set_state(AppState.ERROR, status_message=f"TTS Error: {e}")
                self.tts_engine = None # Discard engine on any error
            finally:
                self.is_speaking.clear()
                self.stop_tts_event.clear()
                if self.current_state == AppState.SPEAKING:
                    self.set_state(AppState.IDLE)

        threading.Thread(target=run_tts, daemon=True).start()

    def process_and_respond(self):
        try:
            self.set_state(AppState.PROCESSING, status_message="Transcribing audio...")
            time.sleep(0.5)

            if not os.path.exists(TEMP_REC_FILE) or os.path.getsize(TEMP_REC_FILE) == 0:
                self.set_state(AppState.IDLE, status_message="Recording was too short.")
                return

            result_list = self.whisper_model.transcribe(TEMP_REC_FILE)
            user_input = "".join(segment.text for segment in result_list).strip()
            if os.path.exists(TEMP_REC_FILE): os.remove(TEMP_REC_FILE)

            if not user_input:
                self.set_state(AppState.IDLE, status_message="Couldn't hear you. Please try again.")
                return

            conversation_history.append({"role": "user", "content": user_input})
            self.update_chat_display()

            self.set_state(AppState.PROCESSING, status_message="AI is thinking...")
            self.last_ai_response = chat_with_ollama(conversation_history)
            conversation_history.append({"role": "assistant", "content": self.last_ai_response})
            self.update_chat_display()

            self.speak(self.last_ai_response)

        except Exception as e:
            self.set_state(AppState.ERROR, status_message=str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantApp(root)
    root.mainloop()
