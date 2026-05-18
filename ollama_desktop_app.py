# pyrefly: ignore [missing-import]
import sys
import os
import subprocess

# Auto-relaunch inside virtual environment if available
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "myenv", "Scripts", "python.exe")
if os.path.exists(venv_python) and "myenv" not in sys.executable.lower():
    subprocess.Popen([venv_python] + sys.argv)
    sys.exit(0)

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import io
import tempfile
import cv2
import ollama
import fitz  # PyMuPDF
from PIL import Image

import socket
import time

try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

def ensure_ollama_running():
    # 1. Quick port check to see if Ollama is already running
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(("127.0.0.1", 11434))
            return True
    except Exception:
        pass
        
    # 2. Try to launch the Ollama app tray executable
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    ollama_app_path = os.path.join(local_app_data, "Programs", "Ollama", "ollama app.exe")
    
    if os.path.exists(ollama_app_path):
        try:
            # Use os.startfile to run it detached and clean
            os.startfile(ollama_app_path)
            # Wait up to 8 seconds for the port to open
            for _ in range(8):
                time.sleep(1)
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.5)
                        s.connect(("127.0.0.1", 11434))
                        return True
                except Exception:
                    pass
        except Exception:
            pass
            
    # 3. Fallback: try starting via subprocess using system path
    try:
        subprocess.Popen(
            'start "" "ollama app.exe"',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        for _ in range(5):
            time.sleep(1)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect(("127.0.0.1", 11434))
                    return True
            except Exception:
                pass
    except Exception:
        pass
        
    return False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Optimization of models")
        self.geometry("1100x750")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Optimization\nof models", font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        self.btn_chat = ctk.CTkButton(self.sidebar_frame, text="1. General Chat", command=lambda: self.select_frame("chat"))
        self.btn_chat.grid(row=1, column=0, padx=20, pady=10)
        
        self.btn_image = ctk.CTkButton(self.sidebar_frame, text="2. Image Describer", command=lambda: self.select_frame("image"))
        self.btn_image.grid(row=2, column=0, padx=20, pady=10)

        self.btn_video_desc = ctk.CTkButton(self.sidebar_frame, text="3. Video Describer", command=lambda: self.select_frame("video_desc"))
        self.btn_video_desc.grid(row=3, column=0, padx=20, pady=10)

        self.btn_video_chat = ctk.CTkButton(self.sidebar_frame, text="4. Video Audio Chat", command=lambda: self.select_frame("video_chat"))
        self.btn_video_chat.grid(row=4, column=0, padx=20, pady=10)

        self.btn_pdf = ctk.CTkButton(self.sidebar_frame, text="5. Chat with PDF", command=lambda: self.select_frame("pdf"))
        self.btn_pdf.grid(row=5, column=0, padx=20, pady=10)

        self.btn_tutor = ctk.CTkButton(self.sidebar_frame, text="6. Coding Tutor", command=lambda: self.select_frame("tutor"))
        self.btn_tutor.grid(row=6, column=0, padx=20, pady=10)

        # Connection Status Indicator
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Ollama: Connecting...", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=7, column=0, padx=20, pady=10, sticky="s")
        threading.Thread(target=self.check_ollama_status, daemon=True).start()

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_optionemenu.set("Dark")

        # Main Frames Dictionary
        self.frames = {}
        
        self.frames["chat"] = ChatFrame(self)
        self.frames["image"] = ImageDescriberFrame(self)
        self.frames["video_desc"] = VideoDescriberFrame(self)
        self.frames["video_chat"] = VideoChatFrame(self)
        self.frames["pdf"] = PDFChatFrame(self)
        self.frames["tutor"] = CodingTutorFrame(self)

        self.select_frame("chat")

    def check_ollama_status(self):
        running = ensure_ollama_running()
        if running:
            self.status_label.configure(text="● Ollama: Connected", text_color="#4CAF50")
        else:
            self.status_label.configure(text="● Ollama: Offline", text_color="#F44336")

    def select_frame(self, name):
        for frame in self.frames.values():
            frame.grid_forget()
        self.frames[name].grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)


class ChatFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=10, fg_color="transparent")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text="General Chat (Llama 3.1)", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, padx=0, pady=(0, 20), sticky="w")

        self.textbox = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(size=14))
        self.textbox.grid(row=1, column=0, padx=0, pady=(0, 20), sticky="nsew")

        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, padx=0, pady=0, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Enter your prompt here...", height=40)
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", lambda e: self.send_message())
        
        self.send_btn = ctk.CTkButton(self.input_frame, text="Send", command=self.send_message, height=40)
        self.send_btn.grid(row=0, column=1)

    def send_message(self):
        prompt = self.entry.get()
        if prompt:
            self.textbox.insert("end", f"You: {prompt}\n\n")
            self.entry.delete(0, "end")
            self.send_btn.configure(state="disabled")
            threading.Thread(target=self.generate_response, args=(prompt,), daemon=True).start()

    def generate_response(self, prompt):
        try:
            response = ollama.generate(model="llama3.1", prompt=prompt)
            clean_resp = response['response'].replace('**', '')
            self.textbox.insert("end", f"Llama 3.1: {clean_resp}\n\n")
        except Exception as e:
            self.textbox.insert("end", f"Error: {e}\n\n")
        finally:
            self.textbox.see("end")
            self.send_btn.configure(state="normal")


class ImageDescriberFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=10, fg_color="transparent")
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text="Image Describer (Llava)", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, padx=0, pady=(0, 20), sticky="w")

        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))

        self.btn_select = ctk.CTkButton(self.controls_frame, text="Select Image", command=self.select_image)
        self.btn_select.grid(row=0, column=0, padx=(0, 10))

        self.lbl_path = ctk.CTkLabel(self.controls_frame, text="No image selected")
        self.lbl_path.grid(row=0, column=1)

        self.textbox = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(size=14))
        self.textbox.grid(row=2, column=0, sticky="nsew")

        self.selected_path = None

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if path:
            self.selected_path = path
            self.lbl_path.configure(text=os.path.basename(path))
            self.textbox.insert("end", f"Selected: {path}\nDescribing...\n\n")
            threading.Thread(target=self.describe, daemon=True).start()

    def describe(self):
        try:
            response = ollama.chat(
                model='llava:7b',
                messages=[{
                    'role': 'user',
                    'content': 'Describe the image.',
                    'images': [self.selected_path]
                }]
            )
            clean_resp = response['message']['content'].replace('**', '')
            self.textbox.insert("end", f"Llava: {clean_resp}\n\n")
        except Exception as e:
            self.textbox.insert("end", f"Error: {e}\n\n")
        finally:
            self.textbox.see("end")


class VideoDescriberFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=10, fg_color="transparent")
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text="Video Describer (Frames -> Llava -> Llama 3.1)", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, padx=0, pady=(0, 20), sticky="w")

        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))

        self.btn_select = ctk.CTkButton(self.controls_frame, text="Select Video", command=self.select_video)
        self.btn_select.grid(row=0, column=0, padx=(0, 10))

        self.lbl_path = ctk.CTkLabel(self.controls_frame, text="No video selected")
        self.lbl_path.grid(row=0, column=1)

        self.textbox = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(size=14))
        self.textbox.grid(row=2, column=0, sticky="nsew")

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if path:
            self.lbl_path.configure(text=os.path.basename(path))
            self.textbox.insert("end", f"Selected: {path}\nExtracting frames...\n")
            threading.Thread(target=self.process_video, args=(path,), daemon=True).start()

    def process_video(self, video_path):
        self.btn_select.configure(state="disabled")
        try:
            # 1. Extract frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.textbox.insert("end", "Error: Could not open video.\n")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * 2) # Every 2 seconds
            
            frame_count = 0
            saved_frames = []
            
            temp_dir = tempfile.mkdtemp()

            while True:
                ret, frame = cap.read()
                if not ret: break
                if frame_count % frame_interval == 0:
                    filename = os.path.join(temp_dir, f"frame_{len(saved_frames)}.jpg")
                    cv2.imwrite(filename, frame)
                    saved_frames.append(filename)
                frame_count += 1
            cap.release()
            
            self.textbox.insert("end", f"Extracted {len(saved_frames)} frames. Analyzing with Llava...\n")
            
            # 2. Analyze frames
            descriptions = ""
            for i, frame_path in enumerate(saved_frames):
                self.textbox.insert("end", f"Analyzing frame {i+1}/{len(saved_frames)}...\n")
                self.textbox.see("end")
                response = ollama.chat(model='llava:7b', messages=[{
                    'role': 'user', 'content': 'Describe the image with one sentence.', 'images': [frame_path]
                }])
                clean_resp = response['message']['content'].replace('**', '')
                descriptions += f"\n{clean_resp}\n"
            
            self.textbox.insert("end", "Summarizing descriptions with Llama 3.1...\n")
            self.textbox.see("end")
            
            # 3. Summarize
            prompt = f"Write a general explaination about what is going on in the video by using the following descriptions of the frames of the video.\n{descriptions}"
            answer = ollama.generate(model='llama3.1', prompt=prompt)
            
            clean_summary = answer['response'].replace('**', '')
            self.textbox.insert("end", f"\n--- Final Summary ---\n{clean_summary}\n\n")
            
        except Exception as e:
            self.textbox.insert("end", f"Error: {e}\n\n")
        finally:
            self.textbox.see("end")
            self.btn_select.configure(state="normal")


class VideoChatFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=10, fg_color="transparent")
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text="Video Chat (Audio -> Whisper -> Llama 3.1)", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, padx=0, pady=(0, 20), sticky="w")

        if not WHISPER_AVAILABLE:
            self.lbl_err = ctk.CTkLabel(self, text="Error: whisper not installed or ffmpeg missing.", text_color="red")
            self.lbl_err.grid(row=1, column=0, sticky="w")
            return

        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))

        self.btn_select = ctk.CTkButton(self.controls_frame, text="Select Video", command=self.select_video)
        self.btn_select.grid(row=0, column=0, padx=(0, 10))

        self.lbl_path = ctk.CTkLabel(self.controls_frame, text="No video selected")
        self.lbl_path.grid(row=0, column=1)

        self.textbox = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(size=14))
        self.textbox.grid(row=2, column=0, sticky="nsew", pady=(0, 20))
        
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=3, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Ask a question about the video...", height=40, state="disabled")
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", lambda e: self.ask_question())
        
        self.send_btn = ctk.CTkButton(self.input_frame, text="Ask", command=self.ask_question, height=40, state="disabled")
        self.send_btn.grid(row=0, column=1)

        self.transcribed_text = ""
        self.whisper_model = None

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if path:
            self.lbl_path.configure(text=os.path.basename(path))
            self.textbox.insert("end", f"Selected: {path}\nLoading Whisper model (this may take a moment)...\n")
            self.btn_select.configure(state="disabled")
            threading.Thread(target=self.process_video, args=(path,), daemon=True).start()

    def process_video(self, video_path):
        try:
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model("base")
            
            self.textbox.insert("end", "Extracting audio...\n")
            self.textbox.see("end")
            
            audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            # Extract audio using ffmpeg directly to bypass pydub/audioop issue in Python 3.13+
            try:
                subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                if "Output file does not contain any stream" in e.stderr:
                    self.textbox.insert("end", "Notice: The selected video does not contain an audio track. Transcription will be empty.\n\n")
                    self.transcribed_text = "[No audio track found in the video]"
                    self.entry.configure(state="normal")
                    self.send_btn.configure(state="normal")
                else:
                    self.textbox.insert("end", f"Error extracting audio:\n{e.stderr}\n\n")
                return
            
            self.textbox.insert("end", "Transcribing audio...\n")
            self.textbox.see("end")
            
            result = self.whisper_model.transcribe(audio_path, fp16=False)
            self.transcribed_text = result['text']
            
            self.textbox.insert("end", f"\n--- Transcription ---\n{self.transcribed_text}\n\n")
            self.textbox.insert("end", "You can now ask questions about the transcription below.\n\n")
            
            self.entry.configure(state="normal")
            self.send_btn.configure(state="normal")
            
        except Exception as e:
            self.textbox.insert("end", f"Error: {e}\n\n")
        finally:
            self.textbox.see("end")
            self.btn_select.configure(state="normal")

    def ask_question(self):
        prompt = self.entry.get()
        if prompt and self.transcribed_text:
            self.textbox.insert("end", f"You: {prompt}\n\n")
            self.entry.delete(0, "end")
            threading.Thread(target=self.generate_response, args=(prompt,), daemon=True).start()

    def generate_response(self, prompt):
        try:
            combined_prompt = f"Based on the following content: {self.transcribed_text}\n\nQuestion: {prompt}"
            response = ollama.generate(model='llama3.1', prompt=combined_prompt)
            clean_resp = response['response'].replace('**', '')
            self.textbox.insert("end", f"Llama 3.1: {clean_resp}\n\n")
        except Exception as e:
            self.textbox.insert("end", f"Error: {e}\n\n")
        finally:
            self.textbox.see("end")


class PDFChatFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=10, fg_color="transparent")
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text="Chat with PDF (PyMuPDF -> Llama 3.1)", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, padx=0, pady=(0, 20), sticky="w")

        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))

        self.btn_select = ctk.CTkButton(self.controls_frame, text="Select PDF", command=self.select_pdf)
        self.btn_select.grid(row=0, column=0, padx=(0, 10))

        self.lbl_path = ctk.CTkLabel(self.controls_frame, text="No PDF selected")
        self.lbl_path.grid(row=0, column=1)

        self.textbox = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(size=14))
        self.textbox.grid(row=2, column=0, sticky="nsew", pady=(0, 20))
        
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=3, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Ask a question about the PDF...", height=40, state="disabled")
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", lambda e: self.ask_question())
        
        self.send_btn = ctk.CTkButton(self.input_frame, text="Ask", command=self.ask_question, height=40, state="disabled")
        self.send_btn.grid(row=0, column=1)

        self.pdf_text = ""

    def select_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if path:
            self.lbl_path.configure(text=os.path.basename(path))
            self.btn_select.configure(state="disabled")
            threading.Thread(target=self.process_pdf, args=(path,), daemon=True).start()

    def process_pdf(self, path):
        try:
            self.textbox.insert("end", f"Extracting text from {os.path.basename(path)}...\n")
            doc = fitz.open(path)
            self.pdf_text = ""
            for page in doc:
                self.pdf_text += page.get_text()
            
            self.textbox.insert("end", "Extraction complete. You can now ask questions.\n\n")
            self.entry.configure(state="normal")
            self.send_btn.configure(state="normal")
            
        except Exception as e:
            self.textbox.insert("end", f"Error: {e}\n\n")
        finally:
            self.textbox.see("end")
            self.btn_select.configure(state="normal")

    def ask_question(self):
        prompt = self.entry.get()
        if prompt and self.pdf_text:
            self.textbox.insert("end", f"You: {prompt}\n\n")
            self.entry.delete(0, "end")
            threading.Thread(target=self.generate_response, args=(prompt,), daemon=True).start()

    def generate_response(self, prompt):
        try:
            combined_prompt = f"Based on the following content: {self.pdf_text}\n\nQuestion: {prompt}"
            response = ollama.generate(model='llama3.1', prompt=combined_prompt)
            clean_resp = response['response'].replace('**', '')
            self.textbox.insert("end", f"Llama 3.1: {clean_resp}\n\n")
        except Exception as e:
            self.textbox.insert("end", f"Error: {e}\n\n")
        finally:
            self.textbox.see("end")


class CodingTutorFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=10, fg_color="transparent")
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text="Auto-correcting Coding Tutor", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, padx=0, pady=(0, 20), sticky="w")

        # Top input
        self.tutor_input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.tutor_input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        self.tutor_input_frame.grid_columnconfigure(0, weight=1)

        self.tutor_entry = ctk.CTkEntry(self.tutor_input_frame, placeholder_text="Ask me to write Python code (e.g. 'Write a function to calculate fibonacci')...", height=40)
        self.tutor_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        
        self.btn_generate = ctk.CTkButton(self.tutor_input_frame, text="Generate & Run", command=self.generate_code, height=40)
        self.btn_generate.grid(row=0, column=1)
        
        self.btn_run_again = ctk.CTkButton(self.tutor_input_frame, text="Run Code Again", command=self.run_working_code, height=40, state="disabled")
        self.btn_run_again.grid(row=0, column=2, padx=(10, 0))

        # Main text area - Split into two panes
        self.split_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.split_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 20))
        self.split_frame.grid_columnconfigure(0, weight=1)
        self.split_frame.grid_columnconfigure(1, weight=1)
        self.split_frame.grid_rowconfigure(0, weight=1)

        self.chat_textbox = ctk.CTkTextbox(self.split_frame, wrap="word", font=ctk.CTkFont(family="Segoe UI", size=14))
        self.chat_textbox.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.code_textbox = ctk.CTkTextbox(self.split_frame, wrap="none", font=ctk.CTkFont(family="Consolas", size=14), fg_color="#1E1E1E", text_color="#D4D4D4")
        self.code_textbox.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Bottom Q&A input
        self.qa_input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.qa_input_frame.grid(row=3, column=0, sticky="ew")
        self.qa_input_frame.grid_columnconfigure(0, weight=1)

        self.qa_entry = ctk.CTkEntry(self.qa_input_frame, placeholder_text="Ask a question about the generated code...", height=40, state="disabled")
        self.qa_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.qa_entry.bind("<Return>", lambda e: self.ask_question())
        
        self.btn_ask = ctk.CTkButton(self.qa_input_frame, text="Ask", command=self.ask_question, height=40, state="disabled")
        self.btn_ask.grid(row=0, column=1)

        self.working_code = ""

    def generate_code(self):
        prompt = self.tutor_entry.get()
        if prompt:
            self.chat_textbox.insert("end", f"You requested: {prompt}\nGenerating and testing code...\n\n")
            self.btn_generate.configure(state="disabled")
            self.qa_entry.configure(state="disabled")
            self.btn_ask.configure(state="disabled")
            threading.Thread(target=self.run_tutor_loop, args=(prompt,), daemon=True).start()

    def run_tutor_loop(self, initial_prompt):
        prompt = initial_prompt + "\nNote: Output must have only Python code, do not add any other explanation."
        try:
            while True:
                response = ollama.generate(model='llama3.1', prompt=prompt)
                code_raw = response["response"]
                code = code_raw.replace("python", "").replace("```", "").strip()
                
                self.code_textbox.delete("0.0", "end")
                self.code_textbox.insert("end", "# Generated Code:\n" + code + "\n\n")
                self.chat_textbox.insert("end", "Code generated. Compiling and running...\n")
                self.chat_textbox.see("end")
                
                # Execution Retry Loop
                retry_execution = True
                import re
                while retry_execution:
                    retry_execution = False
                    
                    # Separate Compiler Process
                    fd, temp_path = tempfile.mkstemp(suffix=".py")
                    with os.fdopen(fd, 'w') as f:
                        f.write(code)
                    
                    exec_error = False
                    output = ""
                    try:
                        proc = subprocess.Popen([sys.executable, temp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        try:
                            # Wait for 3 seconds. If it's a simple script, it will finish.
                            # If it's a game/GUI, it will keep running and raise TimeoutExpired.
                            stdout_data, stderr_data = proc.communicate(timeout=3)
                            output = stdout_data
                            if proc.returncode != 0:
                                output += "\n" + stderr_data
                                exec_error = True
                        except subprocess.TimeoutExpired:
                            output = "Code started successfully and is running in the background (GUI/Game detected)."
                            exec_error = False
                            # We leave the process running so the user can interact with the game window.
                    except Exception as e:
                        output = str(e)
                        exec_error = True
                        
                    # Auto-install missing libraries
                    if exec_error and "ModuleNotFoundError: No module named" in output:
                        match = re.search(r"ModuleNotFoundError: No module named '(\w+)'", output)
                        if match:
                            module_name = match.group(1)
                            # Simple map for common libraries that have different package names
                            pip_name = module_name
                            if module_name == 'cv2': pip_name = 'opencv-python'
                            elif module_name == 'PIL': pip_name = 'Pillow'
                            elif module_name == 'sklearn': pip_name = 'scikit-learn'
                            
                            self.chat_textbox.insert("end", f"Missing library '{module_name}' detected. Auto-installing via pip...\n")
                            self.chat_textbox.see("end")
                            pip_result = subprocess.run([sys.executable, "-m", "pip", "install", pip_name], capture_output=True, text=True)
                            
                            if pip_result.returncode == 0:
                                self.chat_textbox.insert("end", f"Successfully installed '{pip_name}'. Retrying code execution...\n")
                                self.chat_textbox.see("end")
                                retry_execution = True
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                                continue
                            else:
                                output += f"\n\nFailed to auto-install {pip_name}:\n{pip_result.stderr}"

                    # Some compilation errors might be in standard output but return code is 0 (rare but possible)
                    if not exec_error and ("Traceback (most recent call last):" in output):
                        exec_error = True

                    if exec_error:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                        
                if exec_error:
                    self.chat_textbox.insert("end", f"--- Compiler Error ---\n{output}\nCode failed. Sending error back to LLM for fix...\n\n")
                    self.chat_textbox.see("end")
                    prompt = f"The following code has this error:\n{output}\nCode:\n{code}\nFix the code. Do NOT use os.system to install libraries. Note: Output must have only Python code, do not add any other explanation."
                else:
                    self.chat_textbox.insert("end", f"--- Compiler Output ---\n{output}\nSuccess! Code compiled and ran flawlessly.\n\n")
                    self.chat_textbox.see("end")
                    self.working_code = code
                    
                    self.qa_entry.configure(state="normal")
                    self.btn_ask.configure(state="normal")
                    self.btn_run_again.configure(state="normal")
                    break
        except Exception as e:
            self.chat_textbox.insert("end", f"Fatal Error in loop: {e}\n\n")
        finally:
            self.btn_generate.configure(state="normal")
            self.chat_textbox.see("end")

    def run_working_code(self):
        if self.working_code:
            self.chat_textbox.insert("end", "Re-running working code manually...\n")
            self.chat_textbox.see("end")
            self.btn_run_again.configure(state="disabled")
            threading.Thread(target=self._run_working_code_thread, daemon=True).start()
            
    def _run_working_code_thread(self):
        try:
            import tempfile, os, subprocess, sys, re
            code = self.code_textbox.get("0.0", "end").strip()
            code = re.sub(r"^# Generated Code:\n", "", code).strip()
            
            retry = True
            while retry:
                retry = False
                fd, temp_path = tempfile.mkstemp(suffix=".py")
                with os.fdopen(fd, 'w') as f:
                    f.write(code)
                    
                exec_error = False
                output = ""
                try:
                    proc = subprocess.Popen([sys.executable, temp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        stdout_data, stderr_data = proc.communicate(timeout=3)
                        output = stdout_data
                        if proc.returncode != 0:
                            output += "\n" + stderr_data
                            exec_error = True
                    except subprocess.TimeoutExpired:
                        output = "Code ran successfully and is running in the background."
                        exec_error = False
                except Exception as e:
                    output = str(e)
                    exec_error = True

                if exec_error and "ModuleNotFoundError: No module named" in output:
                    match = re.search(r"ModuleNotFoundError: No module named '(\w+)'", output)
                    if match:
                        module_name = match.group(1)
                        pip_name = module_name
                        if module_name == 'cv2': pip_name = 'opencv-python'
                        elif module_name == 'PIL': pip_name = 'Pillow'
                        elif module_name == 'sklearn': pip_name = 'scikit-learn'
                        
                        self.chat_textbox.insert("end", f"Missing library '{module_name}' detected. Auto-installing...\n")
                        self.chat_textbox.see("end")
                        pip_result = subprocess.run([sys.executable, "-m", "pip", "install", pip_name], capture_output=True, text=True)
                        if pip_result.returncode == 0:
                            self.chat_textbox.insert("end", f"Installed '{pip_name}'. Retrying execution...\n")
                            retry = True
                            try: os.remove(temp_path)
                            except: pass
                            continue
                        else:
                            output += f"\nFailed to install {pip_name}:\n{pip_result.stderr}"

                try: os.remove(temp_path)
                except: pass

                if exec_error:
                    self.chat_textbox.insert("end", f"Manual Run Error:\n{output}\n\n")
                else:
                    self.chat_textbox.insert("end", f"Manual Run Output:\n{output}\n\n")
                self.chat_textbox.see("end")

        except Exception as e:
            self.chat_textbox.insert("end", f"Error running code: {e}\n\n")
            self.chat_textbox.see("end")
        finally:
            self.btn_run_again.configure(state="normal")

    def ask_question(self):
        question = self.qa_entry.get()
        if question and self.working_code:
            self.chat_textbox.insert("end", f"Q: {question}\n")
            self.qa_entry.delete(0, "end")
            threading.Thread(target=self.answer_question, args=(question,), daemon=True).start()

    def answer_question(self, question):
        try:
            prompt = f"Can you explain why we used this code \n\"{question}\"\nin the following code in very detail step by step:\n{self.working_code}"
            answer = ollama.generate(model='gemma2:9b', prompt=prompt)
            clean_resp = answer['response'].replace('**', '')
            self.chat_textbox.insert("end", f"A: {clean_resp}\n\n")
        except Exception as e:
            self.chat_textbox.insert("end", f"Error: {e}\n\n")
        finally:
            self.chat_textbox.see("end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
