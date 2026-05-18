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
import threading
import json
import time
from datetime import datetime
from tkinter import messagebox

import socket
import subprocess

# Using the official 'ollama' package which is stable, instead of langchain wrapper
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

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

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

nlp = None
if SPACY_AVAILABLE:
    try:
        nlp = spacy.load("en_core_web_md")
    except Exception:
        nlp = None

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Utility to mimic LangChain's RecursiveCharacterTextSplitter
def split_text_with_overlap(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Workflows & Semantic Search")
        self.geometry("1200x800")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI Features", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 20))

        buttons = [
            ("1. Basic Prompt", "basic"),
            ("2. Template Chain", "template"),
            ("3. Multi-Input", "multi"),
            ("4. Chunking Summarizer", "summary"),
            ("5. Semantic Similarity", "similarity"),
            ("6. Semantic Search", "search"),
            ("7. RAG (Long Text)", "rag"),
            ("8. Chat with Notes", "notes"),
            ("9. Chat with Diary", "diary")
        ]

        self.nav_buttons = {}
        for i, (text, name) in enumerate(buttons):
            btn = ctk.CTkButton(self.sidebar_frame, text=text, anchor="w", command=lambda n=name: self.select_frame(n))
            btn.grid(row=i+1, column=0, padx=20, pady=5, sticky="ew")
            self.nav_buttons[name] = btn

        # Connection Status Indicator
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Ollama: Connecting...", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=11, column=0, padx=20, pady=10, sticky="s")
        threading.Thread(target=self.check_ollama_status, daemon=True).start()

        # Frames Dictionary
        self.frames = {
            "basic": BasicPromptFrame(self),
            "template": TemplateChainFrame(self),
            "multi": MultiInputFrame(self),
            "summary": SummarizerFrame(self),
            "similarity": SemanticSimilarityFrame(self),
            "search": SemanticSearchFrame(self),
            "rag": RAGLongTextFrame(self),
            "notes": ChatWithNotesFrame(self),
            "diary": ChatWithDiaryFrame(self)
        }

        self.select_frame("basic")

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


class BaseFeatureFrame(ctk.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master, corner_radius=10, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.title_lbl = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=24, weight="bold"))
        self.title_lbl.grid(row=0, column=0, pady=(0, 20), sticky="w")
        
        if not OLLAMA_AVAILABLE:
            ctk.CTkLabel(self, text="Error: ollama python package is not installed.", text_color="red").grid(row=1, column=0, sticky="w")
        if not SPACY_AVAILABLE or nlp is None:
            ctk.CTkLabel(self, text="Warning: SpaCy 'en_core_web_md' is not loaded. Semantic search will fail.", text_color="orange").grid(row=1, column=0, sticky="w")


class BasicPromptFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "1. Basic Prompting")
        
        self.input_text = ctk.CTkTextbox(self, height=100)
        self.input_text.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.input_text.insert("0.0", "Write your prompt here...")

        self.btn_run = ctk.CTkButton(self, text="Run", command=self.run_logic)
        self.btn_run.grid(row=2, column=0, sticky="w", pady=(0, 10))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=3, column=0, sticky="nsew")

    def run_logic(self):
        prompt = self.input_text.get("0.0", "end").strip()
        if prompt:
            self.output_text.insert("end", f"You: {prompt}\nProcessing...\n\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(prompt,), daemon=True).start()

    def _process(self, prompt):
        try:
            response = ollama.generate(model='llama3.1', prompt=prompt)
            clean_resp = response['response'].replace('**', '')
            self.output_text.insert("end", f"LLM:\n{clean_resp}\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_run.configure(state="normal")
            self.output_text.see("end")

class TemplateChainFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "2. PromptTemplate Logic")

        self.input_text = ctk.CTkTextbox(self, height=100)
        self.input_text.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.input_text.insert("0.0", "Ask a question (it will be formatted as a list)...")

        self.btn_run = ctk.CTkButton(self, text="Run Template", command=self.run_logic)
        self.btn_run.grid(row=2, column=0, sticky="w", pady=(0, 10))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=3, column=0, sticky="nsew")

    def run_logic(self):
        question = self.input_text.get("0.0", "end").strip()
        if question:
            self.output_text.insert("end", f"Question: {question}\nRunning template...\n\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(question,), daemon=True).start()

    def _process(self, question):
        try:
            prompt = f"You are a helpful assistant. You have been asked the following question:\n\nQuestion: {question}\n\nPlease provide a detailed and thoughtful response as a list with short items."
            response = ollama.generate(model='llama3.1', prompt=prompt)
            clean_resp = response['response'].replace('**', '')
            self.output_text.insert("end", f"Response:\n{clean_resp}\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_run.configure(state="normal")
            self.output_text.see("end")

class MultiInputFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "3. Multi-Input Variables")

        self.name_entry = ctk.CTkEntry(self, placeholder_text="Your Name")
        self.name_entry.grid(row=1, column=0, sticky="ew", pady=5)
        
        self.topic_entry = ctk.CTkEntry(self, placeholder_text="Topic")
        self.topic_entry.grid(row=2, column=0, sticky="ew", pady=5)
        
        self.question_entry = ctk.CTkEntry(self, placeholder_text="Question")
        self.question_entry.grid(row=3, column=0, sticky="ew", pady=5)

        self.inst_entry = ctk.CTkEntry(self, placeholder_text="Instructions")
        self.inst_entry.grid(row=4, column=0, sticky="ew", pady=5)

        self.btn_run = ctk.CTkButton(self, text="Run", command=self.run_logic)
        self.btn_run.grid(row=5, column=0, sticky="w", pady=(10, 10))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=6, column=0, sticky="nsew")
        self.grid_rowconfigure(6, weight=1)

    def run_logic(self):
        name = self.name_entry.get().strip()
        topic = self.topic_entry.get().strip()
        question = self.question_entry.get().strip()
        instructions = self.inst_entry.get().strip()
        
        if name and topic and question and instructions:
            self.output_text.insert("end", f"Processing multi-input prompt...\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(name, topic, question, instructions), daemon=True).start()
        else:
            self.output_text.insert("end", "Please fill all fields.\n")

    def _process(self, name, topic, question, instructions):
        try:
            prompt = f"You are a helpful assistant. {name} has asked a question about {topic}.\n\nQuestion: {question}\n\nAdditional Instructions: {instructions}\n\nPlease provide a detailed and thoughtful response."
            response = ollama.generate(model='llama3.1', prompt=prompt)
            clean_resp = response['response'].replace('**', '')
            self.output_text.insert("end", f"Response:\n{clean_resp}\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_run.configure(state="normal")
            self.output_text.see("end")

class SummarizerFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "4. Text Summarizer w/ Chunking")

        self.input_text = ctk.CTkTextbox(self, height=150)
        self.input_text.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.input_text.insert("0.0", "Paste long text here...")

        self.btn_run = ctk.CTkButton(self, text="Summarize", command=self.run_logic)
        self.btn_run.grid(row=2, column=0, sticky="w", pady=(0, 10))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=3, column=0, sticky="nsew")

    def run_logic(self):
        text = self.input_text.get("0.0", "end").strip()
        if text:
            self.output_text.insert("end", "Starting chunked summarization...\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(text,), daemon=True).start()

    def _process(self, text):
        try:
            chunks = split_text_with_overlap(text, chunk_size=1000, overlap=150)
            
            summaries = []
            for i, chunk in enumerate(chunks):
                self.output_text.insert("end", f"Summarizing chunk {i+1}/{len(chunks)}...\n")
                self.output_text.see("end")
                
                prompt = f"You are a helpful assistant. Please summarize the following text:\n\nText: {chunk}\n\nProvide a concise summary."
                resp = ollama.generate(model='llama3.1', prompt=prompt)
                summaries.append(resp['response'].replace('**', ''))
            
            final_summary = "\n\n".join(summaries)
            self.output_text.insert("end", f"\nGenerating combined final summary...\n")
            self.output_text.see("end")
            
            combined_prompt = f"Combine the following summaries into one concise summary: \n\n{final_summary}"
            combined = ollama.generate(model='llama3.1', prompt=combined_prompt)
            clean_combined = combined['response'].replace('**', '')
            self.output_text.insert("end", f"\n=== Final Summary ===\n{clean_combined}\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_run.configure(state="normal")
            self.output_text.see("end")

class SemanticSimilarityFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "5. Semantic Similarity")

        self.text1 = ctk.CTkTextbox(self, height=100)
        self.text1.grid(row=1, column=0, sticky="ew", pady=5)
        self.text1.insert("0.0", "Text 1...")

        self.text2 = ctk.CTkTextbox(self, height=100)
        self.text2.grid(row=2, column=0, sticky="ew", pady=5)
        self.text2.insert("0.0", "Text 2...")

        self.btn_run = ctk.CTkButton(self, text="Compare", command=self.run_logic)
        self.btn_run.grid(row=3, column=0, sticky="w", pady=(5, 10))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=4, column=0, sticky="nsew")

    def run_logic(self):
        t1 = self.text1.get("0.0", "end").strip()
        t2 = self.text2.get("0.0", "end").strip()
        if t1 and t2:
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(t1, t2), daemon=True).start()

    def _process(self, t1, t2):
        if not nlp:
            self.output_text.insert("end", "Error: SpaCy model 'en_core_web_md' is not loaded. Run: python -m spacy download en_core_web_md\n")
            self.btn_run.configure(state="normal")
            return
        try:
            start = time.time()
            sim = nlp(t1).similarity(nlp(t2))
            end = time.time()
            self.output_text.insert("end", f"Similarity: {sim:.4f} (Calculated in {end-start:.4f}s)\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_run.configure(state="normal")
            self.output_text.see("end")

class SemanticSearchFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "6. Semantic Search")

        self.long_text = ctk.CTkTextbox(self, height=150)
        self.long_text.grid(row=1, column=0, sticky="ew", pady=5)
        self.long_text.insert("0.0", "Database / Long Text...")

        self.query = ctk.CTkEntry(self, placeholder_text="Query / Reference Sentence")
        self.query.grid(row=2, column=0, sticky="ew", pady=5)

        self.btn_run = ctk.CTkButton(self, text="Search", command=self.run_logic)
        self.btn_run.grid(row=3, column=0, sticky="w", pady=(5, 10))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=4, column=0, sticky="nsew")

    def run_logic(self):
        text = self.long_text.get("0.0", "end").strip()
        query = self.query.get().strip()
        if text and query:
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(text, query), daemon=True).start()

    def _process(self, text, query):
        if not nlp:
            self.output_text.insert("end", "Error: SpaCy model not loaded.\n")
            self.btn_run.configure(state="normal")
            return
        try:
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
            
            similarities = []
            q_vec = nlp(query)
            for s in sentences:
                sim = q_vec.similarity(nlp(s))
                similarities.append((sim, s))
            
            ordered = sorted(similarities, key=lambda x: x[0], reverse=True)
            self.output_text.insert("end", f"Query: {query}\n\n")
            for score, s in ordered:
                self.output_text.insert("end", f"[{score:.4f}] {s}\n")
            self.output_text.insert("end", "\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_run.configure(state="normal")
            self.output_text.see("end")

class RAGLongTextFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "7. RAG on Long Text")

        self.long_text = ctk.CTkTextbox(self, height=150)
        self.long_text.grid(row=1, column=0, sticky="ew", pady=5)
        self.long_text.insert("0.0", "Paste source material here...")

        self.query = ctk.CTkEntry(self, placeholder_text="Question")
        self.query.grid(row=2, column=0, sticky="ew", pady=5)

        self.btn_run = ctk.CTkButton(self, text="Answer", command=self.run_logic)
        self.btn_run.grid(row=3, column=0, sticky="w", pady=(5, 10))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=4, column=0, sticky="nsew")

    def run_logic(self):
        text = self.long_text.get("0.0", "end").strip()
        query = self.query.get().strip()
        if text and query:
            self.btn_run.configure(state="disabled")
            self.output_text.insert("end", "Analyzing text and searching for context...\n")
            threading.Thread(target=self._process, args=(text, query), daemon=True).start()

    def _process(self, text, query):
        if not nlp:
            self.output_text.insert("end", "Error: SpaCy model not loaded.\n")
            self.btn_run.configure(state="normal")
            return
        try:
            chunks = split_text_with_overlap(text, chunk_size=1000, overlap=150)
            
            similarities = []
            q_vec = nlp(query)
            for chunk in chunks:
                sim = q_vec.similarity(nlp(chunk))
                similarities.append((sim, chunk))
                
            ordered = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
            
            context = ""
            for score, sent in ordered:
                self.output_text.insert("end", f"Retrieved (score {score:.2f}): {sent[:50]}...\n")
                context += f"{sent}\n\n"
                
            self.output_text.insert("end", "\nGenerating answer...\n")
            self.output_text.see("end")
            
            prompt = f"You are a helpful assistant. Please answer the following question based on the below text:\n\nQuestion: {query}\n\nText: {context}"
            answer = ollama.generate(model='llama3.1', prompt=prompt)
            clean_resp = answer['response'].replace('**', '')
            self.output_text.insert("end", f"\nAnswer:\n{clean_resp}\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_run.configure(state="normal")
            self.output_text.see("end")

class ChatWithNotesFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "8. Chat with Notes")
        self.file_path = "note.text"
        
        # Save note UI
        self.save_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.save_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.save_frame.grid_columnconfigure(0, weight=1)
        
        self.note_text = ctk.CTkTextbox(self.save_frame, height=100)
        self.note_text.grid(row=0, column=0, sticky="ew")
        
        self.btn_save = ctk.CTkButton(self.save_frame, text="Save Note", command=self.save_note)
        self.btn_save.grid(row=1, column=0, sticky="w", pady=(5,0))

        # Ask UI
        self.ask_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ask_frame.grid(row=2, column=0, sticky="ew", pady=(10, 10))
        self.ask_frame.grid_columnconfigure(0, weight=1)
        
        self.query = ctk.CTkEntry(self.ask_frame, placeholder_text="Ask a question about your notes...")
        self.query.grid(row=0, column=0, sticky="ew")
        
        self.btn_ask = ctk.CTkButton(self.ask_frame, text="Ask", command=self.ask_logic)
        self.btn_ask.grid(row=0, column=1, padx=(5,0))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=3, column=0, sticky="nsew")

    def save_note(self):
        note = self.note_text.get("0.0", "end").strip()
        if note:
            if not os.path.exists(self.file_path):
                open(self.file_path, 'w').close()
            with open(self.file_path, 'r') as f:
                content = f.read()
            if note not in content:
                content += f"\n\n{note}"
                with open(self.file_path, 'w') as f:
                    f.write(content.strip())
                self.output_text.insert("end", "Note saved successfully.\n")
            else:
                self.output_text.insert("end", "Note already exists.\n")
            self.note_text.delete("0.0", "end")

    def ask_logic(self):
        query = self.query.get().strip()
        if query and os.path.exists(self.file_path):
            self.btn_ask.configure(state="disabled")
            self.output_text.insert("end", f"\nQ: {query}\nSearching notes...\n")
            threading.Thread(target=self._process, args=(query,), daemon=True).start()

    def _process(self, query):
        if not nlp:
            self.output_text.insert("end", "Error: SpaCy model not loaded.\n")
            self.btn_ask.configure(state="normal")
            return
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
                
            chunks = split_text_with_overlap(content, chunk_size=1000, overlap=150)
            
            similarities = []
            q_vec = nlp(query)
            for chunk in chunks:
                sim = q_vec.similarity(nlp(chunk))
                similarities.append((sim, chunk))
                
            ordered = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
            
            context = ""
            for score, sent in ordered:
                context += f"{sent}\n\n"
                
            prompt = f"You are a helpful assistant. Please answer the following question based on the below text:\n\nQuestion: {query}\n\nText: {context}"
            answer = ollama.generate(model='llama3.1', prompt=prompt)
            clean_resp = answer['response'].replace('**', '')
            self.output_text.insert("end", f"Answer:\n{clean_resp}\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_ask.configure(state="normal")
            self.output_text.see("end")

class ChatWithDiaryFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "9. Chat with Diary")
        self.file_path = "diary.json"
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({}, f)
        
        try:
            import tkcalendar
            TKCALENDAR_AVAILABLE = True
        except ImportError:
            TKCALENDAR_AVAILABLE = False
            
        if not TKCALENDAR_AVAILABLE:
            ctk.CTkLabel(self, text="Error: tkcalendar package is not installed.", text_color="red").grid(row=1, column=0, sticky="w", pady=10)
            ctk.CTkLabel(self, text="Please install it using: pip install tkcalendar", text_color="gray").grid(row=2, column=0, sticky="w", pady=10)
            return

        # Save diary UI
        self.save_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.save_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.save_frame.grid_columnconfigure(1, weight=1)
        
        self.date_entry = tkcalendar.DateEntry(self.save_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.date_entry.grid(row=0, column=0, sticky="w", padx=(0,5))
        
        self.note_text = ctk.CTkTextbox(self.save_frame, height=80)
        self.note_text.grid(row=0, column=1, sticky="ew")
        
        self.btn_save = ctk.CTkButton(self.save_frame, text="Save Entry", command=self.save_diary)
        self.btn_save.grid(row=1, column=1, sticky="w", pady=(5,0))

        # Ask UI
        self.ask_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ask_frame.grid(row=2, column=0, sticky="ew", pady=(10, 10))
        self.ask_frame.grid_columnconfigure(0, weight=1)
        
        self.query = ctk.CTkEntry(self.ask_frame, placeholder_text="Ask a question about your diary...")
        self.query.grid(row=0, column=0, sticky="ew")
        
        self.btn_ask = ctk.CTkButton(self.ask_frame, text="Ask", command=self.ask_logic)
        self.btn_ask.grid(row=0, column=1, padx=(5,0))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=3, column=0, sticky="nsew")

    def save_diary(self):
        date_str = self.date_entry.get().strip()
        note = self.note_text.get("0.0", "end").strip()
        if date_str and note:
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                
                if date_str in data:
                    if note not in data[date_str]:
                        data[date_str] += f"\n\n{note}"
                else:
                    data[date_str] = note
                    
                with open(self.file_path, 'w') as f:
                    json.dump(data, f)
                self.output_text.insert("end", f"Saved entry for {date_str}.\n")
                self.note_text.delete("0.0", "end")
            except Exception as e:
                self.output_text.insert("end", f"Error saving: {e}\n")

    def ask_logic(self):
        query = self.query.get().strip()
        if query:
            self.btn_ask.configure(state="disabled")
            self.output_text.insert("end", f"\nQ: {query}\nSearching diary...\n")
            threading.Thread(target=self._process, args=(query,), daemon=True).start()

    def _process(self, query):
        if not nlp:
            self.output_text.insert("end", "Error: SpaCy model not loaded.\n")
            self.btn_ask.configure(state="normal")
            return
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                
            similarities = []
            q_vec = nlp(query)
            for date_key, diary in data.items():
                sim = q_vec.similarity(nlp(diary))
                similarities.append((sim, f"Date: {date_key}\nDiary: {diary}"))
                
            ordered = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
            
            context = ""
            for score, sent in ordered:
                context += f"{sent}\n\n"
                
            prompt = f"You are a helpful assistant. Please answer the following question based on the below diary:\n\nQuestion: {query}\n\nDiary: {context}"
            answer = ollama.generate(model='llama3.1', prompt=prompt)
            clean_resp = answer['response'].replace('**', '')
            self.output_text.insert("end", f"Answer:\n{clean_resp}\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_ask.configure(state="normal")
            self.output_text.see("end")

if __name__ == "__main__":
    app = App()
    app.mainloop()
