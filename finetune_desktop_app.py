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
import subprocess
import requests
import time
import socket

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

from dataset_gen import generate_dataset, split_train_val

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAL_DATA_PATH = os.path.join(SCRIPT_DIR, "personal_data.json")
PERSONAL_VAL_PATH = os.path.join(SCRIPT_DIR, "personal_data_val.json")
ADAPTER_DIR = os.path.join(SCRIPT_DIR, "personal_model")

# Attempt to use the local 'myenv' virtual environment Python if available
myenv_python = os.path.join(SCRIPT_DIR, "myenv", "Scripts", "python.exe")
PYTHON_EXE = myenv_python if os.path.exists(myenv_python) else sys.executable

SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


def adapter_ready():
    """True if 38_finetune.py produced LoRA adapters."""
    return os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json"))

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("LLM Fine-Tuning Studio (37-46)")
        self.geometry("1200x800")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(sidebar, text="Fine-Tune Studio", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 20))

        nav = [
            ("1. Generate Data (Llama 3.1)", "data"),
            ("2. Train Model (38_finetune)", "train"),
            ("3. Test Model (39_test)", "test"),
            ("4. Export to Ollama (46.py)", "export")
        ]
        
        self.frames = {}
        frame_classes = [DataGenFrame, TrainFrame, TestFrame, ExportFrame]
        
        for i, ((label, key), cls) in enumerate(zip(nav, frame_classes)):
            btn = ctk.CTkButton(sidebar, text=label, anchor="w", command=lambda k=key: self.show(k))
            btn.grid(row=i+1, column=0, padx=15, pady=5, sticky="ew")
            self.frames[key] = cls(self)

        # Connection Status Indicator
        self.status_label = ctk.CTkLabel(sidebar, text="Ollama: Connecting...", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=6, column=0, padx=20, pady=10, sticky="s")
        threading.Thread(target=self.check_ollama_status, daemon=True).start()

        self.show("data")

    def check_ollama_status(self):
        running = ensure_ollama_running()
        if running:
            self.status_label.configure(text="● Ollama: Connected", text_color="#4CAF50")
        else:
            self.status_label.configure(text="● Ollama: Offline", text_color="#F44336")

    def show(self, key):
        for f in self.frames.values():
            f.grid_forget()
        self.frames[key].grid(row=0, column=1, sticky="nsew", padx=20, pady=20)


class BaseProcessFrame(ctk.CTkFrame):
    """A reusable frame that runs a long-running python script and streams its output to the UI."""
    def __init__(self, master, title, desc):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
        ctk.CTkLabel(self, text=desc, text_color="gray").grid(row=1, column=0, sticky="w", pady=(0, 15))

        self.btn = ctk.CTkButton(self, text="Start Process", command=self._start)
        self.btn.grid(row=2, column=0, sticky="w", pady=(0, 10))

        self.out = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(family="Consolas", size=12))
        self.out.grid(row=3, column=0, sticky="nsew")
        
        self.process_target = None  # Override in subclasses
        self._busy = False

    def preflight(self):
        """Return an error string to block start, or None if OK."""
        return None

    def log(self, msg):
        """Thread-safe append to the output textbox."""
        if threading.current_thread() is threading.main_thread():
            self._log_impl(msg)
        else:
            self.after(0, lambda m=msg: self._log_impl(m))

    def _log_impl(self, msg):
        self.out.insert("end", msg)
        self.out.see("end")

    def _start(self):
        if self._busy:
            return
        err = self.preflight()
        self.out.delete("0.0", "end")
        if err:
            self.log(err)
            return
        self._busy = True
        self.btn.configure(state="disabled")
        threading.Thread(target=self._run_process, daemon=True).start()

    def _finish(self):
        self._busy = False
        self.after(0, lambda: self.btn.configure(state="normal"))

    def _run_python_script(self, script_name, intro=""):
        """Run a script with unbuffered stdout so the UI updates live."""
        cmd = [PYTHON_EXE, "-u", script_name]
        if intro:
            self.log(intro)
        self.log(f"Executing: {' '.join(cmd)}\n{'-' * 60}\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=SCRIPT_DIR,
            env=SUBPROCESS_ENV,
        )
        assert process.stdout is not None
        for line in process.stdout:
            self.log(line)
        process.wait()
        self.log(f"\n{'-' * 60}\nProcess finished with exit code {process.returncode}\n")
        return process.returncode

    def _run_process(self):
        try:
            if callable(self.process_target):
                self.process_target()
            elif isinstance(self.process_target, str):
                self._run_python_script(self.process_target)
        except Exception as e:
            self.log(f"\n\nERROR: {str(e)}\n")
        finally:
            self._finish()


# =====================================================================
# 1. Generate Dataset Frame
# =====================================================================
class DataGenFrame(BaseProcessFrame):
    def __init__(self, master):
        super().__init__(
            master,
            "1. Generate Dataset",
            "Llama 3.1 builds diverse Q&A in batches, filters weak rows, shuffled train/val split. "
            "More pairs + optional notes = better fine-tuning.",
        )
        
        # Override the top section to add inputs
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.input_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.input_frame, text="Domain/Topic:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.topic = ctk.CTkEntry(
            self.input_frame, width=250, placeholder_text="e.g. Medical diagnosis for heart surgery"
        )
        self.topic.grid(row=0, column=1, padx=5, sticky="ew")
        
        ctk.CTkLabel(self.input_frame, text="Pairs:").grid(row=0, column=2, padx=5)
        self.pairs = ctk.CTkEntry(self.input_frame, width=50)
        self.pairs.insert(0, "20")
        self.pairs.grid(row=0, column=3, padx=5)

        self.btn = ctk.CTkButton(self.input_frame, text="Generate Dataset", command=self._start)
        self.btn.grid(row=0, column=4, padx=10)

        ctk.CTkLabel(self.input_frame, text="Optional notes (facts, terms, names):").grid(
            row=1, column=0, padx=(0, 5), pady=(8, 0), sticky="nw"
        )
        self.context = ctk.CTkTextbox(self.input_frame, height=70, wrap="word")
        self.context.grid(row=1, column=1, columnspan=4, padx=5, pady=(8, 0), sticky="ew")
        
        self.process_target = self._generate_data

    def _generate_data(self):
        topic = self.topic.get().strip()
        num_pairs_raw = self.pairs.get().strip()
        context = self.context.get("0.0", "end").strip()
        if not topic:
            self.log("Please enter a domain/topic.\n")
            return
        try:
            num_pairs = max(3, int(num_pairs_raw or "20"))
        except ValueError:
            self.log("Pairs must be a number (e.g. 20).\n")
            return

        # Check if Ollama is running; if not, try to start it
        if not ensure_ollama_running():
            self.log("Could not automatically start Ollama. Please open the Ollama app manually.\n")
            return
        else:
            self.log("Ollama is running and connected successfully!\n")

        self.log(
            f"Generating {num_pairs} Q&A pairs with Llama 3.1 (batches of 5, quality filter)...\n"
            f"Topic: {topic}\n"
        )
        if context:
            self.log("Using your optional notes for more accurate answers.\n")
        self.log("\n")

        try:
            dataset = generate_dataset(topic, num_pairs, context=context, log=self.log)
            train_data, val_data = split_train_val(dataset)

            with open(PERSONAL_DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(train_data, f, indent=4, ensure_ascii=False)
            with open(PERSONAL_VAL_PATH, "w", encoding="utf-8") as f:
                json.dump(val_data, f, indent=4, ensure_ascii=False)

            self.log(f"\nDone: {len(dataset)} unique examples.\n")
            self.log(f"  Train: {len(train_data)} -> {PERSONAL_DATA_PATH}\n")
            self.log(f"  Val:   {len(val_data)} -> {PERSONAL_VAL_PATH}\n")
            self.log("\nNext: Tab 2 train, then Tab 3 test, Tab 4 export.\n")
        except Exception as e:
            self.log(
                f"\nFailed to generate dataset. Ensure Ollama is running and llama3.1 is installed.\n"
                f"Error: {e}\n"
            )


# =====================================================================
# 2. Train Model Frame
# =====================================================================
class TrainFrame(BaseProcessFrame):
    def __init__(self, master):
        super().__init__(
            master,
            "2. Fine-Tune Model",
            "Runs 38_finetune.py (Unsloth + QLoRA, low-resource defaults for 8 GB GPUs). "
            "First run downloads ~3 GB. Early stopping usually finishes in a few minutes.",
        )
        self.btn.configure(text="Start Training (38_finetune.py)")
        self.process_target = "38_finetune.py"

    def preflight(self):
        if not os.path.isfile(PERSONAL_DATA_PATH):
            return (
                "Cannot start training:\n\n"
                f"  Missing: {PERSONAL_DATA_PATH}\n\n"
                "Use Tab 1 to generate a dataset, or create personal_data.json manually."
            )
        return None

    def _run_process(self):
        try:
            self._run_python_script(
                self.process_target,
                intro=(
                    "Training will save adapters to personal_model/\n"
                    "Watch for loss lines below. Do not close until exit code 0.\n\n"
                ),
            )
        except Exception as e:
            self.log(f"\n\nERROR: {str(e)}\n")
        finally:
            self._finish()


# =====================================================================
# 3. Test Model Frame
# =====================================================================
class TestFrame(BaseProcessFrame):
    def __init__(self, master):
        super().__init__(
            master,
            "3. Test Fine-Tuned Model",
            "GPU compare (39): base vs fine-tuned on questions from personal_data.json only. "
            "Domain test (47): Llama 3.1 generates questions, then AHQ_quantize answers them.",
        )
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(self.input_frame, text="Domain/topic:").grid(row=0, column=0, padx=(0, 5))
        self.topic = ctk.CTkEntry(
            self.input_frame, width=280, placeholder_text="Same topic as Tab 1 (e.g. your subject)"
        )
        self.topic.grid(row=0, column=1, padx=5)

        ctk.CTkLabel(self.input_frame, text="New Qs:").grid(row=0, column=2, padx=5)
        self.num_q = ctk.CTkEntry(self.input_frame, width=40)
        self.num_q.insert(0, "10")
        self.num_q.grid(row=0, column=3, padx=5)

        self.btn_domain = ctk.CTkButton(
            self.input_frame,
            text="Domain test (Llama 3.1 → AHQ_quantize)",
            command=self._start_domain_test,
        )
        self.btn_domain.grid(row=0, column=4, padx=10)

        self.btn.configure(text="GPU compare (39_test_finetuned.py)")
        self.btn.grid(row=3, column=0, sticky="w", pady=(0, 10))
        self.out.grid(row=4, column=0, sticky="nsew")
        self.grid_rowconfigure(4, weight=1)
        self.process_target = "39_test_finetuned.py"

    def _domain_preflight(self):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return "Ollama is not running. Start the Ollama app, then try again.\n"
        if not any(n == "llama3.1" or n.startswith("llama3.1:") for n in names):
            return (
                "Cannot run domain test: llama3.1 is not installed.\n\n"
                "  Question generation uses Llama 3.1 only (same as Tab 1).\n"
                "  Install: ollama pull llama3.1\n\n"
            )
        model_base = "AHQ_quantize"
        try:
            with open(os.path.join(SCRIPT_DIR, "46.py"), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("OLLAMA_MODEL_NAME"):
                        model_base = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except OSError:
            pass
        if not any(n == model_base or n.startswith(model_base + ":") for n in names):
            modelfile = os.path.join(SCRIPT_DIR, "Modelfile")
            return (
                f"Cannot run domain test: '{model_base}' is not installed in Ollama.\n\n"
                f"  Installed models: {', '.join(names) or '(none)'}\n\n"
                "  Fix: complete Tab 4 (Export), or run:\n"
                f'    ollama create {model_base} --quantize q4_K_M -f "{modelfile}"\n\n'
                f"  Then verify: ollama list\n"
            )
        return None

    def _start_domain_test(self):
        if self._busy:
            return
        topic = self.topic.get().strip()
        self.out.delete("0.0", "end")
        if not topic:
            self.log("Enter the same domain/topic you used when generating training data.\n")
            return
        try:
            num = int(self.num_q.get().strip() or "10")
        except ValueError:
            self.log("New Qs must be a number.\n")
            return
        self._busy = True
        self.btn.configure(state="disabled")
        self.btn_domain.configure(state="disabled")
        threading.Thread(
            target=self._run_domain_test,
            args=(topic, num),
            daemon=True,
        ).start()

    def _run_domain_test(self, topic, num):
        try:
            err = self._domain_preflight()
            if err:
                self.log(err)
                return
            self.log(
                "Domain test: Llama 3.1 generates questions about your topic, "
                "then AHQ_quantize answers them. Requires Tab 4 export.\n\n"
            )
            cmd = [
                PYTHON_EXE,
                "-u",
                "47_test_domain.py",
                "--topic",
                topic,
                "--num",
                str(num),
                "--include-val",
                "--batch-size",
                "3",
            ]
            self.log(f"Executing: {' '.join(cmd)}\n{'-' * 60}\n")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=SCRIPT_DIR,
                env=SUBPROCESS_ENV,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.log(line)
            process.wait()
            self.log(f"\n{'-' * 60}\nProcess finished with exit code {process.returncode}\n")
        except Exception as e:
            self.log(f"\n\nERROR: {str(e)}\n")
        finally:
            self._finish()

    def _finish(self):
        super()._finish()
        if hasattr(self, "btn_domain"):
            self.after(0, lambda: self.btn_domain.configure(state="normal"))

    def preflight(self):
        if not adapter_ready():
            return (
                "Cannot run tests:\n\n"
                f"  Missing trained adapters in: {ADAPTER_DIR}\n"
                "  (expected adapter_config.json)\n\n"
                "Complete Tab 2 (38_finetune.py) successfully first.\n"
                "Training must finish with exit code 0."
            )
        return None

    def _run_process(self):
        try:
            self._run_python_script(
                self.process_target,
                intro=(
                    "Loading models on GPU — this can take several minutes before new lines appear.\n"
                    "The PyTorch 'Redirects' line is a harmless warning.\n\n"
                ),
            )
        except Exception as e:
            self.log(f"\n\nERROR: {str(e)}\n")
        finally:
            self._finish()


# =====================================================================
# 4. Export to Ollama Frame
# =====================================================================
class ExportFrame(BaseProcessFrame):
    def __init__(self, master):
        super().__init__(master, "4. Export to Ollama", "Runs 46.py to merge adapters, then imports the model into Ollama.")
        self.btn.configure(text="Export & Create Ollama Model")
        self.process_target = self._export_and_create

    def preflight(self):
        if not adapter_ready():
            return (
                "Cannot export:\n\n"
                f"  Missing trained adapters in: {ADAPTER_DIR}\n\n"
                "Complete Tab 2 (38_finetune.py) first."
            )
        return None

    def _export_and_create(self):
        # 1. Run 46.py
        self.log("Step 1: Running 46.py to merge LoRA adapters into a full model...\n")
        code = self._run_python_script("46.py")
        
        if code != 0:
            self.log(f"\n46.py failed with exit code {code}. Aborting Ollama import.\n")
            return
            
        # 2. Run ollama create
        # We need to extract OLLAMA_MODEL_NAME from 46.py to know what to build.
        model_name = "AHQ_quantize"  # default fallback; overridden by 46.py OLLAMA_MODEL_NAME
        try:
            with open(os.path.join(SCRIPT_DIR, "46.py"), "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("OLLAMA_MODEL_NAME"):
                        model_name = line.split("=")[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass

        self.log(f"\nStep 2: Importing merged model into Ollama as '{model_name}'...\n")
        modelfile_path = os.path.join(SCRIPT_DIR, "Modelfile")
        
        cmd = ["ollama", "create", model_name, "--quantize", "q4_K_M", "-f", modelfile_path]
        self.log(f"Executing: {' '.join(cmd)}\n")
        
        process2 = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=SCRIPT_DIR,
            env=SUBPROCESS_ENV,
        )
        assert process2.stdout is not None
        for line in process2.stdout:
            self.log(line)
        process2.wait()
        
        if process2.returncode == 0:
            self.log(f"\nSUCCESS! You can now use your model anywhere by running:\n  ollama run {model_name}\n")
        else:
            self.log(f"\nOllama import failed with exit code {process2.returncode}\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()
