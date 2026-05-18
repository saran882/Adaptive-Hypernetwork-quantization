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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Inject SoX path for TTS
sox_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet",
    "Packages", "ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe", "sox-14.4.2")
if os.path.exists(sox_path) and sox_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + sox_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Media AI Workspace (31-36)")
        self.geometry("1200x820")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        sidebar = ctk.CTkFrame(self, width=230, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(7, weight=1)

        ctk.CTkLabel(sidebar, text="Media AI", font=ctk.CTkFont(size=20, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 20))

        nav = [
            ("1. Text Gen (31.py)",  "textgen"),
            ("2. Text Gen v2 (31(1))", "textgen2"),
            ("3. Text-to-Speech (33)", "tts"),
            ("4. Text-to-Image (34)", "tti"),
            ("5. Text-to-Video (35)", "ttv"),
            ("6. Img-to-Video (36)",  "itv"),
        ]
        self.frames = {}
        frame_classes = [TextGenFrame, TextGen2Frame, TTSFrame, TTIFrame, TTVFrame, ITVFrame]
        for i, ((label, key), cls) in enumerate(zip(nav, frame_classes)):
            ctk.CTkButton(sidebar, text=label, anchor="w",
                          command=lambda k=key: self.show(k)).grid(
                row=i+1, column=0, padx=15, pady=4, sticky="ew")
            self.frames[key] = cls(self)

        self.show("textgen")

    def show(self, key):
        for f in self.frames.values():
            f.grid_forget()
        self.frames[key].grid(row=0, column=1, sticky="nsew", padx=20, pady=20)


# ── helpers ──────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


class BaseFrame(ctk.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=22, weight="bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 14))

    def _out(self):
        """Return the output textbox (subclasses expose self.out)."""
        return self.out

    def log(self, msg):
        self.out.insert("end", msg)
        self.out.see("end")


# ── 1. Text Generation — 31.py ────────────────────────────────────────────────

class TextGenFrame(BaseFrame):
    def __init__(self, master):
        super().__init__(master, "1. Local Text Generation  (31.py — LFM2.5-1.2B)")
        self.grid_rowconfigure(3, weight=1)

        self.inp = ctk.CTkTextbox(self, height=90)
        self.inp.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.inp.insert("0.0", "What is the Sonic Boom in Aviation?")

        self.btn = ctk.CTkButton(self, text="Generate", command=self._run)
        self.btn.grid(row=2, column=0, sticky="w", pady=(0, 8))

        self.out = ctk.CTkTextbox(self, wrap="word")
        self.out.grid(row=3, column=0, sticky="nsew")

    def _run(self):
        prompt = self.inp.get("0.0", "end").strip()
        if not prompt:
            return
        self.log(f"You: {prompt}\nGenerating...\n")
        self.btn.configure(state="disabled")
        threading.Thread(target=self._gen, args=(prompt,), daemon=True).start()

    def _gen(self, prompt):
        try:
            from transformers import pipeline
            pipe = pipeline("text-generation", model="LiquidAI/LFM2.5-1.2B-Thinking")
            output = pipe([{"role": "user", "content": prompt}],
                          max_new_tokens=1024, do_sample=True,
                          temperature=0.7, top_k=50, top_p=0.5)
            try:
                res = output[0]["generated_text"][1]["content"].split("</think>")[1]
            except Exception:
                res = str(output)
            self.log(f"\nResult:\n{clean(res)}\n\n")
        except Exception as e:
            self.log(f"\nError: {e}\n\n")
        finally:
            self.btn.configure(state="normal")


# ── 2. Text Generation v2 — 31 (1).py ────────────────────────────────────────
#  Identical model but keeps a persistent pipeline to avoid reloading each time.

class TextGen2Frame(BaseFrame):
    _pipe = None  # class-level cache

    def __init__(self, master):
        super().__init__(master, "2. Text Generation v2  (31(1).py — cached pipeline)")
        self.grid_rowconfigure(4, weight=1)

        self.inp = ctk.CTkTextbox(self, height=90)
        self.inp.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.inp.insert("0.0", "Explain quantum entanglement simply.")

        opts = ctk.CTkFrame(self, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(opts, text="Max tokens:").grid(row=0, column=0, padx=6)
        self.tokens = ctk.CTkEntry(opts, width=80)
        self.tokens.insert(0, "1024")
        self.tokens.grid(row=0, column=1, padx=6)
        ctk.CTkLabel(opts, text="Temp:").grid(row=0, column=2, padx=6)
        self.temp = ctk.CTkEntry(opts, width=60)
        self.temp.insert(0, "0.7")
        self.temp.grid(row=0, column=3, padx=6)

        self.btn = ctk.CTkButton(self, text="Generate", command=self._run)
        self.btn.grid(row=3, column=0, sticky="w", pady=(0, 8))

        self.out = ctk.CTkTextbox(self, wrap="word")
        self.out.grid(row=4, column=0, sticky="nsew")

    def _run(self):
        prompt = self.inp.get("0.0", "end").strip()
        if not prompt:
            return
        self.log(f"You: {prompt}\nGenerating...\n")
        self.btn.configure(state="disabled")
        try:
            mt = int(self.tokens.get())
            tp = float(self.temp.get())
        except ValueError:
            mt, tp = 1024, 0.7
        threading.Thread(target=self._gen, args=(prompt, mt, tp), daemon=True).start()

    def _gen(self, prompt, max_tokens, temperature):
        try:
            from transformers import pipeline
            if TextGen2Frame._pipe is None:
                self.log("(Loading model — first run may take a moment...)\n")
                TextGen2Frame._pipe = pipeline(
                    "text-generation", model="LiquidAI/LFM2.5-1.2B-Thinking")
            output = TextGen2Frame._pipe(
                [{"role": "user", "content": prompt}],
                max_new_tokens=max_tokens, do_sample=True,
                temperature=temperature, top_k=50, top_p=0.5)
            try:
                res = output[0]["generated_text"][1]["content"].split("</think>")[1]
            except Exception:
                res = str(output)
            self.log(f"\nResult:\n{clean(res)}\n\n")
        except Exception as e:
            self.log(f"\nError: {e}\n\n")
        finally:
            self.btn.configure(state="normal")


# ── 3. Text-to-Speech — 33.py ─────────────────────────────────────────────────

class TTSFrame(BaseFrame):
    def __init__(self, master):
        super().__init__(master, "3. Text-to-Speech  (33.py — Qwen3-TTS)")
        self.grid_rowconfigure(5, weight=1)

        self.inp = ctk.CTkTextbox(self, height=90)
        self.inp.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.inp.insert("0.0", "She said she would be here by noon.")

        opts = ctk.CTkFrame(self, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(opts, text="Speaker:").grid(row=0, column=0, padx=6)
        self.speaker = ctk.CTkComboBox(opts, values=["Ryan", "Vivian"], width=120)
        self.speaker.set("Ryan")
        self.speaker.grid(row=0, column=1, padx=6)

        ctk.CTkLabel(opts, text="Language:").grid(row=0, column=2, padx=6)
        self.lang = ctk.CTkComboBox(opts, values=["English", "Chinese", "Auto"], width=120)
        self.lang.set("English")
        self.lang.grid(row=0, column=3, padx=6)

        self.instruct = ctk.CTkEntry(self, placeholder_text="Tone instruction e.g. 'Very happy'")
        self.instruct.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        self.btn = ctk.CTkButton(self, text="Generate Audio", command=self._run)
        self.btn.grid(row=4, column=0, sticky="w", pady=(0, 8))

        self.out = ctk.CTkTextbox(self, wrap="word")
        self.out.grid(row=5, column=0, sticky="nsew")

    def _run(self):
        text = self.inp.get("0.0", "end").strip()
        if not text:
            return
        self.log(f"Synthesizing: {text[:40]}...\n")
        self.btn.configure(state="disabled")
        threading.Thread(target=self._gen,
                         args=(text, self.speaker.get(),
                               self.lang.get(), self.instruct.get().strip()),
                         daemon=True).start()

    def _gen(self, text, speaker, language, instruct):
        try:
            import torch
            import soundfile as sf
            from qwen_tts import Qwen3TTSModel

            self.log("Loading Qwen3-TTS model...\n")
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map="auto", dtype=torch.bfloat16)

            lang_arg = None if language == "Auto" else language
            self.log("Generating audio...\n")
            wavs, sr = model.generate_custom_voice(
                text=text, language=lang_arg, speaker=speaker,
                instruct=instruct, temperature=0.7, top_p=0.5,
                repetition_penalty=1.1, max_new_tokens=4096)

            out_file = os.path.join(SCRIPT_DIR, "output_tts.wav")
            sf.write(out_file, wavs[0], sr)
            self.log(f"\nSaved: {out_file}\n\n")
        except Exception as e:
            self.log(f"\nError: {e}\n\n")
        finally:
            self.btn.configure(state="normal")


# ── 4. Text-to-Image — 34.py ──────────────────────────────────────────────────

class TTIFrame(BaseFrame):
    def __init__(self, master):
        super().__init__(master, "4. Text-to-Image  (34.py — SDXL Lightning 4-step)")
        self.grid_rowconfigure(3, weight=1)

        self.inp = ctk.CTkTextbox(self, height=90)
        self.inp.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.inp.insert("0.0", "A smiling girl in a sunlit garden.")

        self.btn = ctk.CTkButton(self, text="Generate Image", command=self._run)
        self.btn.grid(row=2, column=0, sticky="w", pady=(0, 8))

        self.out = ctk.CTkTextbox(self, wrap="word")
        self.out.grid(row=3, column=0, sticky="nsew")

    def _run(self):
        prompt = self.inp.get("0.0", "end").strip()
        if not prompt:
            return
        self.log(f"Prompt: {prompt}\nLoading SDXL Lightning (requires NVIDIA GPU)...\n")
        self.btn.configure(state="disabled")
        threading.Thread(target=self._gen, args=(prompt,), daemon=True).start()

    def _gen(self, prompt):
        try:
            import torch
            from diffusers import (StableDiffusionXLPipeline,
                                   UNet2DConditionModel, EulerDiscreteScheduler)
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"
            ckpt = "sdxl_lightning_4step_unet.safetensors"

            self.log("Loading UNet...\n")
            unet = UNet2DConditionModel.from_config(
                base, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(
                load_file(hf_hub_download(repo, ckpt), device="cuda"))

            self.log("Building pipeline...\n")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                base, unet=unet, torch_dtype=torch.float16,
                variant="fp16").to("cuda")
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing")

            self.log("Generating image (4 steps)...\n")
            image = pipe(prompt, num_inference_steps=4,
                         guidance_scale=0, height=720, width=1280).images[0]

            out_file = os.path.join(SCRIPT_DIR, "output_image.png")
            image.save(out_file)
            self.log(f"\nSaved: {out_file}\n\n")
        except Exception as e:
            self.log(f"\nError: {e}\n\n")
        finally:
            self.btn.configure(state="normal")


# ── 5. Text-to-Video — 35.py ──────────────────────────────────────────────────

class TTVFrame(BaseFrame):
    def __init__(self, master):
        super().__init__(master, "5. Text-to-Video  (35.py — SkyReels-V2 DF)")
        self.grid_rowconfigure(4, weight=1)

        self.inp = ctk.CTkTextbox(self, height=80)
        self.inp.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.inp.insert("0.0", "Two men fighting in a park.")

        opts = ctk.CTkFrame(self, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(opts, text="Duration (s):").grid(row=0, column=0, padx=6)
        self.dur = ctk.CTkComboBox(opts, values=["5", "10", "15", "30"], width=90)
        self.dur.set("5")
        self.dur.grid(row=0, column=1, padx=6)

        self.btn = ctk.CTkButton(self, text="Generate Video", command=self._run)
        self.btn.grid(row=3, column=0, sticky="w", pady=(0, 8))

        self.out = ctk.CTkTextbox(self, wrap="word")
        self.out.grid(row=4, column=0, sticky="nsew")

    def _run(self):
        prompt = self.inp.get("0.0", "end").strip()
        if not prompt:
            return
        self.log(f"Prompt: {prompt}\nStarting SkyReels-V2 (requires NVIDIA GPU + ~15 GB VRAM)...\n")
        self.btn.configure(state="disabled")
        threading.Thread(target=self._gen,
                         args=(prompt, int(self.dur.get())),
                         daemon=True).start()

    def _gen(self, prompt, duration):
        try:
            import random, torch, imageio

            repo_dir = os.path.join(SCRIPT_DIR, "SkyReels-V2")
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            os.chdir(repo_dir)

            from skyreels_v2_infer import DiffusionForcingPipeline
            from skyreels_v2_infer.modules import download_model

            fps = 24
            num_frames = {5: 97, 10: 257, 15: 377, 30: 737}.get(duration, 97)
            seed = random.randint(0, 2**32 - 1)
            negative_prompt = (
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
                "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
                "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
                "杂乱的背景，三条腿，背景人很多，倒着走"
            )

            self.log("Downloading/loading model...\n")
            model_path = download_model("Skywork/SkyReels-V2-DF-1.3B-540P")
            pipe = DiffusionForcingPipeline(
                model_path, dit_path=model_path,
                device=torch.device("cuda"),
                weight_dtype=torch.bfloat16,
                use_usp=False, offload=True)

            self.log(f"Generating {num_frames} frames (~{round(num_frames/fps,1)}s)...\n")
            with torch.amp.autocast("cuda", dtype=pipe.transformer.dtype), torch.no_grad():
                frames = pipe(
                    prompt=prompt, negative_prompt=negative_prompt,
                    image=None, end_image=None,
                    height=544, width=960,
                    num_frames=num_frames,
                    num_inference_steps=30,
                    shift=8.0, guidance_scale=6.0,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                    overlap_history=17, addnoise_condition=20,
                    base_num_frames=97, ar_step=0,
                    causal_block_size=1, fps=fps)[0]

            out_file = os.path.join(SCRIPT_DIR, "output_video.mp4")
            imageio.mimwrite(out_file, frames, fps=fps, quality=8,
                             output_params=["-loglevel", "error"])
            self.log(f"\nSaved: {out_file}\n\n")
        except Exception as e:
            self.log(f"\nError: {e}\n\n")
        finally:
            os.chdir(SCRIPT_DIR)
            self.btn.configure(state="normal")


# ── 6. Image-to-Video — 36.py ─────────────────────────────────────────────────

class ITVFrame(BaseFrame):
    def __init__(self, master):
        super().__init__(master, "6. Image-to-Video  (36.py — SkyReels-V2 I2V / DF)")
        self.grid_rowconfigure(5, weight=1)

        self.inp = ctk.CTkTextbox(self, height=70)
        self.inp.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.inp.insert("0.0", "A man with dark hair, short beard, wearing a blue suit waves and smiles.")

        opts = ctk.CTkFrame(self, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(opts, text="Duration (s):").grid(row=0, column=0, padx=6)
        self.dur = ctk.CTkComboBox(opts, values=["5", "10", "15", "30"], width=90)
        self.dur.set("5")
        self.dur.grid(row=0, column=1, padx=6)
        ctk.CTkLabel(opts, text="I2V Steps:").grid(row=0, column=2, padx=6)
        self.steps = ctk.CTkComboBox(opts, values=["25", "50"], width=80)
        self.steps.set("25")
        self.steps.grid(row=0, column=3, padx=6)

        self.img_entry = ctk.CTkEntry(self, placeholder_text="Start image path (optional, e.g. start.jpg)")
        self.img_entry.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        self.btn = ctk.CTkButton(self, text="Generate Video", command=self._run)
        self.btn.grid(row=4, column=0, sticky="w", pady=(0, 8))

        self.out = ctk.CTkTextbox(self, wrap="word")
        self.out.grid(row=5, column=0, sticky="nsew")

    def _run(self):
        prompt = self.inp.get("0.0", "end").strip()
        if not prompt:
            return
        image_path = self.img_entry.get().strip() or None
        self.log(f"Prompt: {prompt}\nStarting SkyReels-V2 (requires NVIDIA GPU)...\n")
        self.btn.configure(state="disabled")
        threading.Thread(target=self._gen,
                         args=(prompt, int(self.dur.get()),
                               int(self.steps.get()), image_path),
                         daemon=True).start()

    def _gen(self, prompt, duration, i2v_steps, image_path):
        try:
            import random, torch, imageio

            repo_dir = os.path.join(SCRIPT_DIR, "SkyReels-V2")
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            os.chdir(repo_dir)

            from skyreels_v2_infer.modules import download_model

            fps = 24
            num_frames = {5: 97, 10: 257, 15: 377, 30: 737}.get(duration, 97)
            seed = random.randint(0, 2**32 - 1)
            negative_prompt = (
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
                "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
                "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
                "杂乱的背景，三条腿，背景人很多，倒着走"
            )

            height, width = 544, 960
            image_input = None

            if image_path:
                full = image_path if os.path.isfile(image_path) else os.path.join(SCRIPT_DIR, image_path)
                if os.path.isfile(full):
                    from diffusers.utils import load_image
                    from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop
                    img = load_image(full)
                    if img.height > img.width:
                        height, width = 960, 544
                    image_input = resizecrop(img, height, width).convert("RGB")
                    self.log(f"Using start image: {full}\n")
                else:
                    self.log(f"Warning: image not found at {image_path}, using text-to-video.\n")

            use_i2v = image_input is not None
            if use_i2v:
                model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"
                guidance, shift, steps = 5.0, 5.0, i2v_steps
            else:
                model_id = "Skywork/SkyReels-V2-DF-1.3B-540P"
                guidance, shift, steps = 6.0, 8.0, 30

            self.log("Downloading/loading model...\n")
            model_path = download_model(model_id)

            with torch.amp.autocast("cuda"), torch.no_grad():
                if use_i2v:
                    from skyreels_v2_infer.pipelines.image2video_pipeline import Image2VideoPipeline
                    pipe = Image2VideoPipeline(model_path, dit_path=model_path,
                                              device=torch.device("cuda"),
                                              weight_dtype=torch.bfloat16,
                                              use_usp=False, offload=True)
                    self.log("Generating I2V video...\n")
                    frames = pipe(image=image_input, prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  height=height, width=width,
                                  num_frames=num_frames,
                                  num_inference_steps=steps,
                                  guidance_scale=guidance, shift=shift,
                                  generator=torch.Generator(device="cuda").manual_seed(seed))[0]
                else:
                    from skyreels_v2_infer import DiffusionForcingPipeline
                    pipe = DiffusionForcingPipeline(model_path, dit_path=model_path,
                                                   device=torch.device("cuda"),
                                                   weight_dtype=torch.bfloat16,
                                                   use_usp=False, offload=True)
                    self.log("Generating DF video...\n")
                    frames = pipe(prompt=prompt, negative_prompt=negative_prompt,
                                  image=None, end_image=None,
                                  height=height, width=width,
                                  num_frames=num_frames,
                                  num_inference_steps=steps,
                                  shift=shift, guidance_scale=guidance,
                                  generator=torch.Generator(device="cuda").manual_seed(seed),
                                  overlap_history=17, addnoise_condition=20,
                                  base_num_frames=97, ar_step=0,
                                  causal_block_size=1, fps=fps)[0]

            out_file = os.path.join(SCRIPT_DIR, "output_i2v.mp4")
            imageio.mimwrite(out_file, frames, fps=fps, quality=8,
                             output_params=["-loglevel", "error"])
            self.log(f"\nSaved: {out_file}\n\n")
        except Exception as e:
            self.log(f"\nError: {e}\n\n")
        finally:
            os.chdir(SCRIPT_DIR)
            self.btn.configure(state="normal")


if __name__ == "__main__":
    app = App()
    app.mainloop()
