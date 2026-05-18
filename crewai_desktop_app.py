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
import socket
import time

try:
    from crewai import Agent, Task, Crew, LLM
    # Use CrewAI's native LLM class with proper base_url to connect to local Ollama
    local_llm = LLM(model='ollama/llama3.1', base_url='http://localhost:11434')
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

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

# Suppress OpenAI API key warnings
os.environ["OPENAI_API_KEY"] = "NA"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CrewAI Multi-Agent Workflows")
        self.geometry("1200x800")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="CrewAI Features", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 20))

        buttons = [
            ("1. Basic Assistant", "basic"),
            ("2. Story Writer", "story"),
            ("3. Recipe & Nutrition", "recipe"),
            ("4. Travel Planner", "travel"),
            ("5. Finance Advisor", "finance")
        ]

        self.nav_buttons = {}
        for i, (text, name) in enumerate(buttons):
            btn = ctk.CTkButton(self.sidebar_frame, text=text, anchor="w", command=lambda n=name: self.select_frame(n))
            btn.grid(row=i+1, column=0, padx=20, pady=5, sticky="ew")
            self.nav_buttons[name] = btn

        # Connection Status Indicator
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Ollama: Connecting...", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=6, column=0, padx=20, pady=10, sticky="s")
        threading.Thread(target=self.check_ollama_status, daemon=True).start()

        # Frames Dictionary
        self.frames = {
            "basic": BasicAssistantFrame(self),
            "story": StoryWriterFrame(self),
            "recipe": RecipeFrame(self),
            "travel": TravelFrame(self),
            "finance": FinanceFrame(self)
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
        self.title_lbl = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=24, weight="bold"))
        self.title_lbl.grid(row=0, column=0, pady=(0, 20), sticky="w")
        
        if not CREWAI_AVAILABLE:
            ctk.CTkLabel(self, text="Error: crewai package is not installed.", text_color="red").grid(row=1, column=0, sticky="w")

class BasicAssistantFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "1. Basic Assistant")
        self.grid_rowconfigure(3, weight=1)
        
        self.input_text = ctk.CTkTextbox(self, height=100)
        self.input_text.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.input_text.insert("0.0", "Enter your prompt here...")

        self.btn_run = ctk.CTkButton(self, text="Generate", command=self.run_logic)
        self.btn_run.grid(row=2, column=0, sticky="w", pady=(0, 10))

        self.output_text = ctk.CTkTextbox(self, wrap="word")
        self.output_text.grid(row=3, column=0, sticky="nsew")

    def run_logic(self):
        if not CREWAI_AVAILABLE:
            return
        prompt = self.input_text.get("0.0", "end").strip()
        if prompt:
            self.output_text.insert("end", f"You: {prompt}\nAgent is thinking...\n\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(prompt,), daemon=True).start()

    def _process(self, prompt):
        try:
            agent = Agent(
                role="Assistant",
                goal="Provide helpful responses based on user input.",
                backstory="This agent assists users by generating responses using a local Ollama LLM.",
                llm=local_llm
            )
            task = Task(
                description=f"Generate a response based on user input: {prompt}",
                expected_output="The generated response will be a flat text.",
                agent=agent
            )
            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()
            
            result_str = str(result).replace('**', '')
            self.output_text.insert("end", f"Assistant:\n{result_str}\n\n")
        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n\n")
        finally:
            self.btn_run.configure(state="normal")
            self.output_text.see("end")

class StoryWriterFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "2. Story Writing (Writer + Editor)")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)
        
        self.input_text = ctk.CTkTextbox(self, height=80)
        self.input_text.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.input_text.insert("0.0", "Enter a story prompt...")

        self.btn_run = ctk.CTkButton(self, text="Generate Story", command=self.run_logic)
        self.btn_run.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ctk.CTkLabel(self, text="Writer's Draft:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, sticky="w", pady=(0, 5))
        ctk.CTkLabel(self, text="Editor's Final Version:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(0, 5))

        self.out_draft = ctk.CTkTextbox(self, wrap="word")
        self.out_draft.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        
        self.out_final = ctk.CTkTextbox(self, wrap="word")
        self.out_final.grid(row=4, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))

    def run_logic(self):
        if not CREWAI_AVAILABLE:
            return
        prompt = self.input_text.get("0.0", "end").strip()
        if prompt:
            self.out_draft.insert("end", "Writer is drafting...\n")
            self.out_final.insert("end", "Waiting for draft...\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(prompt,), daemon=True).start()

    def _process(self, prompt):
        try:
            writer = Agent(
                role="Writer",
                goal="Generate a creative story based on the given prompt.",
                backstory="This agent is a skilled writer with a vivid imagination. They will craft an engaging story using the provided prompt as inspiration.",
                llm=local_llm
            )
            editor = Agent(
                role="Editor",
                goal="Review and refine the generated story for clarity, coherence, and flow.",
                backstory="This agent is an experienced editor who will carefully review the story, provide feedback, and make necessary revisions to improve the overall quality.",
                llm=local_llm
            )
            write_story = Task(
                description=f"Write a short story based on the prompt: {prompt}",
                expected_output="A creative short story with a beginning, middle, and end.",
                agent=writer
            )
            edit_story = Task(
                description="Review and edit the generated story to enhance clarity, coherence, and flow.",
                expected_output="The edited story with improved quality.",
                agent=editor
            )
            
            story_crew = Crew(agents=[writer, editor], tasks=[write_story, edit_story])
            final_story = story_crew.kickoff()
            
            draft_str = str(write_story.output).replace('**', '')
            final_str = str(final_story).replace('**', '')
            
            self.out_draft.insert("end", f"\n{draft_str}\n\n")
            self.out_final.insert("end", f"\n{final_str}\n\n")
        except Exception as e:
            self.out_draft.insert("end", f"\nError: {e}\n")
            self.out_final.insert("end", f"\nError: {e}\n")
        finally:
            self.btn_run.configure(state="normal")
            self.out_draft.see("end")
            self.out_final.see("end")

class RecipeFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "3. Recipe & Nutrition (Chef + Nutritionist)")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)
        
        self.input_entry = ctk.CTkEntry(self, placeholder_text="Enter a recipe idea (e.g., High-protein vegan breakfast)")
        self.input_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.btn_run = ctk.CTkButton(self, text="Generate Recipe", command=self.run_logic)
        self.btn_run.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ctk.CTkLabel(self, text="Generated Recipe:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, sticky="w", pady=(0, 5))
        ctk.CTkLabel(self, text="Nutritional Analysis:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(0, 5))

        self.out_recipe = ctk.CTkTextbox(self, wrap="word")
        self.out_recipe.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        
        self.out_nutrition = ctk.CTkTextbox(self, wrap="word")
        self.out_nutrition.grid(row=4, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))

    def run_logic(self):
        if not CREWAI_AVAILABLE:
            return
        prompt = self.input_entry.get().strip()
        if prompt:
            self.out_recipe.insert("end", "Chef is cooking...\n")
            self.out_nutrition.insert("end", "Waiting for recipe...\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(prompt,), daemon=True).start()

    def _process(self, prompt):
        try:
            chef = Agent(
                role="Chef",
                goal="Generate a creative recipe based on the given idea.",
                backstory="This agent is a skilled chef who loves to create delicious recipes.",
                llm=local_llm
            )
            nutritionist = Agent(
                role="Nutritionist",
                goal="Analyze the generated recipe for nutritional value.",
                backstory="This agent is a nutrition expert who evaluates recipes for health benefits.",
                llm=local_llm
            )
            generate_recipe = Task(
                description=f"Create a recipe based on the idea: {prompt}",
                expected_output="A detailed recipe with ingredients and instructions.",
                agent=chef
            )
            analyze_nutrition = Task(
                description="Evaluate the nutritional value of the generated recipe.",
                expected_output="Nutritional analysis of the recipe.",
                agent=nutritionist
            )
            recipe_crew = Crew(agents=[chef, nutritionist], tasks=[generate_recipe, analyze_nutrition])
            final_recipe = recipe_crew.kickoff()
            
            recipe_str = str(generate_recipe.output).replace('**', '')
            nutri_str = str(final_recipe).replace('**', '')
            
            self.out_recipe.insert("end", f"\n{recipe_str}\n\n")
            self.out_nutrition.insert("end", f"\n{nutri_str}\n\n")
        except Exception as e:
            self.out_recipe.insert("end", f"\nError: {e}\n")
            self.out_nutrition.insert("end", f"\nError: {e}\n")
        finally:
            self.btn_run.configure(state="normal")
            self.out_recipe.see("end")
            self.out_nutrition.see("end")

class TravelFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "4. Travel Planner (Planner + Local Expert)")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)
        
        self.input_entry = ctk.CTkEntry(self, placeholder_text="Enter your travel destination (e.g., Tokyo, Japan)")
        self.input_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.btn_run = ctk.CTkButton(self, text="Plan Trip", command=self.run_logic)
        self.btn_run.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ctk.CTkLabel(self, text="Planned Itinerary:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, sticky="w", pady=(0, 5))
        ctk.CTkLabel(self, text="Local Insights:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(0, 5))

        self.out_itinerary = ctk.CTkTextbox(self, wrap="word")
        self.out_itinerary.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        
        self.out_insights = ctk.CTkTextbox(self, wrap="word")
        self.out_insights.grid(row=4, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))

    def run_logic(self):
        if not CREWAI_AVAILABLE:
            return
        prompt = self.input_entry.get().strip()
        if prompt:
            self.out_itinerary.insert("end", "Planner is organizing...\n")
            self.out_insights.insert("end", "Waiting for itinerary...\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(prompt,), daemon=True).start()

    def _process(self, prompt):
        try:
            planner = Agent(
                role="Trip Planner",
                goal="Create a detailed travel itinerary based on the destination.",
                backstory="This agent specializes in planning exciting trips.",
                llm=local_llm
            )
            local_expert = Agent(
                role="Local Expert",
                goal="Provide insights and recommendations about the destination.",
                backstory="This agent knows all the best spots in town!",
                llm=local_llm
            )
            create_itinerary = Task(
                description=f"Plan a trip itinerary for {prompt}.",
                expected_output="A detailed itinerary including activities and timings.",
                agent=planner
            )
            get_local_insights = Task(
                description=f"Provide local insights about {prompt}.",
                expected_output="Recommendations for places to visit and eat.",
                agent=local_expert
            )
            trip_crew = Crew(agents=[planner, local_expert], tasks=[create_itinerary, get_local_insights])
            final_itinerary = trip_crew.kickoff()
            
            itin_str = str(create_itinerary.output).replace('**', '')
            ins_str = str(final_itinerary).replace('**', '')
            
            self.out_itinerary.insert("end", f"\n{itin_str}\n\n")
            self.out_insights.insert("end", f"\n{ins_str}\n\n")
        except Exception as e:
            self.out_itinerary.insert("end", f"\nError: {e}\n")
            self.out_insights.insert("end", f"\nError: {e}\n")
        finally:
            self.btn_run.configure(state="normal")
            self.out_itinerary.see("end")
            self.out_insights.see("end")

class FinanceFrame(BaseFeatureFrame):
    def __init__(self, master):
        super().__init__(master, "5. Finance Advisor (Planner + Investor)")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)
        
        self.input_entry = ctk.CTkEntry(self, placeholder_text="Enter your financial goal (e.g., saving for a house)")
        self.input_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.btn_run = ctk.CTkButton(self, text="Get Advice", command=self.run_logic)
        self.btn_run.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ctk.CTkLabel(self, text="Savings Plan:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, sticky="w", pady=(0, 5))
        ctk.CTkLabel(self, text="Investment Options:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(0, 5))

        self.out_savings = ctk.CTkTextbox(self, wrap="word")
        self.out_savings.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        
        self.out_invest = ctk.CTkTextbox(self, wrap="word")
        self.out_invest.grid(row=4, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))

    def run_logic(self):
        if not CREWAI_AVAILABLE:
            return
        prompt = self.input_entry.get().strip()
        if prompt:
            self.out_savings.insert("end", "Planner is budgeting...\n")
            self.out_invest.insert("end", "Waiting for plan...\n")
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self._process, args=(prompt,), daemon=True).start()

    def _process(self, prompt):
        try:
            planner = Agent(
                role="Financial Planner",
                goal="Create a savings plan to achieve the financial goal.",
                backstory="This agent helps users manage their finances effectively, focusing on budgeting and safe accumulation.",
                llm=local_llm
            )
            investor = Agent(
                role="Investment Advisor",
                goal="Provide investment options based on user’s risk tolerance.",
                backstory="This agent specializes in finding suitable investment opportunities to maximize returns.",
                llm=local_llm
            )
            create_plan = Task(
                description=f"Develop a savings plan for achieving: {prompt}.",
                expected_output="A detailed savings plan with steps to follow.",
                agent=planner
            )
            suggest_investments = Task(
                description=f"Suggest investment options based on user's financial situation.",
                expected_output="A list of recommended investments.",
                agent=investor
            )
            finance_crew = Crew(agents=[planner, investor], tasks=[create_plan, suggest_investments])
            final_advice = finance_crew.kickoff()
            
            plan_str = str(create_plan.output).replace('**', '')
            inv_str = str(final_advice).replace('**', '')
            
            self.out_savings.insert("end", f"\n{plan_str}\n\n")
            self.out_invest.insert("end", f"\n{inv_str}\n\n")
        except Exception as e:
            self.out_savings.insert("end", f"\nError: {e}\n")
            self.out_invest.insert("end", f"\nError: {e}\n")
        finally:
            self.btn_run.configure(state="normal")
            self.out_savings.see("end")
            self.out_invest.see("end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
