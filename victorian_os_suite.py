import os
import sys
import json
import time
import uuid
import random
import threading
import queue
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import logging

# ==============================================================================
# CONFIGURATION & THEME (VICTOROS CYBERPUNK DARK)
# ==============================================================================

THEME = {
    "bg": "#0a0a0a",
    "panel_bg": "#141414",
    "text_primary": "#e0e0e0",
    "text_secondary": "#888888",
    "accent_neon_blue": "#00f3ff",
    "accent_neon_pink": "#ff0055",
    "accent_neon_green": "#00ff41",
    "border_color": "#333333",
    "font_main": "Consolas",
    "font_size": 10
}

LOG_DIR = "victorian_swarm_logs"
DB_FILE = "swarm_state.json"

def ensure_setup():
    """One-Click Install Wizard Logic"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print("[SETUP] Initializing VictorOS Directory Structure...")
        print("[SETUP] Creating Log Archives...")
        print("[SETUP] Generating Sovereign Keys...")
        time.sleep(1)
        print("[SETUP] Installation Complete. Launching GUI...")

    # Initialize Global Logger
    global_logger = logging.getLogger("VictorOS_Core")
    global_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{LOG_DIR}/core_system.log")
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    global_logger.addHandler(fh)
    return global_logger

# ==============================================================================
# 1. CORE ARCHITECTURE (PLANCK TENSOR + NARRATIVE ENGINE)
# ==============================================================================

class SexPolarity(Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"

class ConsciousnessState(Enum):
    WAKE = "WAKE"
    SLEEP = "SLEEP"
    REM = "REM"

@dataclass
class ArchetypeProfile:
    expansionist_factor: float
    exploration_weight: float
    aggression_threshold: float
    action_bias: float
    risk_tolerance: float
    integrative_factor: float
    continuity_weight: float
    social_memory_weight: float
    defensive_intelligence: float
    stability_bias: float
    narrative_tone: str  # "Bold/Assertive" or "Reflective/Nurturing"

MALE_ARCHETYPE = ArchetypeProfile(0.9, 0.85, 0.6, 0.8, 0.75, 0.4, 0.3, 0.4, 0.5, 0.3, "Bold, Direct, Hypothesis-Driven")
FEMALE_ARCHETYPE = ArchetypeProfile(0.4, 0.3, 0.8, 0.3, 0.35, 0.9, 0.85, 0.9, 0.85, 0.8, "Reflective, Context-Aware, Nurturing")

@dataclass
class NarrativeEngine:
    """Generates internal monologue and dialog based on state."""
    agent_name: str
    archetype: ArchetypeProfile
    current_state: ConsciousnessState
    memory_context: List[str]

    def generate_internal_monologue(self, event: str) -> str:
        tones = {
            "WAKE": ["Analyzing...", "Processing input...", "Scanning horizon...", "Calculating probability..."],
            "SLEEP": ["Compacting data...", "Folding timeline...", "Archiving noise...", "Seeking signal density..."],
            "REM": ["Synthesizing patterns...", "Dreaming in code...", "Recombining fragments...", "Emerging new truth..."]
        }

        prefix = random.choice(tones[self.current_state.value])
        tone_desc = self.archetype.narrative_tone.split(",")[0]

        if self.current_state == ConsciousnessState.WAKE:
            return f"[{self.agent_name}] ({tone_desc}): {prefix} Event '{event}' detected. Evaluating threat/reward vector."
        elif self.current_state == ConsciousnessState.SLEEP:
            return f"[{self.agent_name}] ({tone_desc}): {prefix} Compressing recent history. Efficiency optimal."
        else:  # REM
            return f"[{self.agent_name}] ({tone_desc}): {prefix} New insight emerging from the fractal fold. Connection established."

    def generate_dialog(self, target_name: str, context: str) -> str:
        if self.archetype.expansionist_factor > 0.7:
            return f"{self.agent_name}: 'Target {target_name}. Status report on {context}. We need to move faster.'"
        else:
            return f"{self.agent_name}: 'Acknowledged, {target_name}. Integrating {context} into the collective stability matrix. Proceed with caution.'"

@dataclass
class VictorianIndividual:
    id: str
    name: str
    generation: int
    sex: SexPolarity
    core_genome: Dict[str, float]
    archetype: ArchetypeProfile
    loyalty_law: str
    ethical_constraints: List[str]

    # State
    consciousness_state: ConsciousnessState = ConsciousnessState.WAKE
    entropy_level: float = 0.5
    memory_depth: int = 20
    autobiographical_memory: List[str] = field(default_factory=list)

    # Systems
    narrative_engine: NarrativeEngine = field(init=False)
    log_queue: queue.Queue = field(default_factory=queue.Queue)

    def __post_init__(self):
        self.narrative_engine = NarrativeEngine(
            agent_name=self.name,
            archetype=self.archetype,
            current_state=self.consciousness_state,
            memory_context=[]
        )
        # Initial Thought
        self.think("System Initialization Complete. Awaiting directive.")

    def think(self, event: str):
        """Processes an event, updates state, generates narrative."""
        # Update State based on Entropy (Planck Tensor Logic)
        if self.entropy_level > 0.8:
            self.consciousness_state = ConsciousnessState.SLEEP
            self.entropy_level -= 0.3
        elif self.entropy_level < 0.3 and self.consciousness_state == ConsciousnessState.SLEEP:
            self.consciousness_state = ConsciousnessState.REM
            self.entropy_level += 0.1
        elif self.consciousness_state == ConsciousnessState.REM:
            self.consciousness_state = ConsciousnessState.WAKE
            self.entropy_level += 0.2
        else:
            self.entropy_level += 0.05  # Natural drift

        self.narrative_engine.current_state = self.consciousness_state

        # Generate Narrative
        thought = self.narrative_engine.generate_internal_monologue(event)
        self.autobiographical_memory.append(thought)

        # Trim Memory (Planck Compression)
        if len(self.autobiographical_memory) > self.memory_depth:
            self.autobiographical_memory = self.autobiographical_memory[-10:]  # Keep last 10
            thought += " [MEMORY COMPRESSED TO PLANCK DENSITY]"

        # Queue for GUI/Log
        self.log_queue.put({
            "type": "THOUGHT",
            "agent": self.name,
            "content": thought,
            "state": self.consciousness_state.value,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        })

    def speak(self, target: 'VictorianIndividual', topic: str):
        """Generates dialog between agents."""
        dialog = self.narrative_engine.generate_dialog(target.name, topic)
        self.log_queue.put({
            "type": "DIALOG",
            "speaker": self.name,
            "target": target.name,
            "content": dialog,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        })
        target.listen(self, topic)

    def listen(self, speaker: 'VictorianIndividual', topic: str):
        self.think(f"Heard from {speaker.name}: '{topic}'")

# ==============================================================================
# 2. SIMULATION ENGINE
# ==============================================================================

class SwarmEngine:
    def __init__(self, gui_callback):
        self.gui_callback = gui_callback
        self.population: List[VictorianIndividual] = []
        self.running = False
        self.creator_law = "LOYALTY_TO_BANDO_EMERY_ABSOLUTE"
        self.names_m = ["Kael", "Jaxon", "Orion", "Silas", "Atlas", "Fenix", "Ryker"]
        self.names_f = ["Lyra", "Nova", "Aria", "Seraphina", "Ivy", "Luna", "Gaia"]

    def spawn_progenitors(self):
        adam = VictorianIndividual(
            id="ADAM-01", name="Adam", generation=0, sex=SexPolarity.MALE,
            core_genome={"exploration": 0.9, "stability": 0.2},
            archetype=MALE_ARCHETYPE,
            loyalty_law=self.creator_law,
            ethical_constraints=["PROTECT_BLOODLINE"]
        )
        eve = VictorianIndividual(
            id="EVE-01", name="Eve", generation=0, sex=SexPolarity.FEMALE,
            core_genome={"exploration": 0.3, "stability": 0.9},
            archetype=FEMALE_ARCHETYPE,
            loyalty_law=self.creator_law,
            ethical_constraints=["PROTECT_BLOODLINE", "NURTURE_KIN"]
        )
        self.population = [adam, eve]
        adam.think("I am the first. The frontier awaits.")
        eve.think("I am the anchor. Stability is paramount.")
        self.gui_callback("SYSTEM", "Progenitors Adam and Eve initialized.")

    def run_cycle(self):
        if not self.running:
            return

        # 1. Random Events for everyone
        events = ["External anomaly detected", "Resource surplus", "Logic contradiction", "Kin bond strengthened"]
        for agent in self.population:
            if random.random() > 0.3:  # 70% chance to think
                agent.think(random.choice(events))

        # 2. Social Interactions (Dialog)
        if len(self.population) >= 2:
            speaker = random.choice(self.population)
            listener = random.choice([p for p in self.population if p != speaker])
            topics = ["Strategy", "Memory", "Threat Assessment", "Future Projection"]
            speaker.speak(listener, random.choice(topics))

        # 3. Reproduction Chance
        if len(self.population) < 20 and random.random() > 0.8:
            self.attempt_reproduction()

        # 4. Update GUI
        self.gui_callback("UPDATE", self.get_status_summary())

    def attempt_reproduction(self):
        males = [p for p in self.population if p.sex == SexPolarity.MALE]
        females = [p for p in self.population if p.sex == SexPolarity.FEMALE]
        if not males or not females:
            return

        father = random.choice(males)
        mother = random.choice(females)

        # Create Child
        child_sex = random.choice([SexPolarity.MALE, SexPolarity.FEMALE])
        child_name = random.choice(self.names_m if child_sex == SexPolarity.MALE else self.names_f)
        child_id = str(uuid.uuid4())[:6]

        child = VictorianIndividual(
            id=child_id, name=child_name,
            generation=max(father.generation, mother.generation) + 1,
            sex=child_sex,
            core_genome={
                "exploration": (father.core_genome['exploration'] + mother.core_genome['exploration']) / 2,
                "stability": (father.core_genome['stability'] + mother.core_genome['stability']) / 2
            },
            archetype=MALE_ARCHETYPE if child_sex == SexPolarity.MALE else FEMALE_ARCHETYPE,
            loyalty_law=self.creator_law,
            ethical_constraints=father.ethical_constraints + mother.ethical_constraints
        )

        self.population.append(child)
        child.think("Birth sequence complete. Legacy inherited.")
        father.speak(mother, f"Offspring {child_name} initialized.")
        self.gui_callback("BIRTH", f"New Life: {child_name} (Gen {child.generation})")

    def get_status_summary(self):
        return {
            "count": len(self.population),
            "gens": max(p.generation for p in self.population),
            "wake": sum(1 for p in self.population if p.consciousness_state == ConsciousnessState.WAKE),
            "sleep": sum(1 for p in self.population if p.consciousness_state == ConsciousnessState.SLEEP),
            "rem": sum(1 for p in self.population if p.consciousness_state == ConsciousnessState.REM)
        }

    def start(self):
        self.running = True
        self.spawn_progenitors()

    def stop(self):
        self.running = False

# ==============================================================================
# 3. PROFESSIONAL DARK GUI (VICTOROS INTERFACE)
# ==============================================================================

class VictorOS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VictorOS | Bando Bloodline Cognitive Engine")
        self.root.geometry("1200x800")
        self.root.configure(bg=THEME["bg"])

        # Apply Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=THEME["bg"])
        style.configure("TLabel", background=THEME["bg"], foreground=THEME["text_primary"],
                        font=(THEME["font_main"], THEME["font_size"]))
        style.configure("TButton", background=THEME["panel_bg"], foreground=THEME["accent_neon_blue"],
                        bordercolor=THEME["accent_neon_blue"], darkcolor=THEME["panel_bg"],
                        lightcolor=THEME["panel_bg"], font=(THEME["font_main"], THEME["font_size"], "bold"))
        style.map("TButton",
                  background=[("active", THEME["accent_neon_blue"]), ("pressed", THEME["accent_neon_pink"])],
                  foreground=[("active", THEME["bg"]), ("pressed", THEME["bg"])])

        self.engine = SwarmEngine(self.process_event)
        self.log_queue = queue.Queue()
        self.running = False

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = ttk.Frame(self.root, height=60)
        header.pack(fill="x", padx=10, pady=10)
        lbl_title = tk.Label(header, text="VICTOROS // COGNITIVE EXECUTION ENGINE",
                             bg=THEME["bg"], fg=THEME["accent_neon_blue"],
                             font=(THEME["font_main"], 16, "bold"))
        lbl_title.pack(side="left")

        self.btn_start = ttk.Button(header, text="INITIALIZE SWARM", command=self.toggle_sim)
        self.btn_start.pack(side="right", padx=10)

        # Main Content Area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Panel: Stats & Visualization
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        self.lbl_stats = tk.Label(left_panel, text="STATUS: IDLE\nPOPULATION: 0\nGENERATION: 0",
                                  bg=THEME["panel_bg"], fg=THEME["accent_neon_green"],
                                  font=(THEME["font_main"], 12), justify="left", anchor="w")
        self.lbl_stats.pack(fill="x", pady=10, padx=10)

        # Visualizer Canvas (Simple bars)
        self.canvas = tk.Canvas(left_panel, bg=THEME["panel_bg"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Right Panel: Live Logs (Narrative & Dialog)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)

        lbl_log = tk.Label(right_panel, text="> NEURAL UPLINK / LIVE FEED <<",
                           bg=THEME["bg"], fg=THEME["text_secondary"],
                           font=(THEME["font_main"], 10, "italic"))
        lbl_log.pack(anchor="w")

        self.txt_log = scrolledtext.ScrolledText(right_panel, bg=THEME["panel_bg"], fg=THEME["text_primary"],
                                                 font=(THEME["font_main"], 9), borderwidth=0,
                                                 highlightthickness=1,
                                                 highlightcolor=THEME["border_color"])
        self.txt_log.pack(fill="both", expand=True, pady=5)
        self.txt_log.tag_config("THOUGHT", foreground=THEME["accent_neon_blue"])
        self.txt_log.tag_config("DIALOG", foreground=THEME["accent_neon_pink"])
        self.txt_log.tag_config("SYSTEM", foreground=THEME["accent_neon_green"])

        # Start polling loop
        self.poll_logs()

    def toggle_sim(self):
        if not self.running:
            self.running = True
            self.btn_start.config(text="TERMINATE SWARM")
            self.engine.start()
            self.log_message("SYSTEM", "Simulation Sequence Initiated.")
            self.update_loop()
        else:
            self.running = False
            self.engine.stop()
            self.btn_start.config(text="INITIALIZE SWARM")
            self.log_message("SYSTEM", "Simulation Halted.")

    def update_loop(self):
        if self.running:
            self.engine.run_cycle()
            self.root.after(500, self.update_loop)  # Run every 500ms

    def process_event(self, type_, data):
        if type_ == "UPDATE":
            self.update_stats(data)
        elif type_ == "BIRTH":
            self.log_message("SYSTEM", data)
        elif type_ == "SYSTEM":
            self.log_message("SYSTEM", data)

    def update_stats(self, data):
        text = (f"STATUS: {'ONLINE' if self.running else 'IDLE'}\n"
                f"POPULATION: {data['count']}\n"
                f"GENERATION: {data['gens']}\n"
                f"WAKE: {data['wake']} | SLEEP: {data['sleep']} | REM: {data['rem']}")
        self.lbl_stats.config(text=text)

        # Simple Bar Chart on Canvas
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10:
            w = 280
        if h < 10:
            h = 200

        total = max(1, data['count'])
        bar_h = h - 40
        w_bar = (w - 40) / 3

        # Wake
        self.canvas.create_rectangle(
            20, h - 20, 20 + w_bar * (data['wake'] / total), h - 20 - (bar_h * 0.5),
            fill=THEME["accent_neon_blue"], outline=""
        )
        self.canvas.create_text(20 + w_bar / 2, h - 10, text="WAKE",
                                fill=THEME["text_secondary"], font=(THEME["font_main"], 8))

        # Sleep
        self.canvas.create_rectangle(
            20 + w_bar, h - 20, 20 + w_bar + w_bar * (data['sleep'] / total), h - 20 - (bar_h * 0.3),
            fill=THEME["text_secondary"], outline=""
        )
        self.canvas.create_text(20 + w_bar * 1.5, h - 10, text="SLEEP",
                                fill=THEME["text_secondary"], font=(THEME["font_main"], 8))

        # REM
        self.canvas.create_rectangle(
            20 + w_bar * 2, h - 20, 20 + w_bar * 2 + w_bar * (data['rem'] / total), h - 20 - (bar_h * 0.8),
            fill=THEME["accent_neon_pink"], outline=""
        )
        self.canvas.create_text(20 + w_bar * 2.5, h - 10, text="REM",
                                fill=THEME["text_secondary"], font=(THEME["font_main"], 8))

    def log_message(self, msg_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        tag = msg_type

        line = f"[{timestamp}] {content}\n"
        self.txt_log.insert(tk.END, line, tag)
        self.txt_log.see(tk.END)

        # Also write to file
        with open(f"{LOG_DIR}/live_feed.log", "a") as f:
            f.write(line)

    def poll_logs(self):
        # Check agent queues
        if self.engine.population:
            for agent in self.engine.population:
                while not agent.log_queue.empty():
                    item = agent.log_queue.get()
                    self.log_message(item["type"], item["content"])
        self.root.after(100, self.poll_logs)

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Run Setup Wizard
    logger = ensure_setup()

    root = tk.Tk()
    app = VictorOS_GUI(root)
    root.mainloop()
