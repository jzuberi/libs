import time
import threading
from abc import ABC, abstractmethod


class BaseBackgroundAgent(ABC):
    """
    Project-level background orchestrator.
    Manages multiple workflows, each with its own ActorRecord,
    instructions, state, scoring, cooldown, and budget.
    """

    def __init__(self, workflow_loaders, idle_seconds=3600, debug=True):
        """
        workflow_loaders: dict[str, callable -> ActorRecord]
        """
        self.idle_seconds = idle_seconds
        self.debug = debug
        self.last_user_activity = time.time()

        # Load all workflows + their ActorRecords
        self.workflows = {}
        for name, loader in workflow_loaders.items():
            record = loader(
                actor_id=f"background:{name}",
                actor_type="background",
                source="background_agent",
                workflow_name=name,
            )

            self.workflows[name] = {
                "record": record,
                "instructions": self.get_workflow_instructions(record),
                "state": {
                    "last_run": 0,
                    "budget_used": 0,
                    "last_budget_reset": time.time(),   # ⭐ NEW
                    "last_output": None,
                    "errors": 0,
                    "consecutive_errors": 0,
                    "consecutive_idle": 0,
                },
            }

        self.running = False
        self.thread = None

    # ------------------------------------------------------------
    # Abstract: workflow instructions + contextual demand
    # ------------------------------------------------------------
    @abstractmethod
    def get_workflow_instructions(self, record):
        pass

    @abstractmethod
    def compute_contextual_demand(self, record):
        pass

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def start(self):
        if self.running:
            return
        self.running = True
        if self.debug:
            print("[background] starting project-level orchestrator")
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        if self.debug:
            print("[background] stopping project-level orchestrator")
        self.running = False

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def loop(self):

        while self.running:

            if self.debug:
                print("[background] loop tick")
                time.sleep(300)

            try:
                self.run_one_cycle()
            except Exception as e:
                print(f"[background] ❌ Error: {e}")

            time.sleep(30)

    # ------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------
    def run_one_cycle(self):
        wf = self.pick_workflow()
        if not wf:
            if self.debug:
                print("[background] no eligible workflows to run")
            return

        if self.debug:
            print(f"[background] selected workflow: {wf}")

        self.run_workflow(wf)

    # ------------------------------------------------------------
    # Scoring + Eligibility
    # ------------------------------------------------------------
    def pick_workflow(self):
        scores = {}

        if self.debug:
            print("[background] evaluating workflows...")

        now = time.time()

        for name, data in self.workflows.items():
            instr = data["instructions"]
            state = data["state"]
            record = data["record"]

            if self.debug:
                print(f"\n[background] checking workflow: {name}")

            # Enabled?
            if not instr.get("enabled", True):
                if self.debug:
                    print("  - disabled, skipping")
                continue

            # Cooldown
            since = now - state["last_run"]
            cooldown = instr.get("cooldown_seconds", 0)
            if since < cooldown:
                if self.debug:
                    print(f"  - cooldown active ({since:.1f}s < {cooldown}s), skipping")
                continue

            # ⭐ BUDGET WINDOW RESET
            window = 3600  # 1 hour
            if now - state["last_budget_reset"] >= window:
                if self.debug:
                    print("  - resetting budget window")
                state["budget_used"] = 0
                state["last_budget_reset"] = now

            # Budget check
            used = state["budget_used"]
            budget = instr.get("budget_per_hour", 999)
            if self.debug:
                print(f"  - budget used: {used}/{budget}")
            if used >= budget:
                if self.debug:
                    print("  - budget exceeded, skipping")
                continue

            # Scoring components
            static = instr.get("static_priority", 1.0)
            aging = max(since / 3600, 0.01)
            contextual = self.compute_contextual_demand(record)

            score = static * aging * contextual
            scores[name] = score

            if self.debug:
                print(f"  - static_priority:     {static}")
                print(f"  - aging_factor:        {aging:.4f}")
                print(f"  - contextual_demand:   {contextual}")
                print(f"  - score:               {score:.4f}")

        if not scores:
            return None

        best = max(scores, key=scores.get)

        if self.debug:
            print(f"[background] best workflow: {best} (score={scores[best]:.4f})")

        return best

    # ------------------------------------------------------------
    # Run workflow
    # ------------------------------------------------------------
    def run_workflow(self, name):
        data = self.workflows[name]
        record = data["record"]
        state = data["state"]

        entrypoint = "[background_agent]: run it"

        if self.debug:
            print(f"[background] calling workflow entrypoint: {entrypoint}")

        try:
            resp = record.chat(entrypoint)

            assistant_msg = resp["message"]
            assistant_event = resp["assistant_event"]
            handler_name = assistant_event["payload"].get("handler_name")
            item_id = assistant_event.get("item_id")

            # Update bookkeeping
            state["last_output"] = assistant_msg
            state["last_event"] = assistant_event
            state["last_run"] = time.time()

            # ⭐ Consume budget
            state["budget_used"] += 1

        except Exception as e:
            state["errors"] += 1
            state["consecutive_errors"] += 1
            print(f"[background] ERROR in workflow {name}: {e}")

            if state["consecutive_errors"] >= 3:
                state["disabled_by_orchestrator"] = True
                if self.debug:
                    print(f"[background] disabling workflow {name} due to repeated errors")
            return False

        # ------------------------------------------------------------
        # BASIC TERMINATION LOGIC
        # ------------------------------------------------------------
        if handler_name is None:
            if self.debug:
                print(f"[background] workflow {name} returned handler_name=None → idle")
            state["consecutive_idle"] += 1

        elif item_id is None:
            if self.debug:
                print(f"[background] workflow {name} has no active item → idle")
            state["consecutive_idle"] += 1

        elif "not sure what you want" in assistant_msg.lower():
            if self.debug:
                print(f"[background] workflow {name} fallback message → idle")
            state["consecutive_idle"] += 1

        else:
            if self.debug:
                print(f"[background] workflow {name} made progress")
            state["consecutive_idle"] = 0
            state["consecutive_errors"] = 0
            return True

        # ------------------------------------------------------------
        # IDLE HANDLING
        # ------------------------------------------------------------
        if state["consecutive_idle"] >= 3:
            state["disabled_by_orchestrator"] = True
            if self.debug:
                print(f"[background] disabling workflow {name} due to repeated idle")
            return False

        return False
