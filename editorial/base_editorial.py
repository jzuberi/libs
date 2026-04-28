from __future__ import annotations

import ast
import copy
import re
import tempfile
import math
from dataclasses import is_dataclass, asdict
import uuid

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from pydantic import BaseModel, Field
from transitions.extensions import HierarchicalMachine as Machine

from base_models import (
    BaseIdea,
    BaseAsset,
    BaseFinalOutput,
    EngineState,
    IdeaProgressStep,
    IdeaProgress,
)

class BaseEditorialEngine:
    """
    Lifecycle:

        ACQUISITION → INGESTION → IDEATION → APPROVAL → ASSET → EXPORT
    """

    def __init__(self, instance_dir: str, agent_llm: Optional[Callable[[str], str]] = None):
        
        self.instance_path = Path(instance_dir)
        self.instance_path.mkdir(parents=True, exist_ok=True)

        self.ideas_path = self.instance_path / "ideas"
        self.assets_path = self.instance_path / "assets"
        self.outputs_path = self.instance_path / "outputs"
        for p in [self.ideas_path, self.assets_path, self.outputs_path]:
            p.mkdir(exist_ok=True)

        self.state_file = self.instance_path / "engine_state.json"

        self._ideas: Dict[str, BaseIdea] = {}
        self._assets: Dict[str, BaseAsset] = {}
        self._outputs: Dict[str, BaseFinalOutput] = {}

        self.substate_definitions: Dict[str, List[str]] = {
            "ACQUISITION": [],
            "INGESTION": [],
            "IDEATION": [],
            "APPROVAL": [],
            "ASSET": [],
            "EXPORT": [],
        }

        if self.state_file.exists():
            self.engine_state = self._load_engine_state()
        else:
            self.engine_state = EngineState(
                branch="ACQUISITION",
                substate=self._initial_substate("ACQUISITION"),
                context={},
                progress={branch.lower(): {} for branch in self.substate_definitions},
                metadata={
                    "engine_class": self.__class__.__name__,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
            self._save_engine_state()

        self._build_state_machine()
        self._ensure_progress_keys()

        # -------------------------------------------------------
        # NEW: agent_llm + trace
        # -------------------------------------------------------
        self.agent_llm = agent_llm  # used ONLY by the agent
        self.trace: List[Dict[str, Any]] = []
        self.trace_path = self.instance_path / "trace.jsonl"

        self.agent_tools: Dict[str, Callable[..., Any]] = {}
        self._register_default_agent_tools()

    def _register_default_agent_tools(self):
        """
        Register the default set of safe, domain-agnostic tools
        that the agent is allowed to call.
        """
        self.agent_tools = {
            "list_ideas": self._tool_list_ideas,
            "summarize_idea": self._tool_summarize_idea,
            "idea_status": self._tool_idea_status,
            "approve_substate": self._tool_approve_substate,
            "run_step": self._tool_run_step,
            "export": self._tool_export,
        }

    # ---------- TOOL IMPLEMENTATIONS ----------

    def _tool_list_ideas(self) -> List[str]:
        return self.all_ideas()

    def _tool_summarize_idea(self, idea_id: str) -> Dict[str, Any]:
        idea = self._load_or_get_idea(idea_id)
        return self.summarize_idea_structured(idea)

    def _tool_idea_status(self, idea_id: str) -> Dict[str, Any]:
        return self.next_action_for_idea(idea_id)

    def _tool_approve_substate(self, idea_id: str) -> bool:
        wf = self._load_workflow(idea_id)
        wf["approved"] = True
        wf["updated_at"] = datetime.utcnow().isoformat()
        wf_path = self.assets_path / idea_id / "workflow.json"
        wf_path.write_text(json.dumps(wf, indent=2))
        return True

    def _tool_run_step(self, idea_id: str) -> None:
        self.generate_assets(idea_id)

    def _tool_export(self, idea_id: str) -> None:
        self.export(idea_id)




    # -----------------------------------------------------------------------
    # Editorial ledger
    # -----------------------------------------------------------------------
    @staticmethod
    def editorial_step(
        *,
        action: str,
        branch: str | None = None,
        substate: str | None = None,
        actor_param: str = "actor",
    ):
        def decorator(func):
            def wrapper(engine_self, *args, **kwargs):

                # ------------------------------------------------------------
                # 1. Active idea
                # ------------------------------------------------------------
                idea_id = engine_self.engine_state.context.get("current_idea_id")
                if not idea_id:
                    raise ValueError(
                        "No current_idea_id found in engine_state.context. "
                        "Decorated functions require an active idea."
                    )

                # ------------------------------------------------------------
                # 2. Actor
                # ------------------------------------------------------------
                actor = (
                    kwargs.get(actor_param)
                    or engine_self.engine_state.context.get("actor")
                    or "agent"
                )

                # ------------------------------------------------------------
                # 3. Execute wrapped function
                # ------------------------------------------------------------
                result = func(engine_self, *args, **kwargs)

                # ------------------------------------------------------------
                # 4. Extract artifact/details
                # ------------------------------------------------------------
                artifact = None
                details = {}
                if isinstance(result, dict):
                    artifact = result.get("artifact")
                    details = result.get("details", {})

                # ------------------------------------------------------------
                # 5. Extract assets
                # ------------------------------------------------------------
                if isinstance(result, dict):
                    assets = result.get("assets", [])
                elif isinstance(result, list):
                    assets = result
                else:
                    raise RuntimeError(
                        f"editorial_step: function {func.__name__} returned invalid type {type(result)}"
                    )

                # ------------------------------------------------------------
                # 6. LOAD WORKFLOW STATE *BEFORE* logging
                # ------------------------------------------------------------
                wf_before = engine_self._load_workflow(idea_id)
                inferred_branch = branch or wf_before["branch"]
                inferred_substate = substate or wf_before["substate"]

                # ------------------------------------------------------------
                # 7. Log editorial step (ledger)
                # ------------------------------------------------------------
                engine_self.update_idea_progress(
                    idea_id,
                    branch=inferred_branch,
                    substate=inferred_substate,
                    action=action,
                    actor=actor,
                    artifact=artifact,
                    details=details,
                )

                # ------------------------------------------------------------
                # 8. Update workflow.json and advance workflow
                # ------------------------------------------------------------
                engine_self._update_workflow_after_editorial_step(
                    idea_id,
                    branch=inferred_branch,
                    substate=inferred_substate,
                    action=action,
                )

                # ------------------------------------------------------------
                # 9. LOAD WORKFLOW STATE AGAIN (after advancement)
                # ------------------------------------------------------------
                wf_after = engine_self._load_workflow(idea_id)
                # (You may want to use wf_after for debugging or UI updates)

                # ------------------------------------------------------------
                # 10. Return ONLY assets
                # ------------------------------------------------------------
                return assets

            return wrapper
        return decorator

    def _idea_progress_path(self, idea_id: str) -> Path:
        return self.assets_path / idea_id / "idea_progress.json"

    def _load_idea_progress(self, idea_id: str) -> IdeaProgress:
        path = self._idea_progress_path(idea_id)
        if path.exists():
            data = json.loads(path.read_text())
            return IdeaProgress(**data)
        return IdeaProgress(idea_id=idea_id)

    def _save_idea_progress(self, progress: IdeaProgress):
        path = self._idea_progress_path(progress.idea_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(progress.model_dump_json(indent=2, exclude_none=True))

    def list_unexported_approved_ideas(self) -> List[str]:
        """
        Return a list of idea_ids that:
        - are approved
        - have NOT been exported (no manifest.json)
        """

        results = []

        for idea in self._iter_ideas():
            if not idea.approved:
                continue

            manifest_path = self.assets_path / idea.id / "manifest.json"
            if not manifest_path.exists():
                results.append(idea)

        return results


    def restart_asset_pipeline(self, idea_id: str):
        """
        Completely restart the ASSET pipeline for a given idea.

        - Deletes all generated assets for the idea
        - Resets workflow.json to the first ASSET substate
        - Preserves the chosen asset_workflow (carousel/reel)
        - Clears ASSET progress in idea_progress.json
        """

        # ------------------------------------------------------------
        # 1. Load workflow.json so we can preserve asset_workflow
        # ------------------------------------------------------------
        wf = self._load_workflow(idea_id)
        asset_workflow = wf.get("asset_workflow")

        # ------------------------------------------------------------
        # 2. Delete all assets for this idea
        # ------------------------------------------------------------
        asset_dir = self.assets_path / idea_id
        if asset_dir.exists():
            for f in asset_dir.iterdir():
                # Keep workflow.json and idea_progress.json
                if f.name not in ("workflow.json", "idea_progress.json"):
                    if f.is_file():
                        f.unlink()
                    else:
                        shutil.rmtree(f)

        # ------------------------------------------------------------
        # 3. Reset workflow.json to the first ASSET substate
        # ------------------------------------------------------------
        first_substate = self._initial_substate("ASSET")

        # Determine if first substate requires approval
        requires_approval = False
        for s in self.substate_definitions.get("ASSET", []):
            if s["name"] == first_substate:
                requires_approval = s.get("requires_approval", False)
                break

        new_wf = {
            "branch": "ASSET",
            "substate": first_substate,
            "approved": not requires_approval,
            "completed": False,
            "asset_workflow": asset_workflow,  # preserve workflow choice
            "updated_at": datetime.utcnow().isoformat(),
        }

        self._save_workflow(idea_id, new_wf)

        # ------------------------------------------------------------
        # 4. Clear ASSET progress in idea_progress.json
        # ------------------------------------------------------------
        progress = self._load_idea_progress(idea_id)
        progress["ASSET"] = []  # wipe ASSET ledger entries
        self._save_idea_progress(progress)




    def _update_workflow_after_editorial_step(
        self, 
        idea_id: str, 
        branch: str, 
        substate: str, 
        action: str
        ):

        """
        Update workflow.json after an editorial step is completed.
        This keeps workflow.json in sync with idea_progress.json.
        """

        wf = self._load_workflow(idea_id)

        # Only update if this is the current substate
        if wf["branch"] != branch or wf["substate"] != substate:
            return

        # Mark substate as completed (but NOT approved)
        wf["completed"] = True

        # Check if this substate requires approval
        requires_approval = False
        for s in self.substate_definitions.get(branch, []):
            if s["name"] == substate:
                requires_approval = s.get("requires_approval", False)
                break

        # Always save workflow.json before deciding what to do next
        self._save_workflow(idea_id, wf)

        # If approval IS required → STOP HERE
        if requires_approval:
            return  # wait for accept_asset_substate()

        # If approval is NOT required → auto-advance
        self._advance_substate(idea_id)



    def update_idea_progress(
        self,
        idea_id: str,
        *,
        branch: str,                   # maps to EngineState.branch
        substate: str,                 # maps to EngineState.substate
        action: str,                   # created | edited | approved | ...
        actor: str,                    # user | agent
        artifact: Dict[str, Any] | None = None,
        details: dict | None = None,
        merge_with_last: bool = False,
    ):
        """
        Append (or merge) an editorial step into idea_progress.json.
        """
        progress = self._load_idea_progress(idea_id)
        details = details or {}

        step = IdeaProgressStep(
            branch=branch,
            substate=substate,
            action=action,
            actor=actor,
            artifact=artifact,
            details=details,
        )

        if merge_with_last and progress.steps:
            last = progress.steps[-1]
            if (
                last.branch == branch
                and last.substate == substate
                and last.action == action
                and last.actor == actor
            ):
                # merge artifacts
                if artifact:
                    if last.artifact is None:
                        last.artifact = artifact
                    else:
                        last.artifact.update(artifact)

                # merge details
                last.details.update(details)
                last.updated_at = datetime.utcnow()
            else:
                progress.steps.append(step)
        else:
            progress.steps.append(step)

        self._save_idea_progress(progress)


        # After saving the ledger, update workflow.json
        self._update_workflow_after_editorial_step(
            idea_id,
            branch=branch,
            substate=substate,
            action=action,
        )



    # -----------------------------------------------------------------------
    # Init / persistence
    # -----------------------------------------------------------------------

    def _ensure_progress_keys(self):
        for branch in self.substate_definitions:
            self.engine_state.progress.setdefault(branch.lower(), {})
        self._save_engine_state()

    


    def _iter_ideas(self):
        for path in self.ideas_path.glob("*.json"):
            idea_id = path.stem
            yield self._load_or_get_idea(idea_id)

    def list_fresh_ideas(self, max_age_seconds: float = 1.0):
        """
        Return ideas that have not been modified since creation
        (i.e., no approval/rejection yet), ordered by recency.
        """
        fresh: List[BaseIdea] = []

        for idea in self._iter_ideas():
            if idea.approved:
                continue

            delta = abs(idea.updated_at - idea.created_at)
            if delta <= timedelta(seconds=max_age_seconds):
                fresh.append(idea)

        fresh.sort(key=lambda i: i.created_at, reverse=True)
        return fresh

    def append_trace(self, event: Dict[str, Any]) -> None:
        """Append a structured trace event in memory and to trace.jsonl."""
        full_event = {
            "timestamp": datetime.utcnow().isoformat(),
            **event,
        }
        self.trace.append(full_event)

        # Persist to disk as JSONL (one event per line)
        try:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(full_event) + "\n")
        except Exception:
            # For now, fail soft; later we can add logging/metrics
            pass

    def summarize_idea_structured(self, idea: BaseIdea, small: bool = False) -> Dict[str, Any]:
        """
        Domain-agnostic structured summary of an idea.
        Can be overridden by subclasses (e.g., SCOTUS) for richer views.
        """

        if(not small):
            idea_structure = {
                    "id": idea.id,
                    "description": idea.description,
                    "metadata": dict(idea.metadata or {}),
                    "created_at": idea.created_at.isoformat() if hasattr(idea, "created_at") else None,
                    "updated_at": idea.updated_at.isoformat() if hasattr(idea, "updated_at") else None,
                    "approved": getattr(idea, "approved", None),
                }
        else:

            idea_structure = {
                    "id": idea.id,
                    "description": idea.description,
                    "approved": getattr(idea, "approved", None),
                }

        return idea_structure

    def build_context_snapshot(self, focused_idea_id=None, max_chars=None):
        """
        Build a snapshot of the current editorial landscape with fallback tiers
        to ensure the snapshot fits within max_chars if provided.
        """

        import json

        # Local helper for size checking
        def _fits(snap):
            if max_chars is None:
                return True
            return len(json.dumps(snap)) <= max_chars

        # --- Tier 1: Full snapshot ---------------------------------------------
        snap = {
            "engine": self._engine_state_snapshot(),
            "ideas_index": self._build_ideas_index(),
            "ideas": self._build_all_idea_summaries(),
            "focused_idea": self._build_focused_idea(focused_idea_id),
        }
        if _fits(snap):
            return snap

        # --- Tier 2: Index + focused idea --------------------------------------
        snap = {
            "engine": self._engine_state_snapshot(),
            "ideas_index": self._build_ideas_index(),
            "focused_idea": self._build_focused_idea(focused_idea_id),
        }
        if _fits(snap):
            return snap

        # --- Tier 3: Truncated index + focused idea ----------------------------
        full_index = self._build_ideas_index()
        for limit in (50, 20, 10, 5):
            snap = {
                "engine": self._engine_state_snapshot(),
                "ideas_index": full_index[-limit:],
                "focused_idea": self._build_focused_idea(focused_idea_id),
            }
            if _fits(snap):
                return snap

        # --- Tier 4: Minimal snapshot ------------------------------------------
        return {
            "engine": self._engine_state_snapshot(),
            "focused_idea": self._build_focused_idea(focused_idea_id),
        }


    # ---------------------------------------------------------------------------
    # Supporting helpers (still inside BaseEditorialEngine)
    # ---------------------------------------------------------------------------

    def _engine_state_snapshot(self):
        return {
            "branch": self.engine_state.branch,
            "substate": self.engine_state.substate,
        }

    def _build_ideas_index(self):
        return [
            {
                "id": idea.id,
                "description": getattr(idea, "description", None),
                "approved": getattr(idea, "approved", None),
            }
            for idea in self._iter_ideas()
        ]

    def _build_all_idea_summaries(self):
        return [
            {
                "id": idea.id,
                "summary": self.summarize_idea_structured(idea, small=True),
            }
            for idea in self._iter_ideas()
        ]

    def _build_focused_idea(self, idea_id):
        if not idea_id:
            return None
        idea = self.get_idea(idea_id)
        return {
            "id": idea.id,
            "summary": self.summarize_idea_structured(idea, small=False),
            "status": self.next_action_for_idea(idea.id),
        }


    # ================================================================
    # ATOMIC PREDICATES (boolean helpers)
    # ================================================================

    def is_approved(self, idea_id: str) -> bool:
        """Return True if the idea has been approved."""
        return self._load_or_get_idea(idea_id).approved


    def is_exported(self, idea_id: str) -> bool:
        """
        Return True if the idea has been exported.
        Only meaningful for approved ideas.
        """
        if not self.is_approved(idea_id):
            return False
        return (self.assets_path / idea_id / "manifest.json").exists()


    def is_done(self, idea_id: str) -> bool:
        """
        Return True if the idea has completed EXPORT.
        Only meaningful for approved ideas.
        """
        if not self.is_approved(idea_id):
            return False

        wf = self._load_workflow(idea_id)
        return wf["branch"] == "DONE" and wf["substate"] == "complete"


    def requires_approval(self, idea_id: str) -> bool:
        """
        Return True if the current substate requires approval.
        Only meaningful for approved ideas.
        """
        if not self.is_approved(idea_id):
            return False

        wf = self._load_workflow(idea_id)
        branch = wf["branch"]
        sub = wf["substate"]

        for s in self.substate_definitions.get(branch, []):
            if s["name"] == sub:
                return s.get("requires_approval", False) and not wf["approved"]

        return False


    def can_run_step(self, idea_id: str) -> bool:
        """
        Return True if the current substate is approved but not completed.
        Only meaningful for approved ideas.
        """
        if not self.is_approved(idea_id):
            return False

        wf = self._load_workflow(idea_id)
        return wf["approved"] and not wf["completed"]


    # ================================================================
    # ATOMIC SELECTORS (set-theoretic filters)
    # ================================================================

    def all_ideas(self) -> list[str]:
        """Return all idea_ids."""
        return [idea.id for idea in self._iter_ideas()]


    def filter_ideas(self, predicate) -> list[str]:
        """Return all idea_ids for which predicate(idea_id) is True."""
        return [idea_id for idea_id in self.all_ideas() if predicate(idea_id)]


    # ================================================================
    # COMPOSED HELPERS (built from atomic predicates + selectors)
    # ================================================================

    def approved_ideas(self) -> list[str]:
        return self.filter_ideas(self.is_approved)


    def unexported_approved_ideas(self) -> list[str]:
        return self.filter_ideas(lambda i: self.is_approved(i) and not self.is_exported(i))


    def ideas_requiring_approval(self) -> list[str]:
        return self.filter_ideas(
            lambda i: self.is_approved(i) and self.requires_approval(i)
        )


    def ideas_ready_for_llm_action(self) -> list[str]:
        return self.filter_ideas(self.can_run_step)


    def completed_ideas(self) -> list[str]:
        return self.filter_ideas(self.is_done)


    def ideas_ready_for_export(self) -> list[str]:
        """
        Ideas that are approved, not done, and have all assets approved.
        (You can refine this depending on your EXPORT rules.)
        """
        return self.filter_ideas(lambda i: self.is_approved(i) and not self.is_done(i))


    # ================================================================
    # HIGH-LEVEL DECISION HELPER (uses atomic predicates)
    # ================================================================

    def next_action_for_idea(self, idea_id: str) -> dict:
        """
        Return a structured description of what the LLM should do next
        for a given idea.
        """

        wf = self._load_workflow(idea_id)

        branch = wf["branch"]
        substate = wf["substate"]
        approved = wf["approved"]
        completed = wf["completed"]

        # Terminal state (EXPORT complete)
        if branch == "EXPORT" and substate == "complete":
            return {
                "branch": branch,
                "substate": substate,
                "requires_approval": False,
                "approved": True,
                "completed": True,
                "can_run": False,
                "needs_user_approval": False,
                "is_done": True,
            }

        # Determine if this substate requires approval
        requires_approval = False
        for s in self.substate_definitions.get(branch, []):
            if s["name"] == substate:
                requires_approval = s.get("requires_approval", False)
                break

        # Needs approval
        if requires_approval and not approved:
            return {
                "branch": branch,
                "substate": substate,
                "requires_approval": True,
                "approved": False,
                "completed": completed,
                "can_run": False,
                "needs_user_approval": True,
                "is_done": False,
            }

        # Ready to run
        if approved and not completed:
            return {
                "branch": branch,
                "substate": substate,
                "requires_approval": requires_approval,
                "approved": True,
                "completed": False,
                "can_run": True,
                "needs_user_approval": False,
                "is_done": False,
            }

        # Completed → LLM should advance automatically
        if completed:
            return {
                "branch": branch,
                "substate": substate,
                "requires_approval": requires_approval,
                "approved": approved,
                "completed": True,
                "can_run": False,
                "needs_user_approval": False,
                "is_done": False,
            }

        # Fallback
        return {
            "branch": branch,
            "substate": substate,
            "requires_approval": requires_approval,
            "approved": approved,
            "completed": completed,
            "can_run": False,
            "needs_user_approval": False,
            "is_done": False,
        }


    # -----------------------------------------------------------------------
    # State Machine Construction
    # -----------------------------------------------------------------------

    def _initial_substate(self, branch: str) -> Optional[str]:
        substates = self.substate_definitions.get(branch, [])
        if not substates:
            return None
        first = substates[0]
        return first["name"] if isinstance(first, dict) else first



    def _normalize_substates(self, substates):
        normalized = []
        for s in substates:
            if isinstance(s, str):
                normalized.append({"name": s, "requires_approval": False})
            elif isinstance(s, dict):
                normalized.append({
                    "name": s["name"],
                    "requires_approval": s.get("requires_approval", False),
                })
            else:
                raise ValueError("Invalid substate definition")
        return normalized

    def _build_state_machine(self):
        """
        Build a transitions HierarchicalMachine using enriched substate definitions.
        Each substate may be a string or a dict with metadata such as requires_approval.
        Only the 'name' field is used for the state machine.
        """

        states: List[Any] = []

        for branch, substates in self.substate_definitions.items():
            # Normalize substates into dicts
            normalized = []
            for s in substates:
                if isinstance(s, str):
                    normalized.append({"name": s, "requires_approval": False})
                elif isinstance(s, dict):
                    # ensure required fields exist
                    normalized.append({
                        "name": s["name"],
                        "requires_approval": s.get("requires_approval", False),
                    })
                else:
                    raise ValueError(f"Invalid substate definition: {s}")

            # Replace original list with normalized version
            self.substate_definitions[branch] = normalized

            # Extract names for transitions
            substate_names = [s["name"] for s in normalized]

            if substate_names:
                states.append(
                    {
                        "name": branch,
                        "children": substate_names,
                        "initial": substate_names[0],
                    }
                )
            else:
                states.append(branch)

        # Determine initial state
        if self.engine_state.substate:
            initial_state = f"{self.engine_state.branch}.{self.engine_state.substate}"
        else:
            initial_state = self.engine_state.branch

        # Build the machine
        self.machine = Machine(
            model=self,
            states=states,
            initial=initial_state,
            auto_transitions=False,
        )


    # -----------------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------------

    def _load_engine_state(self) -> EngineState:
        data = json.loads(self.state_file.read_text())
        return EngineState(**data)

    def _save_engine_state(self):
        self.engine_state.metadata["updated_at"] = datetime.utcnow().isoformat()
        self.state_file.write_text(self.engine_state.model_dump_json(indent=2))

    def _save_idea(self, idea: BaseIdea):
        self._ideas[idea.id] = idea
        (self.ideas_path / f"{idea.id}.json").write_text(
            idea.model_dump_json(indent=2)
        )

    def _save_asset(self, asset: BaseAsset):
        """
        Save an asset JSON inside the idea-specific asset directory.
        """
        self._assets[asset.id] = asset

        idea_dir = self.assets_path / asset.idea_id
        idea_dir.mkdir(parents=True, exist_ok=True)

        asset_path = idea_dir / f"{asset.id}.json"
        asset_path.write_text(asset.model_dump_json(indent=2))

    def _save_output(self, output: BaseFinalOutput):
        self._outputs[output.id] = output
        (self.outputs_path / f"{output.id}.json").write_text(
            output.model_dump_json(indent=2)
        )

    # -----------------------------------------------------------------------
    # State Helpers
    # -----------------------------------------------------------------------

    def _set_state(self, branch: str, substate: Optional[str]):
        self.engine_state.branch = branch
        self.engine_state.substate = substate
        self._save_engine_state()

        if substate:
            self.state = f"{branch}.{substate}"
        else:
            self.state = branch

    def set_idea_state(self, idea_id: str, branch: str, substate: str):
        """
        Manually set the workflow state for a specific idea.
        This updates workflow.json only (per-idea workflow state).
        """

        # Load workflow.json
        wf = self._load_workflow(idea_id)

        # Validate branch exists
        if branch not in self.substate_definitions:
            raise ValueError(f"Unknown branch '{branch}'")

        # Validate substate exists in that branch
        valid_substates = [s["name"] for s in self.substate_definitions[branch]]
        if substate not in valid_substates:
            raise ValueError(
                f"Unknown substate '{substate}' for branch '{branch}'. "
                f"Valid: {valid_substates}"
            )

        # Determine whether this substate requires approval
        requires_approval = False
        for s in self.substate_definitions[branch]:
            if s["name"] == substate:
                requires_approval = s.get("requires_approval", False)
                break

        # Update workflow.json
        wf["branch"] = branch
        wf["substate"] = substate
        wf["approved"] = not requires_approval
        wf["completed"] = False
        wf["updated_at"] = datetime.utcnow().isoformat()

        self._save_workflow(idea_id, wf)


    def _advance_substate(self, idea_id: str):
        """
        Advance the workflow for a specific idea.

        - Reads workflow.json
        - Advances to next substate if available
        - Otherwise transitions to the next branch (ASSET → EXPORT → DONE)
        - Resets approval/completion flags
        - Never touches engine_state.substate
        """

        wf = self._load_workflow(idea_id)
        branch = wf["branch"]
        current = wf["substate"]

        # ------------------------------------------------------------
        # 1. Get ordered substates for this branch
        # ------------------------------------------------------------
        substates = self.substate_definitions.get(branch, [])
        if not substates:
            return  # No substates defined for this branch

        names = [s["name"] for s in substates]

        if current not in names:
            raise RuntimeError(f"Unknown substate '{current}' for branch '{branch}'")

        idx = names.index(current)

        # ------------------------------------------------------------
        # 2. If there is a next substate, move to it
        # ------------------------------------------------------------
        if idx + 1 < len(names):
            next_sub = names[idx + 1]

            # Determine if next substate requires approval
            requires_approval = False
            for s in substates:
                if s["name"] == next_sub:
                    requires_approval = s.get("requires_approval", False)
                    break

            wf["substate"] = next_sub
            wf["approved"] = not requires_approval
            wf["completed"] = False
            wf["updated_at"] = datetime.utcnow().isoformat()

            self._save_workflow(idea_id, wf)
            return

        # ------------------------------------------------------------
        # 3. No more substates → branch transition
        # ------------------------------------------------------------

        # ASSET → EXPORT
        if branch == "ASSET":
            next_branch = "EXPORT"
            next_sub = self._initial_substate(next_branch)

            # Determine if next substate requires approval
            requires_approval = False
            for s in self.substate_definitions.get(next_branch, []):
                if s["name"] == next_sub:
                    requires_approval = s.get("requires_approval", False)
                    break

            wf["branch"] = next_branch
            wf["substate"] = next_sub
            wf["approved"] = not requires_approval
            wf["completed"] = False
            wf["updated_at"] = datetime.utcnow().isoformat()

            self._save_workflow(idea_id, wf)
            return

        # EXPORT → DONE
        if branch == "EXPORT":
            # You can define DONE as a branch or mark idea complete
            wf["branch"] = "DONE"
            wf["substate"] = "complete"
            wf["approved"] = True
            wf["completed"] = True
            wf["updated_at"] = datetime.utcnow().isoformat()

            self._save_workflow(idea_id, wf)
            return

        # ------------------------------------------------------------
        # 4. Other branches can be added here
        # ------------------------------------------------------------
        return





    # -----------------------------------------------------------------------
    # Preconditions
    # -----------------------------------------------------------------------

    def _ensure_branch(self, expected_branch: str):
        if self.engine_state.branch != expected_branch:
            raise RuntimeError(
                f"Invalid branch: expected {expected_branch}, got {self.engine_state.branch}"
            )

    # -----------------------------------------------------------------------
    # ACQUISITION
    # -----------------------------------------------------------------------

    def acquire_data(self) -> Any:
        self._ensure_branch("ACQUISITION")

        result = self.acquire_data_impl()

        self.engine_state.progress["acquisition"]["completed"] = True
        self._set_state(
            branch="INGESTION",
            substate=self._initial_substate("INGESTION"),
        )

        return result

    def acquire_data_impl(self) -> Any:
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # INGESTION
    # -----------------------------------------------------------------------

    def ingest(self) -> Any:
        self._ensure_branch("INGESTION")

        result = self.ingest_impl()

        self.engine_state.progress["ingestion"]["completed"] = True
        self._set_state(
            branch="IDEATION",
            substate=self._initial_substate("IDEATION"),
        )

        return result

    def ingest_impl(self) -> Any:
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # IDEATION
    # -----------------------------------------------------------------------

    def add_manual_baseidea(
        self,
        description: str,
        metadata: dict | None = None,
        type: str = "manual",
    ) -> str:
        """
        Create a BaseIdea manually and insert it into the engine as a fresh idea.
        """
        idea_id = str(uuid.uuid4())

        idea = BaseIdea(
            id=idea_id,
            description=description,
            type=type,
            metadata=metadata or {},
            approved=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self._save_idea(idea)

        self.engine_state.progress["ideation"]["ideas_generated"] = True
        self._save_engine_state()

        self._set_state(branch="APPROVAL", substate=self._initial_substate("APPROVAL"))

        return idea_id

    def generate_ideas(self) -> List[BaseIdea]:
        self._ensure_branch("IDEATION")

        ideas = self.generate_ideas_impl()

        if not ideas:
            self.engine_state.progress["ideation"]["ideas_generated"] = False
            self._save_engine_state()
            return []

        for idea in ideas:
            self._save_idea(idea)

        self.engine_state.progress["ideation"]["ideas_generated"] = True
        self._save_engine_state()

        self._set_state(branch="APPROVAL", substate=self._initial_substate("APPROVAL"))
        return ideas

    def generate_ideas_impl(self) -> List[BaseIdea]:
        raise NotImplementedError


    # -----------------------------------------------------------------------
    # INITIALIZE IDEA WORKFLOW
    # -----------------------------------------------------------------------

    def _initialize_workflow_for_idea(self, idea_id: str):
        """
        Create a fresh workflow.json for a newly approved idea.
        This is called BEFORE any ledger entries are written.
        """

        first_substate = self._initial_substate("ASSET")

        # Determine whether the first substate requires approval
        requires_approval = False
        for s in self.substate_definitions.get("ASSET", []):
            if s["name"] == first_substate:
                requires_approval = s.get("requires_approval", False)
                break

        workflow = {
            "branch": "ASSET",
            "substate": first_substate,

            # The substate is NOT approved if it requires approval
            "approved": not requires_approval,

            # The substate is NOT completed yet
            "completed": False,

            # Will be filled in by approve_idea()
            "asset_workflow": None,

            "updated_at": datetime.utcnow().isoformat(),
        }

        wf_path = self.assets_path / idea_id / "workflow.json"
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        wf_path.write_text(json.dumps(workflow, indent=2))

    def _load_workflow(self, idea_id: str) -> dict:
        """
        Load workflow.json for an idea.
        If it does not exist (legacy idea), create the assets/<idea_id>/ directory
        and initialize a fresh workflow.json.
        """

        idea_assets_dir = self.assets_path / idea_id
        wf_path = idea_assets_dir / "workflow.json"

        # ------------------------------------------------------------
        # Ensure the idea's asset directory exists (legacy fix #2)
        # ------------------------------------------------------------
        if not idea_assets_dir.exists():
            idea_assets_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # Legacy idea: workflow.json does not exist → auto-migrate
        # ------------------------------------------------------------
        if not wf_path.exists():
            idea = self._load_or_get_idea(idea_id)

            # Determine initial ASSET substate
            initial_substate = self._initial_substate("ASSET")

            # Determine if initial substate requires approval
            requires_approval = False
            for s in self.substate_definitions.get("ASSET", []):
                if s["name"] == initial_substate:
                    requires_approval = s.get("requires_approval", False)
                    break

            wf = {
                "branch": "ASSET",
                "substate": initial_substate,
                "approved": idea.approved and not requires_approval,
                "completed": False,
                "asset_workflow": None,
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Save the newly created workflow.json
            wf_path.write_text(json.dumps(wf, indent=2))
            return wf

        # ------------------------------------------------------------
        # Normal case: workflow.json exists
        # ------------------------------------------------------------
        return json.loads(wf_path.read_text())




    def _save_workflow(self, idea_id: str, workflow: dict):
        workflow["updated_at"] = datetime.utcnow().isoformat()
        wf_path = self.assets_path / idea_id / "workflow.json"
        wf_path.write_text(json.dumps(workflow, indent=2))

    def _set_idea_state(self, idea_id: str, branch: str, substate: str):
        wf = self._load_workflow(idea_id)
        wf["branch"] = branch
        wf["substate"] = substate
        self._save_workflow(idea_id, wf)

    def set_workflow(self, asset_workflow: str, actor: str = "user"):
        """
        Set or change the asset_workflow for a given idea.

        - Validates the workflow value
        - Updates workflow.json for that idea
        - Logs the change in idea_progress
        """

        idea_id = self.engine_state.context.get("current_idea_id")

        if asset_workflow not in ("carousel", "reel"):
            raise ValueError(f"Invalid asset_workflow: {asset_workflow}")

        # Ensure workflow.json exists
        wf = self._load_workflow(idea_id)

        # Update workflow.json
        wf["asset_workflow"] = asset_workflow
        wf["updated_at"] = datetime.utcnow().isoformat()
        wf_path = self.assets_path / idea_id / "workflow.json"
        wf_path.write_text(json.dumps(wf, indent=2))

        # Log in editorial ledger
        self.update_idea_progress(
            idea_id=idea_id,
            branch="ASSET",
            substate=wf["substate"],
            action="workflow_set",
            actor=actor,
            artifact=None,
            details={"asset_workflow": asset_workflow},
        )


    # -----------------------------------------------------------------------
    # APPROVAL
    # -----------------------------------------------------------------------
    def approve_idea(self, idea_id: str, actor: str = "user"):
        """
        Approve an idea to enter the ASSET pipeline.

        - Creates workflow.json FIRST
        - Then logs the approval in idea_progress
        - Does NOT set asset_workflow (user sets it later)
        - Does NOT advance substates
        """

        # Ensure APPROVAL branch is active
        self._ensure_branch("APPROVAL")

        # Load and mark idea as approved
        idea = self._load_or_get_idea(idea_id)
        idea.approved = True
        self._save_idea(idea)

        # Ensure idea directory exists
        idea_dir = self.assets_path / idea_id
        idea_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create workflow.json BEFORE any ledger writes
        self._initialize_workflow_for_idea(idea_id)

        # 2. Log approval in the editorial ledger
        self.update_idea_progress(
            idea_id=idea_id,
            branch="APPROVAL",
            substate="idea",
            action="approved",
            actor=actor,
            artifact=None,
            details={},
        )

        # 3. Track approved ideas in engine_state
        approved_list = (
            self.engine_state.progress
            .setdefault("approval", {})
            .setdefault("approved_idea_ids", [])
        )
        if idea_id not in approved_list:
            approved_list.append(idea_id)

        # 4. Move engine to ASSET branch, but DO NOT set substate here
        #    Substate is now controlled by workflow.json only.
        self._set_state("ASSET", None)

        self._save_engine_state()



    def reject_idea(self, idea_id: str, actor: str = "user"):
        self._ensure_branch("APPROVAL")

        idea = self._load_or_get_idea(idea_id)
        idea.approved = False
        idea.updated_at = datetime.utcnow()
        self._save_idea(idea)

        # Track rejected ideas in engine_state.progress
        rejected = self.engine_state.progress["approval"].setdefault(
            "rejected_idea_ids", []
        )
        if idea_id not in rejected:
            rejected.append(idea_id)

        # IMPORTANT:
        # Do NOT advance branch or substate.
        # Do NOT create asset directories.
        # Do NOT set active idea.
        # The engine stays in APPROVAL until a different idea is approved.

        self._save_engine_state()


    def _load_or_get_idea(self, idea_id: str) -> BaseIdea:
        if idea_id in self._ideas:
            return self._ideas[idea_id]
        path = self.ideas_path / f"{idea_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Idea {idea_id} not found")
        data = json.loads(path.read_text())
        idea = BaseIdea(**data)
        self._ideas[idea_id] = idea
        return idea

    # -----------------------------------------------------------------------
    # ASSET
    # -----------------------------------------------------------------------

    def generate_assets(self, idea_id: str, workflow: str | None = None) -> List[BaseAsset]:
        """
        Run the ASSET pipeline for a specific idea.

        - Ensures the idea is approved
        - Optionally sets the workflow (per-idea)
        - Sets the active idea (UI context only)
        - Dispatches into generate_assets_impl
        - Saves any returned assets
        """

        self._ensure_branch("ASSET")

        idea = self._load_or_get_idea(idea_id)
        if not idea.approved:
            raise RuntimeError(f"Idea {idea_id} is not approved")

        # ------------------------------------------------------------
        # 1. If workflow is provided, set it PER-IDEA
        # ------------------------------------------------------------
        if workflow is not None:
            self.set_workflow(workflow)

        # ------------------------------------------------------------
        # 2. Set active idea (UI context only)
        # ------------------------------------------------------------
        self.set_active_idea(idea_id)

        # ------------------------------------------------------------
        # 3. Run the ASSET substate step
        # ------------------------------------------------------------
        assets = self.generate_assets_impl(idea)

        # ------------------------------------------------------------
        # 4. Save any returned assets
        # ------------------------------------------------------------
        for asset in assets:
            self._save_asset(asset)

        return assets


    def generate_assets_impl(self, idea: BaseIdea) -> List[BaseAsset]:
        """
        Subclasses should implement the ASSET pipeline here, using
        self.engine_state.substate and self.editorial_step where appropriate.
        """
        raise NotImplementedError

    def set_active_idea(self, idea_id: str):
        self.engine_state.context["current_idea_id"] = idea_id
        self._save_engine_state()

    def accept_asset_substate(self, substate: str, actor: str = "user"):
        """
        Approve the current ASSET substate for the active idea.

        - Validates that the idea is in the expected substate
        - Marks the substate as approved + completed in workflow.json
        - Logs the approval in idea_progress
        - Advances to the next substate via _advance_substate
        """

        idea_id = self.engine_state.context.get("current_idea_id")
        if not idea_id:
            raise ValueError("No active idea_id found")

        # Load workflow.json
        wf = self._load_workflow(idea_id)

        # Validate branch + substate
        if wf["branch"] != "ASSET":
            raise ValueError(f"Cannot approve substate outside ASSET branch (current branch: {wf['branch']})")

        if wf["substate"] != substate:
            raise ValueError(
                f"Cannot approve '{substate}'; current substate is '{wf['substate']}'"
            )

        # ------------------------------------------------------------
        # 1. Log approval in the editorial ledger
        # ------------------------------------------------------------
        self.update_idea_progress(
            idea_id=idea_id,
            branch="ASSET",
            substate=substate,
            action="approved",
            actor=actor,
            artifact=None,
            details={},
        )

        # ------------------------------------------------------------
        # 2. Update workflow.json: mark approved + completed
        # ------------------------------------------------------------
        wf["approved"] = True
        wf["completed"] = True
        wf["updated_at"] = datetime.utcnow().isoformat()
        self._save_workflow(idea_id, wf)

        # ------------------------------------------------------------
        # 3. Advance to the next substate
        # ------------------------------------------------------------
        self._advance_substate(idea_id)




    def reset_asset_substate(self, substate: str):
        """Reset to a given ASSET substate so it can be rerun."""
        self._set_state(branch="ASSET", substate=substate)
        self._save_engine_state()


    def approve_asset(self, idea_id: str, asset_id: str):
        self._ensure_branch("ASSET")

        asset = self._load_or_get_asset(idea_id, asset_id)
        asset.approved = True
        asset.updated_at = datetime.utcnow()
        self._save_asset(asset)

        approved = self.engine_state.progress["asset"].setdefault(
            "approved_asset_ids", []
        )
        if asset_id not in approved:
            approved.append(asset_id)

        if approved:
            self._set_state(
                branch="EXPORT",
                substate=self._initial_substate("EXPORT"),
            )

    def reject_asset(self, idea_id: str, asset_id: str):
        self._ensure_branch("ASSET")

        asset = self._load_or_get_asset(idea_id, asset_id)
        asset.approved = False
        asset.updated_at = datetime.utcnow()
        self._save_asset(asset)

    def _load_or_get_asset(self, idea_id: str, asset_id: str) -> BaseAsset:
        if asset_id in self._assets:
            return self._assets[asset_id]

        idea_dir = self.assets_path / idea_id
        asset_path = idea_dir / f"{asset_id}.json"

        if not asset_path.exists():
            raise FileNotFoundError(f"Asset {asset_id} not found for idea {idea_id}")

        data = json.loads(asset_path.read_text())
        asset = BaseAsset(**data)

        self._assets[asset_id] = asset
        return asset


    # -----------------------------------------------------------------------
    # EXPORT
    # -----------------------------------------------------------------------

    def export(self, idea_id: str) -> BaseFinalOutput:
        """
        Run the EXPORT pipeline for a specific idea.
        Reads workflow.json, validates state, runs export_impl,
        saves output, and advances workflow to DONE.
        """

        # ------------------------------------------------------------
        # 1. Load workflow.json and validate branch
        # ------------------------------------------------------------
        wf = self._load_workflow(idea_id)
        if wf["branch"] != "EXPORT":
            raise RuntimeError(
                f"Idea {idea_id} is not in EXPORT branch (current: {wf['branch']})"
            )

        # If EXPORT substates require approval, enforce it
        if wf.get("requires_approval", False) and not wf.get("approved", False):
            raise RuntimeError(
                f"EXPORT substate '{wf['substate']}' is not approved for idea {idea_id}"
            )

        idea = self._load_or_get_idea(idea_id)

        # ------------------------------------------------------------
        # 2. Collect approved assets
        # ------------------------------------------------------------
        approved_assets = [
            a for a in self._iter_assets(idea_id)
            if a.approved
        ]

        if not approved_assets:
            raise RuntimeError(f"No approved assets for idea {idea_id}")

        # ------------------------------------------------------------
        # 3. Set active idea (UI context only)
        # ------------------------------------------------------------
        self.set_active_idea(idea_id)

        # ------------------------------------------------------------
        # 4. Run export_impl
        # ------------------------------------------------------------
        result = self.export_impl(idea, approved_assets)

        if isinstance(result, dict) and "output" in result:
            output = result["output"]
        else:
            output = result  # backward-compatible

        # ------------------------------------------------------------
        # 5. Save output
        # ------------------------------------------------------------
        self._save_output(output)

        # ------------------------------------------------------------
        # 6. Mark workflow as DONE
        # ------------------------------------------------------------
        self._mark_export_done(idea_id)

        return output

    def _mark_export_done(self, idea_id: str):
        """
        Mark the EXPORT pipeline as complete for this idea.
        """

        wf = self._load_workflow(idea_id)

        wf["branch"] = "DONE"
        wf["substate"] = "complete"
        wf["approved"] = True
        wf["completed"] = True
        wf["updated_at"] = datetime.utcnow().isoformat()

        self._save_workflow(idea_id, wf)



    def export_impl(
        self,
        idea: BaseIdea,
        assets: List[BaseAsset],
    ) -> BaseFinalOutput | Dict[str, Any]:
        """
        Subclasses should implement export logic.
        Recommended: return {"artifact": {...}, "details": {...}, "output": BaseFinalOutput(...)}
        so the decorator can log, and export() can persist the output.
        """
        raise NotImplementedError

    def _iter_assets(self, idea_id: str):
        """
        Yield all valid BaseAsset objects for a specific idea.
        Automatically deletes stale asset JSON files whose referenced
        filepath no longer exists on disk.
        """
        idea_dir = self.assets_path / idea_id
        if not idea_dir.exists():
            return

        for f in idea_dir.glob("*.json"):
            try:
                asset = BaseAsset.model_validate_json(f.read_text())
            except Exception:
                continue

            if asset.idea_id != idea_id:
                continue

            filepath = asset.metadata.get("filepath")
            if not filepath:
                continue

            if not Path(filepath).exists():
                try:
                    f.unlink()
                except Exception:
                    pass
                continue

            yield asset

    def _ensure_asset_dir(self, idea_id: str) -> Path:
        asset_dir = self.assets_path / idea_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        return asset_dir

    def _to_jsonable(self, obj):
        if is_dataclass(obj):
            return {k: self._to_jsonable(v) for k, v in asdict(obj).items()}

        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]

        return obj

    def _write_json(self, path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        jsonable = self._to_jsonable(data)

        with path.open("w", encoding="utf-8") as f:
            json.dump(jsonable, f, indent=2, ensure_ascii=False)

    def _read_json(self, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


    def create_user_edits(self, object_name: str, edits: dict):
        """
        Create a new user edit snapshot for a given object (e.g., 'meta'),
        automatically versioning the snapshot file, merging it into the
        materialized current state, and logging the edit.
        """
        idea_id = self.engine_state.context.get("current_idea_id")
        if not idea_id:
            raise ValueError("No active idea_id found.")

        asset_dir = self._ensure_asset_dir(idea_id)

        # 1. Raw object path (e.g., meta.json)
        raw_path = asset_dir / f"{object_name}.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw object not found: {raw_path}")

        # 2. Load current state (raw or materialized)
        current_path = asset_dir / f"{object_name}_current.json"
        if current_path.exists():
            current = self._read_json(current_path)
        else:
            current = self._read_json(raw_path)

        # 3. Merge edits into current state
        updated = copy.deepcopy(current)
        for k, v in edits.items():
            updated[k] = v

        # 4. Create a new snapshot path
        i = 0
        while True:
            snapshot_path = asset_dir / f"{object_name}_edit_{i}.json"
            if not snapshot_path.exists():
                break
            i += 1

        # 5. Write snapshot + materialized current state
        self._write_json(snapshot_path, edits)
        self._write_json(current_path, updated)

        # 6. Log editorial step
        self.update_idea_progress(
            idea_id,
            branch=self.engine_state.branch,
            substate=self.engine_state.substate,
            action="edited",
            actor="user",
            artifact={
                "snapshot_path": str(snapshot_path),
                "current_path": str(current_path),
            },
            details={
                "edits": copy.deepcopy(edits),
                "snapshot_index": i,
            },
        )

        return updated


