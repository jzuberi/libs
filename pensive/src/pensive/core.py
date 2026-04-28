# src/pensive/core.py

import json

from dataclasses import dataclass, field
from typing import List, Dict, Any

import os

from .snapshot import SnapshotIngestor
from .storage import DecisionStore, TurnStore
from .instance import InstanceManager
from .registry import StrategyRegistry, CriteriaRegistry, DecisionPolicyRegistry
from .models import Decision, IdeaSummary


# src/pensive/turn.py

from datetime import datetime
import uuid



@dataclass
class Turn:
    """Represents a single run of the engine."""

    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Raw ideas that passed filtering (before decision policy)
    ideas: List[Any] = field(default_factory=list)

    # Strategy → list of ideas produced
    strategy_outputs: Dict[str, List[Any]] = field(default_factory=dict)

    # idea_key → {criterion_name: bool}
    criteria_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Optional: human editorial rationale (not used by engine)
    review_rationale: Dict[str, str] = field(default_factory=dict)

    # Final editorial decisions
    decisions: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {
            "accepted": [],
            "rejected": [],
            "deferred": [],
        }
    )

    # ------------------------------------------------------------
    # Decision helpers (still useful for manual or policy-driven calls)
    # ------------------------------------------------------------
    def accept(self, idea, rationale=""):
        self.decisions["accepted"].append({
            "decision_id": str(uuid.uuid4()),
            "turn_id": self.turn_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "idea": idea,
            "rationale": rationale,
            "status": "accepted",
        })

    def reject(self, idea, rationale=""):
        self.decisions["rejected"].append({
            "decision_id": str(uuid.uuid4()),
            "turn_id": self.turn_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "idea": idea,
            "rationale": rationale,
            "status": "rejected",
        })

    def defer(self, idea, rationale=""):
        self.decisions["deferred"].append({
            "decision_id": str(uuid.uuid4()),
            "turn_id": self.turn_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "idea": idea,
            "rationale": rationale,
            "status": "deferred",
        })



def default_json_loader(path):
    def loader():
        with open(path, "r") as f:
            return json.load(f)
    return loader



class Pensive:

    """Main engine class."""

    # -----------------------------
    # Constructor
    # -----------------------------
    def __init__(
        self,
        corpus_loader=None,
        corpus_model=None,
        strategies=None,
        criteria=None,
        save_config=None,
        decision_store=None,
        turn_store=None,
        instance_manager=None,
        decision_policy="DefaultDecisionPolicy",
        context=None,
    ):
        self.corpus_loader = corpus_loader
        self.corpus_model = corpus_model
        self.strategies = strategies or []
        self.criteria = criteria or []
        self.save_config = save_config
        self.decision_store = decision_store
        self.turn_store = turn_store
        self.instance_manager = instance_manager
        self.decision_policy = DecisionPolicyRegistry.instantiate(decision_policy)
        self.context = context or {}

    # -----------------------------
    # Initialize a new instance
    # -----------------------------
    @classmethod
    def init(cls, instance_path, force=False, **kwargs):
        # Safety: prevent accidental overwrite
        if os.path.exists(instance_path) and not force:
            raise FileExistsError(
                f"Instance folder '{instance_path}' already exists. "
                f"Use force=True to overwrite."
            )

        # Create engine instance
        instance = cls(**kwargs)

        # ---------------------------------------
        # Build corpus model metadata
        # ---------------------------------------
        corpus_model = kwargs.get("corpus_model")
        if corpus_model is not None:
            try:
                schema = corpus_model.model_json_schema()
            except Exception:
                schema = None

            corpus_model_meta = {
                "name": corpus_model.__name__,
                "schema": schema,
            }
        else:
            corpus_model_meta = None

        # ---------------------------------------
        # Build config dict
        # ---------------------------------------
        config = {
            "engine": {"version": "0.1.0"},
            "strategies": list(StrategyRegistry._registry.keys()),
            "criteria": list(CriteriaRegistry._registry.keys()),
            "corpus_model": corpus_model_meta,
            "save_config": (
                vars(kwargs.get("save_config"))
                if kwargs.get("save_config")
                else {}
            ),
            "decision_policy": kwargs.get("decision_policy", "DefaultDecisionPolicy"),
        }

        # ---------------------------------------
        # Create instance folder
        # ---------------------------------------
        manager = InstanceManager(instance_path)
        manager.init_instance(config)   # <-- MUST write full config

        # Attach manager
        instance.instance_manager = manager

        instance.decision_store = DecisionStore(
            path=os.path.join(instance_path, "decisions")
        )

        instance.turn_store = TurnStore(
            path=os.path.join(instance_path, "turns")
        )

        return instance



    # -----------------------------
    # Load an existing instance
    # -----------------------------
    @classmethod
    def load(cls, instance_path, corpus_loader, corpus_model=None):
        manager = InstanceManager(instance_path)
        config = manager.load_instance()

        decision_policy_name = config.get("decision_policy", "DefaultDecisionPolicy")

        # Rehydrate strategies + criteria
        strategies = [
            StrategyRegistry._registry[name]() for name in config["strategies"]
        ]
        criteria = [
            CriteriaRegistry._registry[name]() for name in config["criteria"]
        ]

        # Rehydrate save config
        from .storage import SaveConfig
        save_config = SaveConfig(**config["save_config"])

        # -----------------------------
        # Corpus model resolution
        # -----------------------------
        # Priority:
        #   1. User-supplied model (ALWAYS wins)
        #   2. Reconstruct from stored schema
        #   3. None
        # -----------------------------
        if corpus_model is not None:
            model = corpus_model

        else:
            meta = config.get("corpus_model")
            if meta is None:
                model = None
            else:
                # Reconstruct a Pydantic model from the stored schema
                from pydantic import create_model

                fields = {}
                schema_props = meta["schema"].get("properties", {})
                required = set(meta["schema"].get("required", []))

                for field_name, field_schema in schema_props.items():
                    # Very simple type mapping (extend later)
                    field_type = {
                        "string": str,
                        "integer": int,
                        "number": float,
                        "boolean": bool,
                    }.get(field_schema.get("type"), Any)

                    default = ... if field_name in required else None
                    fields[field_name] = (field_type, default)

                model = create_model(meta["name"], **fields)

        # Build engine
        return cls(
            corpus_loader=corpus_loader,
            corpus_model=model,
            strategies=strategies,
            criteria=criteria,
            save_config=save_config,
            decision_store=DecisionStore(os.path.join(instance_path, "decisions")),
            turn_store=TurnStore(os.path.join(instance_path, "turns")),
            instance_manager=manager,
            decision_policy=decision_policy_name,
        )



    # -----------------------------
    # Ingest snapshot
    # -----------------------------
    def ingest_snapshot(self):
        ingestor = SnapshotIngestor(
            corpus_loader=self.corpus_loader,
            corpus_model=self.corpus_model,
            decision_store=self.decision_store,
        )
        return ingestor.ingest()

    def canonical_key(self, idea):
        return idea["key"]


    def _filter_against_past_decisions(self, snapshot, ideas):
        past_keys = set()

        for bucket in ("accepted", "rejected", "deferred"):
            for d in snapshot.decisions.get(bucket, []):
                if d["status"] == "accepted":
                    past_keys.add(d["idea"]["key"])

        return [idea for idea in ideas if idea["key"] not in past_keys]

    def _dedupe_ideas(self, ideas):
        seen = {}
        for idea in ideas:
            key = idea["key"]
            # Keep the first occurrence deterministically
            if key not in seen:
                seen[key] = idea
        return list(seen.values())


    # -----------------------------
    # Run a turn
    # -----------------------------
    def run_turn(self, snapshot):
        print(f"Running {len(self.strategies)} strategies...")
        print(f"Running {len(self.criteria)} criteria...")

        strategy_outputs = {}
        all_ideas = []
        criteria_results = {}

        # ------------------------------------------------------------
        # 1. Run strategies and collect ideas
        # ------------------------------------------------------------
        for strategy in self.strategies:

            name = strategy.__class__.__name__
            ideas = strategy.generate(snapshot)
            strategy_outputs[name] = ideas
            all_ideas.extend(ideas)

        # ------------------------------------------------------------
        # 2. Validate that every idea has a 'key'
        # ------------------------------------------------------------
        for idea in all_ideas:

            if "key" not in idea:
                raise ValueError(
                    f"Strategy '{strategy.__class__.__name__}' returned an idea "
                    f"without a 'key' field: {idea}"
                )

        # ------------------------------------------------------------
        # 3. Dedupe ideas by key BEFORE editorial memory filtering
        # ------------------------------------------------------------
        unique_ideas = self._dedupe_ideas(all_ideas)

        # ------------------------------------------------------------
        # 4. Filter ideas against past decisions (editorial memory)
        # ------------------------------------------------------------
        filtered_ideas = self._filter_against_past_decisions(snapshot, unique_ideas)
        
        if not filtered_ideas:
            return Turn(
                ideas=[],
                strategy_outputs=strategy_outputs,
                criteria_results={},
            )


        # ------------------------------------------------------------
        # 5. Run criteria on filtered ideas
        # ------------------------------------------------------------
        for idea in filtered_ideas:
            idea_key = idea["key"]
            criteria_results[idea_key] = {}

            for criterion in self.criteria:

                # 🔥 Inject engine context into each criterion
                criterion.context = self.context

                cname = criterion.__class__.__name__
                result = criterion.evaluate(idea)
                criteria_results[idea_key][cname] = result

        # ------------------------------------------------------------
        # 6. Apply decision policy (per-turn)
        # ------------------------------------------------------------
        turn = Turn(
            ideas=filtered_ideas,
            strategy_outputs=strategy_outputs,
            criteria_results=criteria_results,
        )

        decisions = self.decision_policy.decide_all(filtered_ideas, criteria_results)

        for d in decisions:
            idea = d["idea"]
            status = d["status"]
            rationale = d.get("rationale", "")

            # Let Turn.accept/reject/defer create the full decision dict
            if status == "accepted":
                turn.accept(idea, rationale)
            elif status == "rejected":
                turn.reject(idea, rationale)
            elif status == "deferred":
                turn.defer(idea, rationale)
            else:
                raise ValueError(f"Unknown decision status: {status}")

        return turn

    def _extract_key(self, decision):
        return decision.key

    def _dedupe_bucket(self, bucket_list):
        seen = set()
        deduped = []

        for decision in reversed(bucket_list):
            if decision.key not in seen:
                seen.add(decision.key)
                deduped.append(decision)

        deduped.reverse()
        return deduped

    def seed_manual_decisions(self, records: list[dict]):
        """
        Seed the decision store with manually provided historical decisions.
        Each record must contain:
            - key
            - status ("accepted" | "rejected" | "deferred")
            - rationale
            - timestamp (ISO string)
        """

        # Load existing decisions (may be empty)
        existing = self.decision_store.load()

        # Ensure buckets exist
        for bucket in ["accepted", "rejected", "deferred"]:
            existing.setdefault(bucket, [])

        typed = []

        for r in records:
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                turn_id="manual-seed",
                timestamp=datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")),
                idea=IdeaSummary(key=r["key"], idea=None),
                rationale=r.get("rationale", ""),
                status=r["status"],
            )
            typed.append(decision)

        # Append to correct buckets
        for d in typed:
            existing[d.status].append(d)

        # Dedupe within each bucket
        for bucket in ["accepted", "rejected", "deferred"]:
            existing[bucket] = self._dedupe_bucket(existing[bucket])

        # Save back to JSON
        self.decision_store.save({
            bucket: [d.model_dump(mode="json") for d in existing[bucket]]
            for bucket in ["accepted", "rejected", "deferred"]
        })



    def save_turn(self, turn: Turn):
        # ------------------------------------------------------------
        # 1. Save the turn file (raw, unchanged)
        # ------------------------------------------------------------
        if self.turn_store:
            self.turn_store.save_turn({
                "turn_id": turn.turn_id,
                "timestamp": turn.timestamp,
                "ideas": turn.ideas,
                "strategy_outputs": turn.strategy_outputs,
                "criteria_results": turn.criteria_results,
                "review_rationale": turn.review_rationale,
                "decisions": turn.decisions,   # raw dicts preserved in turn file
            })

        # ------------------------------------------------------------
        # 2. Append decisions to the global decision store
        # ------------------------------------------------------------
        if self.decision_store:
            existing = self.decision_store.load()

            # Ensure buckets exist
            for bucket in ["accepted", "rejected", "deferred"]:
                existing.setdefault(bucket, [])

            # --------------------------------------------------------
            # Normalize *existing* decisions into typed Decision objects
            # --------------------------------------------------------
            for bucket in ["accepted", "rejected", "deferred"]:
                existing[bucket] = [
                    Decision.from_raw(d) if isinstance(d, dict) else d
                    for d in existing[bucket]
                ]

            # --------------------------------------------------------
            # Convert raw dicts → typed Decision objects for this turn
            # --------------------------------------------------------
            new_decisions = {
                bucket: [Decision.from_raw(d) for d in turn.decisions[bucket]]
                for bucket in ["accepted", "rejected", "deferred"]
            }

            # Append typed decisions
            for bucket in ["accepted", "rejected", "deferred"]:
                existing[bucket].extend(new_decisions[bucket])

            # --------------------------------------------------------
            # 3. Dedupe within each bucket (most recent wins)
            # --------------------------------------------------------
            for bucket in ["accepted", "rejected", "deferred"]:
                existing[bucket] = self._dedupe_bucket(existing[bucket])

            # --------------------------------------------------------
            # 4. Save typed decisions back to JSON (JSON-safe)
            # --------------------------------------------------------
            self.decision_store.save({
                bucket: [d.model_dump(mode="json") for d in existing[bucket]]
                for bucket in ["accepted", "rejected", "deferred"]
            })
