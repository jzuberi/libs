# libs/retrieval/src/retrieval/retrieval.py

from pathlib import Path
from datetime import datetime


from typing import Optional, List, Tuple, Any
from .models import RetrievalIntent, IntentType, RetrievalTrace, ChunkTrace
from .batching import adaptive_batches
from .indicators import LocHandler, IndustryHandler, TopicHandler
from .ragstore_backend import RAGStoreBackend
from hashlib import sha1

from .utils import (
    extract_text,
    add_indicators_to_intent,
    debug_banner,
)



class BaseStrategy:
    def build_prompt(self, chunk: str, intent: RetrievalIntent) -> str:
        raise NotImplementedError

    def grade_chunk(self, chunk: Any, intent: RetrievalIntent) -> dict:
        """
        Must return a dict:
        {
            "passed": bool,
            "grader_passed": bool | None,
            "grader_reason": str | None,
            "indicator_results": {name: bool},
            "failure_stage": "grader" | "indicator" | None
        }
        """
        raise NotImplementedError


class TopicFilterStrategy(BaseStrategy):

    def __init__(self, grader, indicator_handlers):
        self.grader = grader
        self.indicator_handlers = indicator_handlers

    def build_prompt(self, chunk: Any, intent: RetrievalIntent, debug=False) -> str:
        text = extract_text(chunk)

        if debug:
            debug_banner("CHUNK TEXT", text)

        return f"""
You are a relevance classifier. Your job is to determine whether the given text
is meaningfully related to the user's topic, and explain your reasoning.

Topic (user intent):
{intent.guidance}

Text to evaluate:
\"\"\"{text}\"\"\"

Guidelines for relevance:
- Consider semantic meaning, not exact keywords.
- If the text discusses events, actors, policies, or developments involving the topic, it IS relevant.
- If the text provides context, background, implications, or related geopolitical/economic issues, it IS relevant.
- If the text is about a country, region, or actor directly connected to the topic, it IS relevant.
- If the text is unrelated, off-topic, or only mentions the topic in passing, it is NOT relevant.
- Ignore metadata, IDs, and technical wrappers. Focus only on the human‑readable content.

You MUST respond in valid JSON with EXACTLY this structure:

{{
  "relevant": true or false,
  "reason": "a very, very short explanation of why you made this decision"
}}

No extra keys. No commentary outside the JSON.
"""

    def grade_chunk(self, chunk: Any, intent: RetrievalIntent) -> dict:

        grader_passed = None
        grader_reason = None

        # Holistic grading
        if intent.grading:
            prompt = self.build_prompt(chunk, intent)
            result = self.grader.grade(prompt)

            grader_passed = result.passed
            grader_reason = result.reason

            if not result.passed:
                return {
                    "passed": False,
                    "grader_passed": grader_passed,
                    "grader_reason": grader_reason,
                    "indicator_results": {},
                    "failure_stage": "grader",
                }

        # Indicator validation
        indicator_results = {}
        for ind in intent.indicators:
            handler = self.indicator_handlers[ind.name]
            ok = handler.validate(chunk, ind.value)
            if ind.negated:
                ok = not ok

            indicator_results[ind.name] = ok

            if not ok:

                return {
                    "passed": False,
                    "grader_passed": grader_passed,
                    "grader_reason": grader_reason,
                    "indicator_results": indicator_results,
                    "failure_stage": "indicator",
                }

        return {
            "passed": True,
            "grader_passed": grader_passed,
            "grader_reason": grader_reason,
            "indicator_results": indicator_results,
            "failure_stage": None,
        }


class SupportStatementStrategy(BaseStrategy):
    def __init__(self, grader, indicator_handlers):
        self.grader = grader
        self.indicator_handlers = indicator_handlers

    def build_prompt(self, chunk: Any, intent: RetrievalIntent) -> str:
        text = extract_text(chunk)

        return f"""
Claim:
{intent.statement}

Text:
\"\"\"{text}\"\"\"

Does this text provide evidence, arguments, or data that SUPPORT the claim?
Respond with "true" or "false".
"""

    def grade_chunk(self, chunk: Any, intent: RetrievalIntent) -> dict:

        prompt = self.build_prompt(chunk, intent)
        result = self.grader.grade(prompt)

        grader_passed = result.passed
        grader_reason = result.reason


        if not result.passed:
            return {
                "passed": False,
                "grader_passed": grader_passed,
                "grader_reason": grader_reason,
                "indicator_results": {},
                "failure_stage": "grader",
            }

        indicator_results = {}
        for ind in intent.indicators:
            handler = self.indicator_handlers[ind.name]
            ok = handler.validate(chunk, ind.value)
            if ind.negated:
                ok = not ok

            indicator_results[ind.name] = ok

            if not ok:
                return {
                    "passed": False,
                    "grader_passed": grader_passed,
                    "grader_reason": grader_reason,
                    "indicator_results": indicator_results,
                    "failure_stage": "indicator",
                }

        return {
            "passed": True,
            "grader_passed": grader_passed,
            "grader_reason": grader_reason,
            "indicator_results": indicator_results,
            "failure_stage": None,
        }
    
class TripleSupportStrategy(BaseStrategy):
    def __init__(self, grader, indicator_handlers):
        self.grader = grader
        self.indicator_handlers = indicator_handlers

    def build_prompt(self, chunk: Any, intent: RetrievalIntent) -> str:
        text = extract_text(chunk)

        prompt = f"""
    You are a strict binary classifier.

    TRIPLE:
    {intent.statement}

    TEXT:
    \"\"\"{text}\"\"\"

    Does the text SUPPORT the triple?

    You MUST respond with EXACTLY one of the following, all lowercase:
    true
    false

    Your response MUST contain ONLY the word "true" or "false".
    No explanations.
    No analysis.
    No markdown.
    No extra text.
    """
        
        return prompt

    def grade_chunk(self, chunk, intent):

        prompt = self.build_prompt(chunk, intent)
        result = self.grader.grade(prompt)

        grader_passed = result.passed
        grader_reason = result.reason

        if not grader_passed:
            return {
                "passed": False,
                "grader_passed": grader_passed,
                "grader_reason": grader_reason,
                "indicator_results": {},
                "failure_stage": "grader",
            }

        # No indicator validation for triple support
        return {
            "passed": True,
            "grader_passed": grader_passed,
            "grader_reason": grader_reason,
            "indicator_results": {},
            "failure_stage": None,
        }

class RetrievalLayer:
    def __init__(self, rag_store, grader, llm_call, threshold: float = 0.1):

        # Store rag_store so we can access time_field + normalization
        self.rag = rag_store

        backend = RAGStoreBackend(rag_store)

        self.backend = backend
        self.grader = grader
        self.threshold = threshold

        # Indicator handlers
        self.indicator_handlers = {
            "is_loc": LocHandler(),
            "is_industry": IndustryHandler(),
            "is_topic": TopicHandler(),
        }

        for handler in self.indicator_handlers.values():
            handler._call_llm = llm_call

        # Strategy map
        self.strategies = {
            IntentType.TOPIC_FILTER: TopicFilterStrategy(
                grader, self.indicator_handlers
            ),
            IntentType.SUPPORT_STATEMENT: SupportStatementStrategy(
                grader, self.indicator_handlers
            ),
            IntentType.TRIPLE_SUPPORT: TripleSupportStrategy(
                grader, self.indicator_handlers
            ),
        }

    # ============================================================
    # NEW: Convert RetrievalIntent → Qdrant filter dict
    # ============================================================
    def intent_to_filter(self, intent: RetrievalIntent) -> dict:
        f = {}

        # -----------------------------
        # 1. Time filtering ONLY
        # -----------------------------
        tf = self.rag.time_field

        if intent.date_after:
            ts = self.rag._normalize_time_value(intent.date_after)
            f.setdefault("$and", []).append({
                tf: { "$gte": ts }
            })

        if intent.date_before:
            ts = self.rag._normalize_time_value(intent.date_before)
            f.setdefault("$and", []).append({
                tf: { "$lte": ts }
            })

        return f



    # ============================================================
    # Main retrieval
    # ============================================================
    def retrieve(
        self,
        query_or_intent,
        grading: Optional[bool] = None,
        indicators: Optional[List[Tuple[str, str, bool]]] = None,
        adaptive: bool = True,
        external_filter: Optional[dict] = None,
        num_results_per_iteration: int = 4,
        diversify: bool = False,
        verbose=False,
    ):

        
        # 1. Query vs intent
        if isinstance(query_or_intent, RetrievalIntent):
            intent = query_or_intent
            query = intent.guidance or intent.statement
            if grading is not None:
                intent.grading = grading
        else:
            query = str(query_or_intent)
            intent = RetrievalIntent(
                type=IntentType.TOPIC_FILTER,
                guidance=query,
                grading=bool(grading),
                indicators=[],
                statement=query,
            )

        # 2. Indicators
        intent = add_indicators_to_intent(intent, indicators)

        # 3. Mode
        has_time = bool(intent.date_after or intent.date_before)
        has_semantic = bool(intent.indicators or intent.guidance)

        if has_time and not has_semantic:
            mode = "recency_only"
        elif has_time and has_semantic:
            mode = "semantic_time_hybrid"
        elif has_semantic and not has_time:
            mode = "semantic_only"
        else:
            mode = "vanilla"

        # 4. Trace
        trace = RetrievalTrace(query=query, intent=intent)
        trace.mode = mode

        # 5. Recency-only fast path
        if mode == "recency_only":
            return self._retrieve_recency_only(intent, trace)

        # 6. Base filter from intent
        if type(external_filter) is not list:

            filter_dict = self.intent_to_filter(intent)

        # 6b. External filter handling
        if external_filter:

            if self.backend.is_chroma:
                
                filter_dict = external_filter

            elif self.backend.is_qdrant:
                # Qdrant path: keep existing $and behavior
                if filter_dict is None:
                    filter_dict = {}
                if "$and" not in filter_dict:
                    filter_dict["$and"] = []
                if "$and" not in external_filter:
                    external_filter = {"$and": [external_filter]}

                filter_dict["$and"].extend(external_filter["$and"])

        # 7. Strategy
        strategy = self.strategies[intent.type]
        results = []

        # 8. Adaptive flag
        use_adaptive = adaptive
        if intent.type == IntentType.TRIPLE_SUPPORT:
            use_adaptive = False

        print('filter_dict')
        print(filter_dict)

        # 9. Non-adaptive path
        if not use_adaptive:

            docs = self.backend.query(
                query or intent.guidance,
                limit=num_results_per_iteration,
                filter=filter_dict,
                diversify=diversify,
            )

            if verbose:
                print("docs")
                print(docs)

            for chunk in docs:
                trace.total_seen += 1

                if grading is True:
                    decision = strategy.grade_chunk(chunk, intent)
                else:
                    decision = {
                        "passed": True,
                        "grader_passed": True,
                        "grader_reason": "grading is False",
                        "indicator_results": {},
                        "failure_stage": None,
                    }

                chunk_id = chunk.get("id") if isinstance(chunk, dict) else None

                trace.chunks.append(
                    ChunkTrace(
                        chunk_id=chunk_id,
                        passed=decision["passed"],
                        grader_passed=decision["grader_passed"],
                        grader_reason=decision["grader_reason"],
                        indicator_results=decision["indicator_results"],
                        failure_stage=decision["failure_stage"],
                    )
                )

                if decision["passed"]:
                    trace.total_kept += 1
                    results.append(chunk)

            self.save_trace(trace)
            return results, trace

        # 10. Adaptive path
        seen_ids = set()

        for batch in adaptive_batches(
            self.backend,
            query or intent.guidance,
            filter_dict=filter_dict,
            batch_size=num_results_per_iteration,
            diversify=diversify,
        ):
            print(f"\n[BATCH] Retrieved {len(batch)} chunks")

            for chunk in batch:
                chunk_id = chunk.get("id") if isinstance(chunk, dict) else None

                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)

                trace.total_seen += 1

                if grading is True:
                    decision = strategy.grade_chunk(chunk, intent)
                else:
                    decision = {
                        "passed": True,
                        "grader_passed": True,
                        "grader_reason": "grading is False",
                        "indicator_results": {},
                        "failure_stage": None,
                    }

                trace.chunks.append(
                    ChunkTrace(
                        chunk_id=chunk_id,
                        passed=decision["passed"],
                        grader_passed=decision["grader_passed"],
                        grader_reason=decision["grader_reason"],
                        indicator_results=decision["indicator_results"],
                        failure_stage=decision["failure_stage"],
                    )
                )

                if decision["passed"]:
                    trace.total_kept += 1
                    results.append(chunk)

            running_mean = trace.total_kept / trace.total_seen if trace.total_seen else 1.0
            print(f"[Running Mean] {running_mean:.3f}")

            if running_mean < self.threshold:
                print("[Early Stop] ❌ Running mean below threshold")
                break

        debug_banner(
            "RETRIEVAL COMPLETE",
            f"Total Seen: {trace.total_seen}\nTotal Kept: {trace.total_kept}",
        )

        self.save_trace(trace)
        return results, trace


    def visualize(self, trace):
        print("\n================ RETRIEVAL TRACE ================\n")
        print(f"Query: {trace.query}")
        print(f"Intent Type: {trace.intent.type}")
        print(f"Guidance: {trace.intent.guidance}")
        print(f"Statement: {trace.intent.statement}")
        print(f"Total Seen: {trace.total_seen}")
        print(f"Total Kept: {trace.total_kept}")

        print("\n---------------- CHUNK RESULTS ----------------\n")

        for i, c in enumerate(trace.chunks, 1):
            status = "✅ PASS" if c.passed else "❌ FAIL"
            print(f"[{i}] Chunk ID: {c.chunk_id} — {status}")

            if c.failure_stage:
                print(f"   Failure Stage: {c.failure_stage}")

            if c.grader_passed is not None:
                print(f"   Grader Passed: {c.grader_passed}")
                print(f"   Grader Reason: {c.grader_reason}")

            if c.indicator_results:
                print("   Indicators:")
                for name, ok in c.indicator_results.items():
                    print(f"      - {name}: {ok}")

            # Optional reranking metadata
            if getattr(c, "similarity", None) is not None:
                print(f"   Similarity: {c.similarity:.3f}")

            if getattr(c, "llm_score", None) is not None:
                print(f"   LLM Score: {c.llm_score:.3f}")

            if getattr(c, "indicator_score", None) is not None:
                print(f"   Indicator Score: {c.indicator_score}")

            if getattr(c, "rerank_score", None) is not None:
                print(f"   Rerank Score: {c.rerank_score:.3f}")

            print()


    def save_trace(self, trace):
        store_path = Path(self.backend.rag.backend.path)

        traces_dir = store_path / "traces"
        traces_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # NEW: short deterministic slug
        slug = sha1(trace.query.encode("utf-8")).hexdigest()[:12]

        filename = f"{timestamp}_{slug}.json"

        with open(traces_dir / filename, "w") as f:
            f.write(trace.model_dump_json(indent=2))

        print(f"[Trace Saved] {filename}")
