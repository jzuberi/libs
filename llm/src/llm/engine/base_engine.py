from pydantic import create_model
import time
import re
import json

from ..parsing.json_parser import DefaultJSONParser
from ..backends.timeout import TimeoutRunnable

from ..schemas.general import GeneralAnswerSchema
from ..schemas.metadata import MetadataSchema
from ..schemas.edit import EditSchema
from ..schemas.classify import ClassificationSchema

from ..logging.decorators import log_engine_call
from ..logging.logger import get_logger

logger = get_logger("llm.engine")

class BaseLLMEngine:
    def __init__(self, backend, parser=None, timeout: int = 10, delay: int = 3, debug: bool = False):
        self.backend = backend
        self.parser = parser or DefaultJSONParser()
        self.timeout = timeout
        self.delay = delay
        self.debug = debug

    # -------------------------
    # INTERNAL DEBUG HELPER
    # -------------------------
    def _debug(self, label: str, value):
        if self.debug:
            print(f"[LLM DEBUG] {label}:\n{value}\n")

    # -------------------------
    # SANITIZATION (NEW)
    # -------------------------
    def _sanitize(self, raw: str) -> str:
        if not raw:
            return ""

        # Remove ```json or ``` fences
        cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")

        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    # -------------------------
    # INTERNAL BACKEND CALL
    # -------------------------
    def _call_backend(self, prompt: str) -> str:
        runner = TimeoutRunnable(self.backend.generate, timeout=self.timeout)
        raw = runner(prompt)

        time.sleep(self.delay)

        # NEW: sanitize once, globally
        cleaned = self._sanitize(raw)

        if self.debug:
            self._debug("SANITIZED OUTPUT", cleaned)

        return cleaned

    # -------------------------
    # GENERAL ANSWER
    # -------------------------
    @log_engine_call("general_answer")
    def general_answer(self, question: str) -> GeneralAnswerSchema:
        prompt = f"""
        You are a thoughtful and strategic expert.

        Your ONLY task is to answer the user's question.

        You MUST respond in valid JSON with EXACTLY this structure:

        {{
          "answer": "your answer here"
        }}

        No preamble. No explanation. No additional keys.

        Question: {question}
        """

        raw = self._call_backend(prompt)
        parsed = self.parser.parse(raw, expected_keys=["answer"])

        if not parsed.get("answer"):
            parsed["answer"] = raw

        return GeneralAnswerSchema(**parsed)

    # -------------------------
    # METADATA (dynamic)
    # -------------------------
    @log_engine_call("metadata")
    def metadata(self, text: str):

        def repair_json(raw: str) -> str:
            # Remove invalid escapes like \'
            raw = raw.replace("\\'", "'")

            # Replace smart quotes with normal quotes
            raw = raw.replace("“", "\"").replace("”", "\"")

            # Replace smart apostrophes and dashes
            raw = raw.replace("’", "'").replace("–", "-")

            # Remove stray backslashes that are not part of valid JSON escapes
            raw = re.sub(r'\\(?=[^"\\/bfnrtu])', '', raw)

            # Remove trailing commas before } or ]
            raw = re.sub(r",\s*([}\]])", r"\1", raw)

            return raw


        prompt = f"""
        You are a careful and precise metadata generator.

        You MUST return valid JSON.
        The JSON may contain ANY keys.
        Do NOT include explanations, preamble, or commentary.

        Text:
        {text}
        """

        raw = self._call_backend(prompt)
        raw = repair_json(raw)
        parsed = self.parser.parse(raw)

        DynamicMetadataSchema = create_model(
            "DynamicMetadataSchema",
            **{key: (type(value), None) for key, value in parsed.items()},
        )

        return DynamicMetadataSchema(**parsed)


    # -------------------------
    # EDIT
    # -------------------------
    @log_engine_call("edit")
    def edit(self, artifact: str, instructions: str) -> EditSchema:
        prompt = f"""
        You are a careful editor.

        Edit the artifact according to the instructions.

        You MUST return valid JSON with EXACTLY this structure:

        {{
          "edited": "the edited text"
        }}

        No preamble. No explanation.

        Artifact:
        {artifact}

        Instructions:
        {instructions}
        """

        raw = self._call_backend(prompt)
        parsed = self.parser.parse(raw, expected_keys=["edited"])

        if not parsed.get("edited"):
            parsed["edited"] = raw

        return EditSchema(**parsed)

    # -------------------------
    # CLASSIFY
    # -------------------------
    @log_engine_call("classify")
    def classify(self, text: str, labels: list[str]) -> ClassificationSchema:
        labels_str = ", ".join(f'"{l}"' for l in labels)

        prompt = f"""
        You are a precise classifier.

        You MUST return valid JSON with EXACTLY this structure:

        {{
          "label": "one of: {labels_str}"
        }}

        No preamble. No explanation.

        Text:
        {text}
        """

        raw = self._call_backend(prompt)
        parsed = self.parser.parse(raw, expected_keys=["label"])

        return ClassificationSchema(**parsed)

    # -------------------------
    # EXTRACTION
    # -------------------------
    @log_engine_call("extract")
    def extract(self, text: str, fields: list[str]):
        fields_str = ", ".join(f'"{f}"' for f in fields)

        prompt = f"""
        You are an information extraction system.

        Extract the following fields from the text:
        {fields_str}

        You MUST return valid JSON with EXACTLY these keys:
        {fields_str}

        If a field cannot be extracted, set it to null.

        No preamble. No explanation.

        Text:
        {text}
        """

        raw = self._call_backend(prompt)
        parsed = self.parser.parse(raw, expected_keys=fields)

        DynamicSchema = create_model(
            "DynamicExtractionSchema",
            **{field: (str | None, None) for field in fields},
        )

        return DynamicSchema(**parsed)
