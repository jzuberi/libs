from pydantic import create_model

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
    def __init__(self, backend, parser=None, timeout: int = 10, debug: bool = True):
        self.backend = backend
        self.parser = parser or DefaultJSONParser()
        self.timeout = timeout
        self.debug = debug

    # -------------------------
    # INTERNAL DEBUG HELPER
    # -------------------------
    def _debug(self, label: str, value):
        if self.debug:
            print(f"[LLM DEBUG] {label}:\n{value}\n")


    # -------------------------
    # INTERNAL BACKEND CALL
    # -------------------------
    

    def _call_backend(self, prompt: str) -> str:
        if self.debug:
            logger.debug(f"PROMPT:\n{prompt}")

        runner = TimeoutRunnable(self.backend.generate, timeout=self.timeout)
        raw = runner(prompt)

        if self.debug:
            logger.debug(f"RAW OUTPUT:\n{raw}")

        return raw


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
        """
        Dynamic metadata extractor.
        The model may return ANY JSON keys.
        We parse them and build a dynamic Pydantic model.
        """

        prompt = f"""
        You are a careful and precise metadata generator.

        You MUST return valid JSON.
        The JSON may contain ANY keys that you believe represent metadata.
        Do NOT include explanations, preamble, or commentary.

        Text:
        {text}
        """

        raw = self._call_backend(prompt)

        # No expected keys → parser returns whatever JSON it finds
        parsed = self.parser.parse(raw)

        # Build a dynamic Pydantic model with fields inferred from the JSON
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
        """
        Extract specific fields from text.
        Returns a dynamic Pydantic model.
        """

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

        # Create a dynamic schema model
        DynamicSchema = create_model(
            "DynamicExtractionSchema",
            **{field: (str | None, None) for field in fields},
        )

        return DynamicSchema(**parsed)
