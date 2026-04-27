# ontologies/ontology.py

from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel


class OntologyType:
    """
    Describes a single object type in a workflow ontology.

    - model: the Pydantic class (e.g., Idea, AssetDefinition)
    - description: human-readable summary of what this object represents
    - fields: which fields are relevant to expose to the LLM
    - metadata: arbitrary extra info (e.g., display hints, relations, tags)
      Common keys we'll use:
        - session_key: where instances live in session.context (e.g. "idea_results")
        - context_key: how they appear in relevant_context (e.g. "ideas")
    """

    def __init__(
        self,
        *,
        name: str,
        model: Type[BaseModel],
        description: str,
        fields: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.model = model
        self.description = description
        self.fields = fields
        self.metadata: Dict[str, Any] = metadata or {}

    def serialize(self, obj: BaseModel) -> Dict[str, Any]:
        """
        Minimal, field-filtered serialization for LLM prompts.
        """
        data = obj.dict()
        return {field: data.get(field) for field in self.fields}


class WorkflowOntologyRegistry:
    """
    Registry for object types that exist in the workflow world.

    Intentionally named 'WorkflowOntologyRegistry' so we can
    introduce other ontologies later (e.g., org-wide, domain-wide).
    """

    def __init__(self):
        self._types: Dict[str, OntologyType] = {}
        # Fast lookup by model type
        self._by_model: Dict[Type[BaseModel], OntologyType] = {}

    def register(
        self,
        *,
        name: str,
        model: Type[BaseModel],
        description: str,
        fields: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ot = OntologyType(
            name=name,
            model=model,
            description=description,
            fields=fields,
            metadata=metadata,
        )
        self._types[name] = ot
        self._by_model[model] = ot

    def get(self, name: str) -> Optional[OntologyType]:
        return self._types.get(name)

    def find_type_for_instance(self, obj: Any) -> Optional[OntologyType]:
        """
        Given an object instance, return the matching OntologyType if any.
        """
        return self._by_model.get(type(obj))

    def all_types(self) -> List[OntologyType]:
        return list(self._types.values())

    def find_by_session_key(self, session_key: str) -> List[OntologyType]:
        """
        Return all ontology types whose metadata.session_key matches
        the given session context key (e.g., "idea_results").
        """
        return [
            ot
            for ot in self._types.values()
            if ot.metadata.get("session_key") == session_key
        ]
