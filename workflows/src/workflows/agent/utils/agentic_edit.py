from typing import Type, Dict, List, Optional, Any, Callable
from pydantic import BaseModel

import json


VALIDATION_HANDLERS = {}

def register_validation_handler(name):
    def decorator(fn):
        VALIDATION_HANDLERS[name] = fn
        return fn
    return decorator


class LocalFieldOntology(BaseModel):
    """
    Semantic description of a single field in a local object.

    This is intentionally *semantic only*:
    - No type information (inferred from the actual object)
    - No nested paths (flat field names only)
    """
    editable: bool = False

    # Human-readable description of what this field represents.
    description: Optional[str] = None

    # Natural language description of what qualifies as a valid value.
    # This will be used in validation prompts.
    qualified_values: Optional[str] = None

    # Natural language aliases for this field (e.g., "summary" for "description").
    aliases: List[str] = []

    validation_handler: Optional[str] = None


class LocalOntology(BaseModel):
    """
    Local, workflow-scoped ontology for a single object.

    Keys in `fields` must correspond to keys in the actual object dict.
    """
    fields: Dict[str, LocalFieldOntology]


class EditResult(BaseModel):
    """
    Structured result of resolving a user edit request.

    This is what `resolve_local_edit` will return.
    """
    success: bool

    # Name of the field being edited (must be a key in LocalOntology.fields).
    field: Optional[str] = None

    # Name of the transformation to apply (e.g., "replace", "edit_value", "remove_key").
    transformation: Optional[str] = None

    # The value to apply for the transformation (semantics depend on transformation).
    value: Optional[Any] = None

    # Internal error detail (for logs/debugging).
    error: Optional[str] = None

    # User-facing message explaining what happened or what went wrong.
    user_friendly_message: Optional[str] = None

def validate_structure(
    ontology: LocalOntology,
    obj: dict
) -> Optional[str]:
    """
    Deterministic structural validation before any LLM involvement.

    Returns:
        None if valid
        user-friendly error message if invalid
    """

    # 1. Ontology fields must exist in the object
    for field_name, meta in ontology.fields.items():
        if field_name not in obj:
            return (
                f"The field '{field_name}' is defined in the ontology but "
                "is missing from the object being edited."
            )

        value = obj[field_name]

        # 2. Editable fields must not be None
        if meta.editable and value is None:
            return (
                f"The field '{field_name}' is marked editable but has no value "
                "in the object. Cannot proceed with editing."
            )

        # 3. Editable fields must be of supported types
        if meta.editable:
            if not isinstance(value, (str, dict, bool, BaseModel)):
                return (
                    f"The field '{field_name}' has an unsupported type "
                    f"({type(value).__name__}). Only string, dict, and boolean "
                    "fields can be edited."
                )

        # 4. No nested paths allowed
        if "." in field_name:
            return (
                f"The field '{field_name}' contains a nested path, which is "
                "not supported in local ontology editing."
            )

    # 5. No extra ontology fields that don't exist in the object
    # (Already covered by #1)

    return None



# Assume you have an LLM call helper:
# llm(prompt: str) -> str


def resolve_field(
    ontology: LocalOntology,
    user_message: str,
    llm
) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve which field the user intends to edit.

    Returns:
        (field_name, error_message)
        - field_name: str if resolved, else None
        - error_message: user-friendly message if resolution failed
    """

    # Build candidate list for the prompt
    field_entries = []
    for field_name, meta in ontology.fields.items():

        if(meta.editable is True):
            entry = {
                "field": field_name,
                "aliases": meta.aliases,
                "description": meta.description,
                "editable": meta.editable,
            }
            field_entries.append(entry)

    # LLM prompt
    prompt = f"""
You are helping to resolve which field of an object the user wants to edit.

Here are the editable fields:

{json.dumps(field_entries, indent=2)}

User request:
"{user_message}"

Instructions:
- Choose exactly ONE field name from the list above.
- Base your choice on field names, aliases, and descriptions.
- If the user request does not clearly refer to any field, return "none".
- Return ONLY the field name or "none".
"""

    print('resolve field prompt')
    print(prompt)


    raw = llm(prompt).strip().strip('"').strip("'")

    print('resolve field answer')
    print(raw)

    # Normalize
    raw_lower = raw.lower()

    # If LLM says "none"
    if raw_lower in ("none", "null", "no field", "unknown"):
        return None, (
            "I couldn't tell which part of the idea you want to edit. "
            "Try referring to a specific part like the summary or the speakers."
        )

    # Validate against ontology
    if raw not in ontology.fields:
        return None, (
            f"I couldn't match '{raw}' to any editable field. "
            "Try using a clearer description or one of the known fields."
        )

    # Check editability
    if not ontology.fields[raw].editable:
        return None, (
            f"You can't edit '{raw}'. "
            "Try editing one of the editable fields instead."
        )

    return raw, None

def resolve_transformation(
    field_name: str,
    ontology: LocalOntology,
    obj: dict,
    user_message: str,
    llm
) -> tuple[Optional[str], Optional[str]]:
    """
    Determine which transformation the user intends to apply.

    Returns:
        (transformation_name, error_message)
    """

    # If the field is a Pydantic model, skip transformation logic.
    value = obj[field_name]
    if isinstance(value, BaseModel):
        return "custom", None


    # 1. Infer type → allowed transformations
    if isinstance(value, str):
        allowed = ["replace"]
    elif isinstance(value, dict):
        allowed = ["edit_value", "remove_key", "rename_key"]
    elif isinstance(value, bool):
        allowed = ["toggle"]
    else:
        return None, (
            f"The field '{field_name}' has an unsupported type "
            f"({type(value).__name__})."
        )

    # 2. Deterministic shortcuts
    msg = user_message.lower()

    if "toggle" in msg and "toggle" in allowed:
        return "toggle", None

    if "remove" in msg or "delete" in msg:
        if "remove_key" in allowed:
            return "remove_key", None

    if "rename" in msg:
        if "rename_key" in allowed:
            return "rename_key", None

    if any(word in msg for word in ["replace", "rewrite", "shorten", "fix", "edit"]):
        if "replace" in allowed:
            return "replace", None
        if "edit_value" in allowed:
            return "edit_value", None


    # 4. LLM fallback
    field_desc = ontology.fields[field_name].description or ""

    prompt = f"""
    You are determining what kind of edit the user wants to perform.

    Field: {field_name}
    Description: {field_desc}

    Allowed transformations: {allowed}

    User request:
    "{user_message}"

    Instructions:
    - Choose exactly ONE transformation from the allowed list.
    - Return ONLY the transformation name.
    - If the request is unclear or clearly not in the allowed list, return "none".
    """

    raw = llm(prompt).strip().strip('"').strip("'")

    if raw.lower() in ("none", "unknown", "null"):
        return None, (
            "I couldn't determine what kind of edit you want to make. "
            f"Try being more specific about how you want to change '{field_name}'."
        )

    if raw not in allowed:
        return None, (
            f"The requested edit doesn't match any valid transformation for '{field_name}'."
        )

    return raw, None


def extract_value_for_string(current_value: str, user_message: str, llm):
    """
    Extract or generate a new string value for a string field.
    Always uses the LLM for extraction or rewrite.
    """

    prompt = f"""
    The user wants to update a string field.

    Current value:
    {current_value}

    User request:
    "{user_message}"

    Your task:
    - If the user clearly provides a new value (e.g., inside quotes or directly stated),
    extract that new value.
    - Otherwise, rewrite the current value according to the user's request.
    - Return ONLY the new value, with no explanation or commentary.

    Examples:
    Update the 'name' field to this: 'Run123/Big Fun'. new value: 'Run123/Big Fun'
    Let's adjust the filepath to /User/chat.jamesz/234902/projects/chicken. new value: '/User/chat.jamesz/234902/projects/chicken'
    Make the description shorter. new value: [process the current value to make it shorter]

    - Return ONLY the new value, with no explanation or commentary.
    """

    new_val = llm(prompt).strip()
    if not new_val:
        return None, "I couldn't generate a new value for this field."

    return new_val, None


def extract_value_for_dict_edit(current_dict: dict, user_message: str, llm):
    """
    Extract (key, new_value) for editing a dict field.
    """

    keys = list(current_dict.keys())

    # 1. Ask LLM which key is being edited
    prompt_key = f"""
You are choosing which key in a dictionary the user wants to edit.

Keys: {keys}

User request:
"{user_message}"

Return ONLY one key from the list, or "none" if unclear.
"""
    raw_key = llm(prompt_key).strip().strip('"').strip("'")

    if raw_key.lower() in ("none", "unknown", "null"):
        return None, "I couldn't determine which part of this field you want to edit."

    if raw_key not in current_dict:
        return None, f"I couldn't match '{raw_key}' to any key in this field."

    key = raw_key
    current_value = current_dict[key]

    # 2. Extract new value for that key
    prompt_val = f"""
The user wants to update the value for the key '{key}'.

Current value:
{current_value}

User request:
"{user_message}"

Your task:
- Extract the new intended value for this key.
- If the user does not provide a new value explicitly, rewrite the current value.
- Return ONLY the new value.
"""
    new_val = llm(prompt_val).strip()

    if not new_val:
        return None, f"I couldn't determine the new value for '{key}'."

    return (key, new_val), None


def extract_value_for_dict_remove(current_dict: dict, user_message: str, llm):
    """
    Extract the key to remove from a dict.
    """

    keys = list(current_dict.keys())

    prompt = f"""
Choose which key the user wants to remove.

Keys: {keys}

User request:
"{user_message}"

Return ONLY the key name or "none".
"""
    raw = llm(prompt).strip().strip('"').strip("'")

    if raw.lower() in ("none", "unknown", "null"):
        return None, "I couldn't determine which key you want to remove."

    if raw not in current_dict:
        return None, f"'{raw}' is not a valid key."

    return raw, None


def extract_value_for_dict_rename(current_dict: dict, user_message: str, llm):
    """
    Extract (old_key, new_key) for renaming a dict key.
    """

    keys = list(current_dict.keys())

    # 1. Determine old key
    prompt_old = f"""
Choose which key the user wants to rename.

Keys: {keys}

User request:
"{user_message}"

Return ONLY the key name or "none".
"""
    raw_old = llm(prompt_old).strip().strip('"').strip("'")

    if raw_old.lower() in ("none", "unknown", "null"):
        return None, "I couldn't determine which key you want to rename."

    if raw_old not in current_dict:
        return None, f"'{raw_old}' is not a valid key."

    old_key = raw_old

    # 2. Extract new key name
    prompt_new = f"""
The user wants to rename the key '{old_key}'.

User request:
"{user_message}"

Return ONLY the new key name (a single word or phrase).
"""
    new_key = llm(prompt_new).strip().strip('"').strip("'")

    if not new_key:
        return None, "I couldn't determine the new key name."

    return (old_key, new_key), None


def extract_value_for_bool_toggle(current_value: bool):
    """
    Toggle a boolean value.
    """
    return not current_value, None


def validate_semantics(field_name, new_value, transformation, ontology, llm):
    """
    Semantic validation using ontology descriptions and qualified_values.

    Returns:
        None if valid
        user-friendly error message if invalid
    """

    field_meta = ontology.fields[field_name]

    # -----------------------------------------
    # 0. Skip semantic validation for structured fields
    # -----------------------------------------
    # If the field value is a Pydantic model, semantic validation does not apply.
    # Custom validation handlers are responsible for validating structured fields.
    if isinstance(new_value, BaseModel):
        return None

    # -----------------------------------------
    # 1. Transformations that should NOT be validated
    # -----------------------------------------

    # Removing a key does not change the field's semantic type
    if transformation == "remove_key":
        return None

    # Renaming a key does not change the semantic type
    if transformation == "rename_key":
        return None

    # Toggling a boolean is always valid
    if transformation == "toggle":
        return None

    # -----------------------------------------
    # 2. Extract the actual value to validate
    # -----------------------------------------

    # edit_value: (key, new_val)
    if transformation == "edit_value":
        _, actual_value = new_value
    else:
        actual_value = new_value

    # -----------------------------------------
    # 3. LLM semantic validation
    # -----------------------------------------

    desc = field_meta.description or ""
    qualified = field_meta.qualified_values or ""

    prompt = f"""
You are validating whether a proposed new value fits the meaning of a field.

Field name: {field_name}
Field description: {desc}
Qualified values: {qualified}

Proposed new value:
{actual_value}

Instructions:
- Answer "yes" if the new value fits the field's meaning.
- Answer "no" if it does not.
- Return ONLY "yes" or "no".
"""

    raw = llm(prompt).strip().lower()

    if raw not in ("yes", "no"):
        return (
            f"I'm not confident the new value fits the meaning of '{field_name}'. "
            "Try rephrasing your edit."
        )

    if raw == "no":
        return (
            f"The new value doesn't seem to fit the meaning of '{field_name}'. "
            f"Try providing a value that matches the field's description:{desc}."
        )

    return None



def resolve_local_edit(
    ontology: LocalOntology,
    obj: dict,
    user_message: str,
    llm: Callable[[str], str],
) -> EditResult:
    """
    Main entry point for resolving a user edit request.
    """

    # -----------------------------
    # Step 0: Structural validation
    # -----------------------------
    error = validate_structure(ontology, obj)
    if error:
        return EditResult(
            success=False,
            error=error,
            user_friendly_message=error,
        )

    # -----------------------------
    # Step 1: Field resolution
    # -----------------------------
    field, error = resolve_field(ontology, user_message, llm)
    if error:
        return EditResult(
            success=False,
            error=error,
            user_friendly_message=error,
        )

    # -----------------------------
    # Step 2: Transformation resolution
    # -----------------------------
    transformation, error = resolve_transformation(
        field, ontology, obj, user_message, llm
    )
    if error:
        return EditResult(
            success=False,
            error=error,
            user_friendly_message=error,
        )

    # -----------------------------
    # Step 3: Value extraction
    # -----------------------------
    current_value = obj[field]

    if transformation == "replace":
        new_value, error = extract_value_for_string(current_value, user_message, llm)

    elif transformation == "edit_value":
        new_value, error = extract_value_for_dict_edit(current_value, user_message, llm)

    elif transformation == "remove_key":
        new_value, error = extract_value_for_dict_remove(current_value, user_message, llm)

    elif transformation == "rename_key":
        new_value, error = extract_value_for_dict_rename(current_value, user_message, llm)

    elif transformation == "toggle":
        new_value, error = extract_value_for_bool_toggle(current_value)

    elif transformation == "custom":
        # For structured fields, the step-specific logic should have already
        # produced the new_value externally. Here we simply pass through.
        new_value = obj[field]  # or whatever your step logic sets
        error = None


    else:
        return EditResult(
            success=False,
            error=f"Unknown transformation '{transformation}'.",
            user_friendly_message="I couldn't understand the requested edit.",
        )

    if error:
        return EditResult(
            success=False,
            error=error,
            user_friendly_message=error,
        )

    # -----------------------------
    # Step 4: Semantic validation
    # -----------------------------
    error = validate_semantics(field, new_value, transformation, ontology, llm)
    if error:
        return EditResult(
            success=False,
            error=error,
            user_friendly_message=error,
        )

    # -----------------------------
    # Step 4.5: Custom validation handler
    # -----------------------------
    ### NEW ###
    ontology_field = ontology.fields[field]
    if ontology_field.validation_handler:
        handler = VALIDATION_HANDLERS.get(ontology_field.validation_handler)
        if handler is None:

            return EditResult(
                success=False,
                error=f"Unknown validation handler '{ontology_field.validation_handler}'.",
                user_friendly_message="Internal validation error.",
            )

        # Build the candidate new object for validation
        candidate_obj = obj.copy()
        candidate_obj[field] = new_value

        ok, err = handler(new_value, candidate_obj, obj)
        if not ok:
            return EditResult(
                success=False,
                error=err,
                user_friendly_message=err,
            )
    ### END NEW ###

    # -----------------------------
    # Step 5: Success
    # -----------------------------
    return EditResult(
        success=True,
        field=field,
        transformation=transformation,
        value=new_value,
    )

def build_edits_from_edit_result(metadata: dict, edit_result):
    """
    Convert a resolve_local_edit() result into an edits dict
    suitable for edit_step_output().
    """

    field = edit_result.field
    transformation = edit_result.transformation
    value = edit_result.value

    if field not in metadata:
        raise ValueError(f"Field '{field}' not found in metadata.")

    current_value = metadata[field]

    # STRING FIELD
    if transformation == "replace":
        return {field: value}

    # DICT FIELD: edit a single key's value
    if transformation == "edit_value":
        key, new_val = value
        new_dict = dict(current_value)
        new_dict[key] = new_val
        return {field: new_dict}

    # DICT FIELD: remove a key
    if transformation == "remove_key":
        key = value
        new_dict = {k: v for k, v in current_value.items() if k != key}
        return {field: new_dict}

    # DICT FIELD: rename a key
    if transformation == "rename_key":
        old_key, new_key = value
        new_dict = dict(current_value)
        new_dict[new_key] = new_dict.pop(old_key)
        return {field: new_dict}

    # BOOLEAN FIELD
    if transformation == "toggle":
        return {field: value}

    raise ValueError(f"Unknown transformation '{transformation}'.")

def ontology_from_model(
    model_cls: Type[BaseModel],
    overrides: Dict[str, Dict[str, Any]] | None = None
) -> LocalOntology:
    """
    Build a LocalOntology from a Pydantic model class, with optional overrides.

    - Every field in the Pydantic model becomes a LocalFieldOntology entry.
    - Defaults: editable=False, description from field metadata, aliases=[field_name]
    - Overrides can specify: editable, description, qualified_values, aliases,
      validation_handler.
    """

    fields = {}

    for name, field in model_cls.model_fields.items():
        # Default ontology entry
        fields[name] = LocalFieldOntology(
            editable=False,
            description=field.description,
            qualified_values=None,
            aliases=[name],
            validation_handler=None,   # <-- ensure default is present
        )

    # Apply overrides
    if overrides:
        for field_name, override in overrides.items():
            if field_name not in fields:
                raise ValueError(
                    f"Override provided for unknown field '{field_name}' "
                    f"in model {model_cls.__name__}"
                )

            # model_copy(update=...) already handles all override keys,
            # including validation_handler, so no special logic needed.
            fields[field_name] = fields[field_name].model_copy(update=override)

    return LocalOntology(fields=fields)

