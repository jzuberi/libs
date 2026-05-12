from pydantic import BaseModel
from typing import Dict, List, Any, Type

from ...engine.models import HandlerMessage
from .agentic_edit import *

class ChoiceConfig(BaseModel):
    step_name: str

    # Candidate domain
    browse_source_key: str | None
    browse_results_key: str
    selection_key: str
    approval_metadata_key: str

    # Optional external persistence
    data_adapter: Any = None
        
        
def make_choice_handlers(config: ChoiceConfig):

    # -----------------------------
    # Helper: load candidate items
    # -----------------------------
    def load_items(agent):
        # 1. External loader
        if hasattr(config.data_adapter, "load_all"):
            try:
                return config.data_adapter.load_all()
            except Exception:
                pass

        # 2. Internal dataset
        if config.browse_source_key is not None:
            return agent.session.context.get(config.browse_source_key) or []

        return []

    # -----------------------------
    # Helper: get identifying text
    # -----------------------------
    def identifying_text(obj):
        # Preferred: model-defined method
        if hasattr(obj, "identifying_text") and callable(obj.identifying_text):
            return obj.identifying_text().lower()

        # Fallback: concatenate all string fields
        try:
            return " ".join(
                str(v).lower()
                for v in obj.__dict__.values()
                if isinstance(v, str)
            )
        except Exception:
            return str(getattr(obj, "id", "")).lower()

    # -----------------------------
    # SEARCH
    # -----------------------------
    def search_handler(agent, intent, item_id, resolution_msg):
        query = intent.parameters.get("query", "").lower()
        items = load_items(agent)

        filtered = [i for i in items if query in identifying_text(i)]

        if not filtered:
            msg = HandlerMessage(
                title="No Matches Found",
                body=f"No items matched '{query}'.",
                success=False
            )
            return {config.browse_results_key: []}, msg

        filtered = filtered[:10]

        bullets = [
            f"{identifying_text(obj)}"
            for obj in filtered
        ]

        msg = HandlerMessage(
            title=f"Found {len(filtered)} Matches",
            body=f"Search query: **{query}**",
            bullets=bullets,
            footer="You can now select a specific item or search again"
        )

        return {config.browse_results_key: filtered}, msg

    # -----------------------------
    # SELECT
    # -----------------------------
    def select_handler(agent, intent, item_id, resolution_msg):
        ref = intent.parameters.get("reference", "").lower().strip()

        items = load_items(agent)

        # Restrict to search results if present
        results = agent.session.context.get(config.browse_results_key) or []
        if results:
            items = [i for i in items if i in results]

        matched = [i for i in items if ref in identifying_text(i)]

        if not matched:
            msg = HandlerMessage(
                title="No Match",
                body=f"No item matched '{ref}'.",
                success=False
            )
            return {config.selection_key: None}, msg

        if len(matched) > 1:
            bullets = [f"{obj.id} — {identifying_text(obj)}" for obj in matched[:10]]
            msg = HandlerMessage(
                title="Multiple Matches",
                body=f"Your reference '{ref}' matched multiple items.",
                bullets=bullets,
                footer="Try being more specific."
            )
            return {config.selection_key: None}, msg

        obj = matched[0]

        msg = HandlerMessage(
            title="Item Selected",
            bullets=[f"ID: {obj.id}", f"Text: {identifying_text(obj)}"],
            footer="You can now approve the selected item to move to the next step or search again"
        )

        return {config.selection_key: obj.id}, msg

    # -----------------------------
    # APPROVE
    # -----------------------------
    def approve_handler(agent, intent, item_id, resolution_msg):
        selected_id = agent.session.context.get(config.selection_key)
        if not selected_id:
            msg = HandlerMessage(
                title="Nothing Selected",
                body="You must select an item before approving.",
                success=False
            )
            return {}, msg

        item = agent.engine.load_item(item_id)

        # Write durable approval
        item.metadata[config.approval_metadata_key] = selected_id
        agent.engine.save_item(item)

        # External adapter approval (optional)
        if hasattr(config.data_adapter, "approve"):
            try:
                config.data_adapter.approve(selected_id)
            except Exception:
                pass

        agent.engine.approve_substate(item_id)

        msg = HandlerMessage(
            title="Approved",
            body=f"Item **{selected_id}** has been approved.",
            footer="Moving to the next step."
        )

        return {}, msg

    return {
        "search": search_handler,
        "select": select_handler,
        "approve": approve_handler,
    }




class EditConfig(BaseModel):
    step_name: str

    # Artifact domain
    artifact_step_name: str
    artifact_ontology: Any
    model_cls: Type[BaseModel]
    edit_key: str
    approval_metadata_key: str

def make_edit_handlers(config: EditConfig):

    # -----------------------------
    # DETAIL
    # -----------------------------
    def detail_handler(agent, intent, item_id, resolution_msg):
        item = agent.engine.load_item(item_id)
        record = item.step_outputs.get(config.artifact_step_name)

        if not record or not record.current:
            msg = HandlerMessage(
                title="No Artifact Available",
                body="There is no artifact to display.",
                success=False
            )
            return {}, msg

        artifact = record.current

        bullets = []
        for key, value in artifact.items():
            if isinstance(value, dict):
                bullets.append(f"**{key}**:")
                for subk, subv in value.items():
                    bullets.append(f"  - {subk}: {subv}")
            else:
                bullets.append(f"**{key}**: {value}")

        msg = HandlerMessage(
            title="Current Artifact Details",
            body="Here is the full artifact as it currently stands:",
            bullets=bullets,
            footer="You can edit any field by saying something like: *change the description*."
        )

        return {}, msg

    # -----------------------------
    # RESET
    # -----------------------------
    def reset_handler(agent, intent, item_id, resolution_msg):
        """
        Reset the artifact back to its original raw value.
        """
        item = agent.engine.load_item(item_id)
        record = item.step_outputs.get(config.artifact_step_name)

        if not record or not record.raw:
            msg = HandlerMessage(
                title="Nothing to Reset",
                body="No original artifact is available to restore.",
                success=False
            )
            return {}, msg

        # record.raw may be a JSON string or a dict
        raw_value = record.raw
        if isinstance(raw_value, str):
            try:
                raw_value = json.loads(raw_value)
            except Exception:
                # If it's not JSON, keep it as-is
                pass

        # Set current = raw
        record.current = raw_value
        agent.engine.save_item(item)

        bullets = []
        for k, v in raw_value.items() if isinstance(raw_value, dict) else []:
            bullets.append(f"**{k}**: {v}")

        msg = HandlerMessage(
            title="Artifact Reset",
            body="The artifact has been restored to its original state.",
            bullets=bullets,
            footer="You can now edit it again or approve it."
        )

        return {config.edit_key: raw_value}, msg

    # -----------------------------
    # EDIT
    # -----------------------------
    def edit_handler(agent, intent, item_id, resolution_msg):
        user_message = intent.user_message

        item = agent.engine.load_item(item_id)
        record = item.step_outputs.get(config.artifact_step_name)

        if not record or not record.current:
            msg = HandlerMessage(
                title="Nothing to Edit",
                body="No artifact is available for editing.",
                success=False
            )
            return {}, msg

        artifact = record.current

        edit_result = resolve_local_edit(
            ontology=config.artifact_ontology,
            obj=artifact,
            user_message=user_message,
            llm=lambda p: agent.engine.agent_llm.general_answer(p).answer,
        )

        if not edit_result.success:
            msg = HandlerMessage(
                title="Edit Not Applied",
                body=edit_result.user_friendly_message,
                success=False
            )
            return {}, msg

        edits = build_edits_from_edit_result(artifact, edit_result)

        updated = agent.engine.edit_step_output(
            item_id=item_id,
            step_name=config.artifact_step_name,
            edits=edits,
        )

        field = list(edits.keys())[0]
        old_value = artifact[field]
        new_value = edits[field]

        msg = HandlerMessage(
            title="Metadata Updated",
            body="Your changes have been applied.",
            bullets=[
                f"Field: **{field}**",
                f"Old: {old_value}",
                f"New: {new_value}",
            ],
            footer="You can continue editing or say *approve this* to finalize."
        )

        return {config.edit_key: updated}, msg

    # -----------------------------
    # APPROVE
    # -----------------------------
    def approve_handler(agent, intent, item_id, resolution_msg):
        item = agent.engine.load_item(item_id)
        record = item.step_outputs.get(config.artifact_step_name)

        if not record or not record.current:
            msg = HandlerMessage(
                title="Nothing to Approve",
                body="No artifact is available for approval.",
                success=False
            )
            return {}, msg

        artifact = record.current

        # 1. Validate with Pydantic model
        try:

            model = config.model_cls(**artifact)

        except Exception as e:

            msg = HandlerMessage(
                title="Invalid Artifact",
                body="The artifact does not satisfy required constraints.",
                bullets=[str(e)],
                success=False
            )
            return {'blocked':True}, msg

        # 3. Save approved artifact
        item.metadata[config.approval_metadata_key] = artifact
        agent.engine.save_item(item)
        agent.engine.approve_substate(item_id)

        msg = HandlerMessage(
            title="Approved",
            body="Your edit has been approved.",
            bullets=[f"Approved key: **{config.approval_metadata_key}**"],
            footer="Moving to the next step."
        )

        return {}, msg


    return {
        "edit": edit_handler,
        "approve": approve_handler,
        "detail": detail_handler,
        "reset": reset_handler,  
    }
