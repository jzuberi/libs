def resolve_item(engine, intent_data):
    """
    Resolve item_id using:
    1. direct match
    2. metadata match
    3. fuzzy match
    4. LLM match
    """
    # 1. direct match
    if intent_data.item_id and intent_data.item_id in engine._items:
        return intent_data.item_id, f"Resolved item {intent_data.item_id}"

    # 2. metadata match
    candidates = []
    for item in engine.list_items():
        if intent_data.item_id and intent_data.item_id.lower() in item.description.lower():
            candidates.append(item.id)

    if len(candidates) == 1:
        return candidates[0], f"Matched by description."

    # 3. fuzzy match
    # (use rapidfuzz or your own scoring)
    # ...

    # 4. LLM match
    if hasattr(engine, "agent_llm") and engine.agent_llm:
        prompt = f"""
        User referred to: {intent_data.item_id}
        Items: {[item.description for item in engine.list_items()]}
        Which item is the user referring to?
        Respond ONLY with valid JSON. 
        Key is item_id. 
        Do not include commentary.
        """
        result = engine.agent_llm.metadata(prompt).dict()
        if isinstance(result, dict) and result.get("item_id"):
            return result["item_id"], "Resolved via LLM."

    return None, "I couldn't determine which item you meant."
