# workflows/agent/context/decorators.py

from functools import wraps

def updates_context(mapping, set_current_item=False):
    """
    mapping = {
        "items": ("items", normalize_item),
        "cases": ("cases", normalize_case),
        "steps": ("steps", normalize_step),
    }
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(agent, intent, item_id, resolution_msg):
            raw_map, user_output = fn(agent, intent, item_id, resolution_msg)

            for ns, (raw_key, normalizer) in mapping.items():

                print(raw_map)
                
                raw_list = raw_map.get(raw_key, [])
                agent.session.context[ns] = [
                    normalizer(obj) for obj in raw_list
                ]
                agent.session.context["_updated_turn"][ns] = agent.session.turn_index

            if set_current_item:
                agent.session.context["current_item_id"] = item_id

            return raw_map, user_output

        return wrapper
    return decorator
