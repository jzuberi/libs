# workflows/agent/context/decorators.py

from functools import wraps
from ...engine.models import HandlerMessage

def updates_context(mapping, set_current_item=False):
    """
    mapping = {
        "context_key": ("raw_key", normalizer)
    }

    Supports:
    - list values
    - scalar values
    - None
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(agent, intent, item_id, resolution_msg):

            raw_map, user_output = fn(agent, intent, item_id, resolution_msg)

            # ----------------------------------------------------
            # NEW: Structured HandlerMessage support
            # ----------------------------------------------------
            if isinstance(user_output, HandlerMessage):
                user_output = user_output.render()

            # ----------------------------------------------------
            # Context updates
            # ----------------------------------------------------
            for ns, (raw_key, normalizer) in mapping.items():

                value = raw_map.get(raw_key)

                # Case 1: None → store None
                if value is None:
                    agent.session.context[ns] = None

                # Case 2: List → normalize each element
                elif isinstance(value, list):
                    agent.session.context[ns] = [
                        normalizer(v) for v in value
                    ]

                # Case 3: Scalar → normalize once
                else:
                    agent.session.context[ns] = normalizer(value)

                # Track update turn
                if "_updated_turn" not in agent.session.context:
                    agent.session.context["_updated_turn"] = {}

                agent.session.context["_updated_turn"][ns] = agent.session.turn_index

            if set_current_item:
                agent.session.last_item_id = item_id

            return raw_map, user_output

        return wrapper
    return decorator
