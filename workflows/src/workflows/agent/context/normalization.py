# workflows/agent/context/normalization.py

def normalize_item(item):
    return {
        "id": item.id,
        "label": item.label,
        "aliases": [],
        "metadata": {
            "status": getattr(item, "status", None),
        }
    }


def normalize_case(case_name: str, docket_number: str, status: str = None):
    """
    Normalize a case extracted from a step output.
    """
    # Build aliases that help the resolver match user input
    aliases = [
        case_name,
        case_name.replace(",", ""),
        case_name.replace("v.", "v"),
        case_name.replace("v", "v."),
        docket_number,
    ]

    return {
        "id": docket_number,          # canonical ID
        "label": case_name,           # human-readable
        "aliases": aliases,
        "metadata": {
            "status": status,
        }
    }


def normalize_step(step_spec):
    """
    Normalize a WorkflowStepSpec.
    """
    return {
        "id": step_spec.name,
        "label": step_spec.human_name or step_spec.name.replace("_", " ").title(),
        "aliases": [
            step_spec.name,
            step_spec.human_name,
            step_spec.human_name.replace(" ", "").lower() if step_spec.human_name else None,
        ],
        "metadata": {
            "description": step_spec.description,
        }
    }
