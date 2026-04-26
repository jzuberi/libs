# file_ops/src/file_ops/utils.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Any, Iterable, Union

import json

# Optional dependencies
try:
    import yaml
except ImportError:
    yaml = None

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None


# ------------------------------------------------------------
# PATH HELPERS
# ------------------------------------------------------------

def ensure_path(path: Union[str, Path]) -> Path:
    """
    Normalize a string or Path into a resolved Path.
    """
    return Path(path).expanduser().resolve()


def is_hidden(path: Union[str, Path]) -> bool:
    """
    Unix-style hidden file check.
    """
    p = Path(path)
    return p.name.startswith(".")


def is_json_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() == ".json"


def is_jsonl_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() == ".jsonl"


def is_yaml_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in {".yaml", ".yml"}


# ------------------------------------------------------------
# JSON SERIALIZATION HELPERS
# ------------------------------------------------------------

def jsonable(obj: Any) -> Any:
    """
    Convert a Pydantic model (or any nested structure containing them)
    into a JSON‑serializable Python object.
    """

    # Pydantic model → dict
    if BaseModel is not None and isinstance(obj, BaseModel):
        return {k: jsonable(v) for k, v in obj.model_dump().items()}

    # Datetime → ISO string
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Enum → value
    if isinstance(obj, Enum):
        return obj.value

    # Path → string
    if isinstance(obj, Path):
        return str(obj)

    # Dict → recursively convert values
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}

    # List / tuple → recursively convert items
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]

    # Primitive → return as‑is
    return obj


# ------------------------------------------------------------
# JSONL HELPERS
# ------------------------------------------------------------

def jsonl_iter(path: Path) -> Iterable[Any]:
    """
    Yield each JSON object from a .jsonl file.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def jsonl_append(path: Path, obj: Any) -> None:
    """
    Append a JSON-serializable object to a .jsonl file.
    """
    serializable = jsonable(obj)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(serializable) + "\n")


# ------------------------------------------------------------
# YAML HELPERS
# ------------------------------------------------------------

def read_yaml(path: Path) -> Any:
    if yaml is None:
        raise ImportError("PyYAML is not installed")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: Any) -> None:
    if yaml is None:
        raise ImportError("PyYAML is not installed")
    serializable = jsonable(data)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False)


# ------------------------------------------------------------
# SAFE WRITE HELPERS
# ------------------------------------------------------------

def atomic_write(path: Path, data: str, encoding: str = "utf-8") -> None:
    """
    Write text to a temporary file and then atomically replace the target.
    Prevents corruption if the process is interrupted.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding=encoding)
    tmp.replace(path)
