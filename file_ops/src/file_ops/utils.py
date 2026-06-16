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


def list_files(
    directory: Union[str, Path],
    remove_hidden: bool = True,
    extensions: Iterable[str] | None = None,
) -> list[Path]:
    """
    List files in a directory with optional filtering.

    Args:
        directory: Path to the directory.
        remove_hidden: If True, exclude hidden files (Unix-style).
        extensions: Optional iterable of extensions to include (e.g. [".json", ".yaml"]).

    Returns:
        A list of Path objects.
    """
    dir_path = ensure_path(directory)

    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    files = []

    for p in dir_path.iterdir():
        if not p.is_file():
            continue

        # Remove hidden files
        if remove_hidden and is_hidden(p):
            continue

        # Filter by extension
        if extensions is not None:
            if p.suffix.lower() not in {ext.lower() for ext in extensions}:
                continue

        files.append(p)

    return sorted(files)

def list_dirs(
    directory: Union[str, Path],
    remove_hidden: bool = True,
    name_contains: str | None = None,
) -> list[Path]:
    """
    List subdirectories in a directory with optional filtering.

    Args:
        directory: Path to the directory.
        remove_hidden: If True, exclude hidden directories (Unix-style).
        name_contains: Optional substring filter for directory names.

    Returns:
        A sorted list of Path objects representing subdirectories.
    """
    dir_path = ensure_path(directory)

    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    dirs = []

    for p in dir_path.iterdir():
        if not p.is_dir():
            continue

        # Remove hidden dirs
        if remove_hidden and is_hidden(p):
            continue

        # Optional substring filter
        if name_contains is not None:
            if name_contains.lower() not in p.name.lower():
                continue

        dirs.append(p)

    return sorted(dirs)
